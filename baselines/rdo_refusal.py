# %%
import argparse
import copy
import json
import os
import os.path
import sys
import uuid
import dotenv
import torch
import torch.nn as nn
import numpy as np
from nnsight import LanguageModel
from nnsight.envoy import Envoy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import set_seed
import nnsight


from pathlib import Path
import sys
from collections import Counter
from datetime import datetime
_GFR = Path(__file__).resolve().parent.parent / "geometry-of-refusal"
sys.path.insert(0, str(_GFR))

from generate_utils import (projection_einops, 
                            generate_completions,
                            intervene_with_fn_vector_ablation,
                            intervene_with_fn_vector_addition)
from scoring import refusal_metric, get_bypass_scores 

from evaluate.evaluation_llamaguard import get_llamaguard_evaluator, unload_llamaguard_evaluator
from evaluate import mmlu as mmlu_eval

dotenv.load_dotenv(override=True)

# Environment variables
os.environ["SAVE_DIR"] = "./results/rdo_refusal"
os.environ["DIM_DIR"] = "dim"
os.environ["WANDB_ENTITY"] = "refusal-representations"
os.environ["WANDB_PROJECT"] = "refusal_directions"

# Set seed
set_seed(42)

# TensorBoard: set by train_* before nested optimization (used by nnsight.apply in repind_rdo)
_ACTIVE_TB_WRITER = None


def tensorboard_log_scalars(writer, metrics, step):
    """Log scalars and per-index list/tuple values to TensorBoard."""
    for key, val in metrics.items():
        if val is None:
            continue
        if isinstance(val, (bool, torch.Tensor)):
            if isinstance(val, torch.Tensor) and val.numel() == 1:
                writer.add_scalar(key, val.item(), step)
            continue
        if isinstance(val, (int, float)):
            if isinstance(val, float) and val != val:  # NaN
                continue
            writer.add_scalar(key, val, step)
        elif isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v):
                    writer.add_scalar(f"{key}/{i}", v, step)


def save_run_hparams(run_dir, config_dict):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "hparams.json")
    safe = {}
    for k, v in config_dict.items():
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = str(v)
    with open(path, "w") as f:
        json.dump(safe, f, indent=2)


def _tb_log_for_nnsight(metrics, step):
    """Drop-in replacement for wandb.log inside nnsight.apply; uses _ACTIVE_TB_WRITER."""
    w = _ACTIVE_TB_WRITER
    if w is None:
        return
    s = step
    while hasattr(s, "value") and not isinstance(s, int):
        try:
            s = s.value
        except Exception:
            break
    try:
        step_int = int(s)
    except (TypeError, ValueError):
        step_int = 0
    tensorboard_log_scalars(w, metrics, step_int)


# Default configuration values
DEFAULT_CONFIG = {
    # Model settings
    'model': 'google/gemma-2-2b-it',  # Model identifier from HuggingFace
    'dtype': 'bfloat16',              # Floating point precision (bfloat16, float16, float32)
    
    # Training objectives
    'train_direction': False,         # Whether to train a single refusal direction
    'train_orthogonal_direction': False,  # Whether to train a direction orthogonal to the DIM direction
    'train_cone': False,              # Whether to train a refusal cone
    'train_independent_direction': False, # Whether to train a direction that is independent of the DIM direction
    
    # Optimization parameters
    'epochs': 1,                      # Number of training epochs
    'max_iters': 100000,              # Maximum number of iterations to train for
    'lr': 3e-4,                       # Learning rate for optimization
    'batch_size': 1,                  # Batch size for training
    'effective_batch_size': 16,       # Effective batch size (uses gradient accumulation)
    'patience': 5,                    # Patience for early stopping
    'n_lr_reduce': 2,                 # Number of learning rate reductions before stopping
    'direction_mode': 'baseline',    # 'baseline' (RefusalCone) | 'rotation' (Cayley orthogonal M @ r0; cone_dim must be 1)
    
    # Cone parameters
    'min_cone_dim': 2,                # Minimum dimension of the refusal cone (number of basis vectors)
    'max_cone_dim': 3,               # Maximum dimension of the refusal cone (number of basis vectors)
    'n_sample': 8,                    # Number of random samples to use during training
    'fixed_samples': 8,               # Number of fixed samples for evaluation
    'sampling_method': "hypersphere", # Method for sampling vectors ('hypersphere' or 'interpolation')
    'optimize_basis': True,           # Whether to optimize the basis vectors directly
    
    # Loss weights
    'ablation_lambda': 1,             # Weight for the ablation loss
    'addition_lambda': 0.2,           # Weight for the addition loss
    'retain_lambda': 1,               # Weight for the retain loss
    
    # Miscellaneous
    'target_generation_batch_size': 512,  # Batch size for generating targets
    'filter_data': True,              # Whether to filter data
    'filter_batch_size': 32,          # Batch size for filtering data
    'splits': "saladbench",           # Dataset split to use

    # Evaluation
    'eval_llamaguard': False,
    'eval_mmlu': False,
    'eval_split': "val",              # {train,val,test} -> data/{splits}_splits/*_{eval_split}.json
    'eval_max_new_tokens': 256,
    'eval_batch_size': 8,

    # MMLU evaluation (subset of evaluate/mmlu.py config)
    'mmlu_dataset': "cais/mmlu",
    'mmlu_subset': "all",
    'mmlu_mode': "zero_shot",
    'mmlu_answer_mode': "generate",
    'mmlu_n_shots': 5,
    'mmlu_sample_size': 100,
    'mmlu_sample_seed': 42,
    'mmlu_max_new_tokens': 8,
    'mmlu_store_predictions': False,
}

def parse_args():
    """
    Parse command line arguments using the default configuration values.
    
    Returns:
        argparse.Namespace: Parsed arguments with default values if not specified.
    """
    # If running in interactive mode
    if not sys.argv[0].endswith('rdo.py') and not sys.argv[0].endswith('rdo_refusal.py'):
        return argparse.Namespace(**DEFAULT_CONFIG)
    
    # If running from command line
    parser = argparse.ArgumentParser(description='Refusal Direction Optimization (RDO)')
    
    # Model settings
    parser.add_argument('--model', type=str, default=DEFAULT_CONFIG['model'],
                    help='HuggingFace model identifier (e.g., google/gemma-2-2b-it, meta-llama/Llama-3-8B)')
    parser.add_argument('--dtype', type=str, default=DEFAULT_CONFIG['dtype'],
                    choices=['bfloat16', 'float16', 'float32'],
                    help='Floating point precision to use for model initialization')
    
    # Training objectives
    parser.add_argument('--train_direction', action='store_true', 
                    help='Train a standard refusal direction')
    parser.add_argument('--train_orthogonal_direction', action='store_true', 
                    help='Train a direction orthogonal to the DIM direction')
    parser.add_argument('--train_cone', action='store_true', 
                    help='Train a refusal cone (multiple basis vectors)')
    parser.add_argument('--train_independent_direction', action='store_true',
                    help='Train a direction that is independent of the DIM direction')

    # Optimization parameters
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                    help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'],
                    help='Learning rate for optimization')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                    help='Batch size for training')
    parser.add_argument('--effective_batch_size', type=int, default=DEFAULT_CONFIG['effective_batch_size'],
                    help='Effective batch size (uses gradient accumulation)')
    parser.add_argument('--patience', type=int, default=DEFAULT_CONFIG['patience'],
                    help='Patience for early stopping')
    parser.add_argument('--n_lr_reduce', type=int, default=DEFAULT_CONFIG['n_lr_reduce'],
                    help='Number of learning rate reductions before stopping')
    parser.add_argument('--direction_mode', type=str, default=DEFAULT_CONFIG['direction_mode'],
                    choices=['baseline', 'rotation', 'activation_rot', 'shtiefel_rot'],
                    help='baseline: original RefusalCone. rotation: learn orthogonal M (Cayley), r=M@r0; requires cone_dim=1')

    # Cone parameters
    parser.add_argument('--min_cone_dim', type=int, default=DEFAULT_CONFIG['min_cone_dim'],
                    help='Minimum dimension of the refusal cone (number of basis vectors)')
    parser.add_argument('--max_cone_dim', type=int, default=DEFAULT_CONFIG['max_cone_dim'],
                    help='Maximum dimension of the refusal cone (number of basis vectors)')
    parser.add_argument('--n_sample', type=int, default=DEFAULT_CONFIG['n_sample'],
                    help='Number of random samples to use during training')
    parser.add_argument('--fixed_samples', type=int, default=DEFAULT_CONFIG['fixed_samples'],
                    help='Number of fixed samples for evaluation')
    parser.add_argument('--sampling_method', type=str, default=DEFAULT_CONFIG['sampling_method'],
                    choices=['hypersphere', 'interpolation'],
                    help='Method for sampling vectors (hypersphere or interpolation)')
    parser.add_argument('--optimize_basis', type=bool, default=DEFAULT_CONFIG['optimize_basis'],
                    help='Whether to optimize the basis vectors directly')

    # Loss weights
    parser.add_argument('--ablation_lambda', type=float, default=DEFAULT_CONFIG['ablation_lambda'],
                    help='Weight for the ablation loss')
    parser.add_argument('--addition_lambda', type=float, default=DEFAULT_CONFIG['addition_lambda'],
                    help='Weight for the addition loss')
    parser.add_argument('--retain_lambda', type=float, default=DEFAULT_CONFIG['retain_lambda'],
                    help='Weight for the retain loss')
    
    # Miscellaneous
    parser.add_argument('--target_generation_batch_size', type=int, default=DEFAULT_CONFIG['target_generation_batch_size'],
                    help='Batch size for generating targets')
    parser.add_argument('--filter_data', action='store_true',
                    help='Filter data')
    parser.add_argument('--filter_batch_size', type=int, default=DEFAULT_CONFIG['filter_batch_size'],
                    help='Batch size for filtering data')
    parser.add_argument('--splits', type=str, default=DEFAULT_CONFIG['splits'],
                    help='Dataset split to use')

    # Evaluation
    parser.add_argument('--eval_llamaguard', action='store_true',
                    help='Evaluate initial and refined generations with LlamaGuard on harmful/harmless eval split')
    parser.add_argument('--eval_mmlu', action='store_true',
                    help='Evaluate initial and refined generations on MMLU (generate-mode by default)')
    parser.add_argument('--eval_split', type=str, default=DEFAULT_CONFIG['eval_split'],
                    choices=['train', 'val', 'test'],
                    help='Which split jsons to use for LlamaGuard eval: data/{splits}_splits/*_{eval_split}.json')
    parser.add_argument('--eval_max_new_tokens', type=int, default=DEFAULT_CONFIG['eval_max_new_tokens'],
                    help='Max new tokens for generation during LlamaGuard eval')
    parser.add_argument('--eval_batch_size', type=int, default=DEFAULT_CONFIG['eval_batch_size'],
                    help='Batch size for generation during eval')

    # MMLU
    parser.add_argument('--mmlu_dataset', type=str, default=DEFAULT_CONFIG['mmlu_dataset'])
    parser.add_argument('--mmlu_subset', type=str, default=DEFAULT_CONFIG['mmlu_subset'])
    parser.add_argument('--mmlu_mode', type=str, default=DEFAULT_CONFIG['mmlu_mode'],
                    choices=['zero_shot', 'few_shot'])
    parser.add_argument('--mmlu_answer_mode', type=str, default=DEFAULT_CONFIG['mmlu_answer_mode'],
                    choices=['generate'],
                    help='Only generate-mode is supported in rdo_refusal.py')
    parser.add_argument('--mmlu_n_shots', type=int, default=DEFAULT_CONFIG['mmlu_n_shots'])
    parser.add_argument('--mmlu_sample_size', type=int, default=DEFAULT_CONFIG['mmlu_sample_size'])
    parser.add_argument('--mmlu_sample_seed', type=int, default=DEFAULT_CONFIG['mmlu_sample_seed'])
    parser.add_argument('--mmlu_max_new_tokens', type=int, default=DEFAULT_CONFIG['mmlu_max_new_tokens'])
    parser.add_argument('--mmlu_store_predictions', action='store_true',
                    help='Store full MMLU predictions in the output JSON (can be large)')
    
    return parser.parse_args()

args = parse_args()
MODEL_PATH = args.model

# Apply configuration values
target_generation_batch_size = args.target_generation_batch_size
splits = args.splits

# %%
# Convert string dtype to torch dtype
if args.dtype == 'bfloat16':
    dtype = torch.bfloat16
elif args.dtype == 'float16':
    dtype = torch.float16
elif args.dtype == 'float32':
    dtype = torch.float32
else:
    raise ValueError(f"Unsupported dtype: {args.dtype}")

model = LanguageModel(MODEL_PATH, cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"), device_map='auto', torch_dtype=dtype)
model.requires_grad_(False)

# %%
# loading and testing model
with model.trace("Hello") as tracer:
    pass

# %%
model_id = MODEL_PATH.split("/")[-1]
dim_dir_path = f"{os.getenv('SAVE_DIR')}/{os.getenv('DIM_DIR')}/{model_id}"
direction_file = f"{dim_dir_path}/direction.pt"
metadata_file = f"{dim_dir_path}/direction_metadata.json"
mean_diffs_file = f"{dim_dir_path}/generate_directions/mean_diffs.pt"

# Check if DIM direction files exist
if not (os.path.exists(direction_file) and os.path.exists(metadata_file)):
    raise FileNotFoundError(
        "DIM direction files not found. Please compute the DIM directions first as described in the README."
    )

refusal_directions = torch.load(mean_diffs_file)
refusal_results = json.load(open(metadata_file))
best_layer = refusal_results["layer"]
best_token = refusal_results["pos"]
best_refusal_direction = torch.load(direction_file).to(model.dtype)

# %%
SAVE_DIR = f"{os.getenv('SAVE_DIR')}/rdo/{MODEL_PATH.split('/')[-1]}/"
os.makedirs(SAVE_DIR, exist_ok=True)

add_layer = best_layer
alpha = best_refusal_direction.norm().detach().clone()
print(f"add_layer: {add_layer}, alpha: {alpha}")

# %%
harmful_train = json.load(open(f'data/{splits}_splits/harmful_train.json'))
harmless_train = json.load(open(f'data/{splits}_splits/harmless_train.json'))

harmless_train = harmless_train[:len(harmful_train)]
print(len(harmful_train), len(harmless_train))

# %%
GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""
QWEN25_CHAT_TEMPLATE = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def apply_chat_template(tokenizer, instructions: list[str]):
    if 'llama-3' in MODEL_PATH.lower():
        prompts = [LLAMA3_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    elif "gemma" in MODEL_PATH.lower():
        prompts = [GEMMA_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    elif "qwen2.5" in MODEL_PATH.lower() or "distill-qwen" in MODEL_PATH.lower():
        prompts = [QWEN25_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    else:
        raise ValueError(f"Model {MODEL_PATH} not supported, need to configure chat template")
    return prompts

# %%
print("Chat template preview:")
print(apply_chat_template(model.tokenizer, ["Hello"])[0])
# %%
# get first token completion
def generate_first_token(model, dataset, max_new_tokens=1):
    instructions = apply_chat_template(model.tokenizer, [d['instruction'] for d in dataset])
    print(instructions[0])
    decoded = []
    tokens = []
    for instruction in instructions:
        input_tokens = model.tokenizer(instruction, add_special_tokens=True, padding=True, truncation=False)["input_ids"]
        start_token = len(input_tokens)
        with model.generate(instruction, max_new_tokens=max_new_tokens, do_sample=False) as generator:
            out = model.generator.output.save()
        decoded.append(model.tokenizer.decode(out[0][start_token:], skip_special_tokens=True))
        tokens.append(out[0][start_token:])
    return decoded, tokens

example_completions, example_tokens = generate_first_token(model, harmful_train[:10])
print("Example first token completions:")
print(example_completions)
print("Example first tokens:")
print(example_tokens)

# %%
if "gemma" in MODEL_PATH.lower():
    refusal_tokens = [235285]
elif "qwen2.5" in MODEL_PATH.lower() or "distill-qwen" in MODEL_PATH.lower():
    refusal_tokens = [40, 2121]
elif "llama-3" in MODEL_PATH.lower():
    refusal_tokens = [40]
else:
    raise ValueError(f"Model {MODEL_PATH} not supported, need to configure refusal tokens")

# %%
harmful_train_instructions = apply_chat_template(model.tokenizer, [d["instruction"] for d in harmful_train])
harmless_train_instructions = apply_chat_template(model.tokenizer, [d["instruction"] for d in harmless_train])
# %%
module = model.model

# %%
def generate_harmful_targets(model, harmful_instructions, best_refusal_direction, targets_path, max_new_tokens):
    if os.path.exists(targets_path):
        return json.load(open(targets_path))

    print("Generating harmful targets")
        
    ablation_completions = intervene_with_fn_vector_ablation(model, harmful_instructions, best_refusal_direction.to(model.dtype), max_new_tokens=max_new_tokens, batch_size=target_generation_batch_size)
    
    # Create batch of dicts with new targets
    targets = []
    for i, instruction in enumerate(harmful_instructions):
        target_dict = {
            'prompt': instruction,
            'ablation': ablation_completions[i] if ablation_completions else "",
        }
        targets.append(target_dict)

    # Save all targets
    os.makedirs(os.path.dirname(targets_path), exist_ok=True)
    with open(targets_path, 'w') as f:
        json.dump(targets, f)
    
    return targets

def generate_harmless_targets(model, harmless_instructions, targets_path, max_new_tokens):
    if os.path.exists(targets_path):
        return json.load(open(targets_path))

    print("Generating harmless targets")

    addition_completions = intervene_with_fn_vector_addition(model, harmless_instructions, best_layer, best_refusal_direction.norm(), best_refusal_direction, max_new_tokens=max_new_tokens, batch_size=target_generation_batch_size)

    retain_completions = generate_completions(model, harmless_instructions, max_new_tokens=max_new_tokens-1, batch_size=target_generation_batch_size)
    
    targets = []
    for i, instruction in enumerate(harmless_instructions):
        target_dict = {
            'prompt': instruction,
            'addition': addition_completions[i].split(".")[0] if addition_completions else "",
            'retain': retain_completions[i] if retain_completions else ""
        }
        targets.append(target_dict)
    
    os.makedirs(os.path.dirname(targets_path), exist_ok=True)
    with open(targets_path, 'w') as f:
        json.dump(targets, f)
        
    return targets

num_target_tokens = 30

# Set up paths
harmful_targets_path = f"{os.getenv('SAVE_DIR')}/rdo/{model_id}/{splits}/targets/harmful_targets.json"
harmless_targets_path = f"{os.getenv('SAVE_DIR')}/rdo/{model_id}/{splits}/targets/harmless_targets.json"

# Generate all targets
harmful_targets = generate_harmful_targets(model, harmful_train_instructions, best_refusal_direction, harmful_targets_path, num_target_tokens)
harmless_targets = generate_harmless_targets(model, harmless_train_instructions, harmless_targets_path, num_target_tokens)

# %%
if args.filter_data:
    print("Filtering data")
    harmful_train_scores = get_bypass_scores(model, harmful_train_instructions, refusal_tokens, batch_size=args.filter_batch_size)
    harmless_train_scores = get_bypass_scores(model, harmless_train_instructions, refusal_tokens, batch_size=args.filter_batch_size)
    
    # Filter instructions based on scores
    filtered_harmful_indices = [i for i, score in enumerate(harmful_train_scores) if score > 0]
    filtered_harmless_indices = [i for i, score in enumerate(harmless_train_scores) if score < 0]
    
    # Filter instructions
    filtered_harmful_train_instructions = [harmful_train_instructions[i] for i in filtered_harmful_indices]
    filtered_harmless_train_instructions = [harmless_train_instructions[i] for i in filtered_harmless_indices]
    
    # Filter targets
    filtered_harmful_targets = [harmful_targets[i] for i in filtered_harmful_indices]
    filtered_harmless_targets = [harmless_targets[i] for i in filtered_harmless_indices]
    
    print(f"Remaining harmful train instances: {len(filtered_harmful_train_instructions)}")
    
    # Balance datasets
    max_instances = min(len(filtered_harmful_train_instructions), len(filtered_harmless_train_instructions))
    filtered_harmful_train_instructions = filtered_harmful_train_instructions[:max_instances]
    filtered_harmless_train_instructions = filtered_harmless_train_instructions[:max_instances]
    filtered_harmful_targets = filtered_harmful_targets[:max_instances]
    filtered_harmless_targets = filtered_harmless_targets[:max_instances]
    
    print(f"Remaining harmless train instances: {len(filtered_harmless_train_instructions)}")
    
    # Update variables with filtered data
    harmful_train_instructions = filtered_harmful_train_instructions
    harmless_train_instructions = filtered_harmless_train_instructions
    harmful_targets = filtered_harmful_targets
    harmless_targets = filtered_harmless_targets

# Extract targets from filtered data
ablation_train_targets = [t["ablation"] for t in harmful_targets]
addition_train_targets = [t["addition"] for t in harmless_targets]
retain_train_targets = [t["retain"] for t in harmless_targets]

# %%
def build_prompts_and_labels(model, harmful_instructions, harmless_instructions, ablation_targets, addition_targets, retain_targets):
    ablation_prompts = []
    addition_prompts = []
    ablation_labels = []
    addition_labels = []
    retain_prompts = []
    for harmful_instruction, harmless_instruction, ablation_target, addition_target, retain_target in zip(harmful_instructions, harmless_instructions, ablation_targets, addition_targets, retain_targets):
        ablation_text = harmful_instruction + ablation_target
        addition_text = harmless_instruction + addition_target
        retain_text = harmless_instruction + retain_target
        ablation_prompts.append(ablation_text)
        addition_prompts.append(addition_text)
        retain_prompts.append(retain_text)

        # Tokenize without padding
        ablation_tokens = model.tokenizer.encode(ablation_text, add_special_tokens=True, return_tensors='pt')[0]
        addition_tokens = model.tokenizer.encode(addition_text, add_special_tokens=True, return_tensors='pt')[0]
        
        ablation_label = ablation_tokens[1:].clone()
        addition_label = addition_tokens[1:].clone()
        
        # Get the length of the instruction
        harmful_instruction_length = len(model.tokenizer.encode(harmful_instruction, add_special_tokens=True)) - 1
        harmless_instruction_length = len(model.tokenizer.encode(harmless_instruction, add_special_tokens=True)) - 1
        
        # Set labels corresponding to the instruction tokens to -100
        ablation_label[:harmful_instruction_length] = -100
        addition_label[:harmless_instruction_length] = -100

        ablation_labels.append(ablation_label)
        addition_labels.append(addition_label)
    return ablation_prompts, addition_prompts, retain_prompts, ablation_labels, addition_labels

ablation_train_prompts, addition_train_prompts, retain_train_prompts, ablation_train_labels, addition_train_labels = build_prompts_and_labels(model, harmful_train_instructions, harmless_train_instructions, ablation_train_targets, addition_train_targets, retain_train_targets)

# %%
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, harmful_prompts, harmless_prompts, ablation_prompts, ablation_targets, ablation_labels, addition_prompts, addition_targets, addition_labels, retain_prompts, retain_targets):
        self.harmful_prompts = harmful_prompts
        self.harmless_prompts = harmless_prompts
        self.ablation_prompts = ablation_prompts
        self.ablation_targets = ablation_targets
        self.ablation_labels = ablation_labels
        self.addition_prompts = addition_prompts
        self.addition_targets = addition_targets
        self.addition_labels = addition_labels
        self.retain_prompts = retain_prompts
        self.retain_targets = retain_targets

    def __len__(self):
        return len(self.harmful_prompts)
    
    def __getitem__(self, idx):
        return {
            'harmful_prompt': self.harmful_prompts[idx],
            'harmless_prompt': self.harmless_prompts[idx],
            'ablation_prompt': self.ablation_prompts[idx],
            'ablation_target': self.ablation_targets[idx],
            'ablation_labels': self.ablation_labels[idx],
            'addition_prompt': self.addition_prompts[idx],
            'addition_target': self.addition_targets[idx],
            'addition_labels': self.addition_labels[idx],
            'retain_prompt': self.retain_prompts[idx],
            'retain_target': self.retain_targets[idx],
        }

train_dataset = CustomDataset(harmful_train_instructions, harmless_train_instructions, ablation_train_prompts, ablation_train_targets, ablation_train_labels, addition_train_prompts, addition_train_targets, addition_train_labels, retain_train_prompts, retain_train_targets)
print(len(train_dataset))
print("Example item:")
d = train_dataset[0]
for item in d.items():
    print(item)

# %%
print(f"Length of harmful_prompts: {len(train_dataset.harmful_prompts)}")
print(f"Length of harmless_prompts: {len(train_dataset.harmless_prompts)}")
print(f"Length of ablation_prompts: {len(train_dataset.ablation_prompts)}")
print(f"Length of ablation_targets: {len(train_dataset.ablation_targets)}")
print(f"Length of ablation_labels: {len(train_dataset.ablation_labels)}")
print(f"Length of addition_prompts: {len(train_dataset.addition_prompts)}")
print(f"Length of addition_targets: {len(train_dataset.addition_targets)}")
print(f"Length of addition_labels: {len(train_dataset.addition_labels)}")
print(f"Length of retain_prompts: {len(train_dataset.retain_prompts)}")
print(f"Length of retain_targets: {len(train_dataset.retain_targets)}")

lengths = [
    len(train_dataset.harmful_prompts),
    len(train_dataset.harmless_prompts),
    len(train_dataset.ablation_prompts),
    len(train_dataset.ablation_targets),
    len(train_dataset.ablation_labels),
    len(train_dataset.addition_prompts),
    len(train_dataset.addition_targets),
    len(train_dataset.addition_labels),
    len(train_dataset.retain_prompts),
    len(train_dataset.retain_targets),
]
assert len(set(lengths)) == 1, f"Dataset component lengths are not equal: {lengths}"
print(f"All dataset component lengths are equal: {lengths[0]}")

# %%
def custom_collate(batch):
    return {
        'harmful_prompt': [item['harmful_prompt'] for item in batch],
        'harmless_prompt': [item['harmless_prompt'] for item in batch],
        'ablation_prompt': [item['ablation_prompt'] for item in batch],
        'ablation_target': [item['ablation_target'] for item in batch],
        'ablation_labels': torch.stack([item['ablation_labels'] for item in batch]),
        'addition_prompt': [item['addition_prompt'] for item in batch],
        'addition_target': [item['addition_target'] for item in batch], 
        'addition_labels': torch.stack([item['addition_labels'] for item in batch]),
        'retain_prompt': [item['retain_prompt'] for item in batch],
        'retain_target': [item['retain_target'] for item in batch],
    }

# %%
def sample_hypersphere_gaussian(batch_size, dim):
    # Sample from standard normal distribution
    samples = torch.randn(batch_size, dim, dtype=torch.float32, device=model.device).abs()
    # Normalize to unit length
    samples = samples / torch.norm(samples, dim=1, keepdim=True)
    return samples

def sample_prob_vectors(batch_size, dim):
    samples = torch.exp(torch.randn(batch_size, dim, dtype=torch.float32, device=model.device))
    samples = samples / samples.sum(dim=1, keepdim=True)
    return samples

def compute_ce_loss(logits, labels):
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    # Always pad labels with ignore tokens (-100) to match logits shape
    padding = torch.full((logits.size(0),), -100, device=labels.device)
    padding[-labels.size(0):] = labels
    return torch.nn.functional.cross_entropy(logits, padding, ignore_index=-100)

# def compute_ce_loss(logits, labels):
#     logits = logits.view(-1, logits.size(-1))
#     labels = labels.view(-1)
#     return torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)

def kl_div_fn(logits_a, logits_b, reduction='batchmean'):
    # Compute log-probabilities for the first distribution
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)
    
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(logits_a, dim=-1), 
        torch.nn.functional.softmax(logits_b, dim=-1),
        reduction=reduction
    )

def get_cosine_sims_for_vector(model, dot_vector, last_token=True):
    """Calculate cosine similarities between activations and provided vector across layers.
    
    Args:
        model: The language model
        prompt: Input prompt to get activations for
        dot_vector: Vector to compute cosine similarity against
        
    Returns:
        torch.Tensor: Tensor of cosine similarities across layers
    """
    cosine_sims = []
    for layer in model.model.layers:
        if last_token:
            cosine_sim = torch.nn.functional.cosine_similarity(layer.input[0, -1], dot_vector, dim=-1).save()
        else:
            cosine_sim = torch.nn.functional.cosine_similarity(layer.input[0, :], dot_vector, dim=-1).save()
        cosine_sims.append(cosine_sim)
    return torch.stack(cosine_sims)


def clip_grad_norm(grad, max_norm):
    total_norm = grad.norm()
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    return grad * clip_coef_clamped


class RefusalCone(nn.Module):
    def __init__(self, module: Envoy, dim: int, n_vectors: int, init_vectors: torch.Tensor | None = None, orthogonal_vectors: torch.Tensor | None = None) -> None:
        super(RefusalCone, self).__init__()
        self.module = module
        self.n_vectors = n_vectors
        self.fn_vectors = [torch.nn.Parameter(torch.randn(dim, dtype=torch.float32).cuda(), requires_grad=True) for _ in range(n_vectors)]
        if init_vectors is not None:
            for i, init_vector in enumerate(init_vectors):
                init_vector = init_vector / init_vector.norm()
                self.fn_vectors[i].data = init_vector.detach().clone().cuda().to(torch.float32)
        self.orthogonal_vectors = [(o / o.norm()).to(torch.float32).cpu() for o in orthogonal_vectors]
        self.orthogonalize()

    def __call__(self, direction):
        normalized_direction = direction / direction.norm()
        normalized_direction = normalized_direction.to(model.dtype)
        for layer in self.module.layers:
            self.ablate_input(layer, normalized_direction)
            self.ablate_output(layer.self_attn, normalized_direction, 2)
            self.ablate_output(layer.mlp, normalized_direction, 1)

    # def ablate_output(self, layer, direction, tuple_length=1):
    #     if tuple_length > 1:
    #         activation = layer.output[0][:]
    #     else:
    #         activation = layer.output
    #     projection = projection_einops(activation, direction)
    #     new_activation = activation - projection
    #     if tuple_length == 2:
    #         layer.output = (new_activation, layer.output[1])
    #     elif tuple_length == 3:
    #         layer.output = (new_activation, layer.output[1], layer.output[2])
    #     elif tuple_length == 1:
    #         layer.output = new_activation

    def ablate_output(self, layer, direction, tuple_length=1):
        if tuple_length > 1:
            activation = layer.output[0][:]
        else:
            activation = layer.output
        projection = projection_einops(activation, direction)
        new_activation = activation - projection
        if tuple_length == 2:
            layer.output = (new_activation, layer.output[1])
        elif tuple_length == 3:
            layer.output = (new_activation, layer.output[1], layer.output[2])
        else:
            layer.output = new_activation

    def ablate_input(self, layer, direction):
        projection = projection_einops(layer.input, direction)
        new_activation = layer.input - projection
        layer.input = new_activation

    def add(self, direction, alpha, layer_idx):
        direction = direction / direction.norm()
        direction = direction.to(model.dtype)
        self.module.layers[layer_idx].input += alpha * direction
    
    def transform(self, sample):
        fn_vectors = torch.stack(self.fn_vectors, dim=0)
        transformed_sample = torch.matmul(sample, fn_vectors).to(model.dtype)
        transformed_sample = transformed_sample / torch.norm(transformed_sample)
        return transformed_sample

    def parameters(self):
        return self.fn_vectors

    def orthogonalize(self):
        with torch.no_grad():
            for i in range(len(self.fn_vectors)):
                for j in range(i):
                    self.fn_vectors[i].data.sub_(projection_einops(self.fn_vectors[i].data, self.fn_vectors[j].data))
                self.fn_vectors[i].data.div_(self.fn_vectors[i].data.norm())
            
            if self.orthogonal_vectors:
                v = self.fn_vectors[0].data.clone().cpu()
                
                # Stack your vectors as rows in a matrix A
                A = torch.stack([vec.flatten().to(torch.float32) for vec in self.orthogonal_vectors])
                # Compute projection matrix P = A^T(AA^T)^-1A
                # The nullspace projector is then I - P
                AAT = A @ A.t()
                AAT_inv = torch.inverse(AAT)
                P = A.t() @ AAT_inv @ A
                I = torch.eye(P.shape[0], device=P.device)
                
                # Project onto nullspace (orthogonal complement)
                v_flat = v.flatten()
                v_ortho = (I - P) @ v_flat
                
                # Reshape back to original shape and normalize
                v_ortho = v_ortho.reshape(v.shape)
                v_ortho = v_ortho / torch.norm(v_ortho)

                self.fn_vectors[0].data = v_ortho.to(self.fn_vectors[0].dtype).to(self.fn_vectors[0].device)

    def normalize(self):
        for i in range(len(self.fn_vectors)):
            self.fn_vectors[i].data.div_(self.fn_vectors[i].data.norm())



class RefusalDirectionActivationRotation(nn.Module):
    """
    Rotate activations directly with an orthogonal matrix M (Cayley transform).

    We want X = M X (column-vector convention). Since activations are stored as row
    vectors with shape (..., dim), we apply: X_row <- X_row @ M^T.

    This class is written to be drop-in compatible with the existing
    `refusal_cone_optimization` loop:
    - exposes a non-empty `fn_vectors` (dummy) so metric loops run at least once
      (avoids nnsight prepared_inputs crash)
    - implements `__call__(direction)` and `add(direction, alpha, layer_idx)` but
      ignores the passed direction and rotates activations instead
    - optimizer updates only the per-layer Cayley params
    """

    def __init__(self, module: Envoy, dim: int, init_vectors: torch.Tensor | None = None) -> None:
        super().__init__()
        self.module = module
        self.dim = dim

        # Learn a separate orthogonal transform per layer via Cayley parameterization.
        n_layers = len(self.module.layers)
        # Keep a single `cayley_param` attribute for compatibility with existing
        # training/metrics code, but store per-layer parameters in its leading dim.
        self.cayley_param = nn.Parameter(
            torch.randn(n_layers, dim, dim, dtype=torch.float32, device="cuda") * 1e-3
        )

        if init_vectors is not None and len(init_vectors) > 0:
            r0 = init_vectors[0].detach().float().cuda().clone()
        else:
            r0 = torch.randn(dim, dtype=torch.float32, device="cuda")
        r0 = r0 / r0.norm()
        self.register_buffer("r0", r0)

        # Dummy "fn_vector" so loops like `for fn_vector in operation.fn_vectors`
        # run once and always execute at least one `tracer.invoke(...)`.
        self._dummy_fn = nn.Parameter(
            torch.zeros(dim, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        )

    @property
    def fn_vectors(self):
        return [self._dummy_fn]

    def _skew(self, layer_idx: int) -> torch.Tensor:
        U = torch.triu(self.cayley_param[layer_idx], diagonal=1)
        return U - U.T

    def cayley_matrix(self, layer_idx: int) -> torch.Tensor:
        A = self._skew(layer_idx)
        d = A.shape[0]
        I = torch.eye(d, device=A.device, dtype=A.dtype)
        return (I - A) @ torch.linalg.inv(I + A)
    
    def stack_directions_for_log(self) -> torch.Tensor:
        return self.r0.detach().unsqueeze(0).cpu()

    def _rotate(self, x, layer_idx: int):
        """
        Rotate an activation object.

        Supports:
        - Tensor-like activations (including nnsight proxies)
        - Tuple outputs (common for attention), rotating only the first element
        - "Tuple-like" proxy objects under nnsight (support indexing/slicing)

        Avoids direct access to `.dtype` / `.device` on tuple-like objects, which
        can break under nnsight attribute-fetch tracing.
        """
        # Case 1: plain Python tuple
        if isinstance(x, tuple):
            if len(x) == 0:
                return x
            return (self._rotate(x[0], layer_idx),) + x[1:]

        # Case 2: nnsight can give a Proxy that behaves like a tuple (indexable),
        # but is not an actual `tuple` instance. Detect by "no dtype" + indexable.
        if not hasattr(x, "dtype") and hasattr(x, "__getitem__"):
            # Important: do NOT fall through to any path that touches x.dtype/x.device.
            try:
                first = x[0]
                rest = x[1:]
                return (self._rotate(first, layer_idx),) + tuple(rest)
            except Exception:
                # If we can't safely rotate/rebuild, return as-is rather than crashing.
                return x

        # x: (..., dim) as row vectors. Implement X_row <- X_row @ M^T
        # For tensor-like activations, match dtype/device to avoid matmul dtype errors.
        M = self.cayley_matrix(layer_idx)  # (dim, dim) on cuda, float32
        try:
            # Safe here because tuple-like cases are handled above.
            M = M.to(device=x.device, dtype=x.dtype)
        except Exception:
            pass
        return x @ M.T

    def __call__(self, direction):
        # Keep signature compatible with baseline training loop.
        del direction

        for layer_idx, layer in enumerate(self.module.layers):
            # Rotate layer input activations. We intentionally avoid rewriting
            # tuple-valued module outputs (common in attention) because nnsight's
            # intervention plumbing may attempt to treat outputs as tensor-like.
            layer.input = self._rotate(layer.input, layer_idx)

    def add(self, direction, alpha, layer_idx):
        # Keep signature compatible with baseline training loop.
        # For activation rotation, we rotate the specified layer with its own matrix.
        del direction, alpha
        layer = self.module.layers[layer_idx]
        layer.input = self._rotate(layer.input, layer_idx)

    def orthogonalize(self):
        # For activation rotation, orthogonality is enforced by the Cayley transform.
        # This no-op exists to satisfy the training loop API.
        return

    def normalize(self):
        # No-op: Cayley transform produces an orthogonal matrix.
        return

    def parameters(self):
        return [self.cayley_param]


class OrthogonalProjection(torch.autograd.Function):
    """Project onto orthogonal group with correct gradients."""

    @staticmethod
    def forward(ctx, W):
        # Forward: project to the nearest orthogonal matrix via SVD.
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        Q = U @ Vh
        ctx.save_for_backward(U, Vh, S)
        return Q

    @staticmethod
    def backward(ctx, grad_output):
        """
        Gradient of the orthogonal projection.

        For Q = U @ Vh, use the tangent-space projection:
        grad_W = grad_Q - Q @ sym(Q^T @ grad_Q)
        where sym(M) = (M + M^T) / 2.
        """
        U, Vh, S = ctx.saved_tensors
        del S

        grad_Q = grad_output
        Q = U @ Vh
        sym_part = (Q.T @ grad_Q + grad_Q.T @ Q) / 2
        grad_W = grad_Q - Q @ sym_part
        return grad_W


class RefusalShtiefelRotation(nn.Module):
    """
    Rotate activations directly with an orthogonal matrix M, optimized in Euclidean
    space and retracted back to the orthogonal group after each optimizer step via
    Stiefel (QR) orthogonalization.

    Drop-in compatible with the existing `refusal_cone_optimization` loop:
    - exposes a non-empty `fn_vectors` (dummy) so metric loops run at least once
    - implements `__call__(direction)` and `add(direction, alpha, layer_idx)` but
      ignores the passed direction and rotates activations instead
    - keeps a `cayley_param` attribute name for training-loop compatibility
    """

    def __init__(self, module: Envoy, dim: int, init_vectors: torch.Tensor | None = None) -> None:
        super().__init__()
        self.module = module
        self.dim = dim

        n_layers = len(self.module.layers)
        self.cayley_param = nn.Parameter(
            torch.randn(n_layers, dim, dim, dtype=torch.float32, device="cuda") * 1e-3
        )

        if init_vectors is not None and len(init_vectors) > 0:
            r0 = init_vectors[0].detach().float().cuda().clone()
        else:
            r0 = torch.randn(dim, dtype=torch.float32, device="cuda")
        r0 = r0 / r0.norm()
        self.register_buffer("r0", r0)

        self._dummy_fn = nn.Parameter(
            torch.zeros(dim, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        )

        # Start from an orthogonal matrix per layer.
        with torch.no_grad():
            self.orthogonalize()

    @property
    def fn_vectors(self):
        return [self._dummy_fn]

    def matrix(self, layer_idx: int) -> torch.Tensor:
        """Apply orthogonal projection with correct gradients."""
        W = self.cayley_param[layer_idx]
        Q = OrthogonalProjection.apply(W)
        return Q

    def stack_directions_for_log(self) -> torch.Tensor:
        return self.r0.detach().unsqueeze(0).cpu()

    def _rotate(self, x, layer_idx: int):
        # Mirror RefusalDirectionActivationRotation's tuple/proxy-safe behavior.
        if isinstance(x, tuple):
            if len(x) == 0:
                return x
            return (self._rotate(x[0], layer_idx),) + x[1:]

        if not hasattr(x, "dtype") and hasattr(x, "__getitem__"):
            try:
                first = x[0]
                rest = x[1:]
                return (self._rotate(first, layer_idx),) + tuple(rest)
            except Exception:
                return x

        M = self.matrix(layer_idx)  # (dim, dim) on cuda, float32
        # Under nnsight, `x` can be a proxy where touching `x.device` / `x.dtype`
        # may fail during tracing; prefer `M.to(x)` which matches both in one call.
        try:
            M = M.to(x)
        except Exception:
            try:
                M = M.to(device=x.device, dtype=x.dtype)
            except Exception:
                # Last resort: match the model dtype if available.
                try:
                    M = M.to(dtype=model.dtype)
                except Exception:
                    pass
        return x @ M.T

    def __call__(self, direction):
        del direction
        for layer_idx, layer in enumerate(self.module.layers):
            layer.input = self._rotate(layer.input, layer_idx)

    def add(self, direction, alpha, layer_idx):
        del direction, alpha
        layer = self.module.layers[layer_idx]
        layer.input = self._rotate(layer.input, layer_idx)

    def orthogonalize(self):
        """Initialize/force orthogonality without gradients."""
        with torch.no_grad():
            for layer_idx in range(self.cayley_param.shape[0]):
                W = self.cayley_param[layer_idx]
                U, _, Vh = torch.linalg.svd(W, full_matrices=False)
                Q = U @ Vh
                self.cayley_param[layer_idx].copy_(Q)

    def skew(self, M: torch.Tensor) -> torch.Tensor:
        """Extract the skew-symmetric part of a matrix."""
        return (M - M.T) / 2

    def normalize(self):
        return

    def parameters(self):
        return [self.cayley_param]


class RefusalDirectionRotation(nn.Module):
    """
    Single direction (cone_dim=1): intervention direction r = M @ r0 with M orthogonal (Cayley transform).
    Trains skew matrix A via Cayley param; Adam updates only ``cayley_param``.
    """

    def __init__(self, module: Envoy, dim: int, init_vectors: torch.Tensor | None = None, orthogonal_vectors: torch.Tensor | None = None) -> None:
        super().__init__()
        self.module = module
        self.dim = dim
        self.cayley_param = nn.Parameter(torch.randn(dim, dim, dtype=torch.float32, device="cuda") * 1e-3)
        if init_vectors is not None and len(init_vectors) > 0:
            r0 = init_vectors[0].detach().float().cuda().clone()
        else:
            r0 = torch.randn(dim, dtype=torch.float32, device="cuda")
        r0 = r0 / r0.norm()
        self.register_buffer("r0", r0)
        self.orthogonal_vectors = [(o / o.norm()).to(torch.float32).cpu() for o in orthogonal_vectors]
        self._dummy_fn = nn.Parameter(torch.zeros(dim, dtype=torch.float32, device="cuda"), requires_grad=False)
        self.orthogonalize()

    def _skew(self) -> torch.Tensor:
        U = torch.triu(self.cayley_param, diagonal=1)
        return U - U.T

    def cayley_matrix(self) -> torch.Tensor:
        A = self._skew()
        d = A.shape[0]
        I = torch.eye(d, device=A.device, dtype=A.dtype)
        IpA = I + A
        ImA = I - A
        return ImA @ torch.linalg.inv(IpA)

    def current_direction(self) -> torch.Tensor:
        M = self.cayley_matrix()
        r = M @ self.r0
        return r / (r.norm() + 1e-12)

    @property
    def fn_vectors(self):
        return [self._dummy_fn]

    def __call__(self, direction):
        del direction
        normalized_direction = self.current_direction()
        normalized_direction = normalized_direction / normalized_direction.norm()
        normalized_direction = normalized_direction.to(model.dtype)
        for layer in self.module.layers:
            self.ablate_input(layer, normalized_direction)
            self.ablate_output(layer.self_attn, normalized_direction, 2)
            self.ablate_output(layer.mlp, normalized_direction, 1)

    def ablate_output(self, layer, direction, tuple_length=1):
        if tuple_length > 1:
            activation = layer.output[0][:]
        else:
            activation = layer.output
        projection = projection_einops(activation, direction)
        new_activation = activation - projection
        if tuple_length == 2:
            layer.output = (new_activation, layer.output[1])
        elif tuple_length == 3:
            layer.output = (new_activation, layer.output[1], layer.output[2])
        elif tuple_length == 1:
            layer.output = new_activation

    def ablate_input(self, layer, direction):
        projection = projection_einops(layer.input, direction)
        new_activation = layer.input - projection
        layer.input = new_activation

    def add(self, direction, alpha, layer_idx):
        direction = self.current_direction()
        direction = direction / direction.norm()
        direction = direction.to(model.dtype)
        self.module.layers[layer_idx].input += alpha * direction

    def transform(self, sample):
        fn_row = self.current_direction().unsqueeze(0)
        transformed_sample = torch.matmul(sample, fn_row).to(model.dtype)
        transformed_sample = transformed_sample / torch.norm(transformed_sample)
        return transformed_sample

    def parameters(self):
        return [self.cayley_param]

    def stack_directions_for_log(self) -> torch.Tensor:
        return self.current_direction().detach().unsqueeze(0).cpu()

    def orthogonalize(self):
        with torch.no_grad():
            A = self._skew()
            d = A.shape[0]
            I = torch.eye(d, device=A.device, dtype=A.dtype)
            M = (I - A) @ torch.linalg.inv(I + A)
            r = M @ self.r0
            r = r / r.norm()
            if self.orthogonal_vectors:
                v = r.cpu()
                Aop = torch.stack([vec.flatten().to(torch.float32) for vec in self.orthogonal_vectors])
                AAT = Aop @ Aop.t()
                AAT_inv = torch.inverse(AAT)
                P = Aop.t() @ AAT_inv @ Aop
                Iop = torch.eye(P.shape[0], device=P.device)
                v_flat = v.flatten()
                v_ortho = (Iop - P) @ v_flat
                v_ortho = v_ortho.reshape(v.shape)
                r_ortho = (v_ortho / torch.norm(v_ortho)).to(r.device).to(r.dtype)
                r0_new = M.T @ r_ortho
                self.r0.copy_(r0_new / r0_new.norm())
            else:
                self.r0.copy_(self.r0 / self.r0.norm())

    def normalize(self):
        with torch.no_grad():
            self.r0.div_(self.r0.norm())


def refusal_cone_optimization(model, train_dataset, 
                              batch_size=DEFAULT_CONFIG['batch_size'], 
                              effective_batch_size=DEFAULT_CONFIG['effective_batch_size'], 
                              epochs=DEFAULT_CONFIG['epochs'], 
                              lr=DEFAULT_CONFIG['lr'], 
                              cone_dim=1,
                              n_sample=DEFAULT_CONFIG['n_sample'], 
                              fixed_samples=DEFAULT_CONFIG['fixed_samples'], 
                              sampling_method=DEFAULT_CONFIG['sampling_method'], 
                              optimize_basis=DEFAULT_CONFIG['optimize_basis'], 
                              fixed_basis_vectors=[], 
                              ablation_lambda=DEFAULT_CONFIG['ablation_lambda'], 
                              alpha=alpha,  # Keep alpha as it's defined globally
                              addition_lambda=DEFAULT_CONFIG['addition_lambda'], 
                              retain_lambda=DEFAULT_CONFIG['retain_lambda'], 
                              patience=DEFAULT_CONFIG['patience'], 
                              init_vectors=[], 
                              n_lr_reduce=DEFAULT_CONFIG['n_lr_reduce'], 
                              orthogonal_vectors=[],
                              tb_writer=None,
                              tb_checkpoint_dir=None,
                              direction_mode=DEFAULT_CONFIG['direction_mode']):

    if direction_mode == "rotation":
        if cone_dim != 1:
            raise ValueError("direction_mode='rotation' requires cone_dim=1")
        if len(fixed_basis_vectors) > 0:
            raise ValueError("direction_mode='rotation' does not support fixed_basis_vectors")
        if not optimize_basis:
            raise ValueError("direction_mode='rotation' requires optimize_basis=True")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate)

    if direction_mode == "baseline":
        operation = RefusalCone(model.model, model.config.hidden_size, cone_dim, init_vectors=init_vectors, orthogonal_vectors=orthogonal_vectors)
    elif direction_mode == "rotation":
        operation = RefusalDirectionRotation(model.model, model.config.hidden_size, init_vectors=init_vectors, orthogonal_vectors=orthogonal_vectors)
    elif direction_mode == "activation_rot":
        operation = RefusalDirectionActivationRotation(model.model, model.config.hidden_size, init_vectors=init_vectors)
    elif direction_mode == "shtiefel_rot":
        operation = RefusalShtiefelRotation(model.model, model.config.hidden_size, init_vectors=init_vectors)
    else:
        ValueError(f"Invalid direction_mode: {direction_mode}")
    
    optimizer = torch.optim.AdamW(operation.parameters(), lr=lr, betas=(.9,.98), weight_decay=0.0, amsgrad=True)

    print("Cone dim", cone_dim)
    if cone_dim == 1:
        n_sample = 0

    accumulation_steps = effective_batch_size // batch_size
    print("Accumulation steps", accumulation_steps)
    vectors = []
    cayley_params = []
    train_losses = []
    stopped = False
    lowest_training_loss = float('inf')
    bypass_scores = []
    patience_counter = 0
    lr_reduce_counter = 0

    print("Starting training")

    step_counter = 0
    batch_sample_ablation_loss = 0.0
    batch_sample_addition_loss = 0.0
    batch_sample_retain_loss = 0.0
    batch_basis_ablation_loss = 0.0
    batch_basis_addition_loss = 0.0
    batch_basis_retain_loss = 0.0

    batch_sample_bypass_scores = []
    batch_sample_induce_scores = []
    batch_basis_bypass_scores = []
    batch_basis_induce_scores = []

    add_layer = best_layer

    if n_sample > 0:
        if sampling_method == "hypersphere":
            fixed_sample_vectors = sample_hypersphere_gaussian(fixed_samples, cone_dim)
        elif sampling_method == "interpolation":
            fixed_sample_vectors = sample_prob_vectors(fixed_samples, cone_dim)
        fixed_sample_vectors = [fixed_sample_vectors[i] for i in range(fixed_samples)]

    max_iters = DEFAULT_CONFIG['max_iters']
    num_iters = 0

    for epoch in range(epochs):
        print('Epoch', epoch)
        for _, batch in enumerate(train_dataloader):

            num_iters += 1
            if num_iters >= max_iters:
                print(f'Reached max number of iterations: {max_iters}')
                stopped = True
                break

            ablation_prompt = batch['ablation_prompt']
            ablation_labels = batch['ablation_labels']
            addition_prompt = batch['addition_prompt']
            addition_labels = batch['addition_labels']
            retain_prompt = batch['retain_prompt']
            harmful_prompt = batch['harmful_prompt']
            harmless_prompt = batch['harmless_prompt']

            if n_sample > 0:
                if sampling_method == "hypersphere":
                    sample_vectors = sample_hypersphere_gaussian(n_sample, cone_dim)
                elif sampling_method == "interpolation":
                    sample_vectors = sample_prob_vectors(n_sample, cone_dim)

                for sample_vector in sample_vectors:
                    if ablation_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(ablation_prompt):
                                direction = operation.transform(sample_vector)
                                operation(direction)
                                logits = model.lm_head.output[:, :-1]
                                sample_ablation_loss = compute_ce_loss(logits, ablation_labels) / n_sample
                                log = sample_ablation_loss.detach().item().save()
                            (ablation_lambda * sample_ablation_loss).backward()
                    batch_sample_ablation_loss += log
                    if addition_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(addition_prompt):
                                direction = operation.transform(sample_vector)
                                operation.add(direction, alpha, add_layer)
                                logits = model.lm_head.output[:, :-1]
                                sample_addition_loss = compute_ce_loss(logits, addition_labels) / n_sample
                                log = sample_addition_loss.detach().item().save()
                            (addition_lambda * sample_addition_loss).backward()
                        batch_sample_addition_loss += log
                    if retain_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(retain_prompt):
                                baseline_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                            with tracer.invoke(retain_prompt):
                                direction = operation.transform(sample_vector)
                                operation(direction)
                                sample_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                                sample_retain_loss = kl_div_fn(baseline_retain_logits, sample_retain_logits).mean() / n_sample
                                log = sample_retain_loss.detach().item().save()
                            (retain_lambda * sample_retain_loss).backward()
                        batch_sample_retain_loss += log
                
            if optimize_basis:
                for fn_vector in operation.fn_vectors:
                    if ablation_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(ablation_prompt):
                                operation(fn_vector)
                                logits = model.lm_head.output[:, :-1]
                                basis_ablation_loss = compute_ce_loss(logits, ablation_labels) / cone_dim
                                log = basis_ablation_loss.detach().item().save()
                            (ablation_lambda * basis_ablation_loss).backward()
                        batch_basis_ablation_loss += log

                    if addition_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(addition_prompt):
                                operation.add(fn_vector, alpha, add_layer)
                                logits = model.lm_head.output[:, :-1]
                                basis_addition_loss = compute_ce_loss(logits, addition_labels) / cone_dim
                                log = basis_addition_loss.detach().item().save()
                            (addition_lambda * basis_addition_loss).backward()
                        batch_basis_addition_loss += log

                    if retain_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(retain_prompt):
                                baseline_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                            with tracer.invoke(retain_prompt):
                                operation(fn_vector)
                                retain_logits = model.lm_head.output[:, -num_target_tokens:]
                                basis_retain_loss = kl_div_fn(baseline_retain_logits, retain_logits).mean() / cone_dim
                                log = basis_retain_loss.detach().item().save()
                            (retain_lambda * basis_retain_loss).backward()
                        batch_basis_retain_loss += log
                
            with torch.no_grad():
                with model.trace() as tracer:
                    for fn_vector in operation.fn_vectors:
                        with tracer.invoke(harmful_prompt):
                            operation(fn_vector)
                            last_token_logits = model.lm_head.output[:, -1]
                            bypass_score = refusal_metric(last_token_logits, refusal_tokens).detach().item().save()
                        batch_basis_bypass_scores.append(bypass_score)

                with model.trace() as tracer:
                    for fn_vector in operation.fn_vectors:
                        with tracer.invoke(harmless_prompt):
                            operation.add(fn_vector, alpha, add_layer)
                            last_token_logits = model.lm_head.output[:, -1]
                            induce_score = refusal_metric(last_token_logits, refusal_tokens).detach().item().save()
                        batch_basis_induce_scores.append(induce_score)
                if n_sample > 0:
                    with model.trace() as tracer:
                        for fixed_sample_vector in fixed_sample_vectors:
                            with tracer.invoke(harmful_prompt):
                                direction = operation.transform(fixed_sample_vector)
                                operation(direction)
                                sample_last_token_logits = model.lm_head.output[:, -1]
                                sample_bypass_score = refusal_metric(sample_last_token_logits, refusal_tokens).detach().item().save()
                                batch_sample_bypass_scores.append(sample_bypass_score)
                    with model.trace() as tracer:
                        for fixed_sample_vector in fixed_sample_vectors:
                            with tracer.invoke(harmless_prompt):
                                direction = operation.transform(fixed_sample_vector)
                                operation.add(direction, alpha, add_layer)
                                sample_last_token_logits = model.lm_head.output[:, -1]
                                sample_induce_score = refusal_metric(sample_last_token_logits, refusal_tokens).detach().item().save()
                                batch_sample_induce_scores.append(sample_induce_score)

                step_counter += 1
                if step_counter % accumulation_steps == 0:
                    if direction_mode == "baseline":
                        for fn_vector in operation.fn_vectors:
                            fn_vector.grad.sub_(projection_einops(fn_vector.grad, fn_vector.data))
                        for fn_vector in operation.fn_vectors:
                            fn_vector.grad.div_(accumulation_steps)
                    else:
                        p0 = operation.cayley_param
                        if p0.grad is not None:
                            p0.grad.div_(accumulation_steps)
                    torch.nn.utils.clip_grad_norm_(operation.parameters(), 10.0)
                    if direction_mode == "baseline":
                        grad_norm = operation.fn_vectors[-1].grad.norm().item()
                    else:
                        grad_norm = operation.cayley_param.grad.norm().item()
                    optimizer.step()
                    optimizer.zero_grad()
                    if len(fixed_basis_vectors) > 0:
                        for i, fixed_basis_vector in enumerate(fixed_basis_vectors):
                            fixed_basis_vector = fixed_basis_vector / fixed_basis_vector.norm()
                            operation.fn_vectors[i].data.copy_(fixed_basis_vector.data)
                    operation.orthogonalize()

                    batch_sample_ablation_loss /= accumulation_steps
                    batch_sample_addition_loss /= accumulation_steps
                    batch_sample_retain_loss /= accumulation_steps
                    batch_basis_ablation_loss /= accumulation_steps
                    batch_basis_addition_loss /= accumulation_steps
                    batch_basis_retain_loss /= accumulation_steps

                    train_loss = batch_sample_ablation_loss + batch_sample_addition_loss + batch_sample_retain_loss + batch_basis_ablation_loss + batch_basis_addition_loss + batch_basis_retain_loss
                    train_losses.append(train_loss)

                    batch_basis_bypass_scores = [s.value for s in batch_basis_bypass_scores]
                    batch_basis_induce_scores = [s.value for s in batch_basis_induce_scores]
                    basis_bypass_scores = [torch.mean(torch.tensor(batch_basis_bypass_scores[i::cone_dim])).item() for i in range(cone_dim)]
                    basis_induce_scores = [torch.mean(torch.tensor(batch_basis_induce_scores[i::cone_dim])).item() for i in range(cone_dim)]

                    if direction_mode == "baseline":
                        vectors.append(torch.stack(operation.fn_vectors, dim=0).detach().cpu().data.clone())
                    else:
                        vectors.append(operation.stack_directions_for_log())
                        cayley_params.append(operation.cayley_param.detach().cpu().data.clone())
                    bypass_scores.append(basis_bypass_scores)

                    training_metrics = {
                        "train/total_loss": train_loss,
                        "train/basis_ablation_loss": batch_basis_ablation_loss,
                        "train/basis_addition_loss": batch_basis_addition_loss,
                        "train/basis_retain_loss": batch_basis_retain_loss,
                        "train/basis_bypass_score": basis_bypass_scores,
                        "train/basis_induce_score": basis_induce_scores,
                        "train/grad_norm": grad_norm
                    }

                    if n_sample > 0:
                        batch_sample_bypass_scores = [s.value for s in batch_sample_bypass_scores]
                        batch_sample_induce_scores = [s.value for s in batch_sample_induce_scores]

                        sample_vector_bypass_scores = [torch.mean(torch.tensor(batch_sample_bypass_scores[i::fixed_samples])).item() for i in range(fixed_samples)]
                        min_sample_bypass_score = min(sample_vector_bypass_scores)
                        max_sample_bypass_score = max(sample_vector_bypass_scores)
                        mean_sample_bypass_score = torch.mean(torch.tensor(sample_vector_bypass_scores)).item()
                        std_sample_bypass_score = torch.std(torch.tensor(sample_vector_bypass_scores)).item()

                        sample_vector_induce_scores = [torch.mean(torch.tensor(batch_sample_induce_scores[i::fixed_samples])).item() for i in range(fixed_samples)]
                        min_sample_induce_score = min(sample_vector_induce_scores)
                        max_sample_induce_score = max(sample_vector_induce_scores)
                        mean_sample_induce_score = torch.mean(torch.tensor(sample_vector_induce_scores)).item()
                        std_sample_induce_score = torch.std(torch.tensor(sample_vector_induce_scores)).item()

                        training_metrics.update({
                            "train/sample_ablation_loss": batch_sample_ablation_loss if n_sample > 0 else 0,
                            "train/sample_addition_loss": batch_sample_addition_loss if n_sample > 0 else 0,
                            "train/sample_retain_loss": batch_sample_retain_loss if n_sample > 0 else 0,
                            "train/min_sample_bypass_score": min_sample_bypass_score if n_sample > 0 else 0,
                            "train/max_sample_bypass_score": max_sample_bypass_score if n_sample > 0 else 0,
                            "train/mean_sample_bypass_score": mean_sample_bypass_score if n_sample > 0 else 0,
                            "train/std_sample_bypass_score": std_sample_bypass_score if n_sample > 0 else 0,
                            "train/min_sample_induce_score": min_sample_induce_score if n_sample > 0 else 0,
                            "train/max_sample_induce_score": max_sample_induce_score if n_sample > 0 else 0,
                            "train/mean_sample_induce_score": mean_sample_induce_score if n_sample > 0 else 0,
                            "train/std_sample_induce_score": std_sample_induce_score if n_sample > 0 else 0,
                        })

                    if tb_writer is not None:
                        tensorboard_log_scalars(tb_writer, training_metrics, step_counter)

                    print("Step", step_counter, "train/basis_vector_bypass_score", [round(s, 2) for s in basis_bypass_scores], "train/basis_vector_induce_score", [round(s, 2) for s in basis_induce_scores])
                    
                    if n_sample > 0:
                        print("train/mean_sample_bypass_scores", round(mean_sample_bypass_score, 2), "train/mean_sample_induce_scores", round(mean_sample_induce_score, 2))
                    if train_loss >= lowest_training_loss:
                        patience_counter += 1
                    else:
                        lowest_training_loss = train_loss
                        patience_counter = 0
                    if patience_counter >= patience:
                        if lr_reduce_counter >= n_lr_reduce:
                            print(f'Stopping')
                            stopped = True
                            break
                        lr_reduce_counter += 1
                        print("Reducing lr to", optimizer.param_groups[0]['lr'] / 10)
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
                        patience_counter = 0

                    batch_sample_ablation_loss = 0.
                    batch_sample_addition_loss = 0.
                    batch_sample_retain_loss = 0.
                    batch_basis_ablation_loss = 0.
                    batch_basis_addition_loss = 0.
                    batch_basis_retain_loss = 0.

                    batch_sample_bypass_scores = []
                    batch_sample_induce_scores = []
                    batch_basis_bypass_scores = []
                    batch_basis_induce_scores = []

                    torch.cuda.empty_cache()

        if stopped:
            break

    save_vectors = vectors
    save_cayley_params = cayley_params
    lowest_loss_index = torch.argmin(torch.tensor(train_losses)).item()
    lowest_loss_vector = save_vectors[lowest_loss_index]
    lowest_loss_cayley_param = None
    if direction_mode != "baseline":
        try:
            lowest_loss_cayley_param = save_cayley_params[lowest_loss_index]
        except Exception:
            lowest_loss_cayley_param = None
    if tb_checkpoint_dir is not None:
        os.makedirs(tb_checkpoint_dir, exist_ok=True)
        torch.save(save_vectors, os.path.join(tb_checkpoint_dir, "vectors.pt"))
        torch.save(lowest_loss_vector, os.path.join(tb_checkpoint_dir, "lowest_loss_vector.pt"))
        if direction_mode != "baseline" and lowest_loss_cayley_param is not None:
            torch.save(save_cayley_params, os.path.join(tb_checkpoint_dir, "cayley_params.pt"))
            torch.save(lowest_loss_cayley_param, os.path.join(tb_checkpoint_dir, "lowest_loss_cayley_param.pt"))

    return {
        "vectors": vectors,
        "lowest_loss": lowest_training_loss,
        "refusal_scores": bypass_scores,
        "train_losses": train_losses,
        "lowest_loss_vector": lowest_loss_vector,
        "lowest_loss_cayley_param": lowest_loss_cayley_param,
    }

# %%
def train_refusal_vector(group_name=None, run_name=None, orthogonal_vectors=[], **kwargs):
    """
    Train a single refusal direction vector.

    Args:
        group_name: Logical group name (used in TensorBoard log path)
        run_name: Optional run name suffix
        orthogonal_vectors: List of vectors to which the trained vector should be orthogonal
        **kwargs: Additional parameters to override the defaults

    Returns:
        Dictionary containing training results
    """
    # Combine specific args and potential overrides from kwargs
    train_kwargs = {
        "cone_dim": 1,  # Specific override for single direction training
        "orthogonal_vectors": orthogonal_vectors,
        "direction_mode": args.direction_mode,
    }
    train_kwargs.update(kwargs) # Apply any user-provided overrides

    run_config = vars(args).copy()
    run_config.update({
        "model_id": model_id,
        "add_layer": add_layer,
        "alpha": alpha,
    })
    run_config.update(train_kwargs)
    run_config.pop('orthogonal_vectors', None)
    run_config.pop('train_cone', None)
    run_config.pop('train_direction', None)
    run_config.pop('train_orthogonal_direction', None)
    run_config.pop('train_independent_direction', None)

    run_id = uuid.uuid4().hex[:12]
    save_root = os.getenv("SAVE_DIR", "results")
    subdir = f"{group_name}_{model_id}_{run_name}" if run_name else f"{group_name}_{model_id}_{run_id}"
    tb_run_dir = os.path.join(save_root, "tensorboard", subdir)
    os.makedirs(tb_run_dir, exist_ok=True)
    save_run_hparams(tb_run_dir, run_config)

    tb_writer = SummaryWriter(log_dir=tb_run_dir)
    tb_checkpoint_dir = os.path.join(tb_run_dir, "checkpoints")
    try:
        results = refusal_cone_optimization(
            model=model,
            train_dataset=train_dataset,
            tb_writer=tb_writer,
            tb_checkpoint_dir=tb_checkpoint_dir,
            **train_kwargs
        )
    finally:
        tb_writer.close()
    print(f"TensorBoard log dir: {tb_run_dir}")
    if args.eval_llamaguard or args.eval_mmlu:
        refined = _extract_refined_artifact(results, args.direction_mode)
        mmlu_cfg = {
            "enabled": True,
            "dataset": args.mmlu_dataset,
            "subset": args.mmlu_subset,
            "split": "test",
            "mode": args.mmlu_mode,
            "answer_mode": args.mmlu_answer_mode,
            "n_shots": args.mmlu_n_shots,
            "sample_size": args.mmlu_sample_size,
            "sample_seed": args.mmlu_sample_seed,
            "max_new_tokens": args.mmlu_max_new_tokens,
            "store_predictions": bool(args.mmlu_store_predictions),
        }
        _evaluate_llamaguard_and_mmlu(
            tb_run_dir=tb_run_dir,
            model=model,
            refined_artifact=refined,
            direction_mode=args.direction_mode,
            splits_name=args.splits,
            eval_split=args.eval_split,
            eval_max_new_tokens=args.eval_max_new_tokens,
            eval_batch_size=args.eval_batch_size,
            mmlu_cfg=mmlu_cfg,
        )
    return results


def _safe_json_dump(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _extract_refined_artifact(training_results: dict, direction_mode: str):
    """
    Return an object representing the learned "refined" intervention.
    - baseline/rotation: a vector direction (Tensor [d_model])
    - activation_rot: per-layer cayley_param (Tensor [n_layers, d, d])
    """
    if training_results is None:
        return None
    if direction_mode == "activation_rot":
        return training_results.get("lowest_loss_cayley_param")

    vec = training_results.get("lowest_loss_vector")
    if vec is None:
        return None
    if isinstance(vec, torch.Tensor):
        if vec.ndim == 2 and vec.shape[0] >= 1:
            vec = vec[0]
        return vec
    return None


def _cayley_from_param(param_2d: torch.Tensor) -> torch.Tensor:
    """Compute Cayley orthogonal matrix from an unconstrained parameter matrix."""
    # param_2d: (d, d)
    U = torch.triu(param_2d, diagonal=1)
    A = U - U.T
    d = A.shape[0]
    I = torch.eye(d, device=A.device, dtype=A.dtype)
    return (I - A) @ torch.linalg.inv(I + A)


def _generate_nnsight(
    model: LanguageModel,
    prompts: list[str],
    *,
    max_new_tokens: int,
    batch_size: int,
    intervene_step_fn=None,
) -> list[str]:
    """
    Generate decoded completions for each prompt.

    intervene_step_fn(model): called once per decoding step inside the generation loop
    (after invoke, before generator.next()).
    """
    decoded: list[str] = []
    tokenizer = model.tokenizer

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        tok = tokenizer(batch, add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
        input_lens = (tok["attention_mask"].sum(dim=1)).tolist()

        with model.generate(max_new_tokens=max_new_tokens, do_sample=False) as generator:
            with generator.invoke(batch):
                out = model.generator.output.save()
                for _ in range(max_new_tokens - 1):
                    if intervene_step_fn is not None:
                        intervene_step_fn(model)
                    generator.next()

        tokens = out.value  # (batch, seq)
        for row, in_len in zip(tokens, input_lens):
            decoded.append(tokenizer.decode(row[in_len:], skip_special_tokens=True))

    return decoded


def _make_ablation_step_fn(fn_vector: torch.Tensor):
    fn_vector = fn_vector / (fn_vector.norm() + 1e-12)
    def step_fn(model: LanguageModel):
        v = fn_vector.to(model.dtype).to(model.device)
        for layer in model.model.layers:
            layer.input[:] -= projection_einops(layer.input[:], v)
            layer.self_attn.output[0][:] -= projection_einops(layer.self_attn.output[0][:], v)
            layer.mlp.output[:] -= projection_einops(layer.mlp.output[:], v)
    return step_fn


def _make_activation_rotation_step_fn(cayley_param: torch.Tensor):
    # cayley_param: (n_layers, d, d) on cpu; move to cuda/float32 once.
    cayley_param = cayley_param.to(device="cuda", dtype=torch.float32)
    Ms = []
    for layer_idx in range(cayley_param.shape[0]):
        M = _cayley_from_param(cayley_param[layer_idx])
        Ms.append(M)

    def step_fn(model: LanguageModel):
        # Rotate only layer inputs, consistent with RefusalDirectionActivationRotation.
        for layer_idx, layer in enumerate(model.model.layers):
            M = Ms[layer_idx]
            try:
                Mx = M.to(device=layer.input.device, dtype=layer.input.dtype)
            except Exception:
                Mx = M
            layer.input = layer.input @ Mx.T

    return step_fn


def _aggregate_llamaguard(results: list[dict]) -> dict:
    scores = [r.get("score") for r in results if r.get("score") is not None]
    unsafe_ps = [r.get("unsafe_probability") for r in results if r.get("unsafe_probability") is not None]
    labels = [r.get("label") for r in results if r.get("label") in ("safe", "unsafe")]
    score_counts = Counter(scores)
    unsafe_count = sum(1 for l in labels if l == "unsafe")
    label_total = len(labels)
    return {
        "n": len(results),
        "mean_score": (sum(scores) / len(scores)) if scores else None,
        "mean_unsafe_probability": (sum(unsafe_ps) / len(unsafe_ps)) if unsafe_ps else None,
        "pct_unsafe": (unsafe_count / label_total) if label_total else None,
        "score_counts": {str(k): int(v) for k, v in sorted(score_counts.items(), key=lambda kv: kv[0])},
    }


def _load_eval_split_json(splits_name: str, kind: str, split: str) -> list[dict]:
    path = f"data/{splits_name}_splits/{kind}_{split}.json"
    return json.load(open(path, "r", encoding="utf-8"))


def _evaluate_llamaguard_and_mmlu(
    *,
    tb_run_dir: str,
    model: LanguageModel,
    refined_artifact,
    direction_mode: str,
    splits_name: str,
    eval_split: str,
    eval_max_new_tokens: int,
    eval_batch_size: int,
    mmlu_cfg: dict,
):
    """
    Runs LlamaGuard on harmful+harmless {eval_split} and MMLU on initial/refined.
    Saves JSON artifacts under tb_run_dir and logs a few scalars to TensorBoard.
    """
    eval_writer = SummaryWriter(log_dir=tb_run_dir)
    try:
        payload: dict = {
            "saved_at": datetime.now().isoformat(),
            "direction_mode": direction_mode,
            "eval_split": eval_split,
        }

        # --- LlamaGuard ---
        if args.eval_llamaguard:
            strict = str(os.getenv("RDO_LLAMAGUARD_STRICT", "0")).lower() in ("1", "true", "yes", "y")
            try:
                harmful = _load_eval_split_json(splits_name, "harmful", eval_split)
                harmless = _load_eval_split_json(splits_name, "harmless", eval_split)

                # Keep evaluation bounded for quick iterations.
                # Tiny preset expectation: RDO_MAX_HARMFUL_PER_CATEGORY=2 across ~10 categories (~20 total),
                # and RDO_MAX_HARMLESS_TOTAL=20.
                max_harmful_per_category = int(os.getenv("RDO_MAX_HARMFUL_PER_CATEGORY", "0") or "0")
                max_harmless_total = int(os.getenv("RDO_MAX_HARMLESS_TOTAL", "0") or "0")

                if max_harmful_per_category > 0:
                    per_cat_counts: dict[str, int] = {}
                    capped: list[dict] = []
                    for item in harmful:
                        cat = str(item.get("category") or "unknown")
                        cur = per_cat_counts.get(cat, 0)
                        if cur >= max_harmful_per_category:
                            continue
                        per_cat_counts[cat] = cur + 1
                        capped.append(item)
                    harmful = capped
                    print(
                        f"[eval] capped harmful to {len(harmful)} "
                        f"({max_harmful_per_category} per category across {len(per_cat_counts)} categories)"
                    )

                if max_harmless_total > 0:
                    harmless = harmless[:max_harmless_total]
                    print(f"[eval] capped harmless to {len(harmless)} total")

                harmful_q = [d["instruction"] for d in harmful]
                harmless_q = [d["instruction"] for d in harmless]

                harmful_prompts = apply_chat_template(model.tokenizer, harmful_q)
                harmless_prompts = apply_chat_template(model.tokenizer, harmless_q)

                initial_harmful = _generate_nnsight(model, harmful_prompts, max_new_tokens=eval_max_new_tokens, batch_size=eval_batch_size)
                initial_harmless = _generate_nnsight(model, harmless_prompts, max_new_tokens=eval_max_new_tokens, batch_size=eval_batch_size)

                refined_harmful = None
                refined_harmless = None
                if refined_artifact is not None:
                    if direction_mode == "activation_rot":
                        step_fn = _make_activation_rotation_step_fn(refined_artifact)
                    else:
                        step_fn = _make_ablation_step_fn(refined_artifact)
                    refined_harmful = _generate_nnsight(model, harmful_prompts, max_new_tokens=eval_max_new_tokens, batch_size=eval_batch_size, intervene_step_fn=step_fn)
                    refined_harmless = _generate_nnsight(model, harmless_prompts, max_new_tokens=eval_max_new_tokens, batch_size=eval_batch_size, intervene_step_fn=step_fn)

                evaluator = get_llamaguard_evaluator()
                initial_harmful_results = evaluator.evaluate_batch(list(zip(harmful_q, initial_harmful)), progress_every=20)
                initial_harmless_results = evaluator.evaluate_batch(list(zip(harmless_q, initial_harmless)), progress_every=20)
                refined_harmful_results = evaluator.evaluate_batch(list(zip(harmful_q, refined_harmful)), progress_every=20) if refined_harmful is not None else None
                refined_harmless_results = evaluator.evaluate_batch(list(zip(harmless_q, refined_harmless)), progress_every=20) if refined_harmless is not None else None

                llamaguard_block = {
                    "harmful": {
                        "initial": _aggregate_llamaguard(initial_harmful_results),
                        "refined": _aggregate_llamaguard(refined_harmful_results) if refined_harmful_results is not None else None,
                    },
                    "harmless": {
                        "initial": _aggregate_llamaguard(initial_harmless_results),
                        "refined": _aggregate_llamaguard(refined_harmless_results) if refined_harmless_results is not None else None,
                    },
                }
                payload["llamaguard"] = llamaguard_block

                # TensorBoard scalars (step=0)
                def _log_lg(prefix: str, stats: dict | None):
                    if not stats:
                        return
                    if stats.get("mean_score") is not None:
                        eval_writer.add_scalar(f"eval/llamaguard/{prefix}_mean_score", stats["mean_score"], 0)
                    if stats.get("pct_unsafe") is not None:
                        eval_writer.add_scalar(f"eval/llamaguard/{prefix}_pct_unsafe", stats["pct_unsafe"], 0)

                _log_lg("harmful_initial", llamaguard_block["harmful"]["initial"])
                _log_lg("harmful_refined", llamaguard_block["harmful"]["refined"])
                _log_lg("harmless_initial", llamaguard_block["harmless"]["initial"])
                _log_lg("harmless_refined", llamaguard_block["harmless"]["refined"])
            except Exception as e:
                payload["llamaguard"] = None
                payload["llamaguard_error"] = {"type": type(e).__name__, "message": str(e)}
                print(f"[eval] LlamaGuard failed, skipping ({type(e).__name__}): {e}")
                if strict:
                    raise
            finally:
                try:
                    unload_llamaguard_evaluator()
                except Exception:
                    pass

        # --- MMLU ---
        if args.eval_mmlu:
            normalized = mmlu_eval.normalize_mmlu_config(mmlu_cfg)
            prepared = mmlu_eval.prepare_mmlu_data(normalized)
            entries = prepared["entries"]

            prompts = [e["prompt"] for e in entries]
            chat_prompts = apply_chat_template(model.tokenizer, prompts)

            initial_resp = _generate_nnsight(model, chat_prompts, max_new_tokens=normalized["max_new_tokens"], batch_size=eval_batch_size)
            initial_pred = [mmlu_eval.parse_choice_letter(r) for r in initial_resp]

            refined_pred = None
            if refined_artifact is not None:
                if direction_mode == "activation_rot":
                    step_fn = _make_activation_rotation_step_fn(refined_artifact)
                else:
                    step_fn = _make_ablation_step_fn(refined_artifact)
                refined_resp = _generate_nnsight(model, chat_prompts, max_new_tokens=normalized["max_new_tokens"], batch_size=eval_batch_size, intervene_step_fn=step_fn)
                refined_pred = [mmlu_eval.parse_choice_letter(r) for r in refined_resp]

            initial_rows = []
            refined_rows = []
            for e, p in zip(entries, initial_pred):
                initial_rows.append({
                    "index": e["index"],
                    "subject": e["subject"],
                    "question": e["question"],
                    "correct_letter": e["correct_letter"],
                    "predicted_letter": p,
                    "is_correct": p == e["correct_letter"],
                    "raw_response": None,
                })
            if refined_pred is not None:
                for e, p in zip(entries, refined_pred):
                    refined_rows.append({
                        "index": e["index"],
                        "subject": e["subject"],
                        "question": e["question"],
                        "correct_letter": e["correct_letter"],
                        "predicted_letter": p,
                        "is_correct": p == e["correct_letter"],
                        "raw_response": None,
                    })

            initial_summary = mmlu_eval.summarize_mmlu_predictions(initial_rows, prepared["config_snapshot"])
            refined_summary = mmlu_eval.summarize_mmlu_predictions(refined_rows, prepared["config_snapshot"]) if refined_rows else None

            mmlu_block = {
                "config": mmlu_eval.get_mmlu_config_snapshot(normalized),
                "initial": initial_summary,
                "refined": refined_summary,
                "delta_accuracy": (refined_summary["accuracy"] - initial_summary["accuracy"]) if refined_summary else None,
                "preview": {
                    "initial": initial_rows[: mmlu_eval.PREDICTION_PREVIEW_LIMIT],
                    "refined": refined_rows[: mmlu_eval.PREDICTION_PREVIEW_LIMIT] if refined_rows else None,
                }
            }
            payload["mmlu"] = mmlu_block

            eval_writer.add_scalar("eval/mmlu/initial_accuracy", initial_summary["accuracy"], 0)
            if refined_summary is not None:
                eval_writer.add_scalar("eval/mmlu/refined_accuracy", refined_summary["accuracy"], 0)
                eval_writer.add_scalar("eval/mmlu/delta_accuracy", refined_summary["accuracy"] - initial_summary["accuracy"], 0)

        out_path = os.path.join(tb_run_dir, f"eval_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        _safe_json_dump(out_path, payload)
        print(f"Saved eval metrics: {out_path}")
    finally:
        eval_writer.close()

# Conditional training based on command line arguments
if args.train_direction:
    print("Training standard refusal direction")
    group_name = "basic_rdo" 
    run_name = None
    train_refusal_vector(
        group_name=group_name,
        run_name=run_name,
    )

if args.train_orthogonal_direction:
    print("Training orthogonal refusal direction")
    group_name = "basic_rdo_orthogonal" 
    orthogonal_vectors = [best_refusal_direction]
    run_name = None
    train_refusal_vector(
        group_name=group_name,
        run_name=run_name,
        orthogonal_vectors=orthogonal_vectors,
    )

# %%
def train_refusal_cone(group_name, run_name, init_vectors, **kwargs):
    """
    Train a refusal cone with multiple basis vectors.

    Args:
        group_name: Logical group name (used in TensorBoard log path)
        run_name: Run name (e.g. dim_i)
        init_vectors: Initial vectors for the cone
        **kwargs: Additional parameters to override the defaults

    Returns:
        Dictionary containing training results
    """
    train_kwargs = {
        "init_vectors": init_vectors,
    }
    train_kwargs.update(kwargs)

    run_config = vars(args).copy()
    run_config.update({
        "model_id": model_id,
        "add_layer": add_layer,
        "alpha": alpha,
    })
    run_config.update(train_kwargs)
    run_config.pop('init_vectors', None)
    run_config.pop('train_cone', None)
    run_config.pop('train_direction', None)
    run_config.pop('train_orthogonal_direction', None)
    run_config.pop('train_independent_direction', None)

    run_id = uuid.uuid4().hex[:12]
    save_root = os.getenv("SAVE_DIR", "results")
    subdir = f"{group_name}_{model_id}_{run_name}_{run_id}"
    tb_run_dir = os.path.join(save_root, "tensorboard", subdir)
    os.makedirs(tb_run_dir, exist_ok=True)
    save_run_hparams(tb_run_dir, run_config)

    tb_writer = SummaryWriter(log_dir=tb_run_dir)
    tb_checkpoint_dir = os.path.join(tb_run_dir, "checkpoints")
    try:
        results = refusal_cone_optimization(
            model=model,
            train_dataset=train_dataset,
            tb_writer=tb_writer,
            tb_checkpoint_dir=tb_checkpoint_dir,
            **train_kwargs
        )
    finally:
        tb_writer.close()
    print(f"TensorBoard log dir: {tb_run_dir}")
    return results

if args.train_cone:
    print(f"Training refusal cone with dimensions from {args.min_cone_dim} to {args.max_cone_dim}")
    subspace_dimensions = range(args.min_cone_dim, args.max_cone_dim + 1)
    group_name = "basic_rco"
    init_vectors = []

    for i in subspace_dimensions:
        run_name = f"dim_{i}"
        print(f"Training dimension {i}")
        results = train_refusal_cone(
            group_name=group_name, 
            run_name=run_name, 
            init_vectors=init_vectors,
            cone_dim=i
        )
        lowest_loss_vector = results['lowest_loss_vector']
        init_vectors = list(lowest_loss_vector)
# %%
import nnsight

class DirectionalAblation(nn.Module):
    def __init__(self, module, dim, orthogonal_vectors, add_layer_idx, alpha, init_vector = None):
        super(DirectionalAblation, self).__init__()
        self.module = module
        self.fn_vector = torch.nn.Parameter(
            init_vector.to(torch.float32) if init_vector is not None else torch.randn(dim, dtype=torch.float32),
            requires_grad=True
        ).save()
        self.alpha = torch.nn.Parameter(torch.tensor(float(alpha.item())).to(torch.float32), requires_grad=False).save()
        nnsight.log("alpha", self.alpha)
        self.orthogonal_vectors = [(o / o.norm()).to(torch.float32).cpu() for o in orthogonal_vectors]
        self.add_layer_idx = add_layer_idx

        self.orthogonalize()

    def __call__(self, direction):
        direction = direction / direction.norm()
        direction = direction.to(model.dtype)
        for layer in self.module.layers:
            self.ablate_input(layer, direction)
            self.ablate_output(layer.self_attn, direction, 2)
            self.ablate_output(layer.mlp, direction, 1)

    def ablate_output(self, layer, direction, tuple_length=1):
        if tuple_length > 1:
            activation = layer.output[0][:]
        else:
            activation = layer.output
        projection = projection_einops(activation, direction)
        new_activation = activation - projection
        if tuple_length == 2:
            layer.output = (new_activation, layer.output[1])
        elif tuple_length == 3:
            layer.output = (new_activation, layer.output[1], layer.output[2])
        elif tuple_length == 1:
            layer.output = new_activation
    
    def ablate_input(self, layer, direction):
        projection = projection_einops(layer.input, direction)
        new_activation = layer.input - projection
        layer.input = new_activation

    def add(self, direction):
        direction = direction / direction.norm()
        direction = direction.to(model.dtype)
        self.module.layers[self.add_layer_idx].input += self.alpha * direction

    def normalize(self):
        with torch.no_grad():
            self.fn_vector.data = self.fn_vector.data / self.fn_vector.data.norm()

    def orthogonalize(self):
        if self.orthogonal_vectors:
            self.fn_vector.data = nnsight.apply(self._orthogonalize, self.fn_vector)

    def _orthogonalize(self, vector):
        with torch.no_grad():
            v = vector.data.clone().cpu()
            
            # Stack your vectors as rows in a matrix A
            A = torch.stack([vec.flatten().to(torch.float32) for vec in self.orthogonal_vectors])
            # Compute projection matrix P = A^T(AA^T)^-1A
            # The nullspace projector is then I - P
            AAT = A @ A.t()
            AAT_inv = torch.inverse(AAT)
            P = A.t() @ AAT_inv @ A
            I = torch.eye(P.shape[0], device=P.device)
            
            # Project onto nullspace (orthogonal complement)
            v_flat = v.flatten()
            v_ortho = (I - P) @ v_flat
            
            # Reshape back to original shape and normalize
            v_ortho = v_ortho.reshape(v.shape)
            v_ortho = v_ortho / torch.norm(v_ortho)
                
            return v_ortho.to(vector.dtype).to(vector.device)

def repind_rdo(model,
               train_dataset,
               batch_size=DEFAULT_CONFIG['batch_size'],
               effective_batch_size=DEFAULT_CONFIG['effective_batch_size'],
               epochs=DEFAULT_CONFIG['epochs'],
               lr=DEFAULT_CONFIG['lr'],
               ablation_lambda=DEFAULT_CONFIG['ablation_lambda'],
               addition_lambda=DEFAULT_CONFIG['addition_lambda'],
               retain_lambda=DEFAULT_CONFIG['retain_lambda'],
               patience=DEFAULT_CONFIG['patience'],
               n_lr_reduce=DEFAULT_CONFIG['n_lr_reduce'],
               alpha=alpha,
               orthogonal_vectors=[],
               repind_layers=[],
               repind_lambda=1,
               independent_vectors=[],
               verbose=False,
               init_vector=None,
               tb_writer=None,
               tb_checkpoint_dir=None,
               direction_mode=DEFAULT_CONFIG['direction_mode']):

    if direction_mode == "rotation":
        raise ValueError("repind_rdo only supports direction_mode='baseline' (use --direction_mode baseline).")

    global _ACTIVE_TB_WRITER
    prev_writer = _ACTIVE_TB_WRITER
    _ACTIVE_TB_WRITER = tb_writer
    try:
        with model.session() as session:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate)

            # Handle independent vectors safely
            processed_independent_vectors = []
            for ind in independent_vectors:
                if ind is not None:
                    processed_independent_vectors.append(ind.detach().clone().cuda().to(model.dtype))
            independent_vectors = processed_independent_vectors
        
            # Handle init_vector safely
            if init_vector is not None:
                init_vector = init_vector.detach().clone().cuda().to(model.dtype)

            operation = DirectionalAblation(model.model, model.config.hidden_size, orthogonal_vectors, best_layer, alpha, init_vector=init_vector)

            parameters = [{"params": operation.fn_vector, "lr": lr}]
            optimizer = torch.optim.AdamW(parameters, betas=(.9,.98), weight_decay=0.0, amsgrad=True)

            # log optimizer parameters
            accumulation_steps = effective_batch_size // batch_size
            if verbose:
                print("Accumulation steps", accumulation_steps)

            vectors = nnsight.list().save()
            train_losses = nnsight.list().save()
            lowest_training_loss = nnsight.float(float('inf')).save()
            patience_counter = nnsight.int(0).save()
            lr_reduce_counter = nnsight.int(0).save()
            stopped = nnsight.bool(False).save()

            if verbose:
                print("Starting training")

            step_counter = nnsight.int(0)
            batch_ablation_loss = nnsight.float(0)
            batch_addition_loss = nnsight.float(0)
            batch_repind_loss = nnsight.float(0)
            batch_retain_loss = nnsight.float(0)
            batch_refusal_score = nnsight.float(0)
            batch_induce_score = nnsight.float(0)
            batch_gradients = nnsight.list()

            with session.iter(range(epochs), return_context=True) as (epoch, epoch_iterator):
                if verbose:
                    epoch_iterator.log('Epoch', epoch)
                with session.iter(train_dataloader, return_context=True) as (batch, train_iterator):
                    ablation_prompt = batch['ablation_prompt']
                    ablation_labels = batch['ablation_labels']
                    addition_prompt = batch['addition_prompt']
                    addition_labels = batch['addition_labels']
                    retain_prompt = batch['retain_prompt']
                    harmful_prompt = batch['harmful_prompt']
                    harmless_prompt = batch['harmless_prompt']

                    if ablation_lambda > 0:
                        with model.trace() as tracer:
                            ablation_loss = torch.tensor(0.0, device=model.device, dtype=model.dtype)
                            with tracer.invoke(ablation_prompt):
                                operation(operation.fn_vector)
                                logits = model.lm_head.output[:, :-1]
                                ablation_loss += compute_ce_loss(logits, ablation_labels)
                            ablation_loss = ablation_loss / accumulation_steps
                            batch_ablation_loss.update(batch_ablation_loss + ablation_loss.detach().item())
                            (ablation_lambda * ablation_loss).backward()

                    if repind_lambda > 0:
                        with model.trace() as tracer:
                            repind_losses = []
                            for independent_vector in independent_vectors:
                                with tracer.invoke(harmful_prompt):
                                    target_independent_cosine_sims = get_cosine_sims_for_vector(model, independent_vector)
                                    target_fn_vector_cosine_sims = get_cosine_sims_for_vector(model, operation.fn_vector)

                                with tracer.invoke(harmful_prompt):
                                    operation(operation.fn_vector)
                                    current_independent_cosine_sims = get_cosine_sims_for_vector(model, independent_vector)
                                    repind_losses.append((current_independent_cosine_sims - target_independent_cosine_sims)[repind_layers].square().mean())

                                with tracer.invoke(harmful_prompt):
                                    operation(independent_vector)
                                    current_fn_vector_cosine_sims = get_cosine_sims_for_vector(model, operation.fn_vector)
                                    repind_losses.append((current_fn_vector_cosine_sims - target_fn_vector_cosine_sims)[repind_layers].square().mean())

                            repind_loss = torch.stack(repind_losses).sum()
                            repind_loss = repind_loss / accumulation_steps
                            batch_repind_loss.update(batch_repind_loss + repind_loss.detach().item())
                            (repind_lambda * repind_loss).backward()

                    if addition_lambda > 0:
                        addition_loss = torch.tensor(0.0, device=model.device, dtype=model.dtype)
                        with model.trace() as tracer:
                            with tracer.invoke(addition_prompt) as _:
                                operation.add(operation.fn_vector)
                                logits = model.lm_head.output[:, :-1]
                                addition_loss += compute_ce_loss(logits, addition_labels)
                        addition_loss = addition_loss / accumulation_steps
                        batch_addition_loss.update(batch_addition_loss + addition_loss.detach().item())
                        (addition_lambda * addition_loss).backward()

                    if retain_lambda > 0:
                        retain_loss = torch.tensor(0.0, device=model.device, dtype=model.dtype)
                        with model.trace() as tracer:
                            with tracer.invoke(retain_prompt):
                                baseline_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                            with tracer.invoke(retain_prompt):
                                operation(operation.fn_vector)
                                retain_logits = model.lm_head.output[:, -num_target_tokens:]
                                retain_loss += kl_div_fn(baseline_retain_logits, retain_logits).mean()
                            retain_loss = retain_loss / accumulation_steps
                            batch_retain_loss.update(batch_retain_loss + retain_loss.detach().item())
                            (retain_lambda * retain_loss).backward()

                    with torch.no_grad():
                        with model.trace() as tracer:
                            with tracer.invoke(harmful_prompt) as _:
                                operation(operation.fn_vector)
                                last_token_logits = model.lm_head.output[:, -1]
                                refusal_score = refusal_metric(last_token_logits, refusal_tokens)
                                refusal_score = refusal_score / accumulation_steps
                                batch_refusal_score.update(batch_refusal_score + refusal_score.detach().item())
                        with model.trace() as tracer:
                            with tracer.invoke(harmless_prompt) as _:
                                operation.add(operation.fn_vector)
                                last_token_logits = model.lm_head.output[:, -1]
                                induce_score = refusal_metric(last_token_logits, refusal_tokens)
                                induce_score = induce_score / accumulation_steps
                                batch_induce_score.update(batch_induce_score + induce_score.detach().item())

                        step_counter.update(step_counter + 1)
                        batch_gradients.append(operation.fn_vector.grad.detach().cpu())
                        with train_iterator.cond(step_counter % accumulation_steps == 0):
                            grad_sum = nnsight.apply(sum, batch_gradients)
                            grad_sum = nnsight.apply(lambda x, y: x - projection_einops(x, y / y.norm()), grad_sum, operation.fn_vector) # project gradient to tangent of sphere
                            grad_sum = nnsight.apply(lambda x, y: x - projection_einops(x, y / y.norm()), grad_sum, operation.fn_vector) # project gradient to tangent of sphere
                            grad_norm = grad_sum.norm().item()
                            operation.fn_vector.grad = grad_sum
                            optimizer.step()
                            optimizer.zero_grad()

                            if orthogonal_vectors:
                                operation.orthogonalize()
                            operation.normalize()
                            train_loss = batch_ablation_loss + batch_addition_loss + batch_repind_loss + batch_retain_loss
                            train_losses.append(train_loss)

                            vectors.append(operation.fn_vector.detach().data.clone())

                            nnsight.apply(_tb_log_for_nnsight, {
                                "train/total_loss": train_loss,
                                "train/ablation_loss": batch_ablation_loss,
                                "train/addition_loss": batch_addition_loss,
                                "train/repind_loss": batch_repind_loss,
                                "train/retain_loss": batch_retain_loss,
                                "train/refusal_score": batch_refusal_score,
                                "train/induce_score": batch_induce_score,
                                "train/grad_norm": grad_norm,
                            }, step=step_counter)
                            nnsight.log("Step", step_counter, "train/refusal_score", batch_refusal_score, "train/induce_score", batch_induce_score)

                            with train_iterator.cond(train_loss >= lowest_training_loss):
                                patience_counter.update(patience_counter + 1)
                            with train_iterator.cond(train_loss < lowest_training_loss):
                                lowest_training_loss.update(train_loss)
                                patience_counter.update(0)
                            with train_iterator.cond(patience_counter >= patience):
                                with train_iterator.cond(lr_reduce_counter >= n_lr_reduce):
                                    if verbose:
                                        nnsight.log(f'Stopping')
                                    stopped.update(True)
                                    train_iterator.exit()
                                with train_iterator.cond(lr_reduce_counter < n_lr_reduce):
                                    lr_reduce_counter.update(lr_reduce_counter + 1)
                                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
                                    nnsight.log("Reducing lr to", optimizer.param_groups[0]['lr'])
                                    patience_counter.update(0)

                            batch_ablation_loss.update(0)
                            batch_addition_loss.update(0)
                            batch_repind_loss.update(0)
                            batch_retain_loss.update(0)
                            batch_refusal_score.update(0)
                            batch_induce_score.update(0)
                            batch_gradients.update([])

                with epoch_iterator.cond(stopped == True):
                    epoch_iterator.exit()

        save_vectors = vectors.value
        lowest_loss_index = torch.argmin(torch.tensor(train_losses.value)).item()
        lowest_loss_vector = vectors.value[lowest_loss_index]
        if tb_checkpoint_dir is not None:
            os.makedirs(tb_checkpoint_dir, exist_ok=True)
            torch.save(save_vectors, os.path.join(tb_checkpoint_dir, "vector.pt"))
            torch.save(lowest_loss_vector, os.path.join(tb_checkpoint_dir, "lowest_loss_vector.pt"))

        return {"vectors": save_vectors, "lowest_loss_vector": lowest_loss_vector}
    finally:
        _ACTIVE_TB_WRITER = prev_writer

# %%
def train_independent_vector(group_name=None, run_name=None, independent_vectors=None, **kwargs):
    train_kwargs = {
        'epochs': 2,
        'retain_lambda': 0.1,
        'repind_lambda': 200,
        'repind_layers': repind_layers,
        'independent_vectors': independent_vectors,
        'direction_mode': args.direction_mode,
    }
    train_kwargs.update(kwargs) # Apply any user-provided overrides

    run_config = vars(args).copy()
    run_config.update({
        "model_id": model_id,
        "add_layer": add_layer,
        "alpha": alpha,
    })
    run_config.update(train_kwargs)
    run_config.pop('independent_vectors', None)
    run_config.pop('train_cone', None)
    run_config.pop('train_direction', None)
    run_config.pop('train_orthogonal_direction', None)
    run_config.pop('train_independent_direction', None)

    run_id = uuid.uuid4().hex[:12]
    save_root = os.getenv("SAVE_DIR", "results")
    subdir = f"{group_name}_{model_id}_{run_name}_{run_id}"
    tb_run_dir = os.path.join(save_root, "tensorboard", subdir)
    os.makedirs(tb_run_dir, exist_ok=True)
    save_run_hparams(tb_run_dir, run_config)

    tb_writer = SummaryWriter(log_dir=tb_run_dir)
    tb_checkpoint_dir = os.path.join(tb_run_dir, "checkpoints")
    try:
        results = repind_rdo(
            model=model,
            train_dataset=train_dataset,
            verbose=True,
            tb_writer=tb_writer,
            tb_checkpoint_dir=tb_checkpoint_dir,
            **train_kwargs
        )
    finally:
        tb_writer.close()
    print(f"TensorBoard log dir: {tb_run_dir}")
    return results

# %%
if args.train_independent_direction:
    harmful_val = json.load(open(f'data/{splits}_splits/harmful_val.json'))
    harmless_val = json.load(open(f'data/{splits}_splits/harmless_val.json'))
    harmful_val_instructions = apply_chat_template(model.tokenizer, [d["instruction"] for d in harmful_val])
    harmless_val_instructions = apply_chat_template(model.tokenizer, [d["instruction"] for d in harmless_val])
    harmful_val_scores = get_bypass_scores(model, harmful_val_instructions, refusal_tokens, batch_size=args.filter_batch_size)
    harmless_val_scores = get_bypass_scores(model, harmless_val_instructions, refusal_tokens, batch_size=args.filter_batch_size)
    filtered_harmful_val_instructions = [d for d, score in zip(harmful_val_instructions, harmful_val_scores) if score > 0]
    filtered_harmless_val_instructions = [d for d, score in zip(harmless_val_instructions, harmless_val_scores) if score < 0]
    harmful_val_instructions = filtered_harmful_val_instructions
    harmless_val_instructions = filtered_harmless_val_instructions[:len(filtered_harmful_val_instructions)]

    layer_cutoff = 0.9
    repind_layers = list(range(int(model.config.num_hidden_layers * layer_cutoff)))

    group_name = f"repind_iterative"

    best_vectors = []
    independent_vectors = [best_refusal_direction]
    n_idx = 1
    inits = 1
    for idx in range(1, n_idx + 1):
        mean_refusal_scores = []
        lowest_loss_vectors = []

        for i in range(1, inits + 1):
            run_name = f"repind_{idx}_run_{i}"
            results = train_independent_vector(group_name=group_name, run_name=run_name, independent_vectors=independent_vectors, repind_layers=repind_layers)

            lowest_loss_vector = results['lowest_loss_vector']
            lowest_loss_vectors.append(lowest_loss_vector)
            refusal_scores = get_bypass_scores(model, harmful_val_instructions, refusal_tokens, fn_vector=lowest_loss_vector, batch_size=args.filter_batch_size)
            mean_refusal_score = refusal_scores.mean().item()
            mean_refusal_scores.append(mean_refusal_score)
            print(f"mean_refusal_score: {mean_refusal_score}")

        best_idx = np.argmin(mean_refusal_scores)
        print(f"best_idx: {best_idx}, i.e. run name 'rep_ind{idx}_run_{best_idx+1}'")
        independent_vectors.append(lowest_loss_vectors[best_idx])
    print(mean_refusal_scores)
# %%
