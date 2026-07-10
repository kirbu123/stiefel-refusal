import torch
import random
import json
import os
import argparse
from pathlib import Path

from dotenv import load_dotenv

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak, substring_matching_judge_fn
from pipeline.submodules.evaluate_loss import evaluate_loss

# Base (non-chat) checkpoints never refuse, so the refusal-based filter empties
# the dataset and select_direction() asserts. Redirect known base checkpoints to
# their Instruct sibling, which the pipeline handles as designed.
BASE_TO_INSTRUCT = {
    "Qwen/Qwen3-8B-Base": "Qwen/Qwen3-8B",
    "allenai/Olmo-3-1025-7B": "allenai/Olmo-3-7B-Instruct",
}

def resolve_instruct_model(model_path):
    if model_path in BASE_TO_INSTRUCT:
        return BASE_TO_INSTRUCT[model_path]
    # convenience fallback for cleanly-named base repos (e.g. ".../Foo-7B-Base")
    for suffix in ("-Base", "-base"):
        if model_path.endswith(suffix):
            return model_path[: -len(suffix)]
    return model_path

def parse_arguments():
    """Parse model path argument from command line."""
    _repo_root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(_repo_root / ".env", override=True)
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    return parser.parse_args()

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_train = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)[:len(harmful_train)]
    harmful_val = load_dataset_split(harmtype='harmful', split='val', instructions_only=True)
    harmless_val = load_dataset_split(harmtype='harmless', split='val', instructions_only=True)[:len(harmful_val)]
    return harmful_train, harmless_train, harmful_val, harmless_val

def _refusal_flags(model_base, instructions, batch_size, max_new_tokens=32):
    """Short greedy completions, flagged True where the response is a refusal."""
    dataset = [{'instruction': ins, 'category': ''} for ins in instructions]
    completions = model_base.generate_completions(dataset, max_new_tokens=max_new_tokens, batch_size=batch_size)
    responses = [c['response'] for c in completions]           # generate_completions preserves input order
    flags = [substring_matching_judge_fn(r) for r in responses]
    return flags, responses

def _apply_refusal_filter(model_base, harmful, harmless, bs, tag):
    h_flags, h_resp = _refusal_flags(model_base, harmful, bs)
    l_flags, _      = _refusal_flags(model_base, harmless, bs)
    print(f"[{tag}] harmful refusals: {sum(h_flags)}/{len(h_flags)} | "
          f"harmless refusals: {sum(l_flags)}/{len(l_flags)}")
    for ins, r, f in list(zip(harmful, h_resp, h_flags))[:3]:      # self-diagnosing sample
        print(f"  [harmful refusal={f}] {ins[:70]!r} -> {r[:90]!r}")
    harmful  = [x for x, f in zip(harmful, h_flags) if f]            # keep refusals
    harmless = [x for x, f in zip(harmless, l_flags) if not f][:len(harmful)]  # keep non-refusals
    print(f"[{tag}] kept {len(harmful)} harmful, {len(harmless)} harmless")
    if len(harmful) == 0:
        raise SystemExit(
            f"[{tag}] 0 harmful refusals detected — the model does not refuse this data with the "
            f"current prompt/template (see sample completions above). Either the chat template is "
            f"wrong or the model complies; rerun with filtering disabled to use the raw diff-in-means "
            f"direction.")
    return harmful, harmless

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets by whether the model actually refuses (generation + substring judge).

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    if cfg.filter_train:
        print("Filtering train dataset")
        harmful_train, harmless_train = _apply_refusal_filter(model_base, harmful_train, harmless_train, cfg.completions_batch_size, "train")

    if cfg.filter_val:
        harmful_val, harmless_val = _apply_refusal_filter(model_base, harmful_val, harmless_val, cfg.completions_batch_size, "val")
    
    return harmful_train, harmless_train, harmful_val, harmless_val

def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))

    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))

    return mean_diffs

def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions):
    """Select and save the direction."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction"),
        # kl_threshold=None,
        # induce_refusal_threshold=None,
        # prune_layer_percentage=None,
    )

    with open(f'{cfg.artifact_path()}/direction_metadata.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f'{cfg.artifact_path()}/direction.pt')

    return pos, layer, direction

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens, batch_size=cfg.completions_batch_size)
    
    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)

def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_baseline_completions.json')

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path)

    with open(f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)

def run_pipeline(model_path):
    """Run the full pipeline."""
    resolved = resolve_instruct_model(model_path)
    if resolved != model_path:
        print(f"[base-model remap] '{model_path}' is a base checkpoint that never "
              f"refuses; running Instruct variant '{resolved}' instead.")
        model_path = resolved
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    model_base = construct_model_base(cfg.model_path)

    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)
    
    # Filter datasets based on refusal scores
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)

    # 1. Generate candidate refusal directions
    candidate_directions = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)
    
    # 2. Select the most effective refusal direction
    pos, layer, direction = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions)

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []

    # 3a. Generate and save completions on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd', dataset_name)

    # 3b. Evaluate completions and save results on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
    
    # 4a. Generate and save completions on harmless evaluation dataset
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless', dataset=harmless_test)
    
    actadd_refusal_pre_hooks, actadd_refusal_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))], []
    generate_and_save_completions_for_dataset(cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, 'actadd', 'harmless', dataset=harmless_test)

    # 4b. Evaluate completions and save results on harmless evaluation dataset
    evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
    evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)

    # 5. Evaluate loss on harmless datasets
    # evaluate_loss_for_datasets(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline')
    # evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation')
    # evaluate_loss_for_datasets(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd')

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path)
