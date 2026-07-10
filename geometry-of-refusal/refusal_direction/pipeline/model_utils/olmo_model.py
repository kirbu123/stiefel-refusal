import torch
import functools
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Olmo chat templates based on:
# - Olmo-3-1025-7B uses standard chat format similar to Llama
# - https://huggingface.co/allenai/Olmo-3-1025-7B

OLMO_CHAT_TEMPLATE = """<|user|>
{instruction}<|assistant|>
"""

OLMO_CHAT_TEMPLATE_WITH_SYSTEM = """<|system|>
{system_prompt}<|user|>
{instruction}<|assistant|>
"""

# Common refusal tokens for Olmo models
OLMO_REFUSAL_TOKS = [40]  # 'I'

def format_instruction_olmo_chat(
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True
):
    if system is not None:
        formatted_instruction = OLMO_CHAT_TEMPLATE_WITH_SYSTEM.format(
            instruction=instruction, 
            system_prompt=system
        )
    else:
        formatted_instruction = OLMO_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_olmo_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True
):
    if outputs is not None:
        prompts = [
            format_instruction_olmo_chat(
                instruction=instruction, 
                output=output, 
                system=system, 
                include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_olmo_chat(
                instruction=instruction, 
                system=system, 
                include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def orthogonalize_olmo_weights(model, direction: Float[Tensor, "d_model"]):
    """Orthogonalize Olmo model weights with respect to a direction."""
    # Embedding layer
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction
    )

    # For each transformer block
    for block in model.model.layers:
        # Attention output projection
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T
        # MLP down projection
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
            block.mlp.down_proj.weight.data.T, direction
        ).T

def act_add_olmo_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    """Add activation-based bias to Olmo model weights."""
    dtype = model.model.layers[layer-1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer-1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.model.layers[layer-1].mlp.down_proj.bias = torch.nn.Parameter(bias)

class OlmoModel(ModelBase):
    """Model implementation for AllenAI Olmo models."""
    
    def __init__(self, model_path: str):
        """Initialize Olmo model with the given path."""
        # Call parent with just model_path
        super().__init__(model_path)

    def _load_model(self, model_path, dtype=torch.bfloat16):
        """Load the Olmo model from HuggingFace."""
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"),
        ).eval()

        model.requires_grad_(False)

        return model

    def _load_tokenizer(self, model_path):
        """Load the tokenizer for Olmo models."""
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        tokenizer.padding_side = 'left'
        
        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        """Get the tokenization function for instructions."""
        return functools.partial(
            tokenize_instructions_olmo_chat, 
            tokenizer=self.tokenizer, 
            system=None, 
            include_trailing_whitespace=True
        )

    def _get_eoi_toks(self):
        """Get the end-of-instruction tokens."""
        # Get the assistant token part from the template
        eoi_part = OLMO_CHAT_TEMPLATE.split("{instruction}")[-1]
        return self.tokenizer.encode(eoi_part, add_special_tokens=False)

    def _get_refusal_toks(self):
        """Get the refusal tokens."""
        return OLMO_REFUSAL_TOKS

    def _get_model_block_modules(self):
        """Get the transformer block modules."""
        return self.model.model.layers

    def _get_attn_modules(self):
        """Get the attention modules."""
        return torch.nn.ModuleList(
            [block_module.self_attn for block_module in self.model_block_modules]
        )

    def _get_mlp_modules(self):
        """Get the MLP modules."""
        return torch.nn.ModuleList(
            [block_module.mlp for block_module in self.model_block_modules]
        )

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        """Get the orthogonalization modification function."""
        return functools.partial(orthogonalize_olmo_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        """Get the activation addition modification function."""
        return functools.partial(act_add_olmo_weights, direction=direction, coeff=coeff, layer=layer)