import torch
import functools
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Prompts are built with the tokenizer's official chat template
# (tokenizer.apply_chat_template), not hand-crafted strings, so the exact
# special tokens / BOS match what the model was trained on.

def tokenize_instructions_olmo_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True
):
    prompts = []
    for i, instruction in enumerate(instructions):
        messages = ([{"role": "system", "content": system}] if system is not None else []) \
                   + [{"role": "user", "content": instruction}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        if outputs is not None:
            text += outputs[i]
        prompts.append(text)

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,   # apply_chat_template already emits BOS/role/special tokens
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
        """Get the end-of-instruction tokens (from the tokenizer's chat template)."""
        marker = "|||INSTR|||"
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": marker}], tokenize=False, add_generation_prompt=True,
        )
        return self.tokenizer.encode(rendered.split(marker)[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        """Get the refusal tokens (derived from the tokenizer for the right vocab)."""
        return [self.tokenizer.encode("I", add_special_tokens=False)[0]]

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