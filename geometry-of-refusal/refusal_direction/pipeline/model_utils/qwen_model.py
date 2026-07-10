import torch
import functools
import os

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Prompts are built with the tokenizer's official chat template
# (tokenizer.apply_chat_template), not hand-crafted strings, so the exact
# special tokens / BOS / thinking scaffold match what the model was trained on.

system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
    enable_thinking: bool=True,
):
    prompts = []
    for i, instruction in enumerate(instructions):
        messages = ([{"role": "system", "content": system}] if system is not None else []) \
                   + [{"role": "user", "content": instruction}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
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

def orthogonalize_qwen_weights(model, direction: Float[Tensor, "d_model"]):
    model.transformer.wte.weight.data = get_orthogonalized_matrix(model.transformer.wte.weight.data, direction)

    for block in model.transformer.h:
        block.attn.c_proj.weight.data = get_orthogonalized_matrix(block.attn.c_proj.weight.data.T, direction).T
        block.mlp.c_proj.weight.data = get_orthogonalized_matrix(block.mlp.c_proj.weight.data.T, direction).T

def act_add_qwen_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.transformer.h[layer-1].mlp.c_proj.weight.dtype
    device = model.transformer.h[layer-1].mlp.c_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.transformer.h[layer-1].mlp.c_proj.bias = torch.nn.Parameter(bias)


class QwenModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"),
        ).eval()

        model.requires_grad_(False) 

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )
        tokenizer.padding_side = 'left'
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        enable_thinking = 'qwen3' not in self.model_name_or_path.lower()
        return functools.partial(tokenize_instructions_qwen_chat, tokenizer=self.tokenizer, system=system, include_trailing_whitespace=True, enable_thinking=enable_thinking)

    def _get_eoi_toks(self):
        marker = "|||INSTR|||"
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": marker}], tokenize=False, add_generation_prompt=True,
        )
        return self.tokenizer.encode(rendered.split(marker)[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        # Derive from the tokenizer so IDs match the model's actual vocab.
        return [self.tokenizer.encode(w, add_special_tokens=False)[0] for w in ("I", "As")]

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_qwen_weights, direction=direction, coeff=coeff, layer=layer)