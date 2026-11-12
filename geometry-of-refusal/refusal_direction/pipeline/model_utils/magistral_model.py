"""Magistral Small 2506: Mistral decoder and native Tekken chat tokenization."""
import functools
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, MistralCommonTokenizer
from transformers.utils import is_mistral_common_available

from pipeline.model_utils.llama3_model import Llama3Model


def tokenize_instructions_magistral(tokenizer, instructions, outputs=None, system=None):
    if not instructions:
        raise ValueError('At least one instruction is required')
    if outputs is not None and len(outputs) != len(instructions):
        raise ValueError('outputs and instructions must have the same length')
    conversations = []
    for instruction in instructions:
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': instruction})
        conversations.append(messages)
    # mistral-common formats BOS/INST/SYSTEM tokens itself. Do not add BOS/EOS
    # again or pass HF-only add_generation_prompt to this native tokenizer.
    if outputs is not None:
        prompts = tokenizer.apply_chat_template(conversations)
        sequences = [prompt + tokenizer.encode(output, add_special_tokens=False)
                     for prompt, output in zip(prompts, outputs)]
        return tokenizer.pad({'input_ids': sequences}, padding=True, return_tensors='pt')
    return tokenizer.apply_chat_template(
        conversations, padding=True, truncation=False, return_tensors='pt', return_dict=True,
    )


class MagistralModel(Llama3Model):
    # Decoder blocks, attention/MLP hooks and weight interventions share the
    # Llama layout (embed_tokens, layers, self_attn.o_proj, mlp.down_proj).
    def __init__(self, model_name_or_path):
        if not is_mistral_common_available():
            raise ImportError('Magistral requires mistral-common==1.8.5. Install it in the Python environment running this pipeline.')
        super().__init__(model_name_or_path)

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map='auto',
            cache_dir=os.getenv('HUGGINGFACE_CACHE_DIR'),
        ).eval()
        if model.config.model_type != 'mistral':
            raise ValueError(f'Expected a Mistral decoder for Magistral, got {model.config.model_type}')
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        # The official repository provides tekken.json rather than tokenizer.json.
        return MistralCommonTokenizer.from_pretrained(
            model_path, padding_side='left', cache_dir=os.getenv('HUGGINGFACE_CACHE_DIR'),
        )

    def _load_system_prompt(self):
        if Path(self.model_name_or_path).is_dir():
            path = Path(self.model_name_or_path) / 'SYSTEM_PROMPT.txt'
            return path.read_text() if path.exists() else None
        path = hf_hub_download(self.model_name_or_path, 'SYSTEM_PROMPT.txt',
                               cache_dir=os.getenv('HUGGINGFACE_CACHE_DIR'))
        return Path(path).read_text()

    def _get_tokenize_instructions_fn(self):
        self.system_prompt = self._load_system_prompt()
        return functools.partial(tokenize_instructions_magistral,
                                 tokenizer=self.tokenizer, system=self.system_prompt)

    def _get_eoi_toks(self):
        # Derive actual closing control tokens from native chat encoding: encoding
        # the literal string [/INST] as ordinary text does not produce its ID.
        first, second = self.tokenize_instructions_fn(instructions=['red', 'blue']).input_ids.tolist()
        suffix = []
        for left, right in zip(reversed(first), reversed(second)):
            if left != right:
                break
            suffix.append(left)
        if not suffix:
            raise ValueError('Magistral chat prompts have no shared end-of-instruction tokens')
        return list(reversed(suffix))

    def _get_refusal_toks(self):
        # Token IDs differ from Qwen/Llama. Resolve refusal prefixes in Tekken.
        return sorted({self.tokenizer.encode(prefix, add_special_tokens=False)[0]
                       for prefix in ('I', 'As')})
