import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.model_utils.olmo_model import OlmoModel


class Falcon3Model(OlmoModel):
    """Falcon3 adapter using the shared Llama-compatible layer operations."""

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"),
        ).eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer_path = (
            "tiiuae/Falcon3-7B-Instruct"
            if model_path == "tiiuae/Falcon3-7B-Base"
            else model_path
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"),
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
