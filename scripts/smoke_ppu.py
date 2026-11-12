"""Small real-backend smoke checks, without downloading pretrained model weights."""
import ast
import functools
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from runtime_support import configure_runtime, check_data_runtime


def load_rdo_operations(model):
    from baselines import additive_rotation_ops
    source = ast.parse((ROOT / 'baselines/rdo_refusal.py').read_text())
    names = {'RefusalDirectionActivationRotation', 'OrthogonalProjection',
             'RefusalDirectionActivationAdditiveRotation', 'RefusalStiefelRotation',
             'RefusalStiefelProjRotation', 'RefusalStiefelAdditiveRotation',
             '_runtime_device', '_layer_scores_from_refusal_directions', '_build_optimized_layer_idxs'}
    nodes = [n for n in source.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    scope = dict(vars(additive_rotation_ops))
    scope.update(torch=torch, nn=nn, np=np, Envoy=object, model=model, time=time,
                 args=SimpleNamespace(), functools=functools)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'rdo-smoke-definitions', 'exec'), scope)
    return scope


def smoke_rdo(device, dtype):
    from nnsight import LanguageModel
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, 'hi': 2}, unk_token='[UNK]')),
        pad_token='[PAD]', unk_token='[UNK]')
    raw = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4)).to(device=device, dtype=dtype).eval()
    raw.requires_grad_(False)
    model = LanguageModel(raw, tokenizer=tokenizer)
    scope = load_rdo_operations(model)
    for name in ['RefusalDirectionActivationAdditiveRotation', 'RefusalStiefelAdditiveRotation']:
        operation = scope[name](model.model, 32, k_proj=8, num_opt_layers=1, best_layer=1)
        parameters = list(operation.parameters())
        before = [p.detach().clone() for p in parameters]
        optimizer = torch.optim.AdamW(parameters, lr=1e-5)
        with model.trace(torch.tensor([[2, 1, 2]])):
            operation(None)
            loss = model.lm_head.output.float().square().mean()
            loss.backward()
        assert any(p.grad is not None for p in parameters)
        assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
        optimizer.step()
        operation.orthogonalize()
        assert any(not torch.equal(p, old) for p, old in zip(parameters, before))
        print(f'[smoke] {name}: real NNsight backward/AdamW/orthogonalization OK', flush=True)


def smoke_dim(device, dtype):
    sys.path.insert(0, str(ROOT / 'geometry-of-refusal/refusal_direction'))
    from transformers import MistralConfig, MistralForCausalLM
    from pipeline.model_utils.magistral_model import MagistralModel
    from pipeline.model_utils.model_factory import construct_model_base
    from pipeline.submodules.select_direction import get_refusal_scores
    raw = MistralForCausalLM(MistralConfig(vocab_size=131072, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            sliding_window=None)).to(device=device, dtype=dtype).eval()
    raw.requires_grad_(False)
    with patch.object(MagistralModel, '_load_model', return_value=raw):
        adapter = construct_model_base('mistralai/Magistral-Small-2506')
    scores = get_refusal_scores(raw, ['Hello'], adapter.tokenize_instructions_fn, adapter.refusal_toks, batch_size=1)
    assert torch.isfinite(scores).all()
    assert adapter.eoi_toks == [4]
    print('[smoke] Magistral native tokenizer/system prompt + Mistral scoring OK', flush=True)


if __name__ == '__main__':
    device, dtype = configure_runtime('cuda')
    check_data_runtime()
    {'rdo': smoke_rdo, 'dim': smoke_dim}[sys.argv[1]](device, dtype)
    torch.cuda.synchronize(device)
    print('[smoke] PASS', flush=True)
