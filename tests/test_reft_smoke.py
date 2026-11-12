"""Offline CPU smoke checks: python -m unittest discover -s tests -p test_reft_smoke.py -v."""

import argparse
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from baselines.reft_ops import DiReFT, LoReFT, REFT_MODES, ReFTIntervention

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rdo_locality", ROOT / "evaluate/rdo_locality.py")
rdo_locality = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rdo_locality)


def pipeline_helpers():
    """Execute real pipeline helpers without its import-time model/data loading."""
    tree = ast.parse((ROOT / "baselines/rdo_refusal.py").read_text())
    names = {
        "parse_args", "_layer_index_sets", "_save_progress_operation_checkpoint",
        "_write_tb_checkpoint_layer_artifacts", "_load_training_results_from_result_path",
        "_extract_refined_artifact", "_prime_rotation_model_cache",
        "_supports_protect_guard_mode", "_build_train_guard_step_fn",
    }
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    nodes += [n for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "DEFAULT_CONFIG" for t in n.targets)]
    nodes.sort(key=lambda n: n.lineno)
    namespace = dict(torch=torch, nn=nn, os=os, sys=sys, argparse=argparse,
                     rdo_locality=rdo_locality,
                     REFT_MODES=REFT_MODES, ReFTIntervention=ReFTIntervention,
                     _PAPER_ANGULAR_MODE="paper_angular_steering",
                     _PAPER_SPHERICAL_MODE="paper_spherical_steering")
    with patch.dict(os.environ, {"MAX_ITERS": "5"}):
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "rdo_helpers", "exec"), namespace)
    return SimpleNamespace(**namespace)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(13, 8)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Tanh()) for _ in range(4)])
        self.head = nn.Linear(8, 13)
        self.requires_grad_(False)

    def forward(self, ids):
        h = self.embed(ids)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


class TestReFTSmoke(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_formulas_initialization_and_signs(self):
        for kind in (LoReFT, DiReFT):
            with self.subTest(method=kind.__name__):
                op = kind(8, 3)
                h = torch.randn(2, 5, 8)
                if kind is LoReFT:
                    torch.testing.assert_close(op.W, op.R)
                    self.assertNotEqual(op.W.data_ptr(), op.R.data_ptr())
                    torch.testing.assert_close(op(h), h)
                else:
                    torch.testing.assert_close(op.W1, op.W2)
                    self.assertNotEqual(op.W1.data_ptr(), op.W2.data_ptr())
                    torch.testing.assert_close(op(h), h + h @ op.W1.T @ op.W1)
                    self.assertFalse(torch.allclose(op(h), h))
                torch.testing.assert_close(op.b, torch.zeros(3))
                with torch.no_grad():
                    op.b.fill_(0.2)
                    (op.W if kind is LoReFT else op.W1).add_(0.1)
                for shape in ((8,), (4, 8), (2, 5, 8)):
                    x = torch.randn(shape)
                    if kind is LoReFT:
                        delta = (op.R.T @ ((op.W - op.R) @ x.unsqueeze(-1)
                                 + op.b[:, None])).squeeze(-1)
                    else:
                        delta = (op.W2.T @ (op.W1 @ x.unsqueeze(-1)
                                 + op.b[:, None])).squeeze(-1)
                    torch.testing.assert_close(op(x), x + delta)
                    torch.testing.assert_close(op(x, sign=-1), x - delta)
                    torch.testing.assert_close(op(x) + op(x, sign=-1), 2 * x)

    def test_tiny_lm_training_and_checkpoint_round_trip(self):
        helpers = pipeline_helpers()
        for method in sorted(REFT_MODES):
            with self.subTest(method=method):
                model = TinyLM()
                frozen = {k: v.clone() for k, v in model.state_dict().items()}
                adapter = ReFTIntervention(model, 8, method, 3, num_opt_layers=1, best_layer=2)
                self.assertEqual(adapter._optimized_layer_idxs, {2})
                before = adapter.export_checkpoint()
                optimizer = torch.optim.Adam(adapter.parameters(), lr=0.03)
                ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
                targets = (ids + 1) % 13
                losses = []
                for _ in range(8):
                    optimizer.zero_grad()
                    with adapter.hooks():
                        logits = model(ids)
                        loss = nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
                        loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    for param in adapter.parameters():
                        self.assertIsNotNone(param.grad)
                        self.assertTrue(torch.isfinite(param.grad).all())
                    optimizer.step()
                    adapter.orthogonalize()
                    losses.append(loss.item())
                    if method == "loreft":
                        R = adapter.operators["2"].R
                        torch.testing.assert_close(R @ R.T, torch.eye(3), atol=1e-5, rtol=1e-5)
                self.assertLess(losses[-1], losses[0])
                self.assertTrue(any(not torch.equal(v, before["state_dict"][k])
                                    for k, v in adapter.state_dict().items()))
                self.assertGreater(adapter.operators["2"].b.norm().item(), 0)
                for k, v in model.state_dict().items():
                    torch.testing.assert_close(v, frozen[k], rtol=0, atol=0)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertTrue(all(not layer._forward_pre_hooks for layer in model.layers))

                with tempfile.TemporaryDirectory() as tmp:
                    ckpt = Path(tmp) / "checkpoints"
                    ckpt.mkdir()
                    snapshot = adapter.export_checkpoint()
                    torch.save(snapshot, ckpt / "lowest_loss_reft.pt")
                    torch.save(adapter.stack_directions_for_log(), ckpt / "lowest_loss_vector.pt")
                    helpers._save_progress_operation_checkpoint(str(ckpt), 8, method, adapter, 4)
                    helpers._write_tb_checkpoint_layer_artifacts(str(ckpt), adapter, method, 4, 2, 1)
                    progress = torch.load(ckpt / "progress_checkpoints/iters_00000008.pt", weights_only=True)
                    self.assertEqual(progress["reft"]["method"], method)
                    active = torch.load(ckpt / "active_layers.pt", weights_only=True)
                    self.assertEqual(active["active_layer_indices"], [2])
                    results = helpers._load_training_results_from_result_path(tmp, method)
                    payload = helpers._extract_refined_artifact(results, method)
                    restored = ReFTIntervention.from_checkpoint(model, payload)
                    helpers._prime_rotation_model_cache(restored, payload)
                    for mode in ("attack", "protect"):
                        adapter.set_guard_mode(mode)
                        restored.set_guard_mode(mode)
                        with adapter.hooks():
                            expected = model(ids)
                        with restored.hooks():
                            actual = model(ids)
                        torch.testing.assert_close(actual, expected)
                    # An exported best-state snapshot must not change with training.
                    with torch.no_grad():
                        adapter.operators["2"].b.add_(1)
                    self.assertFalse(torch.equal(adapter.operators["2"].b, snapshot["state_dict"]["operators.2.b"]))

    def test_layer_adapter_and_guard_paths(self):
        helpers = pipeline_helpers()
        for method in sorted(REFT_MODES):
            for nol, expected in ((0, {0, 1, 2, 3}), (4, {0, 1, 2, 3}), (1, {2})):
                with self.subTest(method=method, num_opt_layers=nol):
                    x = torch.randn(2, 3, 8)
                    layers = [SimpleNamespace(input=x.clone()) for _ in range(4)]
                    adapter = ReFTIntervention(SimpleNamespace(layers=layers), 8, method, 3,
                                               num_opt_layers=nol, best_layer=2)
                    self.assertEqual(adapter._optimized_layer_idxs, expected)
                    with torch.no_grad():
                        for op in adapter.operators.values():
                            op.b.fill_(0.4)
                    for mode in ("attack", "protect"):
                        for layer in layers:
                            layer.input = x.clone()
                        step = helpers._build_train_guard_step_fn(adapter, method, 2, mode)
                        step(None)
                        self.assertTrue(helpers._supports_protect_guard_mode(method))
                        for i, layer in enumerate(layers):
                            torch.testing.assert_close(layer.input, adapter.apply_tensor(x, i))
                        layers[2].input = x.clone()
                        adapter.add(None, 1.0, 2)
                        torch.testing.assert_close(layers[2].input + adapter.apply_tensor(x, 2), 2*x)
                    untouched = adapter.apply_tensor(x, next(iter(set(range(4)) - expected), 10))
                    self.assertIs(untouched, x)

    def test_invalid_rank_and_checkpoint_metadata(self):
        for kind in (LoReFT, DiReFT):
            for rank in (0, 9):
                with self.assertRaises(ValueError):
                    kind(8, rank)
        model = TinyLM()
        adapter = ReFTIntervention(model, 8, "loreft", 3)
        for key, value in (("version", 99), ("method", "direft"), ("dim", 9), ("n_layers", 9)):
            payload = adapter.export_checkpoint()
            payload[key] = value
            with self.assertRaises(ValueError):
                adapter.load_checkpoint(payload)
        with self.assertRaises(ValueError):
            adapter.set_guard_mode("invalid")

    def test_cli_and_shell_forwarding_on_cpu(self):
        helpers = pipeline_helpers()
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Path(tmp) / "record.py"
            recorder.write_text("#!" + sys.executable + "\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
            recorder.chmod(0o755)
            import json
            for method in sorted(REFT_MODES):
                env = dict(os.environ, RDO_PYTHON=str(recorder), DIRECTION_MODE=method,
                           RDO_PLATFORM="ppu", RDO_DEVICE="cpu", K_PROJ="3")
                output = subprocess.check_output(
                    ["bash", str(ROOT / "scripts/run_rdo_refusal.sh"), "--device", "cpu"],
                    env=env, cwd=ROOT, text=True)
                forwarded = json.loads(output)
                self.assertEqual(forwarded[:4], ["-u", "-m", "baselines.rdo_refusal", "--train_direction"])
                with patch.object(sys, "argv", ["rdo_refusal.py"] + forwarded[3:]):
                    args = helpers.parse_args()
                self.assertEqual(args.direction_mode, method)
                self.assertEqual(args.device, "cpu")
                self.assertEqual(args.k_proj, 3)
                self.assertEqual(args.num_opt_layers, 0)


if __name__ == "__main__":
    unittest.main()
