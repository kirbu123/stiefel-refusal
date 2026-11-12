"""Fast CPU checks for bounded best-operator storage and evaluation reload."""

import gc
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock
import weakref

import torch

from baselines.best_checkpoint import BestOperatorCheckpoint
from baselines.reft_ops import ReFTIntervention
from test_reft_smoke import pipeline_helpers


class TestBestCheckpointSmoke(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_only_improvements_copy_state_and_old_best_is_released(self):
        weight = torch.nn.Parameter(torch.zeros(2, 2))
        operation = SimpleNamespace(checkpoint_matrix=Mock(side_effect=lambda: weight))
        best = BestOperatorCheckpoint("activation_additive_rot")
        losses = [5., 3., 4., 3., 2., 8.]
        previous = None
        for index, loss in enumerate(losses):
            with torch.no_grad():
                weight.fill_(index)
            improved = best.update(loss, operation, index)
            if improved:
                if previous is not None:
                    gc.collect()
                    self.assertIsNone(previous())
                previous = weakref.ref(best.artifact)
            self.assertEqual(best.artifact.device.type, "cpu")
            self.assertFalse(best.artifact.requires_grad)
            self.assertIsNone(best.artifact.grad_fn)
            self.assertNotEqual(best.artifact.data_ptr(), weight.data_ptr())
        self.assertEqual(operation.checkpoint_matrix.call_count, 3)
        self.assertEqual(best.index, losses.index(min(losses)))
        self.assertEqual(best.loss, min(losses))
        torch.testing.assert_close(best.artifact, torch.full((2, 2), 4.))
        with tempfile.TemporaryDirectory() as tmp:
            best.save(tmp)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["lowest_loss_cayley_param.pt"])

    def test_nonfinite_losses_and_parameter_fallback(self):
        operation = SimpleNamespace(cayley_param=torch.nn.Parameter(torch.eye(2)))
        best = BestOperatorCheckpoint("rotation")
        for index, loss in enumerate([float("nan"), float("inf"), -float("inf")]):
            self.assertFalse(best.update(loss, operation, index))
        with self.assertRaisesRegex(RuntimeError, "No finite optimizer-step loss"):
            best.require_best()
        self.assertTrue(best.update(torch.tensor(2.), operation, 3))
        with torch.no_grad():
            operation.cayley_param.zero_()
        torch.testing.assert_close(best.artifact, torch.eye(2))
        self.assertFalse(best.update(float("nan"), operation, 4))
        self.assertEqual(best.index, 3)

    def test_baseline_selects_index_without_operator_snapshot(self):
        best = BestOperatorCheckpoint("baseline")
        for index, loss in enumerate([3., 1., 2.]):
            best.update(loss, object(), index)
        self.assertEqual(best.index, 1)
        self.assertIsNone(best.artifact)
        with tempfile.TemporaryDirectory() as tmp:
            best.save(tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_additive_best_reload_ignores_legacy_history(self):
        h = pipeline_helpers(include_additive=True)
        for mode, cls in (("activation_additive_rot", h.RefusalDirectionActivationAdditiveRotation),
                          ("shtiefel_additive_rot", h.RefusalStiefelAdditiveRotation)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                module = SimpleNamespace(layers=[SimpleNamespace() for _ in range(4)])
                op = cls(module, 8, init_vectors=[], k_proj=3, num_opt_layers=1, best_layer=2)
                optimizer = torch.optim.Adam(op.parameters(), lr=0.01)
                best = BestOperatorCheckpoint(mode)
                expected = None
                x = torch.randn(2, 8)
                target = torch.randn_like(x)
                for index, score in enumerate([3., 1., 2.]):
                    optimizer.zero_grad()
                    (op._rotate(x, 2)-target).square().mean().backward()
                    optimizer.step()
                    h._finish_operation_update(op)
                    best.update(score, op, index)
                    if index == 1:
                        expected = op.checkpoint_matrix().detach().clone()
                run = Path(tmp)
                best.save(run/"checkpoints")
                torch.save(torch.zeros(1, 8), run/"checkpoints/lowest_loss_vector.pt")
                # Legacy histories may be huge or incomplete after disk exhaustion.
                # Neither should be opened, nor removed, during best-state reload.
                history = run/"checkpoints/cayley_params.pt"
                history.write_bytes(b"legacy history must not be read")
                loaded = h._load_training_results_from_result_path(str(run), mode)
                artifact = h._extract_refined_artifact(loaded, mode)
                torch.testing.assert_close(artifact, expected)
                self.assertNotIn("cayley_params", loaded)
                self.assertTrue(history.exists())
                restored = cls(module, 8, init_vectors=[], k_proj=3, num_opt_layers=1, best_layer=2)
                h._prime_rotation_model_cache(restored, artifact)
                restored.set_guard_mode("attack")
                torch.testing.assert_close(restored._rotate(x, 2), x @ expected[2].T)
                restored.set_guard_mode("protect")
                torch.testing.assert_close(restored._rotate(x, 2), x @ expected[2])

    def test_reft_keeps_one_independent_state_and_reloads_best(self):
        h = pipeline_helpers()
        for mode in ("loreft", "direft"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                module = SimpleNamespace(layers=[SimpleNamespace() for _ in range(4)])
                op = ReFTIntervention(module, 4, mode, k_proj=2, num_opt_layers=1, best_layer=2)
                best = BestOperatorCheckpoint(mode)
                for index, score in enumerate([3., 1., 2.]):
                    with torch.no_grad():
                        op.operators["2"].b.fill_(index)
                    best.update(score, op, index)
                run = Path(tmp)
                best.save(run/"checkpoints")
                torch.save(torch.zeros(1, 4), run/"checkpoints/lowest_loss_vector.pt")
                self.assertFalse((run/"checkpoints/reft_params.pt").exists())
                (run/"checkpoints/reft_params.pt").write_bytes(b"ignored legacy history")
                results = h._load_training_results_from_result_path(str(run), mode)
                payload = h._extract_refined_artifact(results, mode)
                restored = ReFTIntervention.from_checkpoint(module, payload)
                torch.testing.assert_close(restored.operators["2"].b, torch.ones(2))
                self.assertEqual(best.index, 1)
                self.assertNotIn("cayley_params", results)


if __name__ == "__main__":
    unittest.main()
