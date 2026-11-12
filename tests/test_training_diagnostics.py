"""Offline diagnostics checks; nnsight integration runs when 0.3.7 is installed."""

import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.fake_tensor import FakeTensor

from baselines.training_diagnostics import (
    OPERATOR_FIELDS, TIMING_FIELDS, TrainingDiagnostics, _Span,
    _ACTIVE, current_diagnostics, operator_metrics, steering_begin, steering_end,
    with_training_diagnostics,
)
import test_orthogonalize_flag as orthogonalize_tests
from test_reft_smoke import ROOT, pipeline_helpers

# nnsight 0.3.7 globally patches this constructor using an older signature.
# Keep the native constructor for the independent modern-PyTorch fake-mode test.
_NATIVE_FAKE_NEW = FakeTensor.__new__


def rows(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


class EagerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(4))
        self.steer = True
        self.fail = False

    def forward(self, x):
        if self.steer:
            x, token = steering_begin(x)
            x = steering_end(x @ self.weight, token)
        if self.fail:
            raise RuntimeError("test failure")
        return torch.sin(x)


class TestTrainingDiagnostics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_operator_metrics_orthogonal_raw_and_nonfinite(self):
        basis = torch.eye(5, 3, requires_grad=True)
        rotation = torch.eye(3, requires_grad=True)
        result = operator_metrics(basis, rotation)
        self.assertAlmostEqual(result["b_frobenius"], 3**0.5, places=6)
        self.assertEqual(result["b_orthogonality_error"], 0)
        self.assertEqual(result["q_orthogonality_error"], 0)
        B, Q = basis.detach()*2, rotation.detach()*3
        result = operator_metrics(B, Q)
        self.assertAlmostEqual(result["b_orthogonality_error"], 3*3**0.5, places=5)
        self.assertAlmostEqual(result["q_orthogonality_error"], 8*3**0.5, places=5)
        result = operator_metrics(B.fill_(float("nan")), Q)
        self.assertTrue(torch.isnan(torch.tensor(result["b_frobenius"])))
        self.assertIsNone(basis.grad)
        self.assertIsNone(rotation.grad)

    def test_operator_csv_cadence_layers_final_and_no_reprojection(self):
        helper = orthogonalize_tests.TestOrthogonalizeFlag()
        helper.h = pipeline_helpers(include_additive=True)
        for method in orthogonalize_tests.METHODS:
            for enabled in (True, False):
                with self.subTest(method=method, enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                    op = helper.make_operation(method, enabled)
                    if not enabled:
                        with torch.no_grad():
                            (op.basis_param if method == orthogonalize_tests.METHODS[0] else op.proj_A)[2].mul_(2)
                    with TrainingDiagnostics(tmp, method, gap=2) as diag:
                        diag.attach_operation(op)
                        for step in range(1, 4):
                            diag.log_operators(step)
                    data = rows(Path(tmp)/"operator_metrics.csv")
                    self.assertEqual(list(data[0]), list(OPERATOR_FIELDS))
                    self.assertEqual([int(r["optimizer_step"]) for r in data], [0, 1, 2, 3])
                    self.assertTrue(all(r["layer"] == "2" for r in data))
                    if not enabled:
                        self.assertGreater(float(data[0]["b_orthogonality_error"]), 1)
                    else:
                        self.assertLess(float(data[0]["b_orthogonality_error"]), 1e-5)
                    # Only the final row is forced, and its existing step is not repeated.
                    with tempfile.TemporaryDirectory() as other:
                        with TrainingDiagnostics(other, method, gap=2) as diag:
                            diag.attach_operation(op)
                            diag.log_operators(2)
                        self.assertEqual(len(rows(Path(other)/"operator_metrics.csv")), 2)

    def test_eager_runtime_forward_sampling_and_backward_exclusion(self):
        model = EagerModel()
        with tempfile.TemporaryDirectory() as tmp:
            with TrainingDiagnostics(tmp, "direft", gap=3, model=model) as diag:
                for step in range(1, 5):
                    for micro in (1, 2):
                        diag.begin_microbatch(step, micro)
                        model.steer = micro == 1
                        x = torch.randn(2, 5, 4, requires_grad=True)
                        result = model(x)
                        diag.end_microbatch()
                        result.sum().backward()
                    self.assertFalse(model._forward_pre_hooks)
                    self.assertFalse(model._forward_hooks)
            data = rows(Path(tmp)/"training_forward_timing.csv")
            self.assertEqual(list(data[0]), list(TIMING_FIELDS))
            self.assertEqual([int(r["optimizer_step"]) for r in data], [1, 1, 3, 3])
            self.assertEqual([int(r["steering_call_count"]) for r in data], [1, 0, 1, 0])
            for row in data:
                self.assertGreater(float(row["model_forward_ms"]), 0)
                self.assertGreaterEqual(float(row["steering_application_ms"]), 0)
                self.assertLessEqual(float(row["steering_application_ms"]), float(row["model_forward_ms"]))
                self.assertEqual(row["batch_size"], "2")
                self.assertEqual(row["sequence_length"], "5")
                self.assertEqual(row["timing_backend"], "cpu_perf_counter")
            self.assertFalse((Path(tmp)/"operator_metrics.csv").exists())

    def test_fake_passes_disabled_sampling_and_exception_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = EagerModel()
            with TrainingDiagnostics(tmp, "baseline", gap=100, model=model) as diag:
                diag.begin_microbatch(1, 1)
                with patch.object(FakeTensor, "__new__", staticmethod(_NATIVE_FAKE_NEW)), FakeTensorMode(allow_non_fake_inputs=True):
                    model(torch.randn(2, 4))
                diag.end_microbatch()
            self.assertEqual(list(Path(tmp).iterdir()), [])
            with TrainingDiagnostics(tmp, "baseline", gap=0, model=model) as diag:
                diag.begin_microbatch(1, 1)
                model(torch.randn(2, 4))
            self.assertEqual(list(Path(tmp).iterdir()), [])
            self.assertIsNone(_ACTIVE.get())
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with TrainingDiagnostics(tmp, "baseline", model=model) as diag:
                    diag.begin_microbatch(1, 1)
                    model.fail = True
                    model(torch.randn(2, 4))
            self.assertFalse(model._forward_hooks)
            self.assertFalse(model._forward_pre_hooks)
            self.assertIsNone(_ACTIVE.get())

    def test_existing_files_not_overwritten_and_handles_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = EagerModel()
            path = Path(tmp)/"training_forward_timing.csv"
            path.write_text("existing\n")
            with self.assertRaises(FileExistsError):
                with TrainingDiagnostics(tmp, "baseline", model=model) as diag:
                    diag.begin_microbatch(1, 1)
                    model(torch.randn(2, 4))
            self.assertEqual(path.read_text(), "existing\n")
            self.assertFalse(model._forward_hooks)
            self.assertIsNone(current_diagnostics())

    def test_logits_marker_excludes_inline_backward(self):
        clock = [0.0]
        def now():
            clock[0] += 0.001
            return clock[0]
        class InlineBackward(EagerModel):
            def forward(self, x):
                logits = super().forward(x)
                logits = current_diagnostics().forward_output(logits)
                def backward(grad):
                    clock[0] += 10.0
                    return grad
                logits.register_hook(backward)
                logits.sum().backward()
                return logits
        model = InlineBackward()
        with tempfile.TemporaryDirectory() as tmp, patch("time.perf_counter", side_effect=now):
            with TrainingDiagnostics(tmp, "baseline", model=model) as diag:
                diag.begin_microbatch(1, 1)
                model(torch.randn(2, 4))
                diag.end_microbatch()
            row = rows(Path(tmp)/"training_forward_timing.csv")[0]
            self.assertGreater(clock[0], 10)
            self.assertLess(float(row["model_forward_ms"]), 10)

    def test_decorator_closes_and_flushes_on_training_failure(self):
        @with_training_diagnostics
        def train(model, direction_mode="baseline", tb_checkpoint_dir=None, train_diagnostics_gap=100):
            diag = current_diagnostics()
            diag.begin_microbatch(1, 1)
            model(torch.randn(2, 4))
            raise RuntimeError("training failure")
        with tempfile.TemporaryDirectory() as tmp:
            model = EagerModel()
            with self.assertRaisesRegex(RuntimeError, "training failure"):
                train(model, tb_checkpoint_dir=str(Path(tmp)/"checkpoints"))
            self.assertEqual(len(rows(Path(tmp)/"training_forward_timing.csv")), 1)
            self.assertFalse(model._forward_hooks)
            self.assertFalse(model._forward_pre_hooks)
            self.assertIsNone(current_diagnostics())

    def test_cuda_event_bookkeeping_without_hardware(self):
        event = lambda: unittest.mock.Mock()
        start, end = event(), event()
        start.elapsed_time.return_value = 2.5
        with patch("torch.cuda.device", return_value=contextlib.nullcontext()), \
             patch("torch.cuda.current_stream", return_value="stream"), \
             patch("torch.cuda.Event", side_effect=[start, end]):
            span = _Span("cuda:0")
            span.stop()
            self.assertEqual(span.milliseconds(synchronize=True), 2.5)
        start.record.assert_called_once_with("stream")
        end.record.assert_called_once_with("stream")
        end.synchronize.assert_called_once()

    def test_parser_and_checkpoint_cleanup_preserve_csv(self):
        h = pipeline_helpers()
        for gap in (0, 100):
            with patch.object(sys, "argv", ["rdo_refusal.py", "--train_diagnostics_gap", str(gap)]):
                self.assertEqual(h.parse_args().train_diagnostics_gap, gap)
        with patch.object(sys, "argv", ["rdo_refusal.py", "--train_diagnostics_gap", "-1"]), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            h.parse_args()
        import ast
        tree = ast.parse((ROOT/"baselines/rdo_refusal.py").read_text())
        cleanup = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in {"_clear_experiment_checkpoints", "_safe_json_dump"}]
        namespace = dict(os=__import__("os"), json=__import__("json"), Path=Path,
                         args=__import__("types").SimpleNamespace())
        exec(compile(ast.Module(body=cleanup, type_ignores=[]), "cleanup", "exec"), namespace)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/"checkpoints").mkdir()
            (Path(tmp)/"checkpoints/test.pt").touch()
            for name in ("operator_metrics.csv", "training_forward_timing.csv"):
                (Path(tmp)/name).write_text("keep")
            namespace["_clear_experiment_checkpoints"](tmp)
            self.assertEqual((Path(tmp)/"operator_metrics.csv").read_text(), "keep")
            self.assertEqual((Path(tmp)/"training_forward_timing.csv").read_text(), "keep")

    def test_launcher_diagnostics_environment_and_cli_precedence(self):
        h = pipeline_helpers()
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Path(tmp)/"python"
            recorder.write_text("#!" + sys.executable + "\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
            recorder.chmod(0o755)
            for env_gap, cli_gap, expected in ((None, None, 100), ("17", None, 17), ("17", "0", 0)):
                env = dict(os.environ, RDO_PYTHON=str(recorder), RDO_DEVICE="cpu")
                env.pop("TRAIN_DIAGNOSTICS_GAP", None)
                if env_gap is not None:
                    env["TRAIN_DIAGNOSTICS_GAP"] = env_gap
                command = ["bash", str(ROOT/"scripts/run_rdo_refusal.sh"), "--device", "cpu"]
                if cli_gap is not None:
                    command += ["--train_diagnostics_gap", cli_gap]
                forwarded = json.loads(subprocess.check_output(command, env=env, text=True))
                with patch.object(sys, "argv", ["rdo_refusal.py", *forwarded[3:]]):
                    self.assertEqual(h.parse_args().train_diagnostics_gap, expected)


@unittest.skipUnless(importlib.util.find_spec("nnsight"), "install nnsight==0.3.7 for runtime integration")
class TestNnsightTrainingDiagnostics(unittest.TestCase):
    def test_logits_marker_stops_before_deferred_backward(self):
        from nnsight import NNsight
        class LocalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(4, 4)
            def forward(self, x):
                return self.lm_head(x)
        module = LocalModel()
        model = NNsight(module)
        clock = [0.0]
        def now():
            clock[0] += 0.001
            return clock[0]
        def backward(grad):
            clock[0] += 1000.0
            return grad
        handle = module.lm_head.weight.register_hook(backward)
        try:
            with tempfile.TemporaryDirectory() as tmp, patch("time.perf_counter", side_effect=now):
                with TrainingDiagnostics(tmp, "baseline", model=module) as diag:
                    diag.begin_microbatch(1, 1)
                    with model.trace(torch.randn(2, 4)):
                        logits = diag.forward_output(model.lm_head.output)
                        logits.square().sum().backward()
                    diag.end_microbatch()
                self.assertGreater(clock[0], 1000)
                row = rows(Path(tmp)/"training_forward_timing.csv")[0]
                self.assertLess(float(row["model_forward_ms"]), 100)
        finally:
            handle.remove()

    def test_deferred_runtime_combined_batches_and_gradients(self):
        from nnsight import NNsight
        class LocalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(4, 4, bias=False)
            def forward(self, x):
                return self.layer(x).tanh()
        torch.manual_seed(12)
        module = LocalModel()
        model = NNsight(module)
        x = torch.randn(2, 3, 4)
        shift = nn.Parameter(torch.full((4,), 0.1))
        with tempfile.TemporaryDirectory() as tmp:
            with TrainingDiagnostics(tmp, "baseline", gap=100, model=module) as diag:
                diag.begin_microbatch(1, 1)
                with model.trace() as tracer:
                    with tracer.invoke(x):
                        reference = diag.forward_output(model.output).save()
                    with tracer.invoke(x):
                        activation, token = steering_begin(model.layer.input)
                        model.layer.input = steering_end(activation + shift, token)
                        actual = diag.forward_output(model.output).save()
                        loss = actual.square().sum()
                    loss.backward()
                diag.end_microbatch()
            expected = module.layer(x + shift).tanh()
            torch.testing.assert_close(actual.value, expected)
            torch.testing.assert_close(reference.value, module(x))
            self.assertIsNotNone(shift.grad)
            self.assertGreater(shift.grad.norm().item(), 0)
            expected_gradient, = torch.autograd.grad(expected.square().sum(), shift)
            torch.testing.assert_close(shift.grad, expected_gradient)
            data = rows(Path(tmp)/"training_forward_timing.csv")
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["batch_size"], "4")
            self.assertEqual(data[0]["steering_call_count"], "1")
            self.assertGreater(float(data[0]["steering_application_ms"]), 0)
            self.assertGreater(float(data[0]["model_forward_ms"]), float(data[0]["steering_application_ms"]))

    def test_actual_additive_application_through_nnsight(self):
        from nnsight import NNsight
        from types import SimpleNamespace
        helper = orthogonalize_tests.TestOrthogonalizeFlag()
        helper.h = pipeline_helpers(include_additive=True)
        class LocalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x).tanh()
                return x
        for method in orthogonalize_tests.METHODS:
            for constrained in (True, False):
                with self.subTest(method=method, constrained=constrained), tempfile.TemporaryDirectory() as tmp:
                    module = LocalModel().requires_grad_(False)
                    model = NNsight(module)
                    op = helper.make_operation(method, constrained)
                    op.module = SimpleNamespace(layers=list(model.layers))
                    with TrainingDiagnostics(tmp, method, model=module) as diag:
                        diag.attach_operation(op)
                        diag.begin_microbatch(1, 1)
                        with model.trace(torch.randn(2, 3, 8)):
                            op(None)
                            result = diag.forward_output(model.output).save()
                            result.square().sum().backward()
                        diag.end_microbatch()
                    data = rows(Path(tmp)/"training_forward_timing.csv")
                    self.assertEqual(len(data), 1)
                    self.assertEqual(data[0]["steering_call_count"], "1")
                    self.assertGreater(float(data[0]["steering_application_ms"]), 0)
                    for p in op.parameters():
                        if p.requires_grad:
                            self.assertIsNotNone(p.grad)
                            self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == "__main__":
    unittest.main()
