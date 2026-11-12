"""CPU checks of the actual additive classes, without loading model weights."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_reft_smoke import ROOT, pipeline_helpers


METHODS = ("activation_additive_rot", "shtiefel_additive_rot")


class TestOrthogonalizeFlag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = pipeline_helpers(include_additive=True)
        torch.set_num_threads(1)

    def make_operation(self, method, enabled=True, orth_method="svd"):
        torch.manual_seed(42)
        module = SimpleNamespace(layers=[SimpleNamespace() for _ in range(4)])
        common = dict(k_proj=3, num_opt_layers=1, best_layer=2, init_vectors=[],
                      orthogonalize=enabled)
        if method == METHODS[0]:
            return self.h.RefusalDirectionActivationAdditiveRotation(module, 8, **common)
        return self.h.RefusalStiefelAdditiveRotation(
            module, 8, init_mode="random", orth_method=orth_method, **common)

    def factors(self, op, method):
        if method == METHODS[0]:
            return op._basis(2), self.h.RefusalDirectionActivationRotation.cayley_matrix(op, 2)
        return op._subspace_factors(2)

    def test_initialization_is_identical_for_both_settings(self):
        for method in METHODS:
            for orth_method in ("svd", "qr"):
                with self.subTest(method=method, orth_method=orth_method):
                    constrained = self.make_operation(method, True, orth_method)
                    raw = self.make_operation(method, False, orth_method)
                    for a, b in zip(constrained.parameters(), raw.parameters()):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    basis, _ = self.factors(raw, method)
                    torch.testing.assert_close(basis.T @ basis, torch.eye(3), atol=1e-5, rtol=1e-5)

    def test_default_matches_explicit_true(self):
        for cls in (self.h.RefusalDirectionActivationAdditiveRotation,
                    self.h.RefusalStiefelAdditiveRotation):
            torch.manual_seed(7)
            kwargs = dict(module=SimpleNamespace(layers=[SimpleNamespace() for _ in range(4)]),
                          dim=8, k_proj=3, num_opt_layers=1, best_layer=2, init_vectors=[])
            default = cls(**kwargs)
            torch.manual_seed(7)
            explicit = cls(**kwargs, orthogonalize=True)
            self.assertTrue(default.orthogonalize_enabled)
            for a, b in zip(default.parameters(), explicit.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_training_constraints_enabled_and_disabled(self):
        for method in METHODS:
            for enabled in (True, False):
                for orth_method in ("svd", "qr"):
                    with self.subTest(method=method, enabled=enabled, orth_method=orth_method):
                        op = self.make_operation(method, enabled, orth_method)
                        optimizer = torch.optim.Adam(op.parameters(), lr=0.04)
                        x, target = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
                        # Initialization is allowed to project. False must do no
                        # projection or retraction once training has begun.
                        with contextlib.ExitStack() as stack:
                            if not enabled:
                                stack.enter_context(patch.object(self.h.OrthogonalProjection, "apply", side_effect=AssertionError("forward projection")))
                                stack.enter_context(patch("torch.linalg.svd", side_effect=AssertionError("SVD during unconstrained training")))
                                stack.enter_context(patch("torch.linalg.qr", side_effect=AssertionError("QR during unconstrained training")))
                                stack.enter_context(patch.object(op, "orthogonalize", side_effect=AssertionError("post-step retraction")))
                            for _ in range(5):
                                optimizer.zero_grad()
                                output = op._rotate(x, 2)
                                loss = (output - target).square().mean()
                                self.assertTrue(torch.isfinite(loss))
                                loss.backward()
                                trainable = [op.basis_param, op.cayley_param] if method == METHODS[0] else [op.proj_A, op.proj_B]
                                for p in trainable:
                                    self.assertIsNotNone(p.grad)
                                    self.assertTrue(torch.isfinite(p.grad).all())
                                    self.assertGreater(p.grad[2].norm().item(), 0)
                                optimizer.step()
                                self.h._finish_operation_update(op)
                                basis, rotation = self.factors(op, method)
                                if enabled:
                                    torch.testing.assert_close(basis.T @ basis, torch.eye(3), atol=2e-5, rtol=2e-5)
                                if enabled or method == METHODS[0]:
                                    torch.testing.assert_close(rotation.T @ rotation, torch.eye(3), atol=2e-5, rtol=2e-5)
                                if method == METHODS[1]:
                                    expected = self.h.dense_additive_rotation_matrix(basis, rotation)
                                    torch.testing.assert_close(op.cayley_param[2], expected)
                        if not enabled:
                            self.assertFalse(torch.allclose(basis.T @ basis, torch.eye(3), atol=1e-4))
                            if method == METHODS[1]:
                                torch.testing.assert_close(basis, op.proj_A[2])
                                torch.testing.assert_close(rotation, op.proj_B[2] @ op.proj_A[2])
                                self.assertFalse(torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1e-4))
                            before = [p.detach().clone() for p in op.parameters()]
                            op.orthogonalize()  # Direct calls are also no-ops.
                            for old, new in zip(before, op.parameters()):
                                torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_checkpoint_reload_and_metadata(self):
        for method in METHODS:
            for enabled in (True, False):
                with self.subTest(method=method, enabled=enabled):
                    op = self.make_operation(method, enabled)
                    with torch.no_grad():
                        if method == METHODS[0]:
                            op.basis_param[2].mul_(1.4)
                            op.cayley_param[2].triu_(1).add_(0.2)
                        else:
                            op.proj_A[2].mul_(1.4)
                            op.proj_B[2].mul_(1.2)
                    self.h._finish_operation_update(op)
                    x = torch.randn(2, 3, 8)
                    with tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        checkpoint = op.checkpoint_matrix()
                        torch.save(checkpoint, root / "matrix.pt")
                        self.h._save_progress_operation_checkpoint(tmp, 5, method, op, 4)
                        self.h._write_tb_checkpoint_layer_artifacts(tmp, op, method, 4, 2, 1)
                        for path in (root / "progress_checkpoints/iters_00000005.pt", root / "active_layers.pt"):
                            self.assertIs(torch.load(path, weights_only=True)["orthogonalize"], enabled)
                        for reload_setting in (True, False):
                            restored = self.make_operation(method, reload_setting)
                            self.h._prime_rotation_model_cache(restored, torch.load(root / "matrix.pt", weights_only=True))
                            for mode in ("attack", "protect"):
                                op.set_guard_mode(mode)
                                restored.set_guard_mode(mode)
                                torch.testing.assert_close(restored._rotate(x, 2), op._rotate(x, 2), atol=1e-5, rtol=1e-5)

    def parse(self, *args):
        with patch.object(sys, "argv", ["rdo_refusal.py", *args]):
            return self.h.parse_args()

    def test_parser_defaults_values_and_rejection(self):
        self.assertIs(self.parse().orthogonalize, True)
        for method in METHODS:
            for text, expected in (("true", True), ("false", False)):
                args = self.parse("--direction_mode", method, "--orthogonalize", text)
                self.assertIs(args.orthogonalize, expected)
        for args in (("--orthogonalize", "invalid"), ("--orthogonalize", "false"),
                     ("--direction_mode", "loreft", "--orthogonalize", "false"),
                     ("--direction_mode", "direft", "--orthogonalize", "false")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self.parse(*args)
            self.assertEqual(error.exception.code, 2)

    def test_script_environment_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Path(tmp) / "python"
            recorder.write_text("#!" + sys.executable + "\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
            recorder.chmod(0o755)
            for method in METHODS:
                for env_value, cli_value, expected in ((None, None, True), ("false", None, False),
                                                       ("true", "false", False), ("false", "true", True)):
                    with self.subTest(method=method, env_value=env_value, cli_value=cli_value):
                        env = dict(os.environ, RDO_PYTHON=str(recorder), RDO_DEVICE="cpu", DIRECTION_MODE=method)
                        env.pop("ORTHOGONALIZE", None)
                        if env_value is not None:
                            env["ORTHOGONALIZE"] = env_value
                        command = ["bash", str(ROOT / "scripts/run_rdo_refusal.sh"), "--device", "cpu"]
                        if cli_value is not None:
                            command += ["--orthogonalize", cli_value]
                        forwarded = json.loads(subprocess.check_output(command, env=env, text=True))
                        args = self.parse(*forwarded[3:])
                        self.assertIs(args.orthogonalize, expected)

    def test_container_environment_forwarding_without_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.sif"
            image.touch()
            venv = root / "venv"
            venv.mkdir()
            (venv / "pyvenv.cfg").touch()
            (venv / ".rdo-requirements.sha256").touch()
            for name, body in {
                "sha256sum": "#!/bin/bash\nexit 0\n",
                "apptainer": "#!" + sys.executable + "\nimport os,json\nprint(json.dumps([os.environ['APPTAINERENV_'+k] for k in ['ORTHOGONALIZE','TRAIN_DIAGNOSTICS_GAP','CLEAR_CKPTS','RDO_LLAMAGUARD_STRICT']]))\n",
            }.items():
                script = root / name
                script.write_text(body)
                script.chmod(0o755)
            for value in ("true", "false"):
                env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"],
                           RDO_REPO_ROOT=str(ROOT), RDO_PPU_IMAGE=str(image),
                           RDO_PPU_VENV=str(venv), SLURM_JOB_ID="smoke-test",
                           CUDA_VISIBLE_DEVICES="mock", ORTHOGONALIZE=value, TRAIN_DIAGNOSTICS_GAP="17",
                           CLEAR_CKPTS="false", RDO_LLAMAGUARD_STRICT="1")
                output = subprocess.check_output(
                    ["bash", str(ROOT / "scripts/launch_ppu.sh"), "rdo"], env=env, text=True)
                self.assertEqual(json.loads(output), [value, "17", "false", "1"])


if __name__ == "__main__":
    unittest.main()
