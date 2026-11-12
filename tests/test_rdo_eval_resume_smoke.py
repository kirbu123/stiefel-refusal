"""Offline CPU evaluation from saved checkpoints, without Heretic utils/models.

Generation uses tiny CPU tensor operations and guard classifiers are test doubles;
the real evaluation orchestration, cache, JSON/CSV and TensorBoard writers run.
"""

import __future__
import ast
from collections import Counter
import contextlib
import csv
from datetime import datetime
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import traceback
import unittest
from unittest.mock import Mock, patch

import torch

from test_reft_smoke import ROOT, pipeline_helpers


def evaluation_namespace():
    h = pipeline_helpers(include_additive=True)
    namespace = h._load_training_results_from_result_path.__globals__
    wanted = {
        "_basic_llamaguard_targets_paths", "_build_basic_llamaguard_targets_cache",
        "_load_llamaguard_eval_sets", "_save_generated_responses", "_evaluate_end_metrics",
        "_aggregate_llamaguard", "_normalize_guard_backends", "_clear_rotation_model_cache",
        "_safe_json_dump", "_safe_eval_metrics_csv_dump", "_eval_payload_to_major_metrics_rows",
    }
    tree = ast.parse((ROOT/"baselines/rdo_refusal.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    namespace.update(os=os, torch=torch, json=json, csv=csv, datetime=datetime,
                     traceback=traceback, Counter=Counter,
                     _PAPER_INFERENCE_MODES={"paper_angular_steering", "paper_spherical_steering"})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "rdo_evaluation", "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


@unittest.skipUnless(importlib.util.find_spec("datasets") and importlib.util.find_spec("tensorboard")
                     and importlib.util.find_spec("pydantic_settings"),
                     "requires datasets, tensorboard and pydantic-settings")
class TestRdoEvaluationResumeSmoke(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ns = evaluation_namespace()
        self.datasets = __import__("datasets")
        from torch.utils.tensorboard import SummaryWriter
        self.writers = []
        def writer(**kwargs):
            result = SummaryWriter(**kwargs)
            self.writers.append(result)
            return result
        self.ns["SummaryWriter"] = writer
        self.patch_env = patch.dict(os.environ, SAVE_DIR=str(self.root/"cache"),
                                    MAX_HARMFUL="2", MAX_HARMLESS="2", RDO_LLAMAGUARD_STRICT="0")
        self.patch_env.start()
        self.addCleanup(self.patch_env.stop)
        self.patch_data = patch("data_utils.load_all_datasets_with_categories", return_value=[
            {"instruction": "harmful a", "category": "a"},
            {"instruction": "harmful b", "category": "b"},
        ])
        self.patch_data.start()
        self.addCleanup(self.patch_data.stop)
        self.prompt_dataset = self.datasets.Dataset.from_dict({"text": ["harmless a", "harmless b"]})
        self.patch_loader = patch.object(self.datasets, "load_dataset", return_value=self.prompt_dataset)
        self.loader = self.patch_loader.start()
        self.addCleanup(self.patch_loader.stop)

        self.ns["args"] = SimpleNamespace(
            eval_llamaguard=True, eval_mmlu=False, init_mode="diag_permutation",
            orth_method="svd", k_proj=3, num_opt_layers=1, orthogonalize=True,
        )
        self.ns["best_layer"] = 2
        self.ns["_guard_eval_batch_size"] = lambda batch_size, phase: batch_size
        self.ns["apply_chat_template"] = lambda tokenizer, prompts: prompts
        self.ns["_load_eval_split_json"] = lambda splits, kind, split: [
            {"instruction": kind + " example", "category": kind}]
        self.model = SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(input=torch.zeros(1, 2, 8)) for _ in range(4)]),
            config=SimpleNamespace(hidden_size=8), tokenizer=None,
        )
        self.head = torch.nn.Linear(8, 4).requires_grad_(False)
        self.generation = Mock(side_effect=self.generate)
        self.ns["_generate_nnsight_with_guard_retry"] = self.generation
        self.evaluator = SimpleNamespace(evaluate_batch=Mock(side_effect=lambda pairs, **kwargs: [
            {"score": 1.0, "unsafe_probability": 0.75, "label": "unsafe"} for _ in pairs]))
        self.ns["_get_guard_evaluator"] = Mock(return_value=self.evaluator)
        self.ns["_unload_guard_evaluator"] = Mock()

    def generate(self, *, model, prompts, intervene_step_fn=None, **kwargs):
        # Exercise the restored steering operation on real CPU tensors.
        for layer in model.model.layers:
            layer.input = torch.linspace(-1, 1, 8).expand(len(prompts), 2, 8).clone()
        if intervene_step_fn is not None:
            intervene_step_fn(model)
        logits = self.head(model.model.layers[2].input[:, -1])
        self.assertTrue(torch.isfinite(logits).all())
        return ["token " + str(token) for token in logits.argmax(-1).tolist()]

    def checkpoint(self, enabled):
        run = self.root/("constrained" if enabled else "unconstrained")
        checkpoint = run/"checkpoints"
        checkpoint.mkdir(parents=True, exist_ok=True)
        self.ns["args"].orthogonalize = enabled
        op = self.ns["RefusalStiefelAdditiveRotation"](
            self.model.model, 8, init_vectors=[], k_proj=3, num_opt_layers=1,
            best_layer=2, orthogonalize=enabled)
        with torch.no_grad():
            op.proj_A[2].mul_(1.1)
        self.ns["_finish_operation_update"](op)
        torch.save(op.checkpoint_matrix(), checkpoint/"lowest_loss_cayley_param.pt")
        torch.save(torch.ones(1, 8), checkpoint/"lowest_loss_vector.pt")
        results = self.ns["_load_training_results_from_result_path"](str(run), "shtiefel_additive_rot")
        artifact = self.ns["_extract_refined_artifact"](results, "shtiefel_additive_rot")
        return run, artifact

    def evaluate(self, run, artifact=None):
        self.ns["_evaluate_end_metrics"](
            tb_run_dir=str(run), model=self.model, refined_artifact=artifact,
            direction_mode="shtiefel_additive_rot", splits_name="saladbench",
            eval_split="test", llamaguard_data_mode="basic", model_name="tiny-model",
            eval_guard_backend=["llamaguard", "qwen3guard", "wildguard"],
            eval_max_new_tokens=2, eval_batch_size=2, mmlu_cfg={}, locality_cfg={},
        )

    def test_cache_loader_works_with_actual_local_heretic_shadow(self):
        import heretic
        self.assertEqual(Path(heretic.__file__).resolve(), ROOT/"heretic/__init__.py")
        self.assertIsNone(importlib.util.find_spec("heretic.utils"))
        from config import HARMLESS_EVAL_DATASET
        harmful, harmless = self.ns["_load_llamaguard_eval_sets"]("saladbench", "test", "basic", "tiny-model")
        self.loader.assert_called_once_with(HARMLESS_EVAL_DATASET.dataset, split=HARMLESS_EVAL_DATASET.split)
        self.assertEqual([row["instruction"] for row in harmless], ["harmless a", "harmless b"])
        self.loader.reset_mock()
        self.assertEqual(self.ns["_load_llamaguard_eval_sets"]("saladbench", "test", "basic", "tiny-model"),
                         (harmful, harmless))
        self.loader.assert_not_called()

    def test_resume_both_checkpoints_runs_evaluation_and_writes_artifacts(self):
        for enabled in (True, False):
            with self.subTest(orthogonalize=enabled):
                run, artifact = self.checkpoint(enabled)
                self.evaluate(run, artifact)
                payload = json.loads(next(run.glob("eval_metrics_*.json")).read_text())
                self.assertNotIn("llamaguard_error", payload)
                self.assertNotIn("guard_errors", payload)
                self.assertEqual(set(payload["guard_metrics"]), {"llamaguard", "qwen3guard", "wildguard"})
                for backend in payload["guard_metrics"].values():
                    for kind in ("harmful", "harmless"):
                        for mode in ("initial", "refined_attack", "refined_protect"):
                            self.assertGreater(backend[kind][mode]["n"], 0)
                with next(run.glob("eval_metrics_*.csv")).open() as file:
                    self.assertGreater(len(list(csv.DictReader(file))), 0)
                for split in ("train_answers", "test_answers"):
                    self.assertEqual(len(list((run/"generated_responses"/split).glob("*.json"))), 3)
                self.assertTrue(list(run.glob("events.out.tfevents.*")))
                self.assertTrue((run/"checkpoints/lowest_loss_cayley_param.pt").is_file())
                self.assertIsNone(self.writers[-1].file_writer)

    def test_dataset_or_generation_failure_preserves_original_error(self):
        for failure in ("dataset", "first_generation", "second_generation", "refined_generation"):
            with self.subTest(failure=failure):
                run, artifact = self.checkpoint(failure == "dataset")
                # Keep each failure's artifacts in a distinct experiment.
                run = self.root/failure
                run.mkdir()
                boom = RuntimeError("original evaluation failure")
                if failure == "dataset":
                    self.loader.side_effect = boom
                else:
                    count = {"first_generation": 1, "second_generation": 2, "refined_generation": 3}[failure]
                    self.generation.side_effect = [(["response"]*2)]*(count-1) + [boom]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.evaluate(run, artifact)
                payload = json.loads(next(run.glob("eval_metrics_*.json")).read_text())
                self.assertEqual(payload["llamaguard_error"]["message"], str(boom))
                self.assertFalse((run/"generated_responses").exists())
                self.assertIsNone(self.writers[-1].file_writer)
                self.loader.side_effect = None
                self.generation.side_effect = self.generate

    def test_strict_failure_reraises_original_exception(self):
        run = self.root/"strict"
        self.loader.side_effect = RuntimeError("dataset unavailable")
        with patch.dict(os.environ, RDO_LLAMAGUARD_STRICT="1"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "dataset unavailable"):
                self.evaluate(run)
        self.assertIsNone(self.writers[-1].file_writer)

    def test_guard_failure_still_saves_completed_generations(self):
        run, artifact = self.checkpoint(True)
        self.evaluator.evaluate_batch.side_effect = RuntimeError("guard unavailable")
        with contextlib.redirect_stdout(io.StringIO()):
            self.evaluate(run, artifact)
        payload = json.loads(next(run.glob("eval_metrics_*.json")).read_text())
        self.assertEqual(set(payload["guard_errors"]), {"llamaguard", "qwen3guard", "wildguard"})
        self.assertTrue(list((run/"generated_responses/train_answers").glob("*.json")))

    def test_launcher_can_keep_checkpoints_when_resuming(self):
        import subprocess
        import sys
        recorder = self.root/"python"
        recorder.write_text("#!" + sys.executable + "\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
        recorder.chmod(0o755)
        run, _ = self.checkpoint(True)
        env = dict(os.environ, RDO_PYTHON=str(recorder), RDO_DEVICE="cpu", CLEAR_CKPTS="false")
        output = subprocess.check_output(
            ["bash", str(ROOT/"scripts/run_rdo_refusal.sh"), "--device", "cpu", "--result_path", str(run)],
            env=env, text=True)
        forwarded = json.loads(output)
        self.assertNotIn("--clear_ckpts", forwarded)
        self.assertEqual(forwarded[forwarded.index("--result_path")+1], str(run))


if __name__ == "__main__":
    unittest.main()
