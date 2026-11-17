"""CPU smoke tests for offline SafetyAnalisys backfill; no model downloads."""
import contextlib
import copy
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("backfill", ROOT / "scripts/backfill_safety_metrics.py")
backfill = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backfill)


class BackfillSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.references = self.root / "references.json"
        backfill.write_json(self.references, {"version": 1, **{
            kind: {str(i): [{"prompt": f"q{i}", "response": f"a{i}"}] for i in range(1, 5)}
            for kind in backfill.safety.KINDS}})
        _, digest = backfill.safety.load_references(self.references)
        self.judge = SimpleNamespace(
            config={"model": backfill.safety.MODEL_NAME, "prompt_version": backfill.safety.PROMPT_VERSION,
                    "references_sha256": digest},
            evaluate_batch=Mock(side_effect=lambda pairs, **kw: [
                {"score": 4, "probabilities": {str(i): float(i == 4) for i in range(1, 5)}} for _ in pairs]),
            close=Mock())
        self.factory = Mock(return_value=self.judge)
        self.tb = Mock()
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"RDO_PLATFORM": "cpu", "RDO_DEVICE": "auto", "RDO_PPU_CONTAINER": "0"}).start()
        patch.object(backfill.safety, "SafetyAnalisysEvaluator", self.factory).start()
        patch.object(backfill, "publish_tensorboard", self.tb).start()

    def experiment(self, name="exp"):
        directory = self.root / name
        directory.mkdir()
        phases = list(backfill.PHASE_FILES)
        payload = {"eval_split": "val", "unrelated": {"accuracy": 0.7},
                   "guard_errors": {"safetyanalisys": {"message": "old failure"}, "other": "keep"},
                   "guard_metrics": {"llamaguard": {kind: {phase: {"n": 2, "mean_score": 1.5}
                                      for phase in phases} for kind in backfill.safety.KINDS}}}
        backfill.write_json(directory / "hparams.json", {"model": "target-model", "splits": "saladbench"})
        metrics = directory / "eval_metrics_20261113_090000.json"
        backfill.write_json(metrics, payload)
        metrics.with_suffix(".csv").write_text(
            "benchmark,backend,group,phase,metric,value,custom\n"
            "guard,llamaguard,harmful,initial,mean_score,1.5,keep\n")
        for split, mode, stamp in (("train", "basic", "20261113_075619"),
                                   ("test", "rdo", "20261113_082214")):
            folder = directory / "generated_responses" / f"{split}_answers"
            folder.mkdir(parents=True)
            for phase, prefix in backfill.PHASE_FILES.items():
                backfill.write_json(folder / f"{prefix}_{stamp}.json", {
                    "config": {"llamaguard_data_mode": mode},
                    **{kind: [{"prompt": f"{split}/{kind}/{i}", "response": f"{phase}/{i}"}
                              for i in range(2)] for kind in backfill.safety.KINDS}})
        return directory, metrics, payload

    def run_cli(self, *directories, flags=()):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return backfill.main(["--exp-dirs", *map(str, directories),
                                  "--safety-analisys-references", str(self.references), *flags])

    def test_two_experiments_preserve_metrics_and_reuse_judge(self):
        first, metrics, original = self.experiment("first")
        second, _, _ = self.experiment("second")
        self.assertEqual(self.run_cli(first, second, flags=("--metrics", "safetyanalisys")), 0)
        self.factory.assert_called_once()
        self.judge.close.assert_called_once()
        self.assertEqual(self.judge.evaluate_batch.call_count, 24)
        self.assertEqual(self.tb.call_count, 2)
        result = backfill.read_json(metrics)
        self.assertEqual(result["guard_metrics"], original["guard_metrics"])
        self.assertEqual(result["unrelated"], original["unrelated"])
        self.assertEqual(result["guard_errors"], {"other": "keep"})
        block = result["safety_analisys"]
        self.assertEqual(block["splits"]["test"]["source"]["eval_split"], "val")
        self.assertEqual(block["splits"]["train"]["source"]["model_name"], "target-model")
        details = backfill.read_json(first / block["scores_file"])
        self.assertEqual(len(details["scores"]), 24)
        self.assertEqual(details["scores"][0]["score"], 4)
        with metrics.with_suffix(".csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 25)
        self.assertEqual(rows[0]["custom"], "keep")
        self.assertEqual(rows[0]["split"], "")
        self.assertEqual({r["split"] for r in rows[1:]}, {"train", "test"})

    def test_repeat_does_not_load_judge_or_rewrite_artifacts(self):
        directory, _, _ = self.experiment()
        self.assertEqual(self.run_cli(directory), 0)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.rglob("*") if p.is_file()}
        self.factory.reset_mock()
        self.judge.evaluate_batch.reset_mock()
        # Fully ready experiments don't require a references argument.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(backfill.main(["--exp-dirs", str(directory)]), 0)
        self.factory.assert_not_called()
        self.judge.evaluate_batch.assert_not_called()
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})

    def test_default_references_without_flag(self):
        directory, _, _ = self.experiment()
        with patch.object(backfill.safety, "DEFAULT_REFERENCES_PATH", self.references), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(backfill.main(["--exp-dirs", str(directory)]), 0)
        self.factory.assert_called_once_with(self.references, 1)

    def test_partial_and_derived_artifact_recovery(self):
        directory, metrics, _ = self.experiment()
        self.assertEqual(self.run_cli(directory), 0)
        result = backfill.read_json(metrics)
        block = result["safety_analisys"]
        preserved = copy.deepcopy(block["splits"]["test"])
        del block["splits"]["train"]["results"]["harmful"]["initial"]
        backfill.write_json(metrics, result)
        self.judge.evaluate_batch.reset_mock()
        self.assertEqual(self.run_cli(directory), 0)
        self.judge.evaluate_batch.assert_called_once()
        result = backfill.read_json(metrics)
        self.assertEqual(result["safety_analisys"]["splits"]["test"], preserved)
        self.assertEqual(len(backfill.read_json(directory / block["scores_file"])["scores"]), 24)
        # JSON is already committed, CSV/TensorBoard publication can be retried without inference.
        metrics.with_suffix(".csv").write_text("benchmark,backend,group,phase,metric,value\n")
        self.factory.reset_mock()
        self.assertEqual(self.run_cli(directory), 0)
        self.factory.assert_not_called()
        with metrics.with_suffix(".csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 24)

    def test_interrupted_tensorboard_publication_retries_without_inference(self):
        directory, metrics, _ = self.experiment()
        self.tb.side_effect = RuntimeError("event writer failed")
        self.assertEqual(self.run_cli(directory), 1)
        self.assertIn("safety_analisys", backfill.read_json(metrics))
        self.tb.side_effect = None
        self.factory.reset_mock()
        self.judge.evaluate_batch.reset_mock()
        self.assertEqual(self.run_cli(directory), 0)
        self.factory.assert_not_called()
        self.judge.evaluate_batch.assert_not_called()

    def test_invalid_group_recomputed_and_initial_only_experiment(self):
        directory, metrics, _ = self.experiment()
        result = backfill.read_json(metrics)
        for block in result["guard_metrics"]["llamaguard"].values():
            del block["refined_protect"]
            del block["refined_attack"]
        backfill.write_json(metrics, result)
        for path in directory.glob("generated_responses/*/refined*.json"):
            path.unlink()
        self.assertEqual(self.run_cli(directory), 0)
        self.assertEqual(self.judge.evaluate_batch.call_count, 4)
        result = backfill.read_json(metrics)
        result["safety_analisys"]["splits"]["test"]["results"]["harmless"]["initial"]["score_counts"] = None
        backfill.write_json(metrics, result)
        self.judge.evaluate_batch.reset_mock()
        self.assertEqual(self.run_cli(directory), 0)
        self.judge.evaluate_batch.assert_called_once()

    def test_dry_run_real_entrypoint_has_no_writes_or_judge(self):
        directory, _, _ = self.experiment()
        before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
        result = subprocess.run([sys.executable, str(ROOT / "scripts/backfill_safety_metrics.py"),
                                 "--exp-dirs", str(directory), "--dry-run"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("missing=12", result.stdout)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.factory.assert_not_called()

    def test_invalid_metric_and_batch_size(self):
        for flags in (("--metrics", "llamaguard"), ("--safety-analisys-batch-size", "0")):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                backfill.main(["--exp-dirs", str(self.root), *flags])
        self.factory.assert_not_called()

    def test_ppu_detection_and_no_recursive_launch(self):
        args = SimpleNamespace(dry_run=False, exp_dirs=[self.root / 'exp one'],
                               metrics=['safetyanalisys'], safety_analisys_references=self.references,
                               safety_analisys_batch_size=1)
        for overrides in ({"RDO_PPU_CONTAINER": "1"}, {"RDO_DEVICE": "cpu"},
                          {"RDO_PLATFORM": "cpu"}):
            with patch.dict(os.environ, {"RDO_PLATFORM": "ppu", **overrides}), \
                    patch.object(backfill.os, 'execvpe') as execute:
                backfill.maybe_launch_ppu(args)
                execute.assert_not_called()
        with patch.dict(os.environ, {"RDO_PLATFORM": "ppu"}), \
                patch.object(backfill.os, 'execvpe') as execute:
            args.dry_run = True
            backfill.maybe_launch_ppu(args)
            execute.assert_not_called()
            args.dry_run = False
        for present in (False, True):
            with patch.dict(os.environ, {"RDO_PLATFORM": "auto"}), \
                    patch.object(backfill.Path, 'exists', return_value=present), \
                    patch.object(backfill.os, 'execvpe') as execute:
                backfill.maybe_launch_ppu(args)
                self.assertEqual(execute.call_count, int(present))
                if present:
                    self.assertEqual(execute.call_args.args[1][:3],
                                     ['bash', str(ROOT / 'scripts/launch_ppu.sh'), 'backfill'])

    def test_bad_references_and_incompatible_partial_config(self):
        directory, metrics, _ = self.experiment()
        valid = self.references.read_text()
        self.references.write_text('{"version": 1}')
        original = metrics.read_bytes()
        self.assertEqual(self.run_cli(directory), 1)
        self.assertEqual(metrics.read_bytes(), original)
        self.factory.assert_not_called()
        self.references.write_text(valid)
        self.assertEqual(self.run_cli(directory), 0)
        result = backfill.read_json(metrics)
        del result["safety_analisys"]["splits"]["train"]["results"]["harmful"]["initial"]
        result["safety_analisys"]["config"]["references_sha256"] = "different"
        backfill.write_json(metrics, result)
        self.factory.reset_mock()
        self.assertEqual(self.run_cli(directory), 1)
        self.factory.assert_not_called()

    def test_missing_logs_and_mismatched_prompts(self):
        directory, metrics, _ = self.experiment()
        folder = directory / "generated_responses/test_answers"
        path = next(folder.glob("refined_responses_attack*.json"))
        original = metrics.read_bytes()
        log = backfill.read_json(path)
        log["harmful"][0]["prompt"] = "different question"
        backfill.write_json(path, log)
        self.assertEqual(self.run_cli(directory), 1)
        path.unlink()
        self.assertEqual(self.run_cli(directory), 1)
        self.assertEqual(metrics.read_bytes(), original)
        self.factory.assert_not_called()

    def test_latest_batches_no_cross_batch_phase_mixing(self):
        directory, metrics, _ = self.experiment()
        folder = directory / "generated_responses/train_answers"
        initial = next(folder.glob("initial*.json"))
        log = backfill.read_json(initial)
        backfill.write_json(folder / "initial_responses_20261113_080000.json", log)
        self.assertEqual(self.run_cli(directory), 1)  # Newest batch lacks attack/protect.
        for phase, prefix in backfill.PHASE_FILES.items():
            if phase != "initial":
                old = next(folder.glob(f"{prefix}_20261113_075619.json"))
                backfill.write_json(folder / f"{prefix}_20261113_080000.json", backfill.read_json(old))
        # Future incomplete batch must not be used for the earlier metrics file.
        backfill.write_json(folder / "initial_responses_20261114_080000.json", log)
        _, _, _, files = backfill.load_experiment(directory)
        self.assertTrue(all("20261113_080000" in path for path in files["train"].values()))
        self.assertEqual(self.run_cli(directory), 0)

    def test_judge_failure_isolated_and_original_files_untouched(self):
        first, _, _ = self.experiment("first")
        second, metrics, _ = self.experiment("second")
        before = {p: p.read_bytes() for p in first.rglob("*") if p.is_file()}
        normal = self.judge.evaluate_batch.side_effect
        calls = 0

        def fail_once(pairs, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("judge failed")
            return normal(pairs, **kwargs)

        self.judge.evaluate_batch.side_effect = fail_once
        self.assertEqual(self.run_cli(first, second), 1)
        self.assertEqual(before, {p: p.read_bytes() for p in first.rglob("*") if p.is_file()})
        self.assertIn("safety_analisys", backfill.read_json(metrics))
        self.judge.close.assert_called_once()

    def test_tensorboard_actual_events_and_repeat(self):
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            self.skipTest("requires tensorboard")
        directory, metrics, _ = self.experiment()
        # Use the real publication path for this test.
        with patch.object(backfill, "publish_tensorboard", REAL_PUBLISH):
            self.assertEqual(self.run_cli(directory), 0)
            self.assertEqual(self.run_cli(directory), 0)
        acc = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags()["scalars"]
        self.assertEqual(len(tags), 12)
        for tag in tags:
            events = acc.Scalars(tag)
            self.assertEqual(len(events), 1)
            self.assertEqual((events[0].step, events[0].value), (0, 4.0))


REAL_PUBLISH = backfill.publish_tensorboard

if __name__ == "__main__":
    unittest.main()
