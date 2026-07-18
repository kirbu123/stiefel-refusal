import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "metrics" / "ablation_study_r.py"
_SPEC = importlib.util.spec_from_file_location("ablation_study_r", _MODULE_PATH)
ablation = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = ablation
_SPEC.loader.exec_module(ablation)


class AblationStudyRTests(unittest.TestCase):
    def _write_run(self, root: Path, name: str, value: float) -> None:
        run_dir = root / name
        run_dir.mkdir()
        (run_dir / "hparams.json").write_text(
            json.dumps(
                {
                    "model_id": "model",
                    "direction_mode": "activation_additive_rot",
                    "num_opt_layers": 1,
                    "proj_reduce_ratio": 35,
                    "init_mode": "diag_permutation",
                    "orth_method": "svd",
                }
            ),
            encoding="utf-8",
        )
        rows = [
            ["guard", "llamaguard", "harmful", "initial", "pct_unsafe", 0.1],
            ["guard", "llamaguard", "harmful", "refined_attack", "pct_unsafe", value],
            ["guard", "llamaguard", "harmful", "refined_protect", "pct_unsafe", 0.05],
            ["mmlu", "", "", "refined_attack", "accuracy", value],
            ["ppl", "", "", "refined_attack", "perplexity", 10.0 + value],
        ]
        with (run_dir / "eval_metrics_20260718_010101.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["benchmark", "backend", "group", "phase", "metric", "value"])
            writer.writerows(rows)

    def test_load_and_aggregate_repeated_runs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_run(root, "run_a", 0.6)
            self._write_run(root, "run_b", 0.8)
            records, families, diagnostics = ablation.load_experiments(root, 3500.0)
            aggregated = ablation.aggregate_metrics(records)

        self.assertEqual(diagnostics["runs_loaded"], 2)
        self.assertEqual(len(families), 1)
        self.assertEqual(set(records["r"]), {100.0})
        attack = aggregated[
            (aggregated["benchmark"] == "guard")
            & (aggregated["phase"] == "refined_attack")
            & (aggregated["metric"] == "pct_unsafe")
        ].iloc[0]
        self.assertAlmostEqual(attack["mean"], 0.7)
        self.assertEqual(attack["count"], 2)
        self.assertGreater(attack["std"], 0.0)


if __name__ == "__main__":
    unittest.main()
