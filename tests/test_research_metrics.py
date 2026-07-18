import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "visualization" / "research.py"
_SPEC = importlib.util.spec_from_file_location("research_metrics", _MODULE_PATH)
research = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = research
_SPEC.loader.exec_module(research)


class ResearchMetricSchemaTests(unittest.TestCase):
    def test_json_reader_maps_legacy_refined_to_attack(self):
        payload = {
            "guard_metrics": {
                "llamaguard": {
                    "harmful": {
                        "initial": {"pct_unsafe": 0.1},
                        "refined": {"pct_unsafe": 0.8},
                    },
                    "harmless": {},
                }
            },
            "mmlu": {
                "initial": {"accuracy": 0.5},
                "refined": {"accuracy": 0.4},
                "delta_accuracy": -0.1,
            },
            "locality_metrics": {
                "ppl": {
                    "metric_name": "perplexity",
                    "initial": {"perplexity": 10.0},
                    "refined": {"perplexity": 12.0},
                    "delta": 2.0,
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "eval.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            rows = research._rows_from_eval_json(path)

        phases = {(row["benchmark"], row["phase"]) for row in rows}
        self.assertIn(("guard", "refined_attack"), phases)
        self.assertIn(("mmlu", "refined_attack"), phases)
        self.assertIn(("ppl", "refined_attack"), phases)
        self.assertIn(("mmlu", "delta_attack"), phases)
        self.assertIn(("ppl", "delta_attack"), phases)
        self.assertNotIn(("guard", "refined"), phases)

    def test_json_reader_keeps_canonical_protect_metrics(self):
        payload = {
            "mmlu": {
                "initial": {"accuracy": 0.5},
                "refined_attack": {"accuracy": 0.4},
                "refined_protect": {"accuracy": 0.6},
                "delta_accuracy_attack": -0.1,
                "delta_accuracy_protect": 0.1,
            }
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "eval.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            rows = research._rows_from_eval_json(path)

        phases = {row["phase"] for row in rows if row["benchmark"] == "mmlu"}
        self.assertEqual(
            phases,
            {
                "initial",
                "refined_attack",
                "refined_protect",
                "delta_attack",
                "delta_protect",
            },
        )


if __name__ == "__main__":
    unittest.main()
