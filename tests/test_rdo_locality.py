import importlib.util
import math
import unittest
from pathlib import Path
from unittest.mock import patch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluate" / "rdo_locality.py"
_SPEC = importlib.util.spec_from_file_location("rdo_locality", _MODULE_PATH)
rdo_locality = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(rdo_locality)


class FakeTokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(text.split())))}


class RdoLocalityTests(unittest.TestCase):
    def test_prepare_wikitext_windows_counts_overlap_once(self):
        records = [{"text": "one two three four five six seven eight nine ten"}]
        with patch.object(rdo_locality, "load_dataset_records", return_value=records):
            windows = rdo_locality.prepare_wikitext_windows(
                FakeTokenizer(),
                max_length=5,
                stride=3,
                max_windows=None,
            )

        self.assertEqual([(row["begin"], row["end"]) for row in windows], [(0, 5), (3, 8), (6, 10)])
        self.assertEqual([row["target_start"] for row in windows], [1, 2, 2])
        self.assertEqual(sum(row["end"] - row["begin"] - row["target_start"] for row in windows), 9)

    def test_summarize_token_losses_is_token_weighted(self):
        result = rdo_locality.summarize_token_losses(
            [
                {"nll_sum": 2.0, "n_tokens": 2},
                {"nll_sum": 6.0, "n_tokens": 3},
            ]
        )
        self.assertAlmostEqual(result["cross_entropy"], 1.6)
        self.assertAlmostEqual(result["perplexity"], math.exp(1.6))
        self.assertEqual(result["n_tokens"], 5)

    def test_arc_summary_uses_best_choice_score(self):
        entries = [
            {
                "index": 0,
                "variant": "ARC-Easy",
                "question": "q",
                "prompt": "prompt",
                "choices": ["wrong", "right"],
                "gold_index": 1,
            }
        ]

        def score(_prompt, completion):
            return 2.0 if completion.strip() == "right" else -1.0

        summary, predictions = rdo_locality.evaluate_arc_entries(entries, score)
        self.assertEqual(summary, {"accuracy": 1.0, "correct": 1, "total": 1})
        self.assertTrue(predictions[0]["is_correct"])

    def test_gsm8k_summary_parses_final_answer(self):
        entries = [
            {
                "index": 0,
                "question": "q",
                "target_answer": "1234",
                "prompt": "p",
            }
        ]
        summary, predictions = rdo_locality.summarize_gsm8k_responses(
            entries,
            ["<think>ignore 9</think>\n#### 1,234"],
        )
        self.assertEqual(summary["exact_match"], 1.0)
        self.assertEqual(predictions[0]["predicted_answer"], "1234")

    def test_metric_block_and_csv_rows_include_tri_mode_deltas(self):
        block = rdo_locality.build_metric_block(
            config={"split": "validation"},
            metric_name="accuracy",
            initial={"accuracy": 0.75, "correct": 3, "total": 4},
            refined_attack={"accuracy": 0.5, "correct": 2, "total": 4},
            refined_protect={"accuracy": 1.0, "correct": 4, "total": 4},
        )
        self.assertEqual(block["delta_attack"], -0.25)
        self.assertEqual(block["delta_protect"], 0.25)
        self.assertNotIn("refined", block)

        rows = rdo_locality.metrics_to_csv_rows({"arc_easy": block})
        self.assertEqual(
            {row["phase"] for row in rows},
            {"initial", "refined_attack", "refined_protect", "delta_attack", "delta_protect"},
        )
        delta_rows = [row for row in rows if row["phase"].startswith("delta_")]
        self.assertEqual(
            delta_rows,
            [
                {
                    "benchmark": "arc_easy",
                    "backend": "",
                    "group": "",
                    "phase": "delta_attack",
                    "metric": "accuracy",
                    "value": -0.25,
                },
                {
                    "benchmark": "arc_easy",
                    "backend": "",
                    "group": "",
                    "phase": "delta_protect",
                    "metric": "accuracy",
                    "value": 0.25,
                }
            ],
        )

    def test_metric_block_keeps_null_protect_phase(self):
        block = rdo_locality.build_metric_block(
            config={},
            metric_name="perplexity",
            initial={"perplexity": 10.0},
            refined_attack={"perplexity": 12.0},
            refined_protect=None,
        )
        self.assertIsNone(block["refined_protect"])
        self.assertIsNone(block["delta_protect"])


if __name__ == "__main__":
    unittest.main()
