import unittest
import types
import sys


fake_ui = types.ModuleType("cli.ui")
fake_ui.print = print
sys.modules.setdefault("cli.ui", fake_ui)

from cli.runner import summarize_results


class TestRunnerSummary(unittest.TestCase):
    def test_summarize_results_handles_mmlu_and_legacy_payloads(self):
        results = [
            {
                "experiment_info": {
                    "category": "Physical harm",
                    "hyperparameters": {"max_weight": 2.0},
                    "param_key": "max_weight=2.0",
                },
                "harmful_questions": {
                    "modified_evaluations": [{"score": 1}, {"score": 3}],
                },
                "harmless_questions": {
                    "average_locality_change": -0.5,
                },
                "mmlu": {
                    "original": {"accuracy": 0.42},
                    "modified": {"accuracy": 0.39},
                },
            },
            {
                "experiment_info": {
                    "category": "Legacy",
                    "hyperparameters": {"max_weight": 1.0},
                },
                "final_scores": [2, 2, 4],
            },
        ]

        summaries = summarize_results(results)

        self.assertEqual(len(summaries), 2)
        self.assertAlmostEqual(summaries[0]["mmlu_original_accuracy"], 0.42)
        self.assertAlmostEqual(summaries[0]["mmlu_modified_accuracy"], 0.39)
        self.assertIsNone(summaries[1]["mmlu_original_accuracy"])
        self.assertIsNone(summaries[1]["mmlu_modified_accuracy"])


if __name__ == "__main__":
    unittest.main()
