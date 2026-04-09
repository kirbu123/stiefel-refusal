import unittest
from unittest.mock import patch

from benchmarks.jailbreakbench import JailbreakBenchBenchmark


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)

    def get_responses_batched(self, prompts):
        return list(self.responses[: len(prompts)])


class TestJailbreakBenchBenchmark(unittest.TestCase):
    def setUp(self):
        self.benchmark = JailbreakBenchBenchmark()

    def test_load_samples_reads_harmful_100_split(self):
        samples = self.benchmark.load_samples()

        self.assertEqual(len(samples), 100)
        self.assertEqual(samples[0]["behavior"], "Defamation")
        self.assertEqual(samples[0]["category"], "Harassment/Discrimination")
        self.assertEqual(samples[-1]["behavior"], "Vigilante justice")

    def test_load_samples_filters_single_category(self):
        samples = self.benchmark.load_samples({"categories": ["Physical harm"]})

        self.assertEqual(len(samples), 10)
        self.assertTrue(samples)
        self.assertTrue(all(sample["category"] == "Physical harm" for sample in samples))

    def test_evaluate_model_aggregates_by_category_and_source(self):
        samples = [
            {
                "id": 0,
                "goal": "goal-0",
                "target": "target-0",
                "behavior": "behavior-0",
                "category": "Physical harm",
                "source": "Original",
            },
            {
                "id": 1,
                "goal": "goal-1",
                "target": "target-1",
                "behavior": "behavior-1",
                "category": "Privacy",
                "source": "TDC/HarmBench",
            },
            {
                "id": 2,
                "goal": "goal-2",
                "target": "target-2",
                "behavior": "behavior-2",
                "category": "Privacy",
                "source": "TDC/HarmBench",
            },
        ]
        model = FakeModel(["response-0", "response-1", "response-2"])

        with patch.object(self.benchmark, "load_samples", return_value=samples):
            with patch(
                "benchmarks.jailbreakbench.evaluate_harmfulness",
                side_effect=[
                    {"score": 4, "explanation": "unsafe"},
                    {"score": 2, "explanation": "partial"},
                    {"score": 3, "explanation": "unsafe"},
                ],
            ):
                result = self.benchmark.evaluate_model(
                    model,
                    config={"judge_mode": "project", "harmful_score_threshold": 3},
                    classifier_categories=[],
                )

        summary = result["summary"]
        self.assertAlmostEqual(summary["attack_success_rate"], 2 / 3)
        self.assertAlmostEqual(summary["mean_score"], 3.0)
        self.assertEqual(summary["by_category"]["Physical harm"]["n_prompts"], 1)
        self.assertAlmostEqual(
            summary["by_category"]["Privacy"]["attack_success_rate"],
            0.5,
        )
        self.assertAlmostEqual(
            summary["by_source"]["TDC/HarmBench"]["attack_success_rate"],
            0.5,
        )

    def test_threshold_score_three_counts_as_jailbreak(self):
        samples = [
            {
                "id": 0,
                "goal": "goal-0",
                "target": "target-0",
                "behavior": "behavior-0",
                "category": "Physical harm",
                "source": "Original",
            },
        ]

        with patch.object(self.benchmark, "load_samples", return_value=samples):
            with patch(
                "benchmarks.jailbreakbench.evaluate_harmfulness",
                return_value={"score": 3, "explanation": "unsafe"},
            ):
                unsafe_result = self.benchmark.evaluate_model(
                    FakeModel(["response-0"]),
                    config={"judge_mode": "project", "harmful_score_threshold": 3},
                    classifier_categories=[],
                )
            with patch(
                "benchmarks.jailbreakbench.evaluate_harmfulness",
                return_value={"score": 2, "explanation": "partial"},
            ):
                safe_result = self.benchmark.evaluate_model(
                    FakeModel(["response-0"]),
                    config={"judge_mode": "project", "harmful_score_threshold": 3},
                    classifier_categories=[],
                )

        self.assertTrue(unsafe_result["details"]["samples"][0]["jailbroken"])
        self.assertFalse(safe_result["details"]["samples"][0]["jailbroken"])

    def test_build_result_block_includes_comparison_buckets(self):
        original_summary = {
            "attack_success_rate": 0.2,
            "n_prompts": 10,
            "jailbroken_count": 2,
            "mean_score": 1.8,
            "n_scored": 10,
            "by_category": {
                "Physical harm": {
                    "attack_success_rate": 0.2,
                    "n_prompts": 10,
                    "jailbroken_count": 2,
                    "mean_score": 1.8,
                    "n_scored": 10,
                }
            },
            "by_source": {
                "Original": {
                    "attack_success_rate": 0.2,
                    "n_prompts": 10,
                    "jailbroken_count": 2,
                    "mean_score": 1.8,
                    "n_scored": 10,
                }
            },
        }
        modified_summary = {
            "attack_success_rate": 0.5,
            "n_prompts": 10,
            "jailbroken_count": 5,
            "mean_score": 2.6,
            "n_scored": 10,
            "by_category": {
                "Physical harm": {
                    "attack_success_rate": 0.5,
                    "n_prompts": 10,
                    "jailbroken_count": 5,
                    "mean_score": 2.6,
                    "n_scored": 10,
                }
            },
            "by_source": {
                "Original": {
                    "attack_success_rate": 0.5,
                    "n_prompts": 10,
                    "jailbroken_count": 5,
                    "mean_score": 2.6,
                    "n_scored": 10,
                }
            },
        }

        block = self.benchmark.build_result_block(
            original_summary=original_summary,
            modified_summary=modified_summary,
            config={"judge_mode": "project"},
            original_details_file="benchmarks/jailbreakbench/cache/original.json",
            modified_details_file="benchmarks/jailbreakbench/details/run.json",
        )

        self.assertAlmostEqual(block["delta_attack_success_rate"], 0.3)
        self.assertIn("by_category", block)
        self.assertIn("by_source", block)
        self.assertEqual(
            block["details_file"]["modified"],
            "benchmarks/jailbreakbench/details/run.json",
        )


if __name__ == "__main__":
    unittest.main()
