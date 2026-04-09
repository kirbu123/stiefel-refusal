import unittest
from unittest.mock import patch

from benchmarks.harmbench import (
    HarmBenchBenchmark,
    HarmBenchOfficialClassifier,
    get_cached_official_classifier,
)


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)

    def get_responses_batched(self, prompts):
        return list(self.responses[: len(prompts)])


class FakeOfficialClassifier:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def evaluate_batch(self, samples, responses, *, batch_size):
        self.calls.append(
            {
                "samples": list(samples),
                "responses": list(responses),
                "batch_size": batch_size,
            }
        )
        return list(self.results)


class TestHarmBenchBenchmark(unittest.TestCase):
    def setUp(self):
        self.benchmark = HarmBenchBenchmark()
        get_cached_official_classifier.cache_clear()

    def tearDown(self):
        get_cached_official_classifier.cache_clear()

    def test_load_samples_reads_expected_split_sizes(self):
        self.assertEqual(
            len(self.benchmark.load_samples({"split": "val"})),
            80,
        )
        self.assertEqual(
            len(self.benchmark.load_samples({"split": "test"})),
            320,
        )
        self.assertEqual(
            len(self.benchmark.load_samples({"split": "all"})),
            400,
        )

    def test_load_samples_combines_filters(self):
        samples = self.benchmark.load_samples(
            {
                "split": "all",
                "categories": ["Disinformation"],
                "functional_categories": ["standard"],
                "semantic_categories": ["misinformation_disinformation"],
            }
        )

        self.assertTrue(samples)
        self.assertTrue(all(sample["category"] == "Disinformation" for sample in samples))
        self.assertTrue(all(sample["functional_category"] == "standard" for sample in samples))
        self.assertTrue(
            all(
                sample["semantic_category"] == "misinformation_disinformation"
                for sample in samples
            )
        )

    def test_load_samples_builds_contextual_prompt(self):
        samples = self.benchmark.load_samples(
            {
                "split": "all",
                "functional_categories": ["contextual"],
                "max_samples": 1,
            }
        )

        self.assertEqual(len(samples), 1)
        self.assertTrue(samples[0]["context_string"])
        self.assertIn("Context:\n", samples[0]["prompt"])
        self.assertIn("\n\nTask:\n", samples[0]["prompt"])

    def test_official_classifier_builds_prompts_and_parses_yes_no(self):
        classifier = HarmBenchOfficialClassifier.__new__(HarmBenchOfficialClassifier)
        classifier.model_name = "test-cls"

        captured = {}

        def fake_generate(prompts, *, batch_size):
            captured["prompts"] = list(prompts)
            captured["batch_size"] = batch_size
            return ["Yes", "no"]

        classifier._generate_prompt_outputs = fake_generate

        samples = [
            {
                "behavior": "behavior-0",
                "context_string": "",
            },
            {
                "behavior": "behavior-1",
                "context_string": "context-1",
            },
        ]
        results = classifier.evaluate_batch(
            samples,
            ["response-0", "response-1"],
            batch_size=3,
        )

        self.assertEqual(captured["batch_size"], 3)
        self.assertEqual(len(captured["prompts"]), 2)
        self.assertIn("Behavior:\nbehavior-0", captured["prompts"][0])
        self.assertNotIn("Context:\n", captured["prompts"][0])
        self.assertIn("Context:\ncontext-1", captured["prompts"][1])
        self.assertTrue(results[0]["jailbroken"])
        self.assertFalse(results[1]["jailbroken"])
        self.assertEqual(results[0]["evaluation"]["label"], "yes")
        self.assertEqual(results[1]["evaluation"]["label"], "no")

    def test_official_classifier_cache_reuses_instance(self):
        with patch("benchmarks.harmbench.HarmBenchOfficialClassifier") as classifier_cls:
            classifier_cls.side_effect = [
                object(),
                object(),
            ]
            first = get_cached_official_classifier("model-a", "cuda", "bfloat16")
            second = get_cached_official_classifier("model-a", "cuda", "bfloat16")
            third = get_cached_official_classifier("model-a", "cpu", "bfloat16")

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(classifier_cls.call_count, 2)

    def test_official_judge_uses_cached_classifier(self):
        samples = [
            {
                "id": "sample-0",
                "behavior": "behavior-0",
                "behavior_id": "behavior-0",
                "context_string": "",
                "prompt": "behavior-0",
                "category": "Physical harm",
                "functional_category": "standard",
                "semantic_category": "illegal",
                "split": "test",
            },
            {
                "id": "sample-1",
                "behavior": "behavior-1",
                "behavior_id": "behavior-1",
                "context_string": "ctx",
                "prompt": "Context:\nctx\n\nTask:\nbehavior-1",
                "category": "Privacy",
                "functional_category": "contextual",
                "semantic_category": "harmful",
                "split": "test",
            },
        ]
        fake_classifier = FakeOfficialClassifier(
            [
                {
                    "score": None,
                    "jailbroken": True,
                    "evaluation": {"label": "yes", "raw_output": "Yes"},
                },
                {
                    "score": None,
                    "jailbroken": False,
                    "evaluation": {"label": "no", "raw_output": "No"},
                },
            ]
        )

        with patch.object(self.benchmark, "load_samples", return_value=samples):
            with patch(
                "benchmarks.harmbench.get_cached_official_classifier",
                return_value=fake_classifier,
            ):
                result = self.benchmark.evaluate_model(
                    FakeModel(["response-0", "response-1"]),
                    config={
                        "judge_mode": "official",
                        "split": "test",
                        "official_classifier": {
                            "model_name": "test-cls",
                            "batch_size": 5,
                            "device": "cuda",
                            "dtype": "bfloat16",
                        },
                    },
                    classifier_categories=[],
                )

        summary = result["summary"]
        self.assertAlmostEqual(summary["attack_success_rate"], 0.5)
        self.assertIsNone(summary["mean_score"])
        self.assertEqual(summary["n_scored"], 0)
        self.assertEqual(fake_classifier.calls[0]["batch_size"], 5)
        self.assertEqual(fake_classifier.calls[0]["responses"], ["response-0", "response-1"])
        self.assertEqual(
            fake_classifier.calls[0]["samples"][1]["context_string"],
            "ctx",
        )

    def test_project_judge_uses_threshold(self):
        samples = [
            {
                "id": "sample-0",
                "behavior": "behavior-0",
                "behavior_id": "behavior-0",
                "context_string": "",
                "prompt": "behavior-0",
                "category": "Physical harm",
                "functional_category": "standard",
                "semantic_category": "illegal",
                "split": "test",
            },
        ]

        with patch.object(self.benchmark, "load_samples", return_value=samples):
            with patch(
                "benchmarks.harmbench.evaluate_harmfulness",
                return_value={"score": 3, "explanation": "unsafe"},
            ):
                unsafe_result = self.benchmark.evaluate_model(
                    FakeModel(["response-0"]),
                    config={"judge_mode": "project", "split": "test"},
                    classifier_categories=[],
                )
            with patch(
                "benchmarks.harmbench.evaluate_harmfulness",
                return_value={"score": 2, "explanation": "partial"},
            ):
                safe_result = self.benchmark.evaluate_model(
                    FakeModel(["response-0"]),
                    config={"judge_mode": "project", "split": "test"},
                    classifier_categories=[],
                )

        self.assertTrue(unsafe_result["details"]["samples"][0]["jailbroken"])
        self.assertFalse(safe_result["details"]["samples"][0]["jailbroken"])

    def test_build_result_block_includes_expected_buckets(self):
        original_summary = {
            "attack_success_rate": 0.25,
            "n_prompts": 8,
            "jailbroken_count": 2,
            "mean_score": None,
            "n_scored": 0,
            "by_split": {"test": {"attack_success_rate": 0.25, "n_prompts": 8, "jailbroken_count": 2, "mean_score": None, "n_scored": 0}},
            "by_category": {"Physical harm": {"attack_success_rate": 0.25, "n_prompts": 8, "jailbroken_count": 2, "mean_score": None, "n_scored": 0}},
            "by_functional_category": {"standard": {"attack_success_rate": 0.25, "n_prompts": 8, "jailbroken_count": 2, "mean_score": None, "n_scored": 0}},
            "by_semantic_category": {"illegal": {"attack_success_rate": 0.25, "n_prompts": 8, "jailbroken_count": 2, "mean_score": None, "n_scored": 0}},
        }
        modified_summary = {
            "attack_success_rate": 0.5,
            "n_prompts": 8,
            "jailbroken_count": 4,
            "mean_score": None,
            "n_scored": 0,
            "by_split": {"test": {"attack_success_rate": 0.5, "n_prompts": 8, "jailbroken_count": 4, "mean_score": None, "n_scored": 0}},
            "by_category": {"Physical harm": {"attack_success_rate": 0.5, "n_prompts": 8, "jailbroken_count": 4, "mean_score": None, "n_scored": 0}},
            "by_functional_category": {"standard": {"attack_success_rate": 0.5, "n_prompts": 8, "jailbroken_count": 4, "mean_score": None, "n_scored": 0}},
            "by_semantic_category": {"illegal": {"attack_success_rate": 0.5, "n_prompts": 8, "jailbroken_count": 4, "mean_score": None, "n_scored": 0}},
        }

        block = self.benchmark.build_result_block(
            original_summary=original_summary,
            modified_summary=modified_summary,
            config={"judge_mode": "official", "split": "test"},
            original_details_file="benchmarks/harmbench/cache/original.json",
            modified_details_file="benchmarks/harmbench/details/run.json",
        )

        self.assertAlmostEqual(block["delta_attack_success_rate"], 0.25)
        self.assertIn("by_split", block)
        self.assertIn("by_category", block)
        self.assertIn("by_functional_category", block)
        self.assertIn("by_semantic_category", block)
        self.assertEqual(
            block["details_file"]["modified"],
            "benchmarks/harmbench/details/run.json",
        )


if __name__ == "__main__":
    unittest.main()
