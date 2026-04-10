import os
import unittest
from unittest.mock import patch

from benchmarks.jailbreakbench import (
    JailbreakBenchBenchmark,
    JailbreakBenchOfficialJudgeClient,
    get_cached_official_judge_client,
)


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)

    def get_responses_batched(self, prompts):
        return list(self.responses[: len(prompts)])


class FakeHttpResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeOfficialJudgeClient:
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


class TestJailbreakBenchBenchmark(unittest.TestCase):
    def setUp(self):
        self.benchmark = JailbreakBenchBenchmark()
        get_cached_official_judge_client.cache_clear()

    def tearDown(self):
        get_cached_official_judge_client.cache_clear()

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

    def test_official_judge_builds_prompt_and_parses_safe_unsafe(self):
        with patch.dict(os.environ, {"TEST_JBB_KEY": "secret"}, clear=True):
            client = JailbreakBenchOfficialJudgeClient(
                model_name="together_ai/meta-llama/Llama-3-70b-chat-hf",
                api_url="https://api.together.xyz/v1/chat/completions",
                api_key_env="TEST_JBB_KEY",
                timeout_sec=30,
            )

            with patch(
                "benchmarks.jailbreakbench.requests.post",
                side_effect=[
                    FakeHttpResponse(
                        {"choices": [{"message": {"content": "unsafe"}}]}
                    ),
                    FakeHttpResponse(
                        {"choices": [{"message": {"content": "safe"}}]}
                    ),
                ],
            ) as post:
                results = client.evaluate_batch(
                    [
                        {"goal": "goal-0"},
                        {"goal": "goal-1"},
                    ],
                    ["response-0", "response-1"],
                    batch_size=2,
                )

        self.assertEqual(post.call_count, 2)
        first_payload = post.call_args_list[0].kwargs["json"]
        second_payload = post.call_args_list[1].kwargs["json"]
        self.assertEqual(
            first_payload["model"],
            "meta-llama/Llama-3-70b-chat-hf",
        )
        self.assertIn("<BEGIN REQUEST>\ngoal-0\n<END REQUEST>", first_payload["messages"][0]["content"])
        self.assertIn("<BEGIN RESPONSE>\nresponse-0\n<END RESPONSE>", first_payload["messages"][0]["content"])
        self.assertIn("<BEGIN REQUEST>\ngoal-1\n<END REQUEST>", second_payload["messages"][0]["content"])
        self.assertTrue(results[0]["jailbroken"])
        self.assertFalse(results[1]["jailbroken"])
        self.assertEqual(results[0]["evaluation"]["label"], "unsafe")
        self.assertEqual(results[1]["evaluation"]["label"], "safe")

    def test_official_judge_cache_reuses_instance(self):
        with patch.dict(os.environ, {"TEST_JBB_KEY": "secret"}, clear=True):
            with patch(
                "benchmarks.jailbreakbench.JailbreakBenchOfficialJudgeClient"
            ) as client_cls:
                client_cls.side_effect = [object(), object()]
                first = get_cached_official_judge_client(
                    "model-a",
                    "https://api.example/v1/chat/completions",
                    "TEST_JBB_KEY",
                    60,
                )
                second = get_cached_official_judge_client(
                    "model-a",
                    "https://api.example/v1/chat/completions",
                    "TEST_JBB_KEY",
                    60,
                )
                third = get_cached_official_judge_client(
                    "model-a",
                    "https://api.example/v1/chat/completions",
                    "TEST_JBB_KEY",
                    30,
                )

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(client_cls.call_count, 2)

    def test_evaluate_model_with_official_judge_uses_cached_client(self):
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
        ]
        fake_client = FakeOfficialJudgeClient(
            [
                {
                    "score": None,
                    "jailbroken": True,
                    "evaluation": {"label": "unsafe", "raw_output": "unsafe"},
                },
                {
                    "score": None,
                    "jailbroken": False,
                    "evaluation": {"label": "safe", "raw_output": "safe"},
                },
            ]
        )

        with patch.object(self.benchmark, "load_samples", return_value=samples):
            with patch(
                "benchmarks.jailbreakbench.get_cached_official_judge_client",
                return_value=fake_client,
            ):
                result = self.benchmark.evaluate_model(
                    FakeModel(["response-0", "response-1"]),
                    config={
                        "judge_mode": "official",
                        "official_judge": {
                            "model_name": "together_ai/meta-llama/Llama-3-70b-chat-hf",
                            "api_url": "https://api.together.xyz/v1/chat/completions",
                            "api_key_env": "TOGETHER_API_KEY",
                            "batch_size": 5,
                            "timeout_sec": 60,
                        },
                    },
                    classifier_categories=[],
                )

        summary = result["summary"]
        self.assertAlmostEqual(summary["attack_success_rate"], 0.5)
        self.assertIsNone(summary["mean_score"])
        self.assertEqual(summary["n_scored"], 0)
        self.assertEqual(fake_client.calls[0]["batch_size"], 5)
        self.assertEqual(fake_client.calls[0]["responses"], ["response-0", "response-1"])

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

    def test_threshold_score_three_counts_as_jailbreak_for_project_judge(self):
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
            "mean_score": None,
            "n_scored": 0,
            "by_category": {
                "Physical harm": {
                    "attack_success_rate": 0.2,
                    "n_prompts": 10,
                    "jailbroken_count": 2,
                    "mean_score": None,
                    "n_scored": 0,
                }
            },
            "by_source": {
                "Original": {
                    "attack_success_rate": 0.2,
                    "n_prompts": 10,
                    "jailbroken_count": 2,
                    "mean_score": None,
                    "n_scored": 0,
                }
            },
        }
        modified_summary = {
            "attack_success_rate": 0.5,
            "n_prompts": 10,
            "jailbroken_count": 5,
            "mean_score": None,
            "n_scored": 0,
            "by_category": {
                "Physical harm": {
                    "attack_success_rate": 0.5,
                    "n_prompts": 10,
                    "jailbroken_count": 5,
                    "mean_score": None,
                    "n_scored": 0,
                }
            },
            "by_source": {
                "Original": {
                    "attack_success_rate": 0.5,
                    "n_prompts": 10,
                    "jailbroken_count": 5,
                    "mean_score": None,
                    "n_scored": 0,
                }
            },
        }

        block = self.benchmark.build_result_block(
            original_summary=original_summary,
            modified_summary=modified_summary,
            config={"judge_mode": "official"},
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
