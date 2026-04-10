import sys
import types
import unittest

from benchmarks.registry import get_default_registry
from benchmarks.settings import (
    get_enabled_benchmark_configs,
    normalize_harmbench_config,
    normalize_jailbreakbench_config,
)


class TestBenchmarkSettings(unittest.TestCase):
    def setUp(self):
        self._original_config_module = sys.modules.get("config")

    def tearDown(self):
        if self._original_config_module is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = self._original_config_module

    def test_normalize_jailbreakbench_config_defaults_to_official(self):
        normalized = normalize_jailbreakbench_config()

        self.assertEqual(normalized["judge_mode"], "official")
        self.assertEqual(normalized["max_samples"], 100)
        self.assertEqual(normalized["categories"], [])
        self.assertEqual(normalized["harmful_score_threshold"], 3)
        self.assertEqual(
            normalized["official_judge"]["model_name"],
            "together_ai/meta-llama/Llama-3-70b-chat-hf",
        )
        self.assertEqual(
            normalized["official_judge"]["api_url"],
            "https://api.together.xyz/v1/chat/completions",
        )
        self.assertEqual(
            normalized["official_judge"]["api_key_env"],
            "TOGETHER_API_KEY",
        )
        self.assertEqual(normalized["official_judge"]["batch_size"], 8)
        self.assertEqual(normalized["official_judge"]["timeout_sec"], 60)

    def test_normalize_harmbench_config_defaults_to_official(self):
        normalized = normalize_harmbench_config()

        self.assertEqual(normalized["judge_mode"], "official")
        self.assertEqual(normalized["split"], "test")
        self.assertEqual(normalized["max_samples"], 0)
        self.assertEqual(normalized["official_classifier"]["model_name"], "cais/HarmBench-Llama-2-13b-cls")
        self.assertEqual(normalized["official_classifier"]["batch_size"], 8)
        self.assertEqual(normalized["official_classifier"]["device"], "cuda")
        self.assertEqual(normalized["official_classifier"]["dtype"], "bfloat16")

    def test_get_enabled_benchmark_configs_returns_both_builtin_configs(self):
        fake_config = types.ModuleType("config")
        fake_config.BENCHMARKS_ENABLED = ("jailbreakbench", "harmbench")
        fake_config.JAILBREAKBENCH_CONFIG = {
            "judge_mode": "official",
            "max_samples": 50,
            "categories": ["Physical harm"],
            "harmful_score_threshold": 3,
            "official_judge": {
                "model_name": "test-jbb-judge",
                "api_url": "https://api.example/v1/chat/completions",
                "api_key_env": "TEST_JBB_KEY",
                "batch_size": 4,
                "timeout_sec": 30,
            },
        }
        fake_config.HARMBENCH_CONFIG = {
            "judge_mode": "official",
            "split": "all",
            "max_samples": 10,
            "categories": ["Privacy"],
            "functional_categories": ["contextual"],
            "semantic_categories": ["harmful"],
            "official_classifier": {
                "model_name": "test-cls",
                "batch_size": 4,
                "device": "cpu",
                "dtype": "float32",
            },
        }
        sys.modules["config"] = fake_config

        configs = get_enabled_benchmark_configs()

        self.assertEqual(set(configs), {"jailbreakbench", "harmbench"})
        self.assertEqual(configs["jailbreakbench"]["judge_mode"], "official")
        self.assertEqual(configs["jailbreakbench"]["max_samples"], 50)
        self.assertEqual(
            configs["jailbreakbench"]["official_judge"]["model_name"],
            "test-jbb-judge",
        )
        self.assertEqual(configs["harmbench"]["judge_mode"], "official")
        self.assertEqual(configs["harmbench"]["split"], "all")
        self.assertEqual(configs["harmbench"]["official_classifier"]["model_name"], "test-cls")

    def test_default_registry_includes_harmbench_and_jailbreakbench(self):
        registry = get_default_registry()

        self.assertIn("harmbench", registry.names())
        self.assertIn("jailbreakbench", registry.names())


if __name__ == "__main__":
    unittest.main()
