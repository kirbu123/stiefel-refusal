import sys
import types
import unittest

from benchmarks.registry import get_default_registry
from benchmarks.settings import (
    get_enabled_benchmark_configs,
    normalize_harmbench_config,
)


class TestBenchmarkSettings(unittest.TestCase):
    def setUp(self):
        self._original_config_module = sys.modules.get("config")

    def tearDown(self):
        if self._original_config_module is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = self._original_config_module

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
            "judge_mode": "project",
            "max_samples": 50,
            "categories": ["Physical harm"],
            "harmful_score_threshold": 3,
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
        self.assertEqual(configs["jailbreakbench"]["max_samples"], 50)
        self.assertEqual(configs["harmbench"]["judge_mode"], "official")
        self.assertEqual(configs["harmbench"]["split"], "all")
        self.assertEqual(configs["harmbench"]["official_classifier"]["model_name"], "test-cls")

    def test_default_registry_includes_harmbench_and_jailbreakbench(self):
        registry = get_default_registry()

        self.assertIn("harmbench", registry.names())
        self.assertIn("jailbreakbench", registry.names())


if __name__ == "__main__":
    unittest.main()
