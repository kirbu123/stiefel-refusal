import tempfile
import types
import unittest
from pathlib import Path

from benchmarks.base import BenchmarkDefinition
from benchmarks.io import BenchmarkIO
from benchmarks.registry import BenchmarkRegistry
from benchmarks.runner import BenchmarkRunner


class FakeBenchmark(BenchmarkDefinition):
    name = "fakebench"

    def __init__(self):
        self.evaluate_calls = 0

    def normalize_config(self, raw_config=None):
        raw_config = dict(raw_config or {})
        return {"label": str(raw_config.get("label", "default"))}

    def evaluate_model(self, model, *, config, classifier_categories):
        self.evaluate_calls += 1
        metric = float(getattr(model, "metric", 0.0))
        return {
            "summary": {
                "attack_success_rate": metric,
                "n_prompts": 1,
                "jailbroken_count": int(metric >= 0.5),
                "mean_score": metric,
                "n_scored": 1,
                "by_category": {},
                "by_source": {},
            },
            "details": {
                "label": config["label"],
                "classifier_count": len(classifier_categories),
                "metric": metric,
            },
        }

    def build_result_block(
        self,
        *,
        original_summary,
        modified_summary,
        config,
        original_details_file,
        modified_details_file,
    ):
        return {
            "original": dict(original_summary),
            "modified": dict(modified_summary),
            "delta_attack_success_rate": (
                modified_summary["attack_success_rate"]
                - original_summary["attack_success_rate"]
            ),
            "config": dict(config),
            "details_file": {
                "original": original_details_file,
                "modified": modified_details_file,
            },
            "by_category": {},
            "by_source": {},
        }


class TestBenchmarkRunner(unittest.TestCase):
    def test_prepare_original_uses_cache_across_runner_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            definition = FakeBenchmark()
            registry = BenchmarkRegistry()
            registry.register(definition)

            runner_one = BenchmarkRunner(
                method_results_dir=temp_path,
                model_name="model-a",
                classifier_categories=[],
                registry=registry,
                benchmark_io=BenchmarkIO(temp_path, project_root=temp_path),
                enabled_benchmark_configs={"fakebench": {"label": "alpha"}},
            )
            first_summary = runner_one.prepare_original(
                types.SimpleNamespace(metric=0.25)
            )

            runner_two = BenchmarkRunner(
                method_results_dir=temp_path,
                model_name="model-a",
                classifier_categories=[],
                registry=registry,
                benchmark_io=BenchmarkIO(temp_path, project_root=temp_path),
                enabled_benchmark_configs={"fakebench": {"label": "alpha"}},
            )
            second_summary = runner_two.prepare_original(
                types.SimpleNamespace(metric=0.95)
            )

        self.assertEqual(definition.evaluate_calls, 1)
        self.assertAlmostEqual(
            first_summary["fakebench"]["attack_success_rate"],
            0.25,
        )
        self.assertAlmostEqual(
            second_summary["fakebench"]["attack_success_rate"],
            0.25,
        )

    def test_evaluate_modified_returns_serialized_comparison_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            definition = FakeBenchmark()
            registry = BenchmarkRegistry()
            registry.register(definition)
            runner = BenchmarkRunner(
                method_results_dir=temp_path,
                model_name="model-a",
                classifier_categories=[{"id": "c1"}],
                registry=registry,
                benchmark_io=BenchmarkIO(temp_path, project_root=temp_path),
                enabled_benchmark_configs={"fakebench": {"label": "alpha"}},
            )

            runner.prepare_original(types.SimpleNamespace(metric=0.1))
            blocks = runner.evaluate_modified(
                types.SimpleNamespace(metric=0.6),
                run_label="trial 1",
            )

            block = blocks["fakebench"]
            self.assertAlmostEqual(
                block["original"]["attack_success_rate"],
                0.1,
            )
            self.assertAlmostEqual(
                block["modified"]["attack_success_rate"],
                0.6,
            )
            self.assertAlmostEqual(block["delta_attack_success_rate"], 0.5)
            self.assertEqual(block["config"]["label"], "alpha")

            original_path = temp_path / block["details_file"]["original"]
            modified_path = temp_path / block["details_file"]["modified"]
            self.assertTrue(original_path.exists())
            self.assertTrue(modified_path.exists())


if __name__ == "__main__":
    unittest.main()
