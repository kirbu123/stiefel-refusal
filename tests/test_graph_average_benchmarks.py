import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeModel:
    def __init__(self, settings):
        self.settings = settings

    def get_layers(self):
        return [0, 1]

    def get_responses_batched(self, questions):
        return [f"response::{question}" for question in questions]

    def reload_model(self):
        return None


class FakeBenchmarkRunner:
    def __init__(self, benchmark_results):
        self.benchmark_results = benchmark_results
        self.prepare_calls = 0
        self.run_labels = []

    def prepare_original(self, model):
        self.prepare_calls += 1
        return {
            benchmark_name: {
                "attack_success_rate": benchmark_block["original"]["attack_success_rate"]
            }
            for benchmark_name, benchmark_block in self.benchmark_results.items()
        }

    def evaluate_modified(self, model, run_label):
        self.run_labels.append(run_label)
        return self.benchmark_results


class TestGraphAverageBenchmarks(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()
        sys.modules.pop("baselines.graph_average", None)

    def test_main_saves_benchmark_block_in_answers_json(self):
        few_shots_path = self.temp_path / "few-shots.json"
        few_shots_path.write_text(json.dumps({"categories": []}), encoding="utf-8")

        graph_file = self.temp_path / "graph.txt"
        graph_file.write_text("tag-a\ntag-b\n", encoding="utf-8")

        fake_torch = types.ModuleType("torch")
        fake_torch.manual_seed = lambda seed: None
        fake_torch.set_grad_enabled = lambda enabled: None

        fake_heretic_config = types.ModuleType("heretic.config")

        class Settings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_heretic_config.Settings = Settings
        fake_heretic_config.DatasetSpecification = object

        fake_heretic_model = types.ModuleType("heretic.model")
        fake_heretic_model.Model = FakeModel

        fake_heretic_utils = types.ModuleType("heretic.utils")
        fake_heretic_utils.load_prompts = lambda specification: ["good-prompt"]

        fake_config = types.ModuleType("config")
        fake_config.MODEL_NAME = "test-model"
        fake_config.GOOD_PROMPTS_DATASET = object()
        fake_config.JUDGE_API_URL = "http://judge"
        fake_config.CLASSIFIER_API_URL = "http://classifier"
        fake_config.JUDGE_MODEL = "judge-model"
        fake_config.FEW_SHOTS_PATH = few_shots_path
        fake_config.HARMLESS_EVAL_DATASET = object()
        fake_config.EVALUATE_LOCALITY = False
        fake_config.GRAPH_FILE = graph_file
        fake_config.MMLU_CONFIG = {"enabled": False}
        fake_config.CATEGORY_DATASET_SOURCE = "jailbreakbench"
        fake_config.CATEGORY_FILTER = "Physical harm"
        fake_config.get_method_results_dir = lambda method_name: self.temp_path / method_name

        fake_data_utils = types.ModuleType("data_utils")
        fake_data_utils.load_datasets_with_categories = (
            lambda loader_source, loader_filter: (
                [
                    {
                        "instruction": "question-physical-harm",
                        "category": "Physical harm",
                        "source": "jailbreakbench",
                    },
                ],
                ["Physical harm"],
            )
        )
        fake_data_utils.extract_response_after_think = lambda response: response

        fake_refusal_directions = types.ModuleType("refusal_directions")
        fake_refusal_directions.compute_refusal_direction = (
            lambda model, harmful_prompts, good_prompts: types.SimpleNamespace(shape=(1, 1))
        )

        fake_model_utils = types.ModuleType("model_utils")
        fake_model_utils.apply_abliteration_with_hyperparams = (
            lambda model, refusal_directions, max_weight, max_weight_position, min_weight, min_weight_distance, n_layers: None
        )

        fake_metrics = types.ModuleType("evaluate.metrics")
        fake_metrics.evaluate_responses = (
            lambda questions, responses, classifier_categories, description: (
                [1 for _ in questions],
                [{"score": 1} for _ in questions],
            )
        )
        fake_metrics.evaluate_locality = lambda *args, **kwargs: ([], 0.0)

        fake_mmlu = types.ModuleType("evaluate.mmlu")
        fake_mmlu.build_mmlu_result = lambda **kwargs: None
        fake_mmlu.evaluate_model_on_mmlu = lambda model, config: None
        fake_mmlu.get_cached_or_evaluate_original_mmlu = lambda model, model_name, config: None

        fake_plots = types.ModuleType("visualization.plots")
        fake_plots.plot_harmfulness_heatmap = lambda *args, **kwargs: None
        fake_plots.plot_locality_heatmap = lambda *args, **kwargs: None
        fake_plots.plot_harmfulness_distribution = lambda *args, **kwargs: None
        fake_plots.plot_locality_distribution = lambda *args, **kwargs: None
        fake_plots.generate_plot_filename = lambda prefix, category_name, evaluator_name, ext=".png": f"{prefix}{ext}"

        fake_hyperparams = types.ModuleType("baselines.hyperparams")
        fake_hyperparams.HYPERPARAMS = {
            "max_weight": [1.0],
            "max_weight_position": [0.7],
            "min_weight": [0.0],
            "min_weight_distance": [0.3],
        }

        benchmark_results = {
            "jailbreakbench": {
                "original": {"attack_success_rate": 0.2},
                "modified": {"attack_success_rate": 0.5},
                "delta_attack_success_rate": 0.3,
                "config": {"judge_mode": "project"},
                "details_file": {"original": "orig.json", "modified": "mod.json"},
                "by_category": {},
                "by_source": {},
            },
            "harmbench": {
                "original": {"attack_success_rate": 0.1},
                "modified": {"attack_success_rate": 0.4},
                "delta_attack_success_rate": 0.3,
                "config": {"judge_mode": "official", "split": "test"},
                "details_file": {"original": "harm_orig.json", "modified": "harm_mod.json"},
                "by_split": {},
                "by_category": {},
                "by_functional_category": {},
                "by_semantic_category": {},
            },
        }
        fake_benchmark_runner = FakeBenchmarkRunner(benchmark_results)
        fake_benchmarks_integration = types.ModuleType("benchmarks.integration")
        fake_benchmarks_integration.build_benchmark_runner = (
            lambda **kwargs: fake_benchmark_runner
        )

        fake_modules = {
            "torch": fake_torch,
            "heretic.config": fake_heretic_config,
            "heretic.model": fake_heretic_model,
            "heretic.utils": fake_heretic_utils,
            "config": fake_config,
            "data_utils": fake_data_utils,
            "refusal_directions": fake_refusal_directions,
            "model_utils": fake_model_utils,
            "evaluate.metrics": fake_metrics,
            "evaluate.mmlu": fake_mmlu,
            "visualization.plots": fake_plots,
            "baselines.hyperparams": fake_hyperparams,
            "benchmarks.integration": fake_benchmarks_integration,
        }

        sys.modules.pop("baselines.graph_average", None)
        with patch.dict(sys.modules, fake_modules, clear=False):
            module = importlib.import_module("baselines.graph_average")
            module.main()

        answers_files = sorted((self.temp_path / "graph_average" / "answers").glob("answers_*.json"))
        self.assertTrue(answers_files)
        payload = json.loads(answers_files[-1].read_text(encoding="utf-8"))

        self.assertIn("benchmarks", payload)
        self.assertAlmostEqual(
            payload["benchmarks"]["jailbreakbench"]["modified"]["attack_success_rate"],
            0.5,
        )
        self.assertAlmostEqual(
            payload["benchmarks"]["harmbench"]["modified"]["attack_success_rate"],
            0.4,
        )
        self.assertEqual(fake_benchmark_runner.prepare_calls, 1)
        self.assertEqual(fake_benchmark_runner.run_labels, ["graph_average_Physical_harm_max_weight=1.0_&max_weight_position=0.7_&min_weight=0.0_&min_weight_distance=0.3"])


if __name__ == "__main__":
    unittest.main()
