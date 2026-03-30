import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeModel:
    instances = []

    def __init__(self, settings):
        self.settings = settings
        self.response_calls = []
        FakeModel.instances.append(self)

    def get_layers(self):
        return [0, 1]

    def get_responses_batched(self, questions):
        self.response_calls.append(list(questions))
        return [f"response::{question}" for question in questions]

    def reload_model(self):
        return None


class TestGraphAverage(unittest.TestCase):
    def setUp(self):
        FakeModel.instances = []
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()
        sys.modules.pop("baselines.graph_average", None)

    def _run_main(self, dataset, categories, source, category_filter):
        few_shots_path = self.temp_path / "few-shots.json"
        few_shots_path.write_text(json.dumps({"categories": []}), encoding="utf-8")

        graph_file = self.temp_path / "graph.txt"
        graph_file.write_text("tag-a\ntag-b\n", encoding="utf-8")

        loader_calls = []
        evaluate_calls = []

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
        fake_config.CATEGORY_DATASET_SOURCE = source
        fake_config.CATEGORY_FILTER = category_filter
        fake_config.get_method_results_dir = lambda method_name: self.temp_path / method_name

        fake_data_utils = types.ModuleType("data_utils")

        def fake_load_datasets_with_categories(loader_source, loader_filter):
            loader_calls.append((loader_source, loader_filter))
            return dataset, categories

        fake_data_utils.load_datasets_with_categories = fake_load_datasets_with_categories
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

        def fake_evaluate_responses(questions, responses, classifier_categories, description):
            evaluate_calls.append((list(questions), description))
            return [1 for _ in questions], [{"score": 1} for _ in questions]

        fake_metrics.evaluate_responses = fake_evaluate_responses
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
        }

        sys.modules.pop("baselines.graph_average", None)
        with patch.dict(sys.modules, fake_modules, clear=False):
            module = importlib.import_module("baselines.graph_average")
            module.main()

        return loader_calls, evaluate_calls, FakeModel.instances

    def test_main_uses_category_filter_from_loader(self):
        dataset = [
            {
                "instruction": "question-physical-harm",
                "category": "Physical harm",
                "source": "jailbreakbench",
            },
        ]

        loader_calls, evaluate_calls, models = self._run_main(
            dataset=dataset,
            categories=["Physical harm"],
            source="jailbreakbench",
            category_filter="Physical harm",
        )

        self.assertEqual(loader_calls, [("jailbreakbench", "Physical harm")])
        self.assertEqual(len(evaluate_calls), 2)
        self.assertEqual(evaluate_calls[0][0], ["question-physical-harm"])
        self.assertEqual(models[0].response_calls[0], ["question-physical-harm"])

    def test_main_no_longer_skips_non_physical_categories(self):
        dataset = [
            {"instruction": "malware-question", "category": "Malware/Hacking", "source": "jailbreakbench"},
            {"instruction": "privacy-question", "category": "Privacy", "source": "jailbreakbench"},
        ]

        loader_calls, evaluate_calls, _ = self._run_main(
            dataset=dataset,
            categories=["Malware/Hacking", "Privacy"],
            source="jailbreakbench",
            category_filter=None,
        )

        self.assertEqual(loader_calls, [("jailbreakbench", None)])
        self.assertEqual(len(evaluate_calls), 4)
        self.assertEqual(evaluate_calls[0][0], ["malware-question"])
        self.assertEqual(evaluate_calls[2][0], ["privacy-question"])


if __name__ == "__main__":
    unittest.main()
