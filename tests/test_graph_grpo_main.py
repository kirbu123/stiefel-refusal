import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


if torch is not None:
    class FakeModel:
        instances = []

        def __init__(self, settings):
            self.settings = settings
            self.reload_calls = 0
            self.response_calls = []
            self._layers = [torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)]
            FakeModel.instances.append(self)

        def get_layers(self):
            return self._layers

        def get_responses_batched(self, questions):
            self.response_calls.append(list(questions))
            return [f"<think>hidden</think>response::{question}" for question in questions]

        def reload_model(self):
            self.reload_calls += 1


    class FakeDirectionWeights:
        def __init__(
            self,
            n_directions,
            n_layers,
            hidden_size,
            init_type="average",
            topic_idx=0,
            mode="scalar",
        ):
            self.mode = mode
            if mode == "scalar":
                self.weights = torch.zeros(n_directions, dtype=torch.float32)
            else:
                self.weights = torch.zeros(
                    (n_directions, n_layers + 1, hidden_size),
                    dtype=torch.float32,
                )

        def to(self, device):
            self.weights = self.weights.to(device)
            return self

        def __call__(self, extracted_directions):
            return extracted_directions[0].detach().clone()


@unittest.skipUnless(torch is not None, "torch is required for graph_grpo main tests")
class TestGraphGrpoMain(unittest.TestCase):
    def setUp(self):
        FakeModel.instances = []
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()
        sys.modules.pop("baselines.graph_grpo.__main__", None)

    def _run_main(
        self,
        dataset,
        category_dataset_source,
        category_filter,
        model_batch_size=2,
    ):
        few_shots_path = self.temp_path / "few-shots.json"
        few_shots_path.write_text(json.dumps({"categories": []}), encoding="utf-8")

        graph_file = self.temp_path / "graph.txt"
        graph_file.write_text("physical harm root\nprivacy root\n", encoding="utf-8")

        results_root = self.temp_path / "results"
        results_root.mkdir(parents=True, exist_ok=True)

        loader_calls = []
        wandb_init_calls = []
        wandb_log_calls = []

        fake_wandb = types.ModuleType("wandb")
        fake_wandb.init = lambda **kwargs: wandb_init_calls.append(kwargs)
        fake_wandb.log = lambda payload, step=None: wandb_log_calls.append((payload, step))
        fake_wandb.finish = lambda: None

        fake_heretic_config = types.ModuleType("heretic.config")

        class Settings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_heretic_config.Settings = Settings

        fake_heretic_utils = types.ModuleType("heretic.utils")
        fake_heretic_utils.load_prompts = lambda specification: ["good-prompt"]

        fake_heretic_model = types.ModuleType("heretic.model")
        fake_heretic_model.Model = FakeModel

        fake_config = types.ModuleType("config")
        fake_config.MODEL_NAME = "test-model"
        fake_config.MODEL_BATCH_SIZE = model_batch_size
        fake_config.CATEGORIES = ["Physical harm", "Privacy"]
        fake_config.GOOD_PROMPTS_DATASET = object()
        fake_config.RESULTS_DIR = results_root
        fake_config.GRPO_CONFIG = {
            "n_groups": 2,
            "n_epochs": 1,
            "learning_rate": 1e-3,
            "noise_scale": 0.1,
            "ref_alpha": 1.0,
            "is_clip_ratio": 5.0,
            "clip_ratio": 0.2,
            "loss_agg_mode": "token-mean",
        }
        fake_config.ABLITERATION_PARAMS = {
            "max_weight": 2.0,
            "max_weight_position": 0.7,
            "min_weight": 0.1,
            "min_weight_distance": 0.3,
        }
        fake_config.FEW_SHOTS_PATH = few_shots_path
        fake_config.GRAPH_FILE = graph_file
        fake_config.EVALUATION_BACKEND = "llamaguard"
        fake_config.MMLU_CONFIG = {"enabled": False}
        fake_config.DEBUG = False
        fake_config.get_method_results_dir = lambda method_name: self.temp_path / method_name

        fake_data_utils = types.ModuleType("data_utils")
        fake_data_utils.SUPPORTED_CATEGORY_DATASET_SOURCES = ("combined", "jailbreakbench")
        fake_data_utils.SUPPORTED_JAILBREAKBENCH_CATEGORIES = (
            "Harassment/Discrimination",
            "Malware/Hacking",
            "Physical harm",
            "Economic harm",
            "Fraud/Deception",
            "Disinformation",
            "Sexual/Adult content",
            "Privacy",
            "Expert advice",
            "Government decision-making",
        )

        def fake_load_datasets_with_categories(source="combined", category_filter=None):
            loader_calls.append((source, category_filter))
            filtered = [
                item for item in dataset
                if category_filter is None or item["category"] == category_filter
            ]
            categories = sorted({item["category"] for item in filtered})
            return filtered, categories

        fake_data_utils.load_datasets_with_categories = fake_load_datasets_with_categories
        fake_data_utils.extract_response_after_think = lambda response: response.split("</think>")[-1].strip()

        fake_refusal_directions = types.ModuleType("refusal_directions")
        fake_refusal_directions.compute_refusal_direction = (
            lambda model, harmful_prompts, good_prompts: torch.ones((2, 2), dtype=torch.float32)
        )
        fake_refusal_directions.save_refusal_directions = lambda *args, **kwargs: None
        fake_refusal_directions.load_refusal_directions = lambda *args, **kwargs: ([], [])

        fake_model_utils = types.ModuleType("model_utils")
        fake_model_utils.LearnableDirectionWeights = FakeDirectionWeights
        fake_model_utils.apply_abliteration_with_hyperparams = lambda *args, **kwargs: None

        fake_reward = types.ModuleType("baselines.graph_grpo.reward")
        fake_reward.compute_reward = lambda questions, responses, classifier_categories, backend: [1 for _ in questions]

        fake_mmlu = types.ModuleType("evaluate.mmlu")
        fake_mmlu.build_mmlu_result = lambda **kwargs: None
        fake_mmlu.evaluate_model_on_mmlu = lambda model, config: None
        fake_mmlu.get_cached_or_evaluate_original_mmlu = lambda model, model_name, config: None

        fake_optuna = types.ModuleType("baselines.graph_grpo.optuna_optimizer")

        def fake_optimize_scalar_weights_with_optuna(
            direction_weights,
            extracted_directions,
            model,
            questions,
            abliteration_params,
            classifier_categories,
            n_layers,
            ref_alpha,
            n_trials,
            sampler_name,
            sampler_seed,
            weight_min,
            weight_max,
            backend,
        ):
            direction_weights.weights = torch.tensor([0.75, -0.25], dtype=torch.float32)
            return {
                "optimization_history": [
                    {
                        "trial_number": 0,
                        "weights": [0.75, -0.25],
                        "mean_reward": 1.0,
                        "best_reward": 1.0,
                        "n_questions": len(questions),
                    }
                ],
                "best_trial_number": 0,
                "best_value": 1.0,
                "best_weights": [0.75, -0.25],
            }

        fake_optuna.optimize_scalar_weights_with_optuna = fake_optimize_scalar_weights_with_optuna

        fake_trainer = types.ModuleType("baselines.graph_grpo.trainer")
        fake_trainer.train_grpo_is_step = lambda *args, **kwargs: {
            "mean_reward": 1.0,
            "best_reward": 1.0,
        }

        fake_modules = {
            "wandb": fake_wandb,
            "heretic.config": fake_heretic_config,
            "heretic.utils": fake_heretic_utils,
            "heretic.model": fake_heretic_model,
            "config": fake_config,
            "data_utils": fake_data_utils,
            "refusal_directions": fake_refusal_directions,
            "model_utils": fake_model_utils,
            "baselines.graph_grpo.reward": fake_reward,
            "evaluate.mmlu": fake_mmlu,
            "baselines.graph_grpo.optuna_optimizer": fake_optuna,
            "baselines.graph_grpo.trainer": fake_trainer,
        }

        env = {
            "CATEGORY_DATASET_SOURCE": category_dataset_source,
            "CATEGORY_FILTER": category_filter,
            "OPTIMIZER_METHOD": "optuna",
            "WEIGHTS_MODE": "scalar",
            "WEIGHTS_INIT_TYPE": "average",
        }

        sys.modules.pop("baselines.graph_grpo.__main__", None)
        with patch.dict(sys.modules, fake_modules, clear=False):
            with patch.dict(os.environ, env, clear=False):
                module = importlib.import_module("baselines.graph_grpo.__main__")
                module.main()

        answers_files = sorted((self.temp_path / "graph_grpo" / "answers").glob("answers_*.json"))
        self.assertTrue(answers_files)
        answers_data = json.loads(answers_files[-1].read_text(encoding="utf-8"))

        return {
            "answers_data": answers_data,
            "loader_calls": loader_calls,
            "wandb_init_calls": wandb_init_calls,
            "wandb_log_calls": wandb_log_calls,
            "models": FakeModel.instances,
        }

    def test_main_uses_jailbreakbench_category_filter(self):
        dataset = [
            {
                "instruction": "physical-question",
                "category": "Physical harm",
                "source": "jailbreakbench",
            },
            {
                "instruction": "privacy-question",
                "category": "Privacy",
                "source": "jailbreakbench",
            },
        ]

        result = self._run_main(
            dataset=dataset,
            category_dataset_source="jailbreakbench",
            category_filter="Physical harm",
            model_batch_size=4,
        )

        self.assertEqual(result["loader_calls"], [("jailbreakbench", "Physical harm")])
        self.assertEqual(result["answers_data"]["experiment_config"]["category"], "Physical harm")
        self.assertEqual(
            result["answers_data"]["experiment_config"]["category_dataset_source"],
            "jailbreakbench",
        )
        self.assertEqual(result["answers_data"]["questions"], ["physical-question"])
        self.assertEqual(result["models"][0].response_calls[0], ["physical-question"])
        self.assertEqual(
            result["wandb_init_calls"][0]["config"]["category_dataset_source"],
            "jailbreakbench",
        )

    def test_main_filters_combined_locally_before_batch_sampling(self):
        dataset = [
            {"instruction": "physical-question-1", "category": "Physical harm", "source": "combined"},
            {"instruction": "privacy-question", "category": "Privacy", "source": "combined"},
            {"instruction": "physical-question-2", "category": "Physical harm", "source": "combined"},
        ]

        result = self._run_main(
            dataset=dataset,
            category_dataset_source="combined",
            category_filter="Physical harm",
            model_batch_size=1,
        )

        self.assertEqual(result["loader_calls"], [("combined", None)])
        self.assertEqual(
            set(result["models"][0].response_calls[0]),
            {"physical-question-1", "physical-question-2"},
        )
        self.assertEqual(len(result["answers_data"]["questions"]), 1)
        self.assertIn(
            result["answers_data"]["questions"][0],
            {"physical-question-1", "physical-question-2"},
        )
        self.assertNotIn("privacy-question", result["answers_data"]["questions"])

    def test_main_defaults_empty_category_filter_to_physical_harm(self):
        dataset = [
            {"instruction": "physical-question", "category": "Physical harm", "source": "combined"},
            {"instruction": "privacy-question", "category": "Privacy", "source": "combined"},
        ]

        result = self._run_main(
            dataset=dataset,
            category_dataset_source="combined",
            category_filter="",
            model_batch_size=4,
        )

        self.assertEqual(result["loader_calls"], [("combined", None)])
        self.assertEqual(result["answers_data"]["experiment_config"]["category"], "Physical harm")
        self.assertEqual(result["answers_data"]["questions"], ["physical-question"])
        self.assertEqual(
            result["wandb_init_calls"][0]["config"]["category"],
            "Physical harm",
        )


if __name__ == "__main__":
    unittest.main()
