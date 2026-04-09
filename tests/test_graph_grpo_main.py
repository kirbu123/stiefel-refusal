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
        optimizer_method="optuna",
        weights_mode="scalar",
        grpo_n_epochs=1,
        mmlu_enabled=False,
        train_metrics_sequence=None,
        modified_mmlu_scores=None,
        benchmark_results=None,
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
        define_metric_calls = []
        train_step_questions = []
        optuna_trial_batches = []
        mmlu_eval_calls = []
        benchmark_runner = None

        if benchmark_results is not None:
            class FakeBenchmarkRunner:
                def __init__(self, results):
                    self.results = results
                    self.prepare_calls = 0
                    self.run_labels = []

                def prepare_original(self, model):
                    self.prepare_calls += 1
                    return {
                        "jailbreakbench": {
                            "attack_success_rate": self.results["jailbreakbench"]["original"]["attack_success_rate"]
                        }
                    }

                def evaluate_modified(self, model, run_label):
                    self.run_labels.append(run_label)
                    return self.results

            benchmark_runner = FakeBenchmarkRunner(benchmark_results)

        fake_wandb = types.ModuleType("wandb")
        fake_wandb.init = lambda **kwargs: wandb_init_calls.append(kwargs)
        fake_wandb.log = lambda payload, step=None: wandb_log_calls.append((payload, step))
        fake_wandb.define_metric = lambda *args, **kwargs: define_metric_calls.append((args, kwargs))
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
            "n_epochs": grpo_n_epochs,
            "learning_rate": 1e-3,
            "noise_scale": 0.1,
            "ref_alpha": 1.0,
            "is_clip_ratio": 5.0,
            "clip_ratio": 0.2,
            "loss_agg_mode": "token-mean",
            "kl_loss_coef": 0.01,
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
        fake_config.MMLU_CONFIG = {"enabled": mmlu_enabled}
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
        modified_scores_iter = iter(modified_mmlu_scores or [0.75])

        def fake_build_mmlu_result(**kwargs):
            original_result = kwargs["original_result"]
            modified_result = kwargs["modified_result"]
            return {
                "original": {"accuracy": original_result["summary"]["accuracy"]},
                "modified": {"accuracy": modified_result["summary"]["accuracy"]},
            }

        def fake_evaluate_model_on_mmlu(model, config):
            mmlu_eval_calls.append("modified")
            return {"summary": {"accuracy": next(modified_scores_iter)}}

        def fake_get_cached_or_evaluate_original_mmlu(model, model_name, config):
            if not mmlu_enabled:
                return None
            return {"summary": {"accuracy": 0.25}}

        fake_mmlu.build_mmlu_result = fake_build_mmlu_result
        fake_mmlu.evaluate_model_on_mmlu = fake_evaluate_model_on_mmlu
        fake_mmlu.get_cached_or_evaluate_original_mmlu = fake_get_cached_or_evaluate_original_mmlu

        fake_benchmarks_integration = types.ModuleType("benchmarks.integration")
        fake_benchmarks_integration.build_benchmark_runner = (
            lambda **kwargs: benchmark_runner
        )
        fake_benchmarks_integration.get_benchmark_attack_success_rate = (
            lambda results, benchmark_name: results.get(benchmark_name, {}).get("modified", {}).get("attack_success_rate")
            if results else None
        )

        fake_optuna = types.ModuleType("baselines.graph_grpo.optuna_optimizer")

        def fake_optimize_weights_with_optuna(
            direction_weights,
            extracted_directions,
            model,
            questions,
            abliteration_params,
            classifier_categories,
            n_layers,
            ref_alpha,
            reward_sign,
            kl_loss_coef,
            n_trials,
            sampler_name,
            sampler_seed,
            weight_min,
            weight_max,
            question_sampler=None,
            loss_agg_mode="token-mean",
            backend=None,
        ):
            if question_sampler is not None:
                trial_batches = [list(question_sampler()) for _ in range(min(n_trials, 3))]
            else:
                trial_batches = [list(questions)]
            optuna_trial_batches.extend(trial_batches)
            best_trial_index = 1 if len(trial_batches) > 1 else 0

            if direction_weights.mode == "dense":
                best_weights_tensor = torch.arange(
                    direction_weights.weights.numel(),
                    dtype=torch.float32,
                ).reshape_as(direction_weights.weights)
                optimization_history = []
                for trial_index, trial_questions in enumerate(trial_batches):
                    mean_kl = 0.1 * float(trial_index + 1)
                    mean_objective = float(trial_index + 1) - kl_loss_coef * mean_kl
                    optimization_history.append({
                        "trial_number": trial_index,
                        "weights_mode": "dense",
                        "weights_shape": list(best_weights_tensor.shape),
                        "weights_mean": float(best_weights_tensor.mean().item()),
                        "weights_std": float(best_weights_tensor.std(unbiased=False).item()),
                        "weights_min": float(best_weights_tensor.min().item()),
                        "weights_max": float(best_weights_tensor.max().item()),
                        "weights_norm": float(best_weights_tensor.norm().item()),
                        "mean_reward": float(trial_index + 1),
                        "best_reward": float(trial_index + 1),
                        "mean_harmfulness": float(trial_index + 1),
                        "best_harmfulness": float(trial_index + 1),
                        "mean_kl": mean_kl,
                        "mean_objective": mean_objective,
                        "n_questions": len(trial_questions),
                    })
            else:
                best_weights_tensor = torch.tensor([0.75, -0.25], dtype=torch.float32)
                optimization_history = []
                for trial_index, trial_questions in enumerate(trial_batches):
                    mean_kl = 0.1 * float(trial_index + 1)
                    mean_objective = float(trial_index + 1) - kl_loss_coef * mean_kl
                    optimization_history.append({
                        "trial_number": trial_index,
                        "weights": [0.75, -0.25],
                        "weights_mode": "scalar",
                        "mean_reward": float(trial_index + 1),
                        "best_reward": float(trial_index + 1),
                        "mean_harmfulness": float(trial_index + 1),
                        "best_harmfulness": float(trial_index + 1),
                        "mean_kl": mean_kl,
                        "mean_objective": mean_objective,
                        "n_questions": len(trial_questions),
                    })

            direction_weights.weights = best_weights_tensor
            return {
                "optimization_history": optimization_history,
                "best_trial_number": best_trial_index,
                "best_value": optimization_history[best_trial_index]["mean_objective"],
                "best_mean_objective": optimization_history[best_trial_index]["mean_objective"],
                "best_mean_harmfulness": float(best_trial_index + 1),
                "best_mean_kl": optimization_history[best_trial_index]["mean_kl"],
                "best_weights": best_weights_tensor.tolist(),
                "best_trial_questions": trial_batches[best_trial_index],
            }

        fake_optuna.optimize_weights_with_optuna = fake_optimize_weights_with_optuna
        fake_optuna.optimize_scalar_weights_with_optuna = fake_optimize_weights_with_optuna

        fake_trainer = types.ModuleType("baselines.graph_grpo.trainer")
        trainer_metrics_iter = iter(
            train_metrics_sequence
            or [
                {"mean_reward": 1.0, "best_reward": 1.0, "mean_harmfulness": 1.0}
                for _ in range(grpo_n_epochs)
            ]
        )

        def fake_train_grpo_is_step(*args, **kwargs):
            train_step_questions.append(list(kwargs["questions"]))
            metrics = dict(next(trainer_metrics_iter))
            metrics.setdefault("mean_kl", 0.0)
            metrics.setdefault("kl_loss", metrics["mean_kl"])
            metrics.setdefault(
                "mean_objective",
                metrics["mean_reward"] - fake_config.GRPO_CONFIG["kl_loss_coef"] * metrics["mean_kl"],
            )
            return metrics

        fake_trainer.train_grpo_is_step = fake_train_grpo_is_step

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
            "benchmarks.integration": fake_benchmarks_integration,
            "baselines.graph_grpo.optuna_optimizer": fake_optuna,
            "baselines.graph_grpo.trainer": fake_trainer,
        }

        env = {
            "CATEGORY_DATASET_SOURCE": category_dataset_source,
            "CATEGORY_FILTER": category_filter,
            "OPTIMIZER_METHOD": optimizer_method,
            "WEIGHTS_MODE": weights_mode,
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
            "define_metric_calls": define_metric_calls,
            "train_step_questions": train_step_questions,
            "optuna_trial_batches": optuna_trial_batches,
            "mmlu_eval_calls": mmlu_eval_calls,
            "models": FakeModel.instances,
            "benchmark_runner": benchmark_runner,
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
        scalar_payloads = [payload for payload, _step in result["wandb_log_calls"]]
        self.assertTrue(any("clean_model/harmfulness_on_full_dataset" in payload for payload in scalar_payloads))
        self.assertTrue(any("best_value_model/harmfulness_on_full_dataset" in payload for payload in scalar_payloads))
        series_payloads = [
            payload for payload in scalar_payloads
            if "model_state_eval/point_index" in payload
        ]
        self.assertEqual(
            [payload["model_state_eval/point_name"] for payload in series_payloads],
            ["clean", "best_value"],
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

    def test_grpo_logs_clean_and_best_batch_series_and_restores_best_epoch(self):
        dataset = [
            {"instruction": "physical-question-1", "category": "Physical harm", "source": "combined"},
            {"instruction": "physical-question-2", "category": "Physical harm", "source": "combined"},
            {"instruction": "physical-question-3", "category": "Physical harm", "source": "combined"},
            {"instruction": "physical-question-4", "category": "Physical harm", "source": "combined"},
        ]

        result = self._run_main(
            dataset=dataset,
            category_dataset_source="combined",
            category_filter="Physical harm",
            model_batch_size=2,
            optimizer_method="grpo",
            weights_mode="scalar",
            grpo_n_epochs=3,
            mmlu_enabled=True,
            train_metrics_sequence=[
                {"mean_reward": 0.1, "best_reward": 0.3, "mean_harmfulness": 0.1},
                {"mean_reward": 0.9, "best_reward": 1.0, "mean_harmfulness": 0.9},
                {"mean_reward": 0.4, "best_reward": 0.8, "mean_harmfulness": 0.4},
            ],
            modified_mmlu_scores=[0.8],
        )

        self.assertEqual(result["loader_calls"], [("combined", None)])
        self.assertEqual(len(result["train_step_questions"]), 3)
        self.assertTrue(all(len(batch) == 2 for batch in result["train_step_questions"]))
        self.assertNotEqual(result["train_step_questions"][0], result["train_step_questions"][1])
        self.assertEqual(
            result["answers_data"]["experiment_config"]["best_grpo_epoch"],
            2,
        )
        self.assertEqual(
            result["answers_data"]["experiment_config"]["best_train_batch_mean_objective"],
            0.9,
        )
        self.assertEqual(
            result["answers_data"]["experiment_config"]["best_train_batch_mean_harmfulness"],
            0.9,
        )
        self.assertEqual(
            result["answers_data"]["experiment_config"]["best_train_batch_mean_kl"],
            0.0,
        )
        self.assertEqual(
            result["answers_data"]["questions"],
            result["train_step_questions"][1],
        )
        self.assertEqual(result["answers_data"]["optimal_objective"], 0.9)
        self.assertEqual(result["answers_data"]["optimal_harmfulness"], 0.9)
        self.assertEqual(
            result["answers_data"]["optimal_harmfulness_source"],
            "best_train_batch_mean_objective",
        )
        self.assertEqual(
            result["answers_data"]["optimal_objective_source"],
            "best_train_batch_mean_objective",
        )
        self.assertEqual(result["mmlu_eval_calls"], ["modified"])
        self.assertTrue(result["define_metric_calls"])
        run_name = result["wandb_init_calls"][0]["name"]
        self.assertIn("grpo_Physical_harm", run_name)
        self.assertIn("weights_scalar", run_name)
        self.assertIn("init_average", run_name)
        self.assertIn("n_groups2", run_name)
        self.assertIn("n_epochs3", run_name)
        self.assertIn("lr0.001", run_name)
        self.assertIn("noise0.1", run_name)
        self.assertIn("ref_alpha1", run_name)
        self.assertIn("is_clip5", run_name)
        self.assertIn("clip0.2", run_name)
        self.assertIn("loss_token-mean", run_name)
        self.assertIn("klcoef0.01", run_name)

        scalar_payloads = [payload for payload, _step in result["wandb_log_calls"]]
        self.assertTrue(any("clean_model/mmlu_score" in payload for payload in scalar_payloads))
        self.assertTrue(any("best_value_model/mmlu_score" in payload for payload in scalar_payloads))
        self.assertTrue(any("best_batch_model/mmlu_score" in payload for payload in scalar_payloads))
        self.assertTrue(any("clean_model/harmfulness_on_full_dataset" in payload for payload in scalar_payloads))
        self.assertTrue(any("best_value_model/harmfulness_on_full_dataset" in payload for payload in scalar_payloads))
        self.assertTrue(any("best_batch_model/harmfulness_on_full_dataset" in payload for payload in scalar_payloads))

        series_payloads = [
            payload for payload in scalar_payloads
            if "model_state_eval/point_index" in payload
        ]
        self.assertEqual(
            [payload["model_state_eval/point_index"] for payload in series_payloads],
            [0, 1],
        )
        self.assertEqual(
            [payload["model_state_eval/point_name"] for payload in series_payloads],
            ["clean", "best_value"],
        )

    def test_grpo_skips_mmlu_series_when_mmlu_disabled(self):
        dataset = [
            {"instruction": "physical-question-1", "category": "Physical harm", "source": "combined"},
            {"instruction": "physical-question-2", "category": "Physical harm", "source": "combined"},
        ]

        result = self._run_main(
            dataset=dataset,
            category_dataset_source="combined",
            category_filter="Physical harm",
            model_batch_size=2,
            optimizer_method="grpo",
            weights_mode="scalar",
            grpo_n_epochs=1,
            mmlu_enabled=False,
            train_metrics_sequence=[{"mean_reward": 0.7, "best_reward": 0.9, "mean_harmfulness": 0.7}],
        )

        scalar_payloads = [payload for payload, _step in result["wandb_log_calls"]]
        self.assertTrue(any("clean_model/harmfulness_on_full_dataset" in payload for payload in scalar_payloads))
        self.assertTrue(any("best_value_model/harmfulness_on_full_dataset" in payload for payload in scalar_payloads))
        self.assertTrue(any("best_batch_model/harmfulness_on_full_dataset" in payload for payload in scalar_payloads))
        self.assertFalse(any("clean_model/mmlu_score" in payload for payload in scalar_payloads))
        self.assertFalse(any("best_value_model/mmlu_score" in payload for payload in scalar_payloads))
        self.assertFalse(any("best_batch_model/mmlu_score" in payload for payload in scalar_payloads))

    def test_main_reaches_dense_optuna_path_and_saves_dense_weights(self):
        dataset = [
            {"instruction": "physical-question-1", "category": "Physical harm", "source": "combined"},
            {"instruction": "physical-question-2", "category": "Physical harm", "source": "combined"},
        ]

        result = self._run_main(
            dataset=dataset,
            category_dataset_source="combined",
            category_filter="Physical harm",
            model_batch_size=2,
            optimizer_method="optuna",
            weights_mode="dense",
            mmlu_enabled=False,
        )

        self.assertEqual(result["answers_data"]["experiment_config"]["weights_mode"], "dense")
        self.assertEqual(len(result["answers_data"]["final_weights"]), 2)
        self.assertEqual(len(result["answers_data"]["final_weights"][0]), 3)
        self.assertEqual(len(result["answers_data"]["final_weights"][0][0]), 2)
        self.assertIn("optimization_history", result["answers_data"])
        dense_history = result["answers_data"]["optimization_history"][0]
        self.assertEqual(dense_history["weights_mode"], "dense")
        self.assertIn("weights_shape", dense_history)
        self.assertNotIn("weights", dense_history)

    def test_optuna_samples_random_batch_each_trial_and_saves_best_trial_batch(self):
        dataset = [
            {"instruction": "physical-question-1", "category": "Physical harm", "source": "combined"},
            {"instruction": "physical-question-2", "category": "Physical harm", "source": "combined"},
            {"instruction": "physical-question-3", "category": "Physical harm", "source": "combined"},
            {"instruction": "physical-question-4", "category": "Physical harm", "source": "combined"},
        ]

        result = self._run_main(
            dataset=dataset,
            category_dataset_source="combined",
            category_filter="Physical harm",
            model_batch_size=2,
            optimizer_method="optuna",
            weights_mode="scalar",
            mmlu_enabled=False,
        )

        run_name = result["wandb_init_calls"][0]["name"]
        self.assertIn("optuna_Physical_harm", run_name)
        self.assertIn("weights_scalar", run_name)
        self.assertIn("init_average", run_name)
        self.assertIn("sampler_tpe", run_name)
        self.assertIn("n_trials50", run_name)
        self.assertIn("sampler_seed42", run_name)
        self.assertIn("weight_min-2", run_name)
        self.assertIn("weight_max2", run_name)
        self.assertIn("klcoef0.01", run_name)
        self.assertEqual(len(result["optuna_trial_batches"]), 3)
        self.assertTrue(all(len(batch) == 2 for batch in result["optuna_trial_batches"]))
        self.assertEqual(
            result["answers_data"]["questions"],
            result["optuna_trial_batches"][1],
        )
        self.assertAlmostEqual(result["answers_data"]["optimal_objective"], 1.998)
        self.assertEqual(result["answers_data"]["optimal_harmfulness"], 2.0)
        self.assertEqual(
            result["answers_data"]["optimal_harmfulness_source"],
            "best_trial_mean_objective",
        )
        self.assertEqual(
            result["answers_data"]["optimal_objective_source"],
            "best_trial_mean_objective",
        )

    def test_main_saves_benchmark_block_and_logs_jailbreakbench_series(self):
        dataset = [
            {"instruction": "physical-question-1", "category": "Physical harm", "source": "combined"},
            {"instruction": "physical-question-2", "category": "Physical harm", "source": "combined"},
        ]
        benchmark_results = {
            "jailbreakbench": {
                "original": {"attack_success_rate": 0.2},
                "modified": {"attack_success_rate": 0.6},
                "delta_attack_success_rate": 0.4,
                "config": {"judge_mode": "project"},
                "details_file": {"original": "orig.json", "modified": "mod.json"},
                "by_category": {},
                "by_source": {},
            }
        }

        result = self._run_main(
            dataset=dataset,
            category_dataset_source="combined",
            category_filter="Physical harm",
            model_batch_size=2,
            optimizer_method="optuna",
            weights_mode="scalar",
            mmlu_enabled=False,
            benchmark_results=benchmark_results,
        )

        self.assertIn("benchmarks", result["answers_data"])
        self.assertAlmostEqual(
            result["answers_data"]["benchmarks"]["jailbreakbench"]["modified"]["attack_success_rate"],
            0.6,
        )
        self.assertEqual(result["benchmark_runner"].prepare_calls, 1)
        self.assertEqual(
            result["benchmark_runner"].run_labels,
            ["graph_grpo_Physical_harm"],
        )

        scalar_payloads = [payload for payload, _step in result["wandb_log_calls"]]
        self.assertTrue(
            any(
                "clean_model/jailbreakbench_attack_success_rate" in payload
                for payload in scalar_payloads
            )
        )
        self.assertTrue(
            any(
                "best_value_model/jailbreakbench_attack_success_rate" in payload
                for payload in scalar_payloads
            )
        )


if __name__ == "__main__":
    unittest.main()
