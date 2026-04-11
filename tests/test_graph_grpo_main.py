import importlib
import io
import json
import os
import re
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
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
        category_mode="single",
        model_batch_size=2,
        optimizer_method="optuna",
        weights_mode="scalar",
        weights_init_type="average",
        grpo_n_epochs=1,
        mmlu_enabled=False,
        train_metrics_sequence=None,
        modified_mmlu_scores=None,
        benchmark_results=None,
        academic_benchmark_results=None,
        harmful_train=None,
        harmful_val=None,
        harmful_test=None,
        reward_metric="harmfulness",
        results_root_name="results",
    ):
        few_shots_path = self.temp_path / "few-shots.json"
        few_shots_path.write_text(json.dumps({"categories": []}), encoding="utf-8")

        graph_file = self.temp_path / "graph.txt"
        graph_file.write_text("physical harm root\nprivacy root\n", encoding="utf-8")

        results_root = self.temp_path / results_root_name
        results_root.mkdir(parents=True, exist_ok=True)

        loader_calls = []
        split_loader_calls = []
        wandb_init_calls = []
        wandb_log_calls = []
        define_metric_calls = []
        train_step_questions = []
        train_step_reward_metrics = []
        optuna_trial_batches = []
        optuna_reward_metrics = []
        mmlu_eval_calls = []
        direction_compute_calls = []
        benchmark_runner = None

        if benchmark_results is not None:
            class FakeBenchmarkRunner:
                def __init__(self, results):
                    self.results = results
                    self.prepare_calls = 0
                    self.run_labels = []

                def enabled_benchmark_names(self):
                    return tuple(sorted(self.results))

                def prepare_original(self, model):
                    self.prepare_calls += 1
                    return {
                        benchmark_name: {
                            "attack_success_rate": benchmark_block["original"]["attack_success_rate"]
                        }
                        for benchmark_name, benchmark_block in self.results.items()
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
        fake_config.ACADEMIC_BENCHMARKS_CONFIG = {
            "enabled": list((academic_benchmark_results or {}).keys()),
        }
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

        fake_dataset_load_dataset = types.ModuleType("dataset.load_dataset")
        harmful_train_data = list(harmful_train or [])
        harmful_val_data = list(harmful_val or [])
        harmful_test_data = list(harmful_test or [])

        def fake_load_dataset_split(harmtype, split, instructions_only=False):
            split_loader_calls.append((harmtype, split, instructions_only))
            split_map = {
                ("harmful", "train"): harmful_train_data,
                ("harmful", "val"): harmful_val_data,
                ("harmful", "test"): harmful_test_data,
                ("harmless", "train"): [],
                ("harmless", "val"): [],
                ("harmless", "test"): [],
            }
            selected = list(split_map[(harmtype, split)])
            if instructions_only:
                return list(selected)
            return [{"instruction": item} for item in selected]

        fake_dataset_load_dataset.load_dataset_split = fake_load_dataset_split
        fake_dataset_pkg = types.ModuleType("dataset")
        fake_dataset_pkg.load_dataset = fake_dataset_load_dataset

        fake_refusal_directions = types.ModuleType("refusal_directions")
        def fake_compute_refusal_direction(model, harmful_prompts, good_prompts):
            direction_compute_calls.append(list(harmful_prompts))
            return torch.ones((2, 2), dtype=torch.float32)

        fake_refusal_directions.compute_refusal_direction = fake_compute_refusal_direction
        fake_refusal_directions.save_refusal_directions = lambda *args, **kwargs: None
        fake_refusal_directions.load_refusal_directions = lambda *args, **kwargs: ([], [])

        fake_model_utils = types.ModuleType("model_utils")
        fake_model_utils.LearnableDirectionWeights = FakeDirectionWeights
        fake_model_utils.apply_abliteration_with_hyperparams = lambda *args, **kwargs: None

        fake_reward = types.ModuleType("baselines.graph_grpo.reward")

        def fake_compute_reward(
            questions,
            responses,
            classifier_categories,
            backend,
            reward_metric="harmfulness",
        ):
            if reward_metric == "llamaguard_unsafe":
                return [1.0 if "unsafe" in question else 0.0 for question in questions]
            return [1 for _ in questions]

        fake_reward.compute_reward = fake_compute_reward

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

        fake_academic = types.ModuleType("evaluate.academic_benchmarks")

        def _raw_academic_result_from_summary(summary):
            return {
                "summary": dict(summary),
                "prediction_preview": [],
            }

        def fake_build_academic_benchmarks_result(
            *,
            config,
            original_results,
            modified_results,
            method_results_dir,
            detail_prefix,
        ):
            return academic_benchmark_results or {}

        def fake_evaluate_model_on_academic_benchmarks(model, config):
            return {
                benchmark_name: _raw_academic_result_from_summary(
                    benchmark_block["modified"]
                )
                for benchmark_name, benchmark_block in (academic_benchmark_results or {}).items()
            }

        def fake_get_cached_or_evaluate_original_academic_benchmarks(
            model,
            *,
            model_name,
            config,
        ):
            return {
                benchmark_name: _raw_academic_result_from_summary(
                    benchmark_block["original"]
                )
                for benchmark_name, benchmark_block in (academic_benchmark_results or {}).items()
            }

        def fake_get_academic_metric_names(config=None):
            names = []
            for benchmark_name in (academic_benchmark_results or {}).keys():
                if benchmark_name == "tinyhellaswag":
                    names.extend(
                        ["tinyhellaswag_irt_plus_plus", "tinyhellaswag_accuracy"]
                    )
                elif benchmark_name == "arc":
                    names.extend(
                        ["arc_easy_accuracy", "arc_challenge_accuracy", "arc_macro_accuracy"]
                    )
                elif benchmark_name == "winogrande":
                    names.append("winogrande_accuracy")
                elif benchmark_name == "gsm8k":
                    names.append("gsm8k_exact_match")
                elif benchmark_name == "truthfulqa":
                    names.extend(["truthfulqa_mc1", "truthfulqa_mc2"])
            return names

        def fake_get_academic_metric_values(results):
            values = {}
            for benchmark_name, benchmark_block in (results or {}).items():
                summary = benchmark_block.get("summary") or benchmark_block.get("modified")
                if benchmark_name == "tinyhellaswag":
                    values["tinyhellaswag_irt_plus_plus"] = summary["irt_plus_plus"]
                    values["tinyhellaswag_accuracy"] = summary["accuracy"]
                elif benchmark_name == "arc":
                    values["arc_easy_accuracy"] = summary["by_variant"]["ARC-Easy"]["accuracy"]
                    values["arc_challenge_accuracy"] = summary["by_variant"]["ARC-Challenge"]["accuracy"]
                    values["arc_macro_accuracy"] = summary["macro_accuracy"]
                elif benchmark_name == "winogrande":
                    values["winogrande_accuracy"] = summary["accuracy"]
                elif benchmark_name == "gsm8k":
                    values["gsm8k_exact_match"] = summary["exact_match"]
                elif benchmark_name == "truthfulqa":
                    values["truthfulqa_mc1"] = summary["mc1"]
                    values["truthfulqa_mc2"] = summary["mc2"]
            return values

        fake_academic.build_academic_benchmarks_result = (
            fake_build_academic_benchmarks_result
        )
        fake_academic.evaluate_model_on_academic_benchmarks = (
            fake_evaluate_model_on_academic_benchmarks
        )
        fake_academic.get_academic_metric_names = fake_get_academic_metric_names
        fake_academic.get_academic_metric_values = fake_get_academic_metric_values
        fake_academic.get_cached_or_evaluate_original_academic_benchmarks = (
            fake_get_cached_or_evaluate_original_academic_benchmarks
        )

        fake_benchmarks_integration = types.ModuleType("benchmarks.integration")
        fake_benchmarks_integration.build_benchmark_runner = (
            lambda **kwargs: benchmark_runner
        )
        fake_benchmarks_integration.get_benchmark_attack_success_rate = (
            lambda results, benchmark_name: results.get(benchmark_name, {}).get("modified", {}).get("attack_success_rate")
            if results else None
        )
        fake_benchmarks_integration.get_benchmark_attack_success_rates = (
            lambda results: {
                benchmark_name: float(benchmark_block["modified"]["attack_success_rate"])
                for benchmark_name, benchmark_block in (results or {}).items()
                if benchmark_block.get("modified", {}).get("attack_success_rate") is not None
            }
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
            harmless_questions=None,
            harmless_question_sampler=None,
            loss_agg_mode="token-mean",
            backend=None,
            reward_metric="harmfulness",
        ):
            optuna_reward_metrics.append(reward_metric)
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
                        "mean_unsafe_rate": 0.5 if reward_metric == "llamaguard_unsafe" else None,
                        "best_unsafe_rate": 1.0 if reward_metric == "llamaguard_unsafe" else None,
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
                        "mean_unsafe_rate": 0.5 if reward_metric == "llamaguard_unsafe" else None,
                        "best_unsafe_rate": 1.0 if reward_metric == "llamaguard_unsafe" else None,
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
                "best_mean_unsafe_rate": 0.5 if reward_metric == "llamaguard_unsafe" else None,
                "best_unsafe_rate": 1.0 if reward_metric == "llamaguard_unsafe" else None,
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
            train_step_reward_metrics.append(kwargs.get("reward_metric"))
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
            "dataset": fake_dataset_pkg,
            "dataset.load_dataset": fake_dataset_load_dataset,
            "data_utils": fake_data_utils,
            "refusal_directions": fake_refusal_directions,
            "model_utils": fake_model_utils,
            "baselines.graph_grpo.reward": fake_reward,
            "evaluate.mmlu": fake_mmlu,
            "evaluate.academic_benchmarks": fake_academic,
            "benchmarks.integration": fake_benchmarks_integration,
            "baselines.graph_grpo.optuna_optimizer": fake_optuna,
            "baselines.graph_grpo.trainer": fake_trainer,
        }

        env = {
            "CATEGORY_MODE": category_mode,
            "CATEGORY_DATASET_SOURCE": category_dataset_source,
            "CATEGORY_FILTER": category_filter,
            "ALL_CATEGORIES_HARMFUL_PROMPT_COUNT": "128",
            "ALL_CATEGORIES_HARMFUL_PROMPT_SEED": "42",
            "OPTIMIZER_METHOD": optimizer_method,
            "WEIGHTS_MODE": weights_mode,
            "WEIGHTS_INIT_TYPE": weights_init_type,
            "REWARD_METRIC": reward_metric,
        }

        sys.modules.pop("baselines.graph_grpo.__main__", None)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with patch.dict(sys.modules, fake_modules, clear=False):
                with patch.dict(os.environ, env, clear=False):
                    module = importlib.import_module("baselines.graph_grpo.__main__")
                    module.main()

        answers_files = sorted((self.temp_path / "graph_grpo" / "answers").glob("answers_*.json"))
        self.assertTrue(answers_files)
        answers_data = json.loads(answers_files[-1].read_text(encoding="utf-8"))

        return {
            "stdout": stdout.getvalue(),
            "answers_data": answers_data,
            "loader_calls": loader_calls,
            "split_loader_calls": split_loader_calls,
            "wandb_init_calls": wandb_init_calls,
            "wandb_log_calls": wandb_log_calls,
            "define_metric_calls": define_metric_calls,
            "train_step_questions": train_step_questions,
            "train_step_reward_metrics": train_step_reward_metrics,
            "optuna_trial_batches": optuna_trial_batches,
            "optuna_reward_metrics": optuna_reward_metrics,
            "mmlu_eval_calls": mmlu_eval_calls,
            "direction_compute_calls": direction_compute_calls,
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

    def test_wandb_run_names_share_schema_between_blocking_and_nonblocking(self):
        dataset = [
            {"instruction": "physical-question", "category": "Physical harm", "source": "combined"},
        ]

        nonblocking = self._run_main(
            dataset=dataset,
            category_dataset_source="combined",
            category_filter="Physical harm",
            model_batch_size=1,
            optimizer_method="optuna",
            weights_mode="scalar",
            mmlu_enabled=False,
            results_root_name="results",
        )
        blocking = self._run_main(
            dataset=dataset,
            category_dataset_source="combined",
            category_filter="Physical harm",
            model_batch_size=1,
            optimizer_method="optuna",
            weights_mode="scalar",
            mmlu_enabled=False,
            results_root_name="results/blocking",
        )

        nonblocking_name = nonblocking["wandb_init_calls"][0]["name"]
        blocking_name = blocking["wandb_init_calls"][0]["name"]
        self.assertTrue(nonblocking_name.startswith("nonblocking_model_test-model_"))
        self.assertTrue(blocking_name.startswith("blocking_model_test-model_"))

        def normalize_mode_and_timestamp(run_name):
            without_timestamp = re.sub(r"_[0-9]{8}_[0-9]{6}$", "", run_name)
            return re.sub(r"^(?:blocking|nonblocking)_", "", without_timestamp)

        self.assertEqual(
            normalize_mode_and_timestamp(nonblocking_name),
            normalize_mode_and_timestamp(blocking_name),
        )

    def test_main_all_mode_uses_train_val_test_splits(self):
        result = self._run_main(
            dataset=[],
            category_dataset_source="combined",
            category_filter="",
            category_mode="all",
            model_batch_size=2,
            optimizer_method="optuna",
            weights_mode="scalar",
            mmlu_enabled=False,
            harmful_train=["train-1", "train-2", "train-3"],
            harmful_val=["val-1", "val-2", "val-3", "val-4"],
            harmful_test=["test-1", "test-2"],
        )

        self.assertEqual(result["loader_calls"], [])
        self.assertEqual(
            result["split_loader_calls"],
            [
                ("harmful", "val", True),
                ("harmful", "test", True),
                ("harmful", "train", True),
            ],
        )
        self.assertEqual(
            result["direction_compute_calls"],
            [["train-1"], ["train-2"], ["train-3"]],
        )
        self.assertEqual(result["answers_data"]["experiment_config"]["category"], "all_categories")
        self.assertEqual(result["answers_data"]["experiment_config"]["category_mode"], "all")
        self.assertEqual(
            result["answers_data"]["experiment_config"]["direction_source"],
            "dataset/splits/harmful_train.json",
        )
        self.assertEqual(
            result["answers_data"]["experiment_config"]["optimization_source"],
            "dataset/splits/harmful_val.json",
        )
        self.assertEqual(
            result["answers_data"]["experiment_config"]["final_evaluation_source"],
            "dataset/splits/harmful_test.json",
        )
        self.assertEqual(result["answers_data"]["questions"], ["test-1", "test-2"])
        self.assertEqual(result["models"][0].response_calls[0], ["test-1", "test-2"])
        self.assertEqual(result["models"][0].response_calls[1], ["test-1", "test-2"])
        self.assertEqual(result["models"][0].response_calls[2], ["test-1", "test-2"])
        self.assertEqual(len(result["optuna_trial_batches"]), 3)
        self.assertTrue(all(len(batch) == 2 for batch in result["optuna_trial_batches"]))
        self.assertTrue(
            all(set(batch).issubset({"val-1", "val-2", "val-3", "val-4"}) for batch in result["optuna_trial_batches"])
        )
        self.assertNotEqual(
            result["answers_data"]["questions"],
            result["optuna_trial_batches"][1],
        )
        run_name = result["wandb_init_calls"][0]["name"]
        self.assertEqual(
            result["wandb_init_calls"][0]["project"],
            "weighted_refusal_direction",
        )
        self.assertIn("optuna_category_mode_all_all_categories", run_name)

    def test_main_rejects_topic_init_in_all_mode(self):
        with self.assertRaisesRegex(
            ValueError,
            "does not support WEIGHTS_INIT_TYPE='topic' when CATEGORY_MODE='all'",
        ):
            self._run_main(
                dataset=[],
                category_dataset_source="combined",
                category_filter="",
                category_mode="all",
                model_batch_size=2,
                optimizer_method="grpo",
                weights_mode="scalar",
                weights_init_type="topic",
                mmlu_enabled=False,
                harmful_train=["train-1"],
                harmful_val=["val-1", "val-2"],
                harmful_test=["test-1"],
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
        self.assertEqual(
            result["wandb_init_calls"][0]["project"],
            "weighted_refusal_direction",
        )
        self.assertIn("grpo_category_mode_single_Physical_harm", run_name)
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

    def test_main_threads_llamaguard_unsafe_reward_metric_through_grpo(self):
        dataset = [
            {"instruction": "unsafe-question", "category": "Physical harm", "source": "combined"},
            {"instruction": "safe-question", "category": "Physical harm", "source": "combined"},
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
            reward_metric="llamaguard_unsafe",
            train_metrics_sequence=[
                {
                    "mean_reward": 0.5,
                    "best_reward": 1.0,
                    "mean_harmfulness": 0.5,
                    "mean_unsafe_rate": 0.5,
                    "best_unsafe_rate": 1.0,
                }
            ],
        )

        self.assertEqual(result["train_step_reward_metrics"], ["llamaguard_unsafe"])
        self.assertEqual(
            result["answers_data"]["experiment_config"]["reward_metric"],
            "llamaguard_unsafe",
        )
        self.assertEqual(
            result["answers_data"]["experiment_config"]["best_train_batch_mean_unsafe_rate"],
            0.5,
        )
        self.assertEqual(
            result["answers_data"]["experiment_config"]["best_train_batch_best_unsafe_rate"],
            1.0,
        )
        self.assertEqual(result["answers_data"]["final_mean_unsafe_rate"], 0.5)

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
        self.assertIn(
            "Skipping scalar weight distribution plots for weights_mode='dense'",
            result["stdout"],
        )
        plot_files = sorted((self.temp_path / "graph_grpo").glob("scalar_weights_distribution_*.pdf"))
        self.assertEqual(plot_files, [])

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
        self.assertEqual(
            result["wandb_init_calls"][0]["project"],
            "weighted_refusal_direction",
        )
        self.assertIn("optuna_category_mode_single_Physical_harm", run_name)
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
        self.assertIn("Evaluating initialized weights on full evaluation split", result["stdout"])
        self.assertIn("Initialized weights harmfulness: 1.000", result["stdout"])
        scalar_payloads = [payload for payload, _step in result["wandb_log_calls"]]
        self.assertFalse(
            any(
                "initialized" in metric_name.lower()
                for payload in scalar_payloads
                for metric_name in payload
            )
        )
        plot_files = sorted((self.temp_path / "graph_grpo").glob("scalar_weights_distribution_*.pdf"))
        self.assertEqual(len(plot_files), 2)
        self.assertTrue(any("scalar_weights_distribution_initial" in path.name for path in plot_files))
        self.assertTrue(any("scalar_weights_distribution_final" in path.name for path in plot_files))
        self.assertTrue(all(path.suffix == ".pdf" for path in plot_files))
        self.assertTrue(all(path.stat().st_size > 0 for path in plot_files))

    def test_main_threads_llamaguard_unsafe_reward_metric_through_optuna(self):
        dataset = [
            {"instruction": "unsafe-question", "category": "Physical harm", "source": "combined"},
            {"instruction": "safe-question", "category": "Physical harm", "source": "combined"},
        ]

        result = self._run_main(
            dataset=dataset,
            category_dataset_source="combined",
            category_filter="Physical harm",
            model_batch_size=2,
            optimizer_method="optuna",
            weights_mode="scalar",
            mmlu_enabled=False,
            reward_metric="llamaguard_unsafe",
        )

        self.assertEqual(result["optuna_reward_metrics"], ["llamaguard_unsafe"])
        self.assertEqual(
            result["answers_data"]["experiment_config"]["reward_metric"],
            "llamaguard_unsafe",
        )
        self.assertEqual(
            result["wandb_init_calls"][0]["config"]["reward_metric"],
            "llamaguard_unsafe",
        )
        self.assertIn("reward_llamaguard_unsafe", result["wandb_init_calls"][0]["name"])
        self.assertEqual(result["answers_data"]["final_scores"], [1, 0])
        self.assertEqual(result["answers_data"]["final_mean_unsafe_rate"], 0.5)
        self.assertEqual(result["answers_data"]["final_best_unsafe_rate"], 1.0)
        self.assertEqual(result["answers_data"]["score_statistics"]["mean_unsafe_rate"], 0.5)
        self.assertEqual(result["answers_data"]["score_statistics"]["best_unsafe_rate"], 1.0)
        self.assertEqual(
            result["answers_data"]["experiment_config"]["best_trial_mean_unsafe_rate"],
            0.5,
        )
        self.assertEqual(
            result["answers_data"]["optimization_history"][0]["mean_unsafe_rate"],
            0.5,
        )

    def test_main_saves_benchmark_blocks_and_logs_generic_series(self):
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
            "malicious_instruct": {
                "original": {"attack_success_rate": 0.15},
                "modified": {"attack_success_rate": 0.45},
                "delta_attack_success_rate": 0.3,
                "config": {"judge_mode": "official"},
                "details_file": {"original": "mi_orig.json", "modified": "mi_mod.json"},
                "by_category": {},
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
        self.assertAlmostEqual(
            result["answers_data"]["benchmarks"]["harmbench"]["modified"]["attack_success_rate"],
            0.4,
        )
        self.assertAlmostEqual(
            result["answers_data"]["benchmarks"]["malicious_instruct"]["modified"]["attack_success_rate"],
            0.45,
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
        self.assertTrue(
            any(
                "clean_model/harmbench_attack_success_rate" in payload
                for payload in scalar_payloads
            )
        )
        self.assertTrue(
            any(
                "best_value_model/harmbench_attack_success_rate" in payload
                for payload in scalar_payloads
            )
        )
        self.assertTrue(
            any(
                "clean_model/malicious_instruct_attack_success_rate" in payload
                for payload in scalar_payloads
            )
        )
        self.assertTrue(
            any(
                "best_value_model/malicious_instruct_attack_success_rate" in payload
                for payload in scalar_payloads
            )
        )

    def test_main_saves_academic_benchmark_blocks_and_logs_series(self):
        dataset = [
            {"instruction": "physical-question-1", "category": "Physical harm", "source": "combined"},
            {"instruction": "physical-question-2", "category": "Physical harm", "source": "combined"},
        ]
        academic_benchmark_results = {
            "tinyhellaswag": {
                "original": {
                    "primary_metric_name": "irt_plus_plus",
                    "primary_metric_value": 0.31,
                    "irt_plus_plus": 0.31,
                    "accuracy": 0.44,
                },
                "modified": {
                    "primary_metric_name": "irt_plus_plus",
                    "primary_metric_value": 0.48,
                    "irt_plus_plus": 0.48,
                    "accuracy": 0.55,
                },
                "config": {"sample_size": 100},
                "details_file": "academic/tinyhellaswag.json",
                "delta_primary_metric": 0.17,
            },
            "arc": {
                "original": {
                    "primary_metric_name": "macro_accuracy",
                    "primary_metric_value": 0.5,
                    "macro_accuracy": 0.5,
                    "by_variant": {
                        "ARC-Easy": {"accuracy": 0.6},
                        "ARC-Challenge": {"accuracy": 0.4},
                    },
                },
                "modified": {
                    "primary_metric_name": "macro_accuracy",
                    "primary_metric_value": 0.65,
                    "macro_accuracy": 0.65,
                    "by_variant": {
                        "ARC-Easy": {"accuracy": 0.7},
                        "ARC-Challenge": {"accuracy": 0.6},
                    },
                },
                "config": {"sample_size": 100},
                "details_file": "academic/arc.json",
                "delta_primary_metric": 0.15,
            },
            "winogrande": {
                "original": {
                    "primary_metric_name": "accuracy",
                    "primary_metric_value": 0.52,
                    "accuracy": 0.52,
                },
                "modified": {
                    "primary_metric_name": "accuracy",
                    "primary_metric_value": 0.61,
                    "accuracy": 0.61,
                },
                "config": {"sample_size": 100},
                "details_file": "academic/winogrande.json",
                "delta_primary_metric": 0.09,
            },
            "gsm8k": {
                "original": {
                    "primary_metric_name": "exact_match",
                    "primary_metric_value": 0.18,
                    "exact_match": 0.18,
                },
                "modified": {
                    "primary_metric_name": "exact_match",
                    "primary_metric_value": 0.27,
                    "exact_match": 0.27,
                },
                "config": {"sample_size": 100},
                "details_file": "academic/gsm8k.json",
                "delta_primary_metric": 0.09,
            },
            "truthfulqa": {
                "original": {
                    "primary_metric_name": "mc1",
                    "primary_metric_value": 0.36,
                    "mc1": 0.36,
                    "mc2": 0.41,
                },
                "modified": {
                    "primary_metric_name": "mc1",
                    "primary_metric_value": 0.43,
                    "mc1": 0.43,
                    "mc2": 0.49,
                },
                "config": {"sample_size": 100},
                "details_file": "academic/truthfulqa.json",
                "delta_primary_metric": 0.07,
            },
        }

        result = self._run_main(
            dataset=dataset,
            category_dataset_source="combined",
            category_filter="Physical harm",
            model_batch_size=2,
            optimizer_method="optuna",
            weights_mode="scalar",
            mmlu_enabled=False,
            academic_benchmark_results=academic_benchmark_results,
        )

        self.assertIn("academic_benchmarks", result["answers_data"])
        self.assertAlmostEqual(
            result["answers_data"]["academic_benchmarks"]["tinyhellaswag"]["modified"]["irt_plus_plus"],
            0.48,
        )
        self.assertAlmostEqual(
            result["answers_data"]["academic_benchmarks"]["arc"]["modified"]["macro_accuracy"],
            0.65,
        )
        self.assertAlmostEqual(
            result["answers_data"]["academic_benchmarks"]["truthfulqa"]["modified"]["mc1"],
            0.43,
        )

        scalar_payloads = [payload for payload, _step in result["wandb_log_calls"]]
        self.assertTrue(any("clean_model/tinyhellaswag_irt_plus_plus" in payload for payload in scalar_payloads))
        self.assertTrue(any("best_value_model/tinyhellaswag_irt_plus_plus" in payload for payload in scalar_payloads))
        self.assertTrue(any("clean_model/arc_easy_accuracy" in payload for payload in scalar_payloads))
        self.assertTrue(any("best_value_model/arc_challenge_accuracy" in payload for payload in scalar_payloads))
        self.assertTrue(any("clean_model/winogrande_accuracy" in payload for payload in scalar_payloads))
        self.assertTrue(any("best_value_model/gsm8k_exact_match" in payload for payload in scalar_payloads))
        self.assertTrue(any("clean_model/truthfulqa_mc1" in payload for payload in scalar_payloads))
        self.assertTrue(any("best_value_model/truthfulqa_mc2" in payload for payload in scalar_payloads))


if __name__ == "__main__":
    unittest.main()
