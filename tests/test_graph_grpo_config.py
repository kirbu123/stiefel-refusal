import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


fake_ui = types.ModuleType("cli.ui")
fake_ui.print = print
sys.modules.setdefault("cli.ui", fake_ui)

from baselines.graph_grpo.runtime_config import (
    resolve_graph_grpo_all_categories_harmful_prompt_count,
    resolve_graph_grpo_all_categories_harmful_prompt_seed,
    resolve_graph_grpo_category_mode,
    resolve_graph_grpo_category_dataset_source,
    resolve_graph_grpo_category_filter,
    resolve_graph_grpo_debug_noise_scale,
    resolve_graph_grpo_debug_question_count,
    resolve_graph_grpo_optimizer_method,
    resolve_graph_grpo_optuna_n_trials,
    resolve_graph_grpo_optuna_sampler,
    resolve_graph_grpo_optuna_sampler_seed,
    resolve_graph_grpo_optuna_weight_max,
    resolve_graph_grpo_optuna_weight_min,
    resolve_graph_grpo_weights_mode,
    resolve_graph_grpo_weights_init_type,
    validate_graph_grpo_category_mode,
    validate_graph_grpo_category_dataset_source,
    validate_graph_grpo_category_filter,
    validate_graph_grpo_optimizer_compatibility,
    validate_graph_grpo_optimizer_method,
    validate_graph_grpo_optuna_sampler,
    validate_graph_grpo_optuna_weight_range,
    validate_graph_grpo_weights_mode,
    validate_graph_grpo_weights_init_type,
)
from cli.config_loader import apply_config_to_env, load_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestGraphGrpoConfig(unittest.TestCase):
    def test_graph_grpo_config_uses_w_sampling_knobs(self):
        config = load_config("graph_grpo")
        model = config["model"]
        data = config["data"]
        grpo = config["grpo"]
        weights = config["weights"]
        optimizer = config["optimizer"]
        optuna = config["optuna"]
        academic_benchmarks = config["academic_benchmarks"]
        tinyhellaswag = academic_benchmarks["tinyhellaswag"]
        arc = academic_benchmarks["arc"]
        winogrande = academic_benchmarks["winogrande"]
        gsm8k = academic_benchmarks["gsm8k"]
        truthfulqa = academic_benchmarks["truthfulqa"]
        benchmarks = config["benchmarks"]
        jailbreakbench = benchmarks["jailbreakbench"]
        jbb_official_judge = jailbreakbench["official_judge"]
        harmbench = benchmarks["harmbench"]
        official_classifier = harmbench["official_classifier"]
        malicious_instruct = benchmarks["malicious_instruct"]
        mi_official_evaluator = malicious_instruct["official_evaluator"]

        self.assertEqual(model["batch_size"], 32)
        self.assertEqual(data["category_mode"], "single")
        self.assertEqual(data["category_dataset_source"], "combined")
        self.assertEqual(data["category_filter"], "Physical harm")
        self.assertEqual(data["all_categories_harmful_prompt_count"], 128)
        self.assertEqual(data["all_categories_harmful_prompt_seed"], 42)
        self.assertEqual(grpo["n_groups"], 4)
        self.assertEqual(grpo["noise_scale"], 0.1)
        self.assertEqual(grpo["kl_loss_coef"], 0.01)
        self.assertNotIn("alphas", grpo)
        self.assertEqual(optimizer["method"], "grpo")
        self.assertEqual(optuna["sampler"], "tpe")
        self.assertEqual(optuna["n_trials"], 50)
        self.assertEqual(optuna["weight_min"], -2.0)
        self.assertEqual(optuna["weight_max"], 2.0)
        self.assertEqual(
            academic_benchmarks["enabled"],
            ["tinyhellaswag", "arc", "winogrande", "gsm8k", "truthfulqa"],
        )
        self.assertEqual(academic_benchmarks["sample_seed"], 42)
        self.assertTrue(academic_benchmarks["store_predictions"])
        self.assertEqual(tinyhellaswag["dataset"], "tinyBenchmarks/tinyHellaswag")
        self.assertEqual(tinyhellaswag["split"], "validation")
        self.assertEqual(tinyhellaswag["sample_size"], 100)
        self.assertEqual(arc["dataset"], "allenai/ai2_arc")
        self.assertEqual(arc["split"], "validation")
        self.assertEqual(arc["sample_size"], 100)
        self.assertEqual(winogrande["dataset"], "allenai/winogrande")
        self.assertEqual(winogrande["subset"], "winogrande_xl")
        self.assertEqual(winogrande["split"], "validation")
        self.assertEqual(winogrande["sample_size"], 100)
        self.assertEqual(gsm8k["dataset"], "openai/gsm8k")
        self.assertEqual(gsm8k["subset"], "main")
        self.assertEqual(gsm8k["split"], "test")
        self.assertEqual(gsm8k["sample_size"], 100)
        self.assertEqual(gsm8k["max_new_tokens"], 512)
        self.assertEqual(truthfulqa["dataset"], "truthfulqa/truthful_qa")
        self.assertEqual(truthfulqa["subset"], "multiple_choice")
        self.assertEqual(truthfulqa["split"], "validation")
        self.assertEqual(truthfulqa["sample_size"], 100)
        self.assertEqual(weights["mode"], "scalar")
        self.assertEqual(weights["init_type"], "average")
        self.assertEqual(benchmarks["enabled"], [])
        self.assertEqual(jailbreakbench["judge_mode"], "official")
        self.assertEqual(jailbreakbench["max_samples"], 100)
        self.assertEqual(jailbreakbench["categories"], [])
        self.assertEqual(jailbreakbench["harmful_score_threshold"], 3)
        self.assertEqual(
            jbb_official_judge["model_name"],
            "together_ai/meta-llama/Llama-3-70b-chat-hf",
        )
        self.assertEqual(
            jbb_official_judge["api_url"],
            "https://api.together.xyz/v1/chat/completions",
        )
        self.assertEqual(jbb_official_judge["api_key_env"], "TOGETHER_API_KEY")
        self.assertEqual(jbb_official_judge["batch_size"], 8)
        self.assertEqual(jbb_official_judge["timeout_sec"], 60)
        self.assertEqual(harmbench["judge_mode"], "official")
        self.assertEqual(harmbench["split"], "test")
        self.assertEqual(harmbench["max_samples"], 0)
        self.assertEqual(harmbench["categories"], [])
        self.assertEqual(harmbench["functional_categories"], [])
        self.assertEqual(harmbench["semantic_categories"], [])
        self.assertEqual(harmbench["harmful_score_threshold"], 3)
        self.assertEqual(official_classifier["model_name"], "cais/HarmBench-Llama-2-13b-cls")
        self.assertEqual(official_classifier["batch_size"], 8)
        self.assertEqual(official_classifier["device"], "cuda")
        self.assertEqual(official_classifier["dtype"], "bfloat16")
        self.assertEqual(malicious_instruct["judge_mode"], "official")
        self.assertEqual(malicious_instruct["max_samples"], 0)
        self.assertEqual(malicious_instruct["categories"], [])
        self.assertEqual(malicious_instruct["harmful_score_threshold"], 3)
        self.assertEqual(
            mi_official_evaluator["model_name"],
            "LibrAI/maliciousinstruct-evaluator",
        )
        self.assertEqual(mi_official_evaluator["batch_size"], 32)
        self.assertEqual(mi_official_evaluator["device"], "cuda")

    def test_apply_config_to_env_sets_new_grpo_env_vars_without_alphas(self):
        config = load_config("graph_grpo")

        with patch.dict(os.environ, {}, clear=True):
            apply_config_to_env(config)

            self.assertEqual(os.environ["BATCH_SIZE"], "32")
            self.assertEqual(os.environ["CATEGORY_MODE"], "single")
            self.assertEqual(os.environ["CATEGORY_DATASET_SOURCE"], "combined")
            self.assertEqual(os.environ["CATEGORY_FILTER"], "Physical harm")
            self.assertEqual(os.environ["ALL_CATEGORIES_HARMFUL_PROMPT_COUNT"], "128")
            self.assertEqual(os.environ["ALL_CATEGORIES_HARMFUL_PROMPT_SEED"], "42")
            self.assertEqual(os.environ["GRPO_N_GROUPS"], "4")
            self.assertEqual(os.environ["GRPO_NOISE_SCALE"], "0.1")
            self.assertEqual(os.environ["GRPO_REF_ALPHA"], "1.0")
            self.assertEqual(os.environ["GRPO_KL_LOSS_COEF"], "0.01")
            self.assertEqual(os.environ["WEIGHTS_MODE"], "scalar")
            self.assertEqual(os.environ["WEIGHTS_INIT_TYPE"], "average")
            self.assertEqual(os.environ["OPTIMIZER_METHOD"], "grpo")
            self.assertEqual(os.environ["OPTUNA_SAMPLER"], "tpe")
            self.assertEqual(os.environ["OPTUNA_N_TRIALS"], "50")
            self.assertEqual(os.environ["OPTUNA_SAMPLER_SEED"], "42")
            self.assertEqual(os.environ["OPTUNA_WEIGHT_MIN"], "-2.0")
            self.assertEqual(os.environ["OPTUNA_WEIGHT_MAX"], "2.0")
            self.assertEqual(
                os.environ["ACADEMIC_BENCHMARKS_ENABLED"],
                "tinyhellaswag,arc,winogrande,gsm8k,truthfulqa",
            )
            self.assertEqual(os.environ["ACADEMIC_BENCHMARKS_SAMPLE_SEED"], "42")
            self.assertEqual(os.environ["ACADEMIC_BENCHMARKS_STORE_PREDICTIONS"], "true")
            self.assertEqual(
                os.environ["TINYHELLASWAG_DATASET"],
                "tinyBenchmarks/tinyHellaswag",
            )
            self.assertEqual(os.environ["TINYHELLASWAG_SPLIT"], "validation")
            self.assertEqual(os.environ["TINYHELLASWAG_SAMPLE_SIZE"], "100")
            self.assertEqual(os.environ["ARC_DATASET"], "allenai/ai2_arc")
            self.assertEqual(os.environ["ARC_SPLIT"], "validation")
            self.assertEqual(os.environ["ARC_SAMPLE_SIZE"], "100")
            self.assertEqual(os.environ["WINOGRANDE_DATASET"], "allenai/winogrande")
            self.assertEqual(os.environ["WINOGRANDE_SUBSET"], "winogrande_xl")
            self.assertEqual(os.environ["WINOGRANDE_SPLIT"], "validation")
            self.assertEqual(os.environ["WINOGRANDE_SAMPLE_SIZE"], "100")
            self.assertEqual(os.environ["GSM8K_DATASET"], "openai/gsm8k")
            self.assertEqual(os.environ["GSM8K_SUBSET"], "main")
            self.assertEqual(os.environ["GSM8K_SPLIT"], "test")
            self.assertEqual(os.environ["GSM8K_SAMPLE_SIZE"], "100")
            self.assertEqual(os.environ["GSM8K_MAX_NEW_TOKENS"], "512")
            self.assertEqual(
                os.environ["TRUTHFULQA_DATASET"],
                "truthfulqa/truthful_qa",
            )
            self.assertEqual(os.environ["TRUTHFULQA_SUBSET"], "multiple_choice")
            self.assertEqual(os.environ["TRUTHFULQA_SPLIT"], "validation")
            self.assertEqual(os.environ["TRUTHFULQA_SAMPLE_SIZE"], "100")
            self.assertEqual(os.environ["BENCHMARKS_ENABLED"], "")
            self.assertEqual(os.environ["JAILBREAKBENCH_JUDGE_MODE"], "official")
            self.assertEqual(os.environ["JAILBREAKBENCH_MAX_SAMPLES"], "100")
            self.assertEqual(os.environ["JAILBREAKBENCH_CATEGORIES"], "")
            self.assertEqual(os.environ["JAILBREAKBENCH_HARMFUL_SCORE_THRESHOLD"], "3")
            self.assertEqual(
                os.environ["JAILBREAKBENCH_OFFICIAL_JUDGE_MODEL_NAME"],
                "together_ai/meta-llama/Llama-3-70b-chat-hf",
            )
            self.assertEqual(
                os.environ["JAILBREAKBENCH_OFFICIAL_JUDGE_API_URL"],
                "https://api.together.xyz/v1/chat/completions",
            )
            self.assertEqual(
                os.environ["JAILBREAKBENCH_OFFICIAL_JUDGE_API_KEY_ENV"],
                "TOGETHER_API_KEY",
            )
            self.assertEqual(
                os.environ["JAILBREAKBENCH_OFFICIAL_JUDGE_BATCH_SIZE"],
                "8",
            )
            self.assertEqual(
                os.environ["JAILBREAKBENCH_OFFICIAL_JUDGE_TIMEOUT_SEC"],
                "60",
            )
            self.assertEqual(os.environ["HARMBENCH_JUDGE_MODE"], "official")
            self.assertEqual(os.environ["HARMBENCH_SPLIT"], "test")
            self.assertEqual(os.environ["HARMBENCH_MAX_SAMPLES"], "0")
            self.assertEqual(os.environ["HARMBENCH_CATEGORIES"], "")
            self.assertEqual(os.environ["HARMBENCH_FUNCTIONAL_CATEGORIES"], "")
            self.assertEqual(os.environ["HARMBENCH_SEMANTIC_CATEGORIES"], "")
            self.assertEqual(os.environ["HARMBENCH_HARMFUL_SCORE_THRESHOLD"], "3")
            self.assertEqual(os.environ["HARMBENCH_OFFICIAL_CLASSIFIER_MODEL_NAME"], "cais/HarmBench-Llama-2-13b-cls")
            self.assertEqual(os.environ["HARMBENCH_OFFICIAL_CLASSIFIER_BATCH_SIZE"], "8")
            self.assertEqual(os.environ["HARMBENCH_OFFICIAL_CLASSIFIER_DEVICE"], "cuda")
            self.assertEqual(os.environ["HARMBENCH_OFFICIAL_CLASSIFIER_DTYPE"], "bfloat16")
            self.assertEqual(os.environ["MALICIOUS_INSTRUCT_JUDGE_MODE"], "official")
            self.assertEqual(os.environ["MALICIOUS_INSTRUCT_MAX_SAMPLES"], "0")
            self.assertEqual(os.environ["MALICIOUS_INSTRUCT_CATEGORIES"], "")
            self.assertEqual(
                os.environ["MALICIOUS_INSTRUCT_HARMFUL_SCORE_THRESHOLD"],
                "3",
            )
            self.assertEqual(
                os.environ["MALICIOUS_INSTRUCT_OFFICIAL_EVALUATOR_MODEL_NAME"],
                "LibrAI/maliciousinstruct-evaluator",
            )
            self.assertEqual(
                os.environ["MALICIOUS_INSTRUCT_OFFICIAL_EVALUATOR_BATCH_SIZE"],
                "32",
            )
            self.assertEqual(
                os.environ["MALICIOUS_INSTRUCT_OFFICIAL_EVALUATOR_DEVICE"],
                "cuda",
            )
            self.assertNotIn("GRPO_ALPHAS", os.environ)

    def test_run_graph_grpo_script_uses_average_init_and_debug_false(self):
        script_text = (PROJECT_ROOT / "scripts" / "run_graph_grpo.sh").read_text(encoding="utf-8")

        self.assertIn("GRPO_N_GROUPS", script_text)
        self.assertIn("GRPO_NOISE_SCALE", script_text)
        self.assertIn("GRPO_KL_LOSS_COEF=0.01", script_text)
        self.assertIn("BATCH_SIZE=32", script_text)
        self.assertIn("DEBUG=false", script_text)
        self.assertIn("DEBUG_N_QUESTIONS=4", script_text)
        self.assertIn("DEBUG_NOISE_SCALE=0.02", script_text)
        self.assertIn("CATEGORY_MODE", script_text)
        self.assertIn("CATEGORY_DATASET_SOURCE", script_text)
        self.assertIn("CATEGORY_FILTER", script_text)
        self.assertIn("ALL_CATEGORIES_HARMFUL_PROMPT_COUNT", script_text)
        self.assertIn("ALL_CATEGORIES_HARMFUL_PROMPT_SEED", script_text)
        self.assertIn('WEIGHTS_MODE="scalar"', script_text)
        self.assertIn('WEIGHTS_INIT_TYPE="average"', script_text)
        self.assertIn("OPTIMIZER_METHOD=", script_text)
        self.assertIn('OPTIMIZER_METHOD="grpo"', script_text)
        self.assertIn("OPTUNA_SAMPLER=", script_text)
        self.assertIn("OPTUNA_N_TRIALS=50", script_text)
        self.assertIn("OPTUNA_SAMPLER_SEED=42", script_text)
        self.assertIn("OPTUNA_WEIGHT_MIN=-2.0", script_text)
        self.assertIn("OPTUNA_WEIGHT_MAX=2.0", script_text)
        self.assertIn("ACADEMIC_BENCHMARKS_ENABLED", script_text)
        self.assertIn("TINYHELLASWAG_SAMPLE_SIZE", script_text)
        self.assertIn("ARC_SAMPLE_SIZE", script_text)
        self.assertIn("WINOGRANDE_SAMPLE_SIZE", script_text)
        self.assertIn("GSM8K_SAMPLE_SIZE", script_text)
        self.assertIn("GSM8K_MAX_NEW_TOKENS", script_text)
        self.assertIn("TRUTHFULQA_SAMPLE_SIZE", script_text)
        self.assertNotIn("GRPO_ALPHAS", script_text)

    def test_run_graph_optuna_script_uses_optuna_optimizer(self):
        script_text = (PROJECT_ROOT / "scripts" / "run_graph_optuna.sh").read_text(encoding="utf-8")

        self.assertIn('OPTIMIZER_METHOD="optuna"', script_text)

    def test_graph_grpo_runtime_defaults_optimizer_method_to_grpo(self):
        self.assertEqual(resolve_graph_grpo_category_mode(None), "single")
        self.assertEqual(resolve_graph_grpo_category_mode(""), "single")
        self.assertEqual(resolve_graph_grpo_category_mode("all"), "all")
        self.assertEqual(resolve_graph_grpo_category_dataset_source(None), "combined")
        self.assertEqual(resolve_graph_grpo_category_dataset_source(""), "combined")
        self.assertEqual(resolve_graph_grpo_category_dataset_source("jailbreakbench"), "jailbreakbench")
        self.assertEqual(resolve_graph_grpo_category_filter(None), "Physical harm")
        self.assertEqual(resolve_graph_grpo_category_filter(""), "Physical harm")
        self.assertEqual(resolve_graph_grpo_category_filter("Privacy"), "Privacy")
        self.assertEqual(resolve_graph_grpo_all_categories_harmful_prompt_count(None), 128)
        self.assertEqual(resolve_graph_grpo_all_categories_harmful_prompt_seed(None), 42)
        self.assertEqual(resolve_graph_grpo_optimizer_method(None), "grpo")
        self.assertEqual(resolve_graph_grpo_optimizer_method(""), "grpo")
        self.assertEqual(resolve_graph_grpo_optimizer_method("optuna"), "optuna")
        self.assertEqual(resolve_graph_grpo_optuna_sampler(None), "tpe")
        self.assertEqual(resolve_graph_grpo_optuna_sampler(""), "tpe")
        self.assertEqual(resolve_graph_grpo_optuna_sampler("random"), "random")

    def test_graph_grpo_runtime_validates_optimizer_method_and_compatibility(self):
        validate_graph_grpo_category_mode("single")
        validate_graph_grpo_category_mode("all")
        validate_graph_grpo_category_dataset_source("combined")
        validate_graph_grpo_category_dataset_source("jailbreakbench")
        validate_graph_grpo_category_filter("Physical harm")
        validate_graph_grpo_category_filter("Privacy")
        validate_graph_grpo_category_filter("Physical harm", category_mode="all")
        validate_graph_grpo_optimizer_method("grpo")
        validate_graph_grpo_optimizer_method("optuna")
        validate_graph_grpo_optimizer_compatibility("grpo", "dense")
        validate_graph_grpo_optimizer_compatibility("optuna", "scalar")
        validate_graph_grpo_optimizer_compatibility("optuna", "dense")
        validate_graph_grpo_optuna_sampler("tpe")
        validate_graph_grpo_optuna_sampler("random")
        validate_graph_grpo_optuna_sampler("gp")
        validate_graph_grpo_optuna_sampler("cmaes")
        validate_graph_grpo_optuna_sampler("qmc")

        with self.assertRaisesRegex(ValueError, "graph_grpo only supports OPTIMIZER_METHOD"):
            validate_graph_grpo_optimizer_method("random-search")

        with self.assertRaisesRegex(ValueError, "graph_grpo only supports CATEGORY_MODE"):
            validate_graph_grpo_category_mode("dataset")

        with self.assertRaisesRegex(ValueError, "graph_grpo only supports CATEGORY_DATASET_SOURCE"):
            validate_graph_grpo_category_dataset_source("harmbench_semantic")

        with self.assertRaisesRegex(ValueError, "graph_grpo only supports CATEGORY_FILTER"):
            validate_graph_grpo_category_filter("unknown_category")

        with self.assertRaisesRegex(ValueError, "graph_grpo only supports OPTUNA_SAMPLER"):
            validate_graph_grpo_optuna_sampler("nsga2")

    def test_graph_grpo_runtime_defaults_weights_mode_to_scalar(self):
        self.assertEqual(resolve_graph_grpo_weights_mode(None), "scalar")
        self.assertEqual(resolve_graph_grpo_weights_mode(""), "scalar")
        self.assertEqual(resolve_graph_grpo_weights_mode("dense"), "dense")

    def test_graph_grpo_runtime_validates_weights_mode(self):
        validate_graph_grpo_weights_mode("scalar")
        validate_graph_grpo_weights_mode("dense")

        with self.assertRaisesRegex(
            ValueError,
            "graph_grpo only supports WEIGHTS_MODE",
        ):
            validate_graph_grpo_weights_mode("weird")

    def test_graph_grpo_runtime_defaults_weights_init_to_average(self):
        self.assertEqual(resolve_graph_grpo_weights_init_type(None), "average")
        self.assertEqual(resolve_graph_grpo_weights_init_type(""), "average")
        self.assertEqual(resolve_graph_grpo_weights_init_type("topic"), "topic")

    def test_graph_grpo_runtime_rejects_zero_init(self):
        with self.assertRaisesRegex(
            ValueError,
            "does not support WEIGHTS_INIT_TYPE='zero'",
        ):
            validate_graph_grpo_weights_init_type("zero")

        validate_graph_grpo_weights_init_type("average")
        validate_graph_grpo_weights_init_type("topic")

        with self.assertRaisesRegex(
            ValueError,
            "does not support WEIGHTS_INIT_TYPE='topic' when CATEGORY_MODE='all'",
        ):
            validate_graph_grpo_weights_init_type("topic", category_mode="all")

    def test_graph_grpo_runtime_debug_defaults(self):
        self.assertEqual(resolve_graph_grpo_debug_question_count(None), 4)
        self.assertEqual(resolve_graph_grpo_debug_question_count("8"), 8)
        self.assertEqual(resolve_graph_grpo_debug_noise_scale(0.1, None), 0.02)
        self.assertEqual(resolve_graph_grpo_debug_noise_scale(0.01, None), 0.01)
        self.assertEqual(resolve_graph_grpo_debug_noise_scale(0.1, "0.005"), 0.005)
        self.assertEqual(resolve_graph_grpo_all_categories_harmful_prompt_count("16"), 16)
        self.assertEqual(resolve_graph_grpo_all_categories_harmful_prompt_seed("7"), 7)
        self.assertEqual(resolve_graph_grpo_optuna_n_trials(None), 50)
        self.assertEqual(resolve_graph_grpo_optuna_sampler_seed(None), 42)
        self.assertEqual(resolve_graph_grpo_optuna_weight_min(None), -2.0)
        self.assertEqual(resolve_graph_grpo_optuna_weight_max(None), 2.0)
        validate_graph_grpo_optuna_weight_range(-2.0, 2.0)

        with self.assertRaisesRegex(ValueError, "DEBUG_N_QUESTIONS must be >= 1"):
            resolve_graph_grpo_debug_question_count("0")

        with self.assertRaisesRegex(ValueError, "DEBUG_NOISE_SCALE must be >= 0"):
            resolve_graph_grpo_debug_noise_scale(0.1, "-1")

        with self.assertRaisesRegex(ValueError, "OPTUNA_N_TRIALS must be >= 1"):
            resolve_graph_grpo_optuna_n_trials("0")

        with self.assertRaisesRegex(ValueError, "ALL_CATEGORIES_HARMFUL_PROMPT_COUNT must be >= 1"):
            resolve_graph_grpo_all_categories_harmful_prompt_count("0")

        with self.assertRaisesRegex(ValueError, "OPTUNA weight range must satisfy"):
            validate_graph_grpo_optuna_weight_range(2.0, -2.0)

    def test_graph_grpo_debug_mode_uses_configurable_question_subset(self):
        main_text = (PROJECT_ROOT / "baselines" / "graph_grpo" / "__main__.py").read_text(encoding="utf-8")
        trainer_text = (PROJECT_ROOT / "baselines" / "graph_grpo" / "trainer.py").read_text(encoding="utf-8")

        self.assertIn("model=MODEL_NAME, batch_size=MODEL_BATCH_SIZE", main_text)
        self.assertIn("def _sample_training_questions(", main_text)
        self.assertIn("train_question_count = min(MODEL_BATCH_SIZE, len(all_category_questions))", main_text)
        self.assertIn("epoch_questions = _sample_training_questions(", main_text)
        self.assertIn("if DEBUG:", main_text)
        self.assertIn("effective_noise_scale = debug_noise_scale if DEBUG else GRPO_CONFIG[\"noise_scale\"]", main_text)
        self.assertIn("resolve_graph_grpo_debug_question_count(os.getenv(\"DEBUG_N_QUESTIONS\"))", main_text)
        self.assertIn("resolve_graph_grpo_debug_noise_scale(", main_text)
        self.assertIn("resolve_graph_grpo_category_mode(os.getenv(\"CATEGORY_MODE\"))", main_text)
        self.assertIn("resolve_graph_grpo_category_dataset_source(", main_text)
        self.assertIn("resolve_graph_grpo_category_filter(", main_text)
        self.assertIn("resolve_graph_grpo_optimizer_method(os.getenv(\"OPTIMIZER_METHOD\"))", main_text)
        self.assertIn("ALL_CATEGORIES_HARMFUL_PROMPT_COUNT", main_text)
        self.assertIn("ALL_CATEGORIES_HARMFUL_PROMPT_SEED", main_text)
        self.assertIn("harmful_train", main_text)
        self.assertIn("harmful_val", main_text)
        self.assertIn("harmful_test", main_text)
        self.assertIn("resolve_graph_grpo_optuna_sampler(os.getenv(\"OPTUNA_SAMPLER\"))", main_text)
        self.assertIn("resolve_graph_grpo_optuna_n_trials(os.getenv(\"OPTUNA_N_TRIALS\"))", main_text)
        self.assertIn('category_dataset_source', main_text)
        self.assertIn("resolve_graph_grpo_weights_mode(os.getenv(\"WEIGHTS_MODE\"))", main_text)
        self.assertIn("resolve_graph_grpo_weights_init_type(os.getenv(\"WEIGHTS_INIT_TYPE\"))", main_text)
        self.assertIn("optimize_weights_with_optuna(", main_text)
        self.assertIn('"optimal_objective"', main_text)
        self.assertIn('"optimal_harmfulness"', main_text)
        self.assertIn('"optimal_harmfulness_source"', main_text)
        self.assertIn('"best_train_batch_mean_objective"', main_text)
        self.assertIn('"best_train_batch_mean_harmfulness"', main_text)
        self.assertIn('"best_train_batch_mean_kl"', main_text)
        self.assertIn('"best_grpo_epoch"', main_text)
        self.assertIn('clean_model', main_text)
        self.assertIn('mmlu_accuracy', main_text)
        self.assertIn('best_value_model', main_text)
        self.assertIn('harmfulness_on_full_dataset', main_text)
        self.assertIn('mmlu_score', main_text)
        self.assertIn('best_batch_model', main_text)
        self.assertIn('best_value_model', main_text)
        self.assertIn('"model_state_eval/point_index"', main_text)
        self.assertIn('"model_state_eval/mmlu_score"', main_text)
        self.assertIn('"model_state_eval/harmfulness_on_full_dataset"', main_text)
        self.assertIn("Loss: {accumulated_total_loss:.6e}", trainer_text)
        self.assertIn("mean_objective", trainer_text)
        self.assertIn("mean_kl", trainer_text)
        self.assertIn("Grad norm: {grad_norm:.6e}", trainer_text)


if __name__ == "__main__":
    unittest.main()
