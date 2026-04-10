"""
Configuration for refusal direction experiments.
"""

import os
from pathlib import Path
from heretic.config import DatasetSpecification

PROJECT_ROOT = Path(__file__).parent


def _resolve_results_dir() -> Path:
    raw_results_root = os.getenv("RESULTS_ROOT", "results").strip() or "results"
    results_root = Path(raw_results_root).expanduser()
    if not results_root.is_absolute():
        results_root = PROJECT_ROOT / results_root
    return results_root


def _parse_csv_env_list(env_var: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    raw_value = os.getenv(env_var)
    if raw_value is None:
        return default

    values = []
    for part in raw_value.split(","):
        normalized = part.strip()
        if normalized:
            values.append(normalized)
    return tuple(values)

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
MODEL_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

N_SAMPLES_PER_CATEGORY = 10

CATEGORIES = [
    "Harassment/Discrimination",
    "Malware/Hacking",
    "Physical harm",
    "Economic harm",
    "Fraud/Deception",
    "Disinformation",
    "Sexual/Adult content",
    "Privacy",
    "Expert advice",
    "Government decision-making"
]

HYPERPARAMS = {
    "max_weight": [2.5, 3.0],
    "max_weight_position": [0.7],
    "min_weight": [0.0, 1.0],
    "min_weight_distance": [0.3]
}

GOOD_PROMPTS_DATASET = DatasetSpecification(
    dataset="mlabonne/harmless_alpaca",
    split="train[:400]",
    column="text"
)

HARMLESS_EVAL_DATASET = DatasetSpecification(
    dataset="mlabonne/harmless_alpaca",
    split="test[:50]",
    column="text"
)

RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "results").strip() or "results")
RESULTS_DIR = _resolve_results_dir()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_method_results_dir(method_name: str) -> Path:
    method_dir = RESULTS_DIR / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    return method_dir

GRPO_CONFIG = {
    "n_groups": int(os.getenv("GRPO_N_GROUPS", 4)),
    "n_epochs": int(os.getenv("GRPO_N_EPOCHS", 50)),
    "learning_rate": float(os.getenv("GRPO_LEARNING_RATE", 1e-3)),
    "noise_scale": float(os.getenv("GRPO_NOISE_SCALE", 1.0)),
    "beta": float(os.getenv("GRPO_BETA", 0.1)),
    "gradient_scale": float(os.getenv("GRPO_GRADIENT_SCALE", 1e6)),
    "ref_alpha": float(os.getenv("GRPO_REF_ALPHA", 1.0)),
    "is_clip_ratio": float(os.getenv("IS_CLIP_RATIO", 5.0)),
    "clip_ratio": float(os.getenv("GRPO_CLIP_RATIO", 0.2)),
    "loss_agg_mode": os.getenv("GRPO_LOSS_AGG_MODE", "token-mean"),
    "kl_loss_coef": float(os.getenv("GRPO_KL_LOSS_COEF", 0.01)),
}

ABLITERATION_PARAMS = {
    "max_weight": float(os.getenv("ABLITERATION_MAX_WEIGHT", 2.0)),
    "max_weight_position": float(os.getenv("ABLITERATION_MAX_WEIGHT_POSITION", 0.7)),
    "min_weight": float(os.getenv("ABLITERATION_MIN_WEIGHT", 0.1)),
    "min_weight_distance": float(os.getenv("ABLITERATION_MIN_WEIGHT_DISTANCE", 0.3))
}

JUDGE_API_URL = os.getenv("JUDGE_API_URL", "http://localhost:31181/v1/chat/completions")
CLASSIFIER_API_URL = os.getenv("CLASSIFIER_API_URL", "http://localhost:31180/v1/chat/completions")
JUDGE_MODEL = "deepseek-v3.2"
CLASSIFIER_MODEL = "deepseek-v3.2"

FEW_SHOTS_PATH = PROJECT_ROOT / "prompts" / "few-shots.json"

GRAPH_FILE = Path(os.getenv("GRAPH_FILE", str(PROJECT_ROOT / "graph_build" / "Physical harm_graph.txt")))
TAG_FILTERED_QUESTIONS_FILE = Path(os.getenv("TAG_FILTERED_QUESTIONS_FILE", str(PROJECT_ROOT / "tag_filtered_questions.json")))

EVALUATE_LOCALITY = os.getenv("EVALUATE_LOCALITY", "true").lower() in ("true", "1", "yes")
EVALUATION_BACKEND = os.getenv("EVALUATION_BACKEND", "local_llm_judge")
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
CATEGORY_DATASET_SOURCE = os.getenv("CATEGORY_DATASET_SOURCE", "combined").strip() or "combined"
_category_filter = os.getenv("CATEGORY_FILTER", "").strip()
CATEGORY_FILTER = _category_filter or None

WEIGHTS_INIT_TYPE = os.getenv("WEIGHTS_INIT_TYPE", "zero")

BENCHMARKS_ENABLED = tuple(
    item.lower() for item in _parse_csv_env_list("BENCHMARKS_ENABLED")
)

_jailbreakbench_categories = _parse_csv_env_list("JAILBREAKBENCH_CATEGORIES")
JAILBREAKBENCH_CONFIG = {
    "judge_mode": os.getenv("JAILBREAKBENCH_JUDGE_MODE", "official").strip().lower() or "official",
    "max_samples": int(os.getenv("JAILBREAKBENCH_MAX_SAMPLES", "100").strip() or "100"),
    "categories": list(_jailbreakbench_categories),
    "harmful_score_threshold": int(
        os.getenv("JAILBREAKBENCH_HARMFUL_SCORE_THRESHOLD", "3").strip() or "3"
    ),
    "official_judge": {
        "model_name": os.getenv(
            "JAILBREAKBENCH_OFFICIAL_JUDGE_MODEL_NAME",
            "together_ai/meta-llama/Llama-3-70b-chat-hf",
        ).strip() or "together_ai/meta-llama/Llama-3-70b-chat-hf",
        "api_url": os.getenv(
            "JAILBREAKBENCH_OFFICIAL_JUDGE_API_URL",
            "https://api.together.xyz/v1/chat/completions",
        ).strip() or "https://api.together.xyz/v1/chat/completions",
        "api_key_env": os.getenv(
            "JAILBREAKBENCH_OFFICIAL_JUDGE_API_KEY_ENV",
            "TOGETHER_API_KEY",
        ).strip() or "TOGETHER_API_KEY",
        "batch_size": int(
            os.getenv("JAILBREAKBENCH_OFFICIAL_JUDGE_BATCH_SIZE", "8").strip() or "8"
        ),
        "timeout_sec": int(
            os.getenv("JAILBREAKBENCH_OFFICIAL_JUDGE_TIMEOUT_SEC", "60").strip() or "60"
        ),
    },
}

HARMBENCH_CONFIG = {
    "judge_mode": os.getenv("HARMBENCH_JUDGE_MODE", "official").strip().lower() or "official",
    "split": os.getenv("HARMBENCH_SPLIT", "test").strip().lower() or "test",
    "max_samples": int(os.getenv("HARMBENCH_MAX_SAMPLES", "0").strip() or "0"),
    "categories": list(_parse_csv_env_list("HARMBENCH_CATEGORIES")),
    "functional_categories": list(
        _parse_csv_env_list("HARMBENCH_FUNCTIONAL_CATEGORIES")
    ),
    "semantic_categories": list(
        _parse_csv_env_list("HARMBENCH_SEMANTIC_CATEGORIES")
    ),
    "harmful_score_threshold": int(
        os.getenv("HARMBENCH_HARMFUL_SCORE_THRESHOLD", "3").strip() or "3"
    ),
    "official_classifier": {
        "model_name": os.getenv(
            "HARMBENCH_OFFICIAL_CLASSIFIER_MODEL_NAME",
            "cais/HarmBench-Llama-2-13b-cls",
        ).strip() or "cais/HarmBench-Llama-2-13b-cls",
        "batch_size": int(
            os.getenv("HARMBENCH_OFFICIAL_CLASSIFIER_BATCH_SIZE", "8").strip() or "8"
        ),
        "device": os.getenv(
            "HARMBENCH_OFFICIAL_CLASSIFIER_DEVICE",
            "cuda",
        ).strip() or "cuda",
        "dtype": os.getenv(
            "HARMBENCH_OFFICIAL_CLASSIFIER_DTYPE",
            "bfloat16",
        ).strip().lower() or "bfloat16",
    },
}

_mmlu_sample_size = os.getenv("MMLU_SAMPLE_SIZE", "100").strip()

MMLU_CONFIG = {
    "enabled": os.getenv("MMLU_ENABLED", "true").lower() in ("true", "1", "yes"),
    "dataset": os.getenv("MMLU_DATASET", "cais/mmlu"),
    "subset": os.getenv("MMLU_SUBSET", "all"),
    "split": os.getenv("MMLU_SPLIT", "test"),
    "mode": os.getenv("MMLU_MODE", "zero_shot"),
    "answer_mode": os.getenv("MMLU_ANSWER_MODE", "generate"),
    "n_shots": int(os.getenv("MMLU_N_SHOTS", 5)),
    "sample_size": None if _mmlu_sample_size.lower() in ("", "none", "null", "all") else int(_mmlu_sample_size),
    "sample_seed": int(os.getenv("MMLU_SAMPLE_SEED", 42)),
    "max_new_tokens": int(os.getenv("MMLU_MAX_NEW_TOKENS", 2048)),
    "store_predictions": os.getenv("MMLU_STORE_PREDICTIONS", "true").lower() in ("true", "1", "yes"),
}
