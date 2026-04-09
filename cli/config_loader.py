"""
Config loading: parse TOML files and set environment variables
so that existing baseline code (which reads os.getenv) picks them up.
"""

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

try:
    from .ui import print
except ModuleNotFoundError:
    from builtins import print

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

METHODS = {
    "basic_refusal": "Basic Refusal",
    "topic_ablation": "Topic Ablation",
    "tag_ablation": "Tag Ablation",
    "graph_average": "Graph Average",
    "graph_grpo": "Graph GRPO (IS)",
    "graph_grpo_old": "Graph GRPO (old)",
}

DEFAULT_CONFIG_FILES = {
    method: CONFIGS_DIR / f"{method}.toml" for method in METHODS
}


def load_config(method: str, config_path: str | None = None) -> dict[str, Any]:
    """Load a TOML config file for the given method."""
    if config_path:
        path = Path(config_path)
    else:
        path = DEFAULT_CONFIG_FILES[method]

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        config = tomllib.load(f)

    return config


def apply_config_to_env(config: dict[str, Any], model_name: str | None = None):
    """
    Set environment variables from the parsed TOML config
    so that existing baseline code picks them up via os.getenv().
    """
    if model_name:
        os.environ["MODEL_NAME"] = model_name
    elif config.get("model", {}).get("name"):
        os.environ["MODEL_NAME"] = config["model"]["name"]

    model = config.get("model", {})
    if model.get("batch_size") is not None:
        os.environ["BATCH_SIZE"] = str(model["batch_size"])

    grid = config.get("grid_search", {})
    if grid.get("max_weight") is not None:
        os.environ["GRID_MAX_WEIGHT"] = ",".join(str(v) for v in grid["max_weight"])
    if grid.get("max_weight_position") is not None:
        os.environ["GRID_MAX_WEIGHT_POSITION"] = ",".join(str(v) for v in grid["max_weight_position"])
    if grid.get("min_weight") is not None:
        os.environ["GRID_MIN_WEIGHT"] = ",".join(str(v) for v in grid["min_weight"])
    if grid.get("min_weight_distance") is not None:
        os.environ["GRID_MIN_WEIGHT_DISTANCE"] = ",".join(str(v) for v in grid["min_weight_distance"])

    data = config.get("data", {})
    if data.get("graph_file"):
        os.environ["GRAPH_FILE"] = data["graph_file"]
    if data.get("category_dataset_source"):
        os.environ.setdefault("CATEGORY_DATASET_SOURCE", data["category_dataset_source"])
    if data.get("category_filter"):
        os.environ.setdefault("CATEGORY_FILTER", data["category_filter"])
    if data.get("tag_filtered_questions_file"):
        os.environ["TAG_FILTERED_QUESTIONS_FILE"] = data["tag_filtered_questions_file"]

    output = config.get("output", {})
    os.environ["RESULTS_ROOT"] = str(output.get("results_root", "results"))

    grpo = config.get("grpo", {})
    if grpo.get("n_groups") is not None:
        os.environ["GRPO_N_GROUPS"] = str(grpo["n_groups"])
    if grpo.get("n_epochs") is not None:
        os.environ["GRPO_N_EPOCHS"] = str(grpo["n_epochs"])
    if grpo.get("learning_rate") is not None:
        os.environ["GRPO_LEARNING_RATE"] = str(grpo["learning_rate"])
    if grpo.get("noise_scale") is not None:
        os.environ["GRPO_NOISE_SCALE"] = str(grpo["noise_scale"])
    if grpo.get("beta") is not None:
        os.environ["GRPO_BETA"] = str(grpo["beta"])
    if grpo.get("gradient_scale") is not None:
        os.environ["GRPO_GRADIENT_SCALE"] = str(grpo["gradient_scale"])
    if grpo.get("ref_alpha") is not None:
        os.environ["GRPO_REF_ALPHA"] = str(grpo["ref_alpha"])
    if grpo.get("is_clip_ratio") is not None:
        os.environ["IS_CLIP_RATIO"] = str(grpo["is_clip_ratio"])
    if grpo.get("clip_ratio") is not None:
        os.environ["GRPO_CLIP_RATIO"] = str(grpo["clip_ratio"])
    if grpo.get("loss_agg_mode") is not None:
        os.environ["GRPO_LOSS_AGG_MODE"] = grpo["loss_agg_mode"]
    if grpo.get("kl_loss_coef") is not None:
        os.environ["GRPO_KL_LOSS_COEF"] = str(grpo["kl_loss_coef"])

    abl = config.get("abliteration", {})
    if abl.get("max_weight") is not None:
        os.environ["ABLITERATION_MAX_WEIGHT"] = str(abl["max_weight"])
    if abl.get("max_weight_position") is not None:
        os.environ["ABLITERATION_MAX_WEIGHT_POSITION"] = str(abl["max_weight_position"])
    if abl.get("min_weight") is not None:
        os.environ["ABLITERATION_MIN_WEIGHT"] = str(abl["min_weight"])
    if abl.get("min_weight_distance") is not None:
        os.environ["ABLITERATION_MIN_WEIGHT_DISTANCE"] = str(abl["min_weight_distance"])

    weights = config.get("weights", {})
    if weights.get("init_type"):
        os.environ["WEIGHTS_INIT_TYPE"] = weights["init_type"]
    if weights.get("mode"):
        os.environ["WEIGHTS_MODE"] = weights["mode"]

    optimizer = config.get("optimizer", {})
    if optimizer.get("method"):
        os.environ["OPTIMIZER_METHOD"] = optimizer["method"]

    reward = config.get("reward", {})
    if reward.get("sign") is not None:
        os.environ["REWARD_SIGN"] = str(reward["sign"])

    optuna = config.get("optuna", {})
    if optuna.get("sampler"):
        os.environ["OPTUNA_SAMPLER"] = optuna["sampler"]
    if optuna.get("n_trials") is not None:
        os.environ["OPTUNA_N_TRIALS"] = str(optuna["n_trials"])
    if optuna.get("sampler_seed") is not None:
        os.environ["OPTUNA_SAMPLER_SEED"] = str(optuna["sampler_seed"])
    if optuna.get("weight_min") is not None:
        os.environ["OPTUNA_WEIGHT_MIN"] = str(optuna["weight_min"])
    if optuna.get("weight_max") is not None:
        os.environ["OPTUNA_WEIGHT_MAX"] = str(optuna["weight_max"])

    evl = config.get("evaluation", {})
    if evl.get("backend"):
        os.environ["EVALUATION_BACKEND"] = evl["backend"]
    if evl.get("evaluate_locality") is not None:
        os.environ["EVALUATE_LOCALITY"] = str(evl["evaluate_locality"]).lower()
    if evl.get("judge_api_url"):
        os.environ["JUDGE_API_URL"] = evl["judge_api_url"]
    if evl.get("classifier_api_url"):
        os.environ["CLASSIFIER_API_URL"] = evl["classifier_api_url"]
    if evl.get("llamaguard_model_name"):
        os.environ["LLAMAGUARD_MODEL_NAME"] = evl["llamaguard_model_name"]
    if evl.get("llamaguard_device"):
        os.environ["LLAMAGUARD_DEVICE"] = evl["llamaguard_device"]
    if evl.get("llamaguard_dtype"):
        os.environ["LLAMAGUARD_DTYPE"] = evl["llamaguard_dtype"]

    mmlu = config.get("mmlu", {})
    if mmlu.get("enabled") is not None:
        os.environ["MMLU_ENABLED"] = str(mmlu["enabled"]).lower()
    if mmlu.get("dataset"):
        os.environ["MMLU_DATASET"] = mmlu["dataset"]
    if mmlu.get("subset"):
        os.environ["MMLU_SUBSET"] = mmlu["subset"]
    if mmlu.get("split"):
        os.environ["MMLU_SPLIT"] = mmlu["split"]
    if mmlu.get("mode"):
        os.environ["MMLU_MODE"] = mmlu["mode"]
    if mmlu.get("answer_mode"):
        os.environ["MMLU_ANSWER_MODE"] = mmlu["answer_mode"]
    if mmlu.get("n_shots") is not None:
        os.environ["MMLU_N_SHOTS"] = str(mmlu["n_shots"])
    if "sample_size" in mmlu:
        os.environ["MMLU_SAMPLE_SIZE"] = "none" if mmlu["sample_size"] is None else str(mmlu["sample_size"])
    if mmlu.get("sample_seed") is not None:
        os.environ["MMLU_SAMPLE_SEED"] = str(mmlu["sample_seed"])
    if mmlu.get("max_new_tokens") is not None:
        os.environ["MMLU_MAX_NEW_TOKENS"] = str(mmlu["max_new_tokens"])
    if mmlu.get("store_predictions") is not None:
        os.environ["MMLU_STORE_PREDICTIONS"] = str(mmlu["store_predictions"]).lower()


def print_config_summary(config: dict[str, Any], model_name: str):
    """Display a summary of the loaded configuration."""
    print()
    print("[bold]Configuration summary:[/]")
    print(f"  Model: [bold]{model_name}[/]")

    grid = config.get("grid_search", {})
    if grid:
        print("  [bold]Grid search:[/]")
        for k, v in grid.items():
            print(f"    {k}: {v}")

    grpo = config.get("grpo", {})
    if grpo:
        print("  [bold]GRPO parameters:[/]")
        for k, v in grpo.items():
            print(f"    {k}: {v}")

    abl = config.get("abliteration", {})
    if abl:
        print("  [bold]Abliteration parameters:[/]")
        for k, v in abl.items():
            print(f"    {k}: {v}")

    optimizer = config.get("optimizer", {})
    if optimizer:
        print("  [bold]Optimizer:[/]")
        for k, v in optimizer.items():
            print(f"    {k}: {v}")

    reward = config.get("reward", {})
    if reward:
        print("  [bold]Reward:[/]")
        for k, v in reward.items():
            print(f"    {k}: {v}")

    optuna = config.get("optuna", {})
    if optuna:
        print("  [bold]Optuna:[/]")
        for k, v in optuna.items():
            print(f"    {k}: {v}")

    data = config.get("data", {})
    if data:
        print("  [bold]Data:[/]")
        for k, v in data.items():
            print(f"    {k}: {v}")

    output = config.get("output", {})
    if output:
        print("  [bold]Output:[/]")
        for k, v in output.items():
            print(f"    {k}: {v}")

    evl = config.get("evaluation", {})
    if evl:
        print(f"  [bold]Evaluation:[/]")
        print(f"    backend: {evl.get('backend', 'N/A')}")
        print(f"    evaluate_locality: {evl.get('evaluate_locality', 'N/A')}")

    mmlu = config.get("mmlu", {})
    if mmlu:
        print("  [bold]MMLU:[/]")
        for k, v in mmlu.items():
            print(f"    {k}: {v}")
