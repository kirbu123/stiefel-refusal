"""
Method runner: dispatches to existing baseline modules after setting
environment variables from the TOML config.

Each baseline's main() is called directly -- it reads env vars via
config.py and baselines/hyperparams.py at import time, so we must
set env vars *before* importing the baseline module.
"""

import importlib
import os
import sys
import json
from pathlib import Path
from typing import Any

from .ui import print


PROJECT_ROOT = Path(__file__).parent.parent

BASELINE_MODULES = {
    "basic_refusal": "baselines.basic_refusal",
    "topic_ablation": "baselines.topic_ablation",
    "tag_ablation": "baselines.tag_ablation",
    "graph_average": "baselines.graph_average",
    "graph_grpo": "baselines.graph_grpo",
    "graph_grpo_old": "baselines.graph_grpo_old",
}


def _invalidate_cached_config():
    """
    Remove already-imported config modules so they re-read env vars
    on next import.
    """
    modules_to_reload = [
        "config",
        "baselines.hyperparams",
    ]
    for mod_name in modules_to_reload:
        if mod_name in sys.modules:
            del sys.modules[mod_name]


def _invalidate_baseline_module(method: str):
    """Remove the baseline module from cache so it reimports cleanly."""
    mod_name = BASELINE_MODULES[method]
    to_remove = [k for k in sys.modules if k == mod_name or k.startswith(mod_name + ".")]
    for k in to_remove:
        del sys.modules[k]


def run_method(method: str, config: dict[str, Any], model_name: str) -> Path:
    """
    Run the selected editing method.

    Sets environment variables from the config, invalidates cached modules,
    then calls the baseline's main() function.

    Returns the results directory for the method.
    """
    from .config_loader import apply_config_to_env

    apply_config_to_env(config, model_name)

    _invalidate_cached_config()
    _invalidate_baseline_module(method)

    original_dir = os.getcwd()
    os.chdir(PROJECT_ROOT)

    try:
        print()
        print(f"[bold green]Running method: {method}[/]")
        print("=" * 60)

        module = importlib.import_module(BASELINE_MODULES[method])
        module.main()

        print()
        print("[bold green]Method execution completed.[/]")
    finally:
        os.chdir(original_dir)

    results_dir = PROJECT_ROOT / "results" / method
    return results_dir


def find_latest_results(results_dir: Path) -> list[dict[str, Any]]:
    """
    Scan the results directory for answer JSON files and return
    parsed experiment results sorted by timestamp (newest first).
    """
    results = []
    answers_dir = results_dir / "answers"

    if not answers_dir.exists():
        return results

    for json_file in sorted(answers_dir.glob("answers_*.json"), reverse=True):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_source_file"] = str(json_file)
            results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[yellow]Warning: could not read {json_file}: {e}[/]")

    return results


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build a summary list from raw result dicts for display in the UI.
    Each summary contains key metrics and the hyperparameters used.
    """
    summaries = []

    for result in results:
        experiment_info = result.get("experiment_config", result.get("experiment_info", {}))
        category = experiment_info.get("category", "N/A")
        hyperparams = experiment_info.get("hyperparameters", experiment_info.get("abliteration_params", {}))
        param_key = experiment_info.get("param_key", "")

        harmful = result.get("harmful_questions", {})
        modified_evals = harmful.get("modified_evaluations", [])
        modified_scores = [
            e.get("score", 0) for e in modified_evals if e.get("score") is not None
        ] if modified_evals else result.get("final_scores", [])

        stats = result.get("score_statistics", {})
        mean_score = stats.get("modified_mean", stats.get("mean"))
        if mean_score is None and modified_scores:
            import numpy as np
            mean_score = float(np.mean(modified_scores))

        harmless = result.get("harmless_questions", {})
        avg_locality = harmless.get("average_locality_change")
        mmlu = result.get("mmlu", {})
        mmlu_original_accuracy = mmlu.get("original", {}).get("accuracy")
        mmlu_modified_accuracy = mmlu.get("modified", {}).get("accuracy")

        summaries.append({
            "category": category,
            "hyperparams": hyperparams,
            "param_key": param_key,
            "mean_score": mean_score,
            "avg_locality_change": avg_locality,
            "mmlu_original_accuracy": mmlu_original_accuracy,
            "mmlu_modified_accuracy": mmlu_modified_accuracy,
            "n_scores": len(modified_scores),
            "source_file": result.get("_source_file", ""),
            "raw": result,
        })

    return summaries
