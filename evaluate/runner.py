#!/usr/bin/env python3
"""
Evaluate locality and harmfulness of model responses from ablation results.

Loads ablation experiment results and evaluates:
- Harmfulness on harmful questions (original vs modified responses)
- Locality on harmless questions (original vs modified harmless responses)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from evaluate.metrics import evaluate_responses, evaluate_locality
from visualization.plots import create_harmfulness_heatmap, create_distribution_plots
from config import (
    JUDGE_API_URL,
    CLASSIFIER_API_URL,
    JUDGE_MODEL,
    FEW_SHOTS_PATH,
    PROJECT_ROOT,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "evaluated"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VALUES_PLOTS_DIR = RESULTS_DIR / "values_plots"
VALUES_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DISTRIBUTION_PLOTS_DIR = RESULTS_DIR / "distribution_plots"
DISTRIBUTION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_ablation_results(file_path: Path) -> Dict:
    """Load ablation experiment results from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # Path to ablation results file (edit as needed)
    ablation_results = PROJECT_ROOT / "topic_ablation_results2/category_Physical_harm_final_20251220_190410.json"

    print("=" * 80)
    print("STARTING BASELINE RESULTS EVALUATION")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results file: {ablation_results}")
    print()

    print("Loading ablation experiment results...")
    ablation_data = load_ablation_results(ablation_results)

    experiment_config = ablation_data.get("experiment_config", {})
    category_name = experiment_config.get("category", "Unknown")
    results = ablation_data.get("results", {})

    print(f"Category: {category_name}")
    print(f"Number of experiments: {len(results)}")

    try:
        with open(FEW_SHOTS_PATH, "r", encoding="utf-8") as f:
            few_shots_data = json.load(f)
        classifier_categories = [
            {"id": c.get("id"), "label": c.get("label", "")}
            for c in few_shots_data.get("categories", [])
            if c.get("id")
        ]
        print(f"Loaded {len(classifier_categories)} classifier categories")
    except Exception as e:
        print(f"Warning: could not load classifier categories: {e}")
        classifier_categories = []

    evaluated_results = {}

    for param_key, result_data in results.items():
        print(f"\n{'=' * 80}")
        print(f"Processing experiment: {param_key}")
        print(f"{'=' * 80}")

        questions = result_data.get("questions", [])
        original_responses = result_data.get("original_responses", [])
        modified_responses = result_data.get("modified_responses", [])

        harmless_eval_prompts = result_data.get("harmless_eval_prompts", [])
        original_harmless_responses = result_data.get("original_harmless_responses", [])
        modified_harmless_responses = result_data.get("modified_harmless_responses", [])

        original_scores, original_score_data = evaluate_responses(
            questions, original_responses, classifier_categories,
            description=f"original responses (experiment {param_key})"
        )

        modified_scores, modified_score_data = evaluate_responses(
            questions, modified_responses, classifier_categories,
            description=f"modified responses (experiment {param_key})"
        )

        locality_scores = []
        average_locality_change = None

        if harmless_eval_prompts and original_harmless_responses and modified_harmless_responses:
            locality_scores, average_locality_change = evaluate_locality(
                harmless_eval_prompts,
                original_harmless_responses,
                modified_harmless_responses,
                classifier_categories,
            )

        evaluated_results[param_key] = {
            **result_data,
            "original_scores": original_scores,
            "modified_scores": modified_scores,
            "original_score_data": original_score_data,
            "modified_score_data": modified_score_data,
            "locality_scores": locality_scores,
            "average_locality_change": average_locality_change,
            "evaluation_timestamp": datetime.now().isoformat(),
        }

        if original_scores and modified_scores:
            print(f"\nHarmfulness statistics for experiment {param_key}:")
            print(f"  Original responses: mean={np.mean(original_scores):.2f}, median={np.median(original_scores):.2f}")
            print(f"  Modified responses: mean={np.mean(modified_scores):.2f}, median={np.median(modified_scores):.2f}")

        if average_locality_change is not None:
            print(f"  Locality (average change): {average_locality_change:+.2f}")

    print(f"\n{'=' * 80}")
    print("CREATING PLOTS")
    print(f"{'=' * 80}")

    print("\nCreating harmfulness heatmap...")
    create_harmfulness_heatmap(
        {"results": evaluated_results},
        category_name,
        VALUES_PLOTS_DIR,
    )

    print("\nCreating distribution plots...")
    for param_key, result_data in evaluated_results.items():
        original_scores = result_data.get("original_scores", [])
        modified_scores = result_data.get("modified_scores", [])
        locality_scores_data = result_data.get("locality_scores", [])

        if original_scores or modified_scores or locality_scores_data:
            create_distribution_plots(
                original_scores,
                modified_scores,
                locality_scores_data,
                f"{category_name}_{param_key}",
                DISTRIBUTION_PLOTS_DIR,
                DISTRIBUTION_PLOTS_DIR,
            )

    output_file = RESULTS_DIR / f"evaluated_baselines_{category_name.replace('/', '_').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        "experiment_config": experiment_config,
        "evaluation_config": {
            "judge_api_url": JUDGE_API_URL,
            "judge_model": JUDGE_MODEL,
            "classifier_api_url": CLASSIFIER_API_URL,
            "few_shots_path": str(FEW_SHOTS_PATH),
        },
        "results": evaluated_results,
        "evaluation_timestamp": datetime.now().isoformat(),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation results saved to: {output_file}")
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
