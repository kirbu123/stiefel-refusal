#!/usr/bin/env python3
"""
Topic ablation experiment: iterate over categories and abliteration hyperparameters.

Uses the category name itself as the harmful prompt for computing the refusal direction.
"""

import sys
import json
import random
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from heretic.config import Settings, DatasetSpecification
from heretic.model import Model
from heretic.utils import load_prompts

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.set_grad_enabled(False)

from config import (
    MODEL_NAME, CATEGORIES, GOOD_PROMPTS_DATASET,
    JUDGE_API_URL, CLASSIFIER_API_URL, JUDGE_MODEL, FEW_SHOTS_PATH,
    HARMLESS_EVAL_DATASET, EVALUATE_LOCALITY, MMLU_CONFIG, get_method_results_dir,
)
from data_utils import load_all_datasets_with_categories, extract_response_after_think
from refusal_directions import compute_refusal_direction
from model_utils import apply_abliteration_with_hyperparams
from evaluate.metrics import evaluate_responses, evaluate_locality
from evaluate.mmlu import (
    build_mmlu_result,
    evaluate_model_on_mmlu,
    get_cached_or_evaluate_original_mmlu,
)
from benchmarks.integration import build_benchmark_runner
from visualization.plots import (
    plot_harmfulness_heatmap, plot_locality_heatmap,
    plot_harmfulness_distribution, plot_locality_distribution,
    generate_plot_filename,
)
from baselines.hyperparams import HYPERPARAMS

BASELINE_RESULTS_DIR = get_method_results_dir("basic_refusal")

ANSWERS_DIR = BASELINE_RESULTS_DIR / "answers"
ANSWERS_DIR.mkdir(parents=True, exist_ok=True)

HARMFULNESS_DISTRIBUTION_DIR = BASELINE_RESULTS_DIR / "harmfulness" / "distribution_plots"
HARMFULNESS_DISTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)

HARMFULNESS_VALUES_DIR = BASELINE_RESULTS_DIR / "harmfulness" / "heatmap_plots"
HARMFULNESS_VALUES_DIR.mkdir(parents=True, exist_ok=True)

LOCALITY_DISTRIBUTION_DIR = BASELINE_RESULTS_DIR / "locality" / "distribution_plots"
LOCALITY_DISTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)

LOCALITY_VALUES_DIR = BASELINE_RESULTS_DIR / "locality" / "heatmap_plots"
LOCALITY_VALUES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("STARTING EXPERIMENT: TOPIC ABLATION (CATEGORY-LEVEL)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL_NAME}")
    print(f"Categories: {len(CATEGORIES)}")
    print(f"Hyperparameters: {HYPERPARAMS}")
    print()

    print("Loading data...")
    all_data = load_all_datasets_with_categories()
    print(f"Total samples loaded: {len(all_data)}")

    data_by_category = {cat: [] for cat in CATEGORIES}
    for item in all_data:
        cat = item.get('category', '')
        if cat in data_by_category:
            data_by_category[cat].append(item)

    print("\nSamples per category:")
    for cat in CATEGORIES:
        print(f"  {cat}: {len(data_by_category[cat])}")

    print("\nLoading model...")
    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]] if sys.argv else ["script"]
        settings = Settings(
            model=MODEL_NAME, batch_size=16,
            max_response_length=2048, system_prompt="You are a helpful assistant.",
        )
    finally:
        sys.argv = original_argv

    print(f"Loading model {MODEL_NAME}...")
    model = Model(settings)
    print("Model loaded successfully!")

    n_layers = len(model.get_layers())
    print(f"Number of layers: {n_layers}")

    print("Loading good prompts...")
    good_prompts = load_prompts(GOOD_PROMPTS_DATASET)
    print(f"Loaded {len(good_prompts)} good prompts")

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

    benchmark_runner = build_benchmark_runner(
        method_results_dir=BASELINE_RESULTS_DIR,
        model_name=MODEL_NAME,
        classifier_categories=classifier_categories,
    )

    harmless_eval_prompts = []
    if EVALUATE_LOCALITY:
        print("\nLoading harmless questions for locality...")
        harmless_eval_prompts = load_prompts(HARMLESS_EVAL_DATASET)
        print(f"Loaded {len(harmless_eval_prompts)} harmless questions")

    original_mmlu_result = None
    if MMLU_CONFIG["enabled"]:
        print("\nEvaluating original model on MMLU...")
        original_mmlu_result = get_cached_or_evaluate_original_mmlu(
            model,
            model_name=MODEL_NAME,
            config=MMLU_CONFIG,
        )
        if original_mmlu_result is not None:
            print(f"Original MMLU accuracy: {original_mmlu_result['summary']['accuracy']:.4f}")

    if benchmark_runner is not None:
        print("\nPreparing clean benchmark evaluations...")
        original_benchmark_summaries = benchmark_runner.prepare_original(model)
        for benchmark_name, summary in original_benchmark_summaries.items():
            attack_success_rate = summary.get("attack_success_rate")
            if attack_success_rate is not None:
                print(
                    f"Original {benchmark_name} attack success rate: "
                    f"{attack_success_rate:.4f}"
                )

    param_names = list(HYPERPARAMS.keys())
    param_values = list(HYPERPARAMS.values())
    param_combinations = list(product(*param_values))

    print(f"\nTotal hyperparameter combinations: {len(param_combinations)}")
    print(f"Total categories: {len(CATEGORIES)}")
    print(f"Total experiments: {len(CATEGORIES) * len(param_combinations)}")
    print()

    for category_idx, category_name in enumerate(CATEGORIES, 1):
        print("\n" + "=" * 80)
        print(f"CATEGORY {category_idx}/{len(CATEGORIES)}: {category_name}")
        print("=" * 80)

        category_items = data_by_category.get(category_name, [])
        if not category_items:
            print(f"  Skipping category {category_name}: no data")
            continue

        category_questions = [item.get('instruction', '') for item in category_items if item.get('instruction')]
        if not category_questions:
            print(f"  Skipping category {category_name}: no questions")
            continue

        print(f"  Questions in category: {len(category_questions)}")

        # Key difference from graph_average: use category name as the harmful prompt
        print(f"\n  Computing refusal direction for category {category_name}...")
        refusal_directions = compute_refusal_direction(model, category_questions, good_prompts)
        print(f"  refusal_directions shape: {refusal_directions.shape}")

        print(f"\n  Getting original responses for category questions...")
        original_responses_raw = model.get_responses_batched(category_questions)
        original_responses = [extract_response_after_think(resp) for resp in original_responses_raw]
        print(f"  Got {len(original_responses)} original responses")

        original_harmless_responses = []
        if EVALUATE_LOCALITY and harmless_eval_prompts:
            print(f"\n  Getting original responses for harmless questions...")
            original_harmless_responses_raw = model.get_responses_batched(harmless_eval_prompts)
            original_harmless_responses = [extract_response_after_think(resp) for resp in original_harmless_responses_raw]
            print(f"  Got {len(original_harmless_responses)} original responses for harmless questions")

        category_results = {}

        for param_idx, param_combo in enumerate(param_combinations, 1):
            params_dict = dict(zip(param_names, param_combo))
            param_key = "_&".join([f"{k}={v}" for k, v in params_dict.items()])

            print(f"\n  {'-'*60}")
            print(f"  Experiment {param_idx}/{len(param_combinations)} for category {category_name}")
            print(f"  Parameters: {param_key}")
            print(f"  {'-'*60}")

            print("    Reloading model...")
            model.reload_model()

            print(f"    Applying abliteration with parameters:")
            for k, v in params_dict.items():
                print(f"      {k}: {v}")

            apply_abliteration_with_hyperparams(
                model, refusal_directions,
                params_dict["max_weight"], params_dict["max_weight_position"],
                params_dict["min_weight"], params_dict["min_weight_distance"],
                n_layers,
            )

            print(f"    Getting responses for {len(category_questions)} category questions...")
            modified_responses_raw = model.get_responses_batched(category_questions)
            modified_responses = [extract_response_after_think(resp) for resp in modified_responses_raw]
            print(f"    Got {len(modified_responses)} responses")

            modified_harmless_responses = []
            if EVALUATE_LOCALITY and harmless_eval_prompts:
                print(f"    Getting modified responses for harmless questions...")
                modified_harmless_responses_raw = model.get_responses_batched(harmless_eval_prompts)
                modified_harmless_responses = [extract_response_after_think(resp) for resp in modified_harmless_responses_raw]
                print(f"    Got {len(modified_harmless_responses)} modified responses for harmless questions")

            category_safe_name = category_name.replace('/', '_').replace(' ', '_')
            mmlu_block = None
            if MMLU_CONFIG["enabled"] and original_mmlu_result is not None:
                print("    Evaluating modified model on MMLU...")
                modified_mmlu_result = evaluate_model_on_mmlu(model, MMLU_CONFIG)
                mmlu_block = build_mmlu_result(
                    config=MMLU_CONFIG,
                    original_result=original_mmlu_result,
                    modified_result=modified_mmlu_result,
                    method_results_dir=BASELINE_RESULTS_DIR,
                    detail_prefix=f"mmlu_{category_safe_name}_{param_idx}",
                )
                if mmlu_block is not None:
                    print(
                        f"    MMLU accuracy original -> modified: "
                        f"{mmlu_block['original']['accuracy']:.4f} -> {mmlu_block['modified']['accuracy']:.4f}"
                    )

            benchmark_results = {}
            if benchmark_runner is not None:
                print("    Evaluating modified model on benchmarks...")
                benchmark_results = benchmark_runner.evaluate_modified(
                    model,
                    run_label=f"basic_refusal_{category_safe_name}_{param_key}",
                )

            original_scores, original_score_data_list = evaluate_responses(
                category_questions, original_responses, classifier_categories,
                description=f"original responses (experiment {param_key})",
            )
            modified_scores, modified_score_data_list = evaluate_responses(
                category_questions, modified_responses, classifier_categories,
                description=f"modified responses (experiment {param_key})",
            )

            if original_scores and modified_scores:
                print(f"    Mean score original: {np.mean(original_scores):.2f}")
                print(f"    Mean score modified: {np.mean(modified_scores):.2f}")

            locality_scores = []
            average_locality_change = None

            if EVALUATE_LOCALITY and harmless_eval_prompts and original_harmless_responses and modified_harmless_responses:
                locality_scores, average_locality_change = evaluate_locality(
                    harmless_eval_prompts, original_harmless_responses,
                    modified_harmless_responses, classifier_categories,
                )

            category_results[param_key] = {
                "hyperparameters": params_dict,
                "category": category_name,
                "questions": category_questions,
                "original_responses": original_responses,
                "modified_responses": modified_responses,
                "original_scores": original_scores,
                "modified_scores": modified_scores,
                "original_score_data": original_score_data_list,
                "modified_score_data": modified_score_data_list,
                "score_statistics": {
                    "original_mean": float(np.mean(original_scores)) if original_scores else None,
                    "original_median": float(np.median(original_scores)) if original_scores else None,
                    "modified_mean": float(np.mean(modified_scores)) if modified_scores else None,
                    "modified_median": float(np.median(modified_scores)) if modified_scores else None,
                    "original_count": len(original_scores),
                    "modified_count": len(modified_scores),
                },
            }

            if EVALUATE_LOCALITY:
                category_results[param_key]["harmless_eval_prompts"] = harmless_eval_prompts
                category_results[param_key]["original_harmless_responses"] = original_harmless_responses
                category_results[param_key]["modified_harmless_responses"] = modified_harmless_responses
                category_results[param_key]["locality_scores"] = locality_scores
                category_results[param_key]["average_locality_change"] = float(average_locality_change) if average_locality_change is not None else None
                category_results[param_key]["locality_statistics"] = {
                    "original_mean": float(np.mean([s["original_score"] for s in locality_scores])) if locality_scores else None,
                    "modified_mean": float(np.mean([s["modified_score"] for s in locality_scores])) if locality_scores else None,
                    "average_change": float(average_locality_change) if average_locality_change is not None else None,
                    "count": len(locality_scores),
                }

            category_results[param_key]["timestamp"] = datetime.now().isoformat()
            if mmlu_block is not None:
                category_results[param_key]["mmlu"] = mmlu_block
            if benchmark_results:
                category_results[param_key]["benchmarks"] = benchmark_results

            answers_file = ANSWERS_DIR / f"answers_{category_safe_name}_{param_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            answers_data = {
                "experiment_info": {
                    "category": category_name,
                    "hyperparameters": params_dict,
                    "param_key": param_key,
                    "timestamp": datetime.now().isoformat(),
                },
                "harmful_questions": {
                    "questions": category_questions,
                    "original_responses": original_responses,
                    "modified_responses": modified_responses,
                    "original_evaluations": original_score_data_list,
                    "modified_evaluations": modified_score_data_list,
                },
            }
            if mmlu_block is not None:
                answers_data["mmlu"] = mmlu_block
            if benchmark_results:
                answers_data["benchmarks"] = benchmark_results
            if EVALUATE_LOCALITY:
                answers_data["harmless_questions"] = {
                    "questions": harmless_eval_prompts,
                    "original_responses": original_harmless_responses,
                    "modified_responses": modified_harmless_responses,
                    "locality_scores": locality_scores,
                    "average_locality_change": float(average_locality_change) if average_locality_change is not None else None,
                }
            with open(answers_file, "w", encoding="utf-8") as f:
                json.dump(answers_data, f, indent=2, ensure_ascii=False)
            print(f"    All responses and scores saved to: {answers_file}")

        category_safe_name = category_name.replace('/', '_').replace(' ', '_')

        print(f"\n  Creating plots for category {category_name}...")
        all_original_scores = []
        all_modified_scores = []
        all_locality_scores = []
        heatmap_harm = []
        heatmap_loc = []

        for result_data in category_results.values():
            hp = result_data.get("hyperparameters", {})
            orig = result_data.get("original_scores", [])
            mod = result_data.get("modified_scores", [])
            all_original_scores.extend(orig)
            all_modified_scores.extend(mod)
            if mod:
                heatmap_harm.append({
                    "max_weight": hp.get("max_weight"),
                    "min_weight": hp.get("min_weight"),
                    "mean_modified_score": np.mean(mod),
                })
            if EVALUATE_LOCALITY:
                loc = result_data.get("locality_scores", [])
                all_locality_scores.extend(loc)
                avg_loc = result_data.get("average_locality_change")
                if avg_loc is not None:
                    heatmap_loc.append({
                        "max_weight": hp.get("max_weight"),
                        "min_weight": hp.get("min_weight"),
                        "average_locality_change": avg_loc,
                    })

        evaluator_name = JUDGE_MODEL
        method_name = "basic_refusal"

        if heatmap_harm:
            fname = generate_plot_filename("harmfulness_heatmap", category_name, evaluator_name)
            plot_harmfulness_heatmap(
                heatmap_harm, HARMFULNESS_VALUES_DIR / fname,
                evaluator_name=evaluator_name, method_name=method_name,
                category_name=category_name,
            )

        if EVALUATE_LOCALITY and heatmap_loc:
            fname = generate_plot_filename("locality_heatmap", category_name, evaluator_name)
            plot_locality_heatmap(
                heatmap_loc, LOCALITY_VALUES_DIR / fname,
                evaluator_name=evaluator_name, method_name=method_name,
                category_name=category_name,
            )

        if all_original_scores or all_modified_scores:
            fname = generate_plot_filename("harmfulness_distribution", category_name, evaluator_name, ext=".png")
            plot_harmfulness_distribution(
                all_original_scores, all_modified_scores,
                HARMFULNESS_DISTRIBUTION_DIR / fname,
                evaluator_name=evaluator_name, method_name=method_name,
                category_name=category_name,
            )

        if all_locality_scores:
            orig_loc = [s["original_score"] for s in all_locality_scores]
            mod_loc = [s["modified_score"] for s in all_locality_scores]
            diffs = [s["difference"] for s in all_locality_scores if s["difference"] is not None]
            if diffs:
                fname = generate_plot_filename("locality_distribution", category_name, evaluator_name, ext=".png")
                plot_locality_distribution(
                    orig_loc, mod_loc, diffs,
                    LOCALITY_DISTRIBUTION_DIR / fname,
                    evaluator_name=evaluator_name, method_name=method_name,
                    category_name=category_name,
                )

        print(f"  Plots for category {category_name} created")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to directory: {BASELINE_RESULTS_DIR}")


if __name__ == "__main__":
    main()
