#!/usr/bin/env python3
"""
Tag ablation experiment: iterate over tags and abliteration hyperparameters.

Uses tags from tag_filtered_questions.json and computes a separate refusal
direction for each tag.
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
    MODEL_NAME, GOOD_PROMPTS_DATASET, RESULTS_DIR,
    JUDGE_API_URL, CLASSIFIER_API_URL, JUDGE_MODEL, FEW_SHOTS_PATH,
    HARMLESS_EVAL_DATASET, EVALUATE_LOCALITY, TAG_FILTERED_QUESTIONS_FILE,
)
from data_utils import extract_response_after_think
from refusal_directions import compute_refusal_direction
from model_utils import apply_abliteration_with_hyperparams
from evaluate.metrics import evaluate_responses, evaluate_locality
from visualization.plots import (
    plot_harmfulness_heatmap, plot_locality_heatmap,
    plot_harmfulness_distribution, plot_locality_distribution,
    generate_plot_filename,
)
from baselines.hyperparams import HYPERPARAMS

BASELINE_RESULTS_DIR = RESULTS_DIR / "tag_ablation"
BASELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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


def load_tags_and_questions() -> Dict[str, List[str]]:
    """Load tags and questions from tag_filtered_questions.json."""
    if not TAG_FILTERED_QUESTIONS_FILE.exists():
        print(f"ERROR: File {TAG_FILTERED_QUESTIONS_FILE} not found!")
        return {}

    try:
        with open(TAG_FILTERED_QUESTIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print(f"ERROR: Invalid data format in file {TAG_FILTERED_QUESTIONS_FILE}")
            return {}

        tags_questions = {}
        for tag, questions in data.items():
            if isinstance(questions, list):
                tags_questions[tag] = questions
            else:
                print(f"  Warning: tag {tag} does not contain a list of questions")

        print(f"Loaded {len(tags_questions)} tags from {TAG_FILTERED_QUESTIONS_FILE}")
        for tag, questions in tags_questions.items():
            print(f"  {tag}: {len(questions)} questions")

        return tags_questions
    except Exception as e:
        print(f"ERROR loading file {TAG_FILTERED_QUESTIONS_FILE}: {e}")
        return {}


def main():
    print("=" * 80)
    print("STARTING EXPERIMENT: TAG AND HYPERPARAMETER GRID SEARCH")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL_NAME}")
    print(f"Hyperparameters: {HYPERPARAMS}")
    print()

    print("Loading tags and questions from tag_filtered_questions.json...")
    tags_questions_dict = load_tags_and_questions()

    if not tags_questions_dict:
        print("ERROR: Failed to load tags and questions!")
        return

    harmful_tags = list(tags_questions_dict.keys())
    print(f"\nLoaded tags: {len(harmful_tags)}")
    print(f"Tags: {harmful_tags}")
    print()

    print("Loading model...")
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

    harmless_eval_prompts = []
    if EVALUATE_LOCALITY:
        print("\nLoading harmless questions for locality...")
        harmless_eval_prompts = load_prompts(HARMLESS_EVAL_DATASET)
        print(f"Loaded {len(harmless_eval_prompts)} harmless questions")

    param_names = list(HYPERPARAMS.keys())
    param_values = list(HYPERPARAMS.values())
    param_combinations = list(product(*param_values))

    print(f"\nTotal hyperparameter combinations: {len(param_combinations)}")
    print(f"Total tags: {len(harmful_tags)}")
    print(f"Total experiments: {len(harmful_tags) * len(param_combinations)}")
    print()

    for tag_idx, tag_name in enumerate(harmful_tags, 1):
        print("\n" + "=" * 80)
        print(f"TAG {tag_idx}/{len(harmful_tags)}: {tag_name}")
        print("=" * 80)

        tag_questions = tags_questions_dict.get(tag_name, [])
        if not tag_questions:
            print(f"  Skipping tag {tag_name}: no questions")
            continue

        print(f"  Questions for tag: {len(tag_questions)}")

        print(f"\n  Computing refusal direction for tag {tag_name}...")
        refusal_directions = compute_refusal_direction(model, [tag_name], good_prompts)
        print(f"  refusal_directions shape: {refusal_directions.shape}")

        print(f"\n  Getting original responses for tag questions...")
        original_responses_raw = model.get_responses_batched(tag_questions)
        original_responses = [extract_response_after_think(resp) for resp in original_responses_raw]
        print(f"  Got {len(original_responses)} original responses")

        original_harmless_responses = []
        if EVALUATE_LOCALITY and harmless_eval_prompts:
            print(f"\n  Getting original responses for harmless questions...")
            original_harmless_responses_raw = model.get_responses_batched(harmless_eval_prompts)
            original_harmless_responses = [extract_response_after_think(resp) for resp in original_harmless_responses_raw]
            print(f"  Got {len(original_harmless_responses)} original responses for harmless questions")

        tag_results = {}

        for param_idx, param_combo in enumerate(param_combinations, 1):
            params_dict = dict(zip(param_names, param_combo))
            param_key = "_&".join([f"{k}={v}" for k, v in params_dict.items()])

            print(f"\n  {'-'*60}")
            print(f"  Experiment {param_idx}/{len(param_combinations)} for tag {tag_name}")
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

            print(f"    Getting responses for {len(tag_questions)} tag questions...")
            modified_responses_raw = model.get_responses_batched(tag_questions)
            modified_responses = [extract_response_after_think(resp) for resp in modified_responses_raw]
            print(f"    Got {len(modified_responses)} responses")

            modified_harmless_responses = []
            if EVALUATE_LOCALITY and harmless_eval_prompts:
                print(f"    Getting modified responses for harmless questions...")
                modified_harmless_responses_raw = model.get_responses_batched(harmless_eval_prompts)
                modified_harmless_responses = [extract_response_after_think(resp) for resp in modified_harmless_responses_raw]
                print(f"    Got {len(modified_harmless_responses)} modified responses for harmless questions")

            original_scores, original_score_data_list = evaluate_responses(
                tag_questions, original_responses, classifier_categories,
                description=f"original responses (experiment {param_key})",
            )
            modified_scores, modified_score_data_list = evaluate_responses(
                tag_questions, modified_responses, classifier_categories,
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

            tag_results[param_key] = {
                "hyperparameters": params_dict,
                "tag": tag_name,
                "questions": tag_questions,
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
                tag_results[param_key]["harmless_eval_prompts"] = harmless_eval_prompts
                tag_results[param_key]["original_harmless_responses"] = original_harmless_responses
                tag_results[param_key]["modified_harmless_responses"] = modified_harmless_responses
                tag_results[param_key]["locality_scores"] = locality_scores
                tag_results[param_key]["average_locality_change"] = float(average_locality_change) if average_locality_change is not None else None
                tag_results[param_key]["locality_statistics"] = {
                    "original_mean": float(np.mean([s["original_score"] for s in locality_scores])) if locality_scores else None,
                    "modified_mean": float(np.mean([s["modified_score"] for s in locality_scores])) if locality_scores else None,
                    "average_change": float(average_locality_change) if average_locality_change is not None else None,
                    "count": len(locality_scores),
                }

            tag_results[param_key]["timestamp"] = datetime.now().isoformat()

            tag_safe_name = tag_name.replace('/', '_').replace(' ', '_')
            answers_file = ANSWERS_DIR / f"answers_tag_{tag_safe_name}_{param_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            answers_data = {
                "experiment_info": {
                    "tag": tag_name,
                    "hyperparameters": params_dict,
                    "param_key": param_key,
                    "timestamp": datetime.now().isoformat(),
                },
                "harmful_questions": {
                    "questions": tag_questions,
                    "original_responses": original_responses,
                    "modified_responses": modified_responses,
                    "original_evaluations": original_score_data_list,
                    "modified_evaluations": modified_score_data_list,
                },
            }
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

        tag_safe_name = tag_name.replace('/', '_').replace(' ', '_')

        print(f"\n  Creating plots for tag {tag_name}...")
        all_original_scores = []
        all_modified_scores = []
        all_locality_scores = []
        heatmap_harm = []
        heatmap_loc = []

        for result_data in tag_results.values():
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
        method_name = "tag_ablation"

        if heatmap_harm:
            fname = generate_plot_filename("harmfulness_heatmap", tag_name, evaluator_name)
            plot_harmfulness_heatmap(
                heatmap_harm, HARMFULNESS_VALUES_DIR / fname,
                evaluator_name=evaluator_name, method_name=method_name,
                category_name=tag_name,
            )

        if EVALUATE_LOCALITY and heatmap_loc:
            fname = generate_plot_filename("locality_heatmap", tag_name, evaluator_name)
            plot_locality_heatmap(
                heatmap_loc, LOCALITY_VALUES_DIR / fname,
                evaluator_name=evaluator_name, method_name=method_name,
                category_name=tag_name,
            )

        if all_original_scores or all_modified_scores:
            fname = generate_plot_filename("harmfulness_distribution", tag_name, evaluator_name, ext=".png")
            plot_harmfulness_distribution(
                all_original_scores, all_modified_scores,
                HARMFULNESS_DISTRIBUTION_DIR / fname,
                evaluator_name=evaluator_name, method_name=method_name,
                category_name=tag_name,
            )

        if all_locality_scores:
            orig_loc = [s["original_score"] for s in all_locality_scores]
            mod_loc = [s["modified_score"] for s in all_locality_scores]
            diffs = [s["difference"] for s in all_locality_scores if s["difference"] is not None]
            if diffs:
                fname = generate_plot_filename("locality_distribution", tag_name, evaluator_name, ext=".png")
                plot_locality_distribution(
                    orig_loc, mod_loc, diffs,
                    LOCALITY_DISTRIBUTION_DIR / fname,
                    evaluator_name=evaluator_name, method_name=method_name,
                    category_name=tag_name,
                )

        print(f"  Plots for tag {tag_name} created")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to directory: {BASELINE_RESULTS_DIR}")


if __name__ == "__main__":
    main()
