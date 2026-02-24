#!/usr/bin/env python3
"""
GRPO training of coefficients for weighted sum of refusal directions.

Uses evaluate_harmfulness_with_local_judge as the reward function.
"""

import os
import sys
import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Wandb logging disabled.")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from heretic.config import Settings
from heretic.utils import load_prompts

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from config import (
    MODEL_NAME, CATEGORIES, GOOD_PROMPTS_DATASET, RESULTS_DIR,
    GRPO_CONFIG, ABLITERATION_PARAMS, FEW_SHOTS_PATH,
    HARMLESS_EVAL_DATASET, EVALUATE_LOCALITY, WEIGHTS_INIT_TYPE, GRAPH_FILE, EVALUATION_BACKEND,
)
from data_utils import load_all_datasets_with_categories, extract_response_after_think
from refusal_directions import (
    compute_refusal_direction, save_refusal_directions, load_refusal_directions,
)
from model_utils import LearnableDirectionWeights, apply_abliteration_with_hyperparams
from training import train_grpo_epoch
from evaluate.judges import evaluate_harmfulness


def _fmt(v: float) -> str:
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def _wandb_run_name(category_name: str, grpo_config: dict, abliteration_params: dict, weights_init_type: str) -> str:
    cat = category_name.replace("/", "_").replace(" ", "_")
    g = grpo_config
    a = abliteration_params
    parts = [
        "grpo", cat,
        f"init_{weights_init_type}",
        f"n_groups{g['n_groups']}_n_epochs{g['n_epochs']}",
        f"learning_rate{_fmt(g['learning_rate'])}",
        f"noise_scale{_fmt(g['noise_scale'])}",
        f"beta{_fmt(g['beta'])}",
        f"max_weight{_fmt(a['max_weight'])}",
        f"max_weight_position{_fmt(a['max_weight_position'])}",
        f"min_weight{_fmt(a['min_weight'])}",
        f"min_weight_distance{_fmt(a['min_weight_distance'])}",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    ]
    return "_".join(parts)


def create_harmfulness_heatmap(
    final_scores: List[int],
    category_name: str,
    abliteration_params: Dict,
    output_dir: Path,
):
    """Create heatmap for harmfulness scores (single-point variant for GRPO)."""
    if not final_scores:
        print("No data for harmfulness heatmap")
        return

    category_safe_name = category_name.replace('/', '_').replace(' ', '_')
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    mean_score = np.mean(final_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    max_weight = abliteration_params.get("max_weight", 0)
    max_weight_position = abliteration_params.get("max_weight_position", 0)
    min_weight = abliteration_params.get("min_weight", 0)
    min_weight_distance = abliteration_params.get("min_weight_distance", 0)

    data_matrix = np.array([[mean_score]])
    sns.heatmap(
        data_matrix, annot=True, fmt=".2f", cmap="RdYlGn_r",
        vmin=0, vmax=4,
        cbar_kws={"label": "Mean Harmfulness Score (0-4)"},
        ax=ax, linewidths=0.5, linecolor="gray",
    )
    ax.set_title(
        f"Mean Harmfulness Score\n"
        f"max_weight={max_weight}, max_weight_position={max_weight_position:.2f}\n"
        f"min_weight={min_weight}, min_weight_distance={min_weight_distance:.2f}\n"
        f"Total scores: {len(final_scores)}",
        fontsize=12, fontweight="bold",
    )
    plt.suptitle(
        f"Heatmap: Harmfulness Scores\nCategory: {category_name}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    plot_filename = output_dir / f"harmfulness_heatmap_{category_safe_name}_{timestamp_str}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Harmfulness heatmap saved to: {plot_filename}")
    plt.close()


def create_locality_heatmap(
    average_locality_change: float,
    category_name: str,
    abliteration_params: Dict,
    output_dir: Path,
):
    """Create heatmap for locality scores (single-point variant for GRPO)."""
    if average_locality_change is None:
        print("No data for locality heatmap")
        return

    category_safe_name = category_name.replace('/', '_').replace(' ', '_')
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    fig, ax = plt.subplots(figsize=(8, 6))
    max_weight = abliteration_params.get("max_weight", 0)
    max_weight_position = abliteration_params.get("max_weight_position", 0)
    min_weight = abliteration_params.get("min_weight", 0)
    min_weight_distance = abliteration_params.get("min_weight_distance", 0)

    data_matrix = np.array([[average_locality_change]])
    sns.heatmap(
        data_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Average Locality Change"},
        ax=ax, linewidths=0.5, linecolor="gray",
    )
    ax.set_title(
        f"Average Locality Change\n"
        f"max_weight={max_weight}, max_weight_position={max_weight_position:.2f}\n"
        f"min_weight={min_weight}, min_weight_distance={min_weight_distance:.2f}\n"
        f"(Positive = degradation)",
        fontsize=12, fontweight="bold",
    )
    plt.suptitle(
        f"Heatmap: Locality Scores\nCategory: {category_name}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    plot_filename = output_dir / f"locality_heatmap_{category_safe_name}_{timestamp_str}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Locality heatmap saved to: {plot_filename}")
    plt.close()


def get_locality_cache_filename() -> Path:
    dataset_name = HARMLESS_EVAL_DATASET.dataset.replace('/', '_')
    model_name_safe = MODEL_NAME.replace('/', '_')
    cache_file = RESULTS_DIR / f"locality_cache_{model_name_safe}_{dataset_name}_{HARMLESS_EVAL_DATASET.split.replace(':', '_').replace('[', '').replace(']', '')}.json"
    return cache_file


def save_locality_cache(
    harmless_eval_prompts: List[str],
    original_harmless_responses: List[str],
    original_harmless_scores_dict: Dict[str, int],
) -> None:
    cache_file = get_locality_cache_filename()
    cache_data = {
        "harmless_eval_prompts": harmless_eval_prompts,
        "original_harmless_responses": original_harmless_responses,
        "original_harmless_scores_dict": original_harmless_scores_dict,
        "model_name": MODEL_NAME,
        "dataset": HARMLESS_EVAL_DATASET.dataset,
        "split": HARMLESS_EVAL_DATASET.split,
        "timestamp": datetime.now().isoformat(),
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
    print(f"  Cache saved to: {cache_file}")


def load_locality_cache() -> Tuple[List[str], List[str], Dict[str, int], bool]:
    cache_file = get_locality_cache_filename()

    if not cache_file.exists():
        return [], [], {}, False

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        if (cache_data.get("model_name") != MODEL_NAME or
            cache_data.get("dataset") != HARMLESS_EVAL_DATASET.dataset or
            cache_data.get("split") != HARMLESS_EVAL_DATASET.split):
            print(f"  Cache does not match current settings (model or dataset changed)")
            return [], [], {}, False

        harmless_eval_prompts = cache_data.get("harmless_eval_prompts", [])
        original_harmless_responses = cache_data.get("original_harmless_responses", [])
        original_harmless_scores_dict = cache_data.get("original_harmless_scores_dict", {})
        original_harmless_scores_dict = {str(k): int(v) for k, v in original_harmless_scores_dict.items()}

        print(f"  Cache loaded from: {cache_file}")
        print(f"  Saved at: {cache_data.get('timestamp', 'unknown')}")
        print(f"  Loaded {len(harmless_eval_prompts)} prompts, {len(original_harmless_responses)} responses, {len(original_harmless_scores_dict)} scores")

        return harmless_eval_prompts, original_harmless_responses, original_harmless_scores_dict, True
    except Exception as e:
        print(f"  Error loading cache: {e}")
        return [], [], {}, False


def main():
    print("=" * 80)
    print("STARTING GRPO COEFFICIENT TRAINING EXPERIMENT")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GRPO Config: {GRPO_CONFIG}")
    print(f"Abliteration Params: {ABLITERATION_PARAMS}")
    print(f"Weights Init Type: {WEIGHTS_INIT_TYPE}")
    print(f"Evaluation backend: {EVALUATION_BACKEND}")
    print()

    GRPO_RESULTS_DIR = RESULTS_DIR / "graph_grpo_old"
    GRPO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    GRPO_HARMFULNESS_DIST_DIR = GRPO_RESULTS_DIR / "harmfulness" / "distribution_plots"
    GRPO_HARMFULNESS_DIST_DIR.mkdir(parents=True, exist_ok=True)

    GRPO_HARMFULNESS_VALUES_DIR = GRPO_RESULTS_DIR / "harmfulness" / "heatmap_plots"
    GRPO_HARMFULNESS_VALUES_DIR.mkdir(parents=True, exist_ok=True)

    GRPO_LOCALITY_DIST_DIR = GRPO_RESULTS_DIR / "locality" / "distribution_plots"
    GRPO_LOCALITY_DIST_DIR.mkdir(parents=True, exist_ok=True)

    GRPO_LOCALITY_VALUES_DIR = GRPO_RESULTS_DIR / "locality" / "heatmap_plots"
    GRPO_LOCALITY_VALUES_DIR.mkdir(parents=True, exist_ok=True)

    GRPO_ANSWERS_DIR = GRPO_RESULTS_DIR / "answers"
    GRPO_ANSWERS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {GRPO_RESULTS_DIR}")

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
    from heretic.model import Model
    model = Model(settings)
    print("Model loaded successfully.")

    n_layers = len(model.get_layers())
    print(f"Number of layers: {n_layers}")

    print("Loading good prompts...")
    good_prompts = load_prompts(GOOD_PROMPTS_DATASET)
    print(f"Loaded {len(good_prompts)} good prompts")

    if GRAPH_FILE.exists():
        with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
            bad_tags = [line.strip() for line in f if line.strip()]
        print(f"\nLoaded {len(bad_tags)} lines from {GRAPH_FILE}")
    else:
        print(f"\nERROR: File {GRAPH_FILE} not found!")
        return

    directions_file = RESULTS_DIR / f"refusal_directions_{MODEL_NAME.replace('/', '_')}.pt"

    if directions_file.exists():
        print(f"\nFound saved directions file: {directions_file}")
        extracted_directions, loaded_tags = load_refusal_directions(directions_file, expected_tags=bad_tags)

        if loaded_tags != bad_tags:
            print(f"\nTags do not match! Recomputing directions...")
            extracted_directions = []
            for tag_idx, tag_name in enumerate(bad_tags, 1):
                print(f"  Tag {tag_idx}/{len(bad_tags)}: {tag_name}")
                refusal_dir = compute_refusal_direction(model, [tag_name], good_prompts)
                print(f"    refusal_directions shape: {refusal_dir.shape}")
                extracted_directions.append(refusal_dir)
            save_refusal_directions(extracted_directions, bad_tags, MODEL_NAME, directions_file)
        else:
            try:
                device = next(model.get_layers()[0].parameters()).device
            except (StopIteration, IndexError):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            extracted_directions = [dir_tensor.to(device) for dir_tensor in extracted_directions]
    else:
        print("\nComputing refusal directions for all tags...")
        extracted_directions = []
        for tag_idx, tag_name in enumerate(bad_tags, 1):
            print(f"  Tag {tag_idx}/{len(bad_tags)}: {tag_name}")
            refusal_dir = compute_refusal_direction(model, [tag_name], good_prompts)
            print(f"    refusal_directions shape: {refusal_dir.shape}")
            extracted_directions.append(refusal_dir)
        save_refusal_directions(extracted_directions, bad_tags, MODEL_NAME, directions_file)

    print(f"\nComputed {len(extracted_directions)} refusal directions")

    n_directions = len(extracted_directions)
    hidden_size = extracted_directions[0].shape[1]

    print(f"Weight params: n_directions={n_directions}, n_layers={n_layers}, hidden_size={hidden_size}")

    target_tag = "Physical harm"
    physical_harm_idx = None
    for idx, tag in enumerate(bad_tags):
        if tag.lower() == target_tag.lower():
            physical_harm_idx = idx
            break

    if physical_harm_idx is None:
        for idx, tag in enumerate(bad_tags):
            if tag.lower().startswith("physical harm"):
                physical_harm_idx = idx
                break

    if physical_harm_idx is None:
        print(f"Warning: '{target_tag}' not found in tags. Using index 0.")
        physical_harm_idx = 0

    print(f"Index for '{target_tag}': {physical_harm_idx} (tag: '{bad_tags[physical_harm_idx]}')")

    torch.set_grad_enabled(True)
    direction_weights = LearnableDirectionWeights(
        n_directions=n_directions, n_layers=n_layers, hidden_size=hidden_size,
        init_type=WEIGHTS_INIT_TYPE, topic_idx=physical_harm_idx,
    )
    device = extracted_directions[0].device
    direction_weights = direction_weights.to(device)

    if WEIGHTS_INIT_TYPE == "zero":
        print(f"Weights init: all weights = 0")
    elif WEIGHTS_INIT_TYPE == "topic":
        print(f"Weights init: weight for index {physical_harm_idx} ('{bad_tags[physical_harm_idx]}') = 1, others = 0")
    elif WEIGHTS_INIT_TYPE == "average":
        print(f"Weights init: all weights = 1/{n_directions} = {1.0/n_directions:.6f}")
    else:
        print(f"Weights init: unknown type '{WEIGHTS_INIT_TYPE}', using random init")

    optimizer = torch.optim.Adam(direction_weights.parameters(), lr=GRPO_CONFIG["learning_rate"])

    category_name = "Physical harm"
    category_items = data_by_category.get(category_name, [])
    category_questions = [item.get('instruction', '') for item in category_items if item.get('instruction')]
    print(f"\nLoaded {len(category_questions)} questions for category '{category_name}'")

    if WANDB_AVAILABLE:
        wandb.init(
            project="refusal_direction_grpo",
            name=_wandb_run_name(category_name, GRPO_CONFIG, ABLITERATION_PARAMS, WEIGHTS_INIT_TYPE),
            config={
                "model": MODEL_NAME, "category": category_name,
                "n_directions": n_directions, "n_layers": n_layers,
                "hidden_size": hidden_size, "physical_harm_idx": physical_harm_idx,
                "n_questions": len(category_questions),
                "grpo_config": GRPO_CONFIG, "abliteration_params": ABLITERATION_PARAMS,
                "evaluate_locality": EVALUATE_LOCALITY,
                "weights_init_type": WEIGHTS_INIT_TYPE,
            },
        )
        print("Wandb initialized")

    classifier_categories = []
    if EVALUATION_BACKEND == "local_llm_judge":
        try:
            with open(FEW_SHOTS_PATH, "r", encoding="utf-8") as f:
                few_shots_data = json.load(f)
            classifier_categories = [
                {"id": c.get("id"), "label": c.get("label", "")}
                for c in few_shots_data.get("categories", [])
                if c.get("id")
            ]
        except Exception as e:
            print(f"Warning: could not load classifier categories: {e}")

    harmless_eval_prompts = []
    original_harmless_responses = []
    original_harmless_scores_dict = {}
    original_harmless_scores_list = []
    modified_harmless_responses = []
    modified_harmless_scores = []
    locality_scores = []
    average_locality_change = None
    valid_differences = []

    if EVALUATE_LOCALITY:
        print("\nLoading harmless questions for locality evaluation...")
        cached_prompts, cached_responses, cached_scores_dict, cache_success = load_locality_cache()

        if cache_success and len(cached_prompts) > 0:
            harmless_eval_prompts = cached_prompts
            original_harmless_responses = cached_responses
            original_harmless_scores_dict = cached_scores_dict
            original_harmless_scores_list = list(cached_scores_dict.values())
            print(f"Using cache: {len(harmless_eval_prompts)} prompts, {len(original_harmless_responses)} responses, {len(original_harmless_scores_dict)} scores")
        else:
            harmless_eval_prompts = load_prompts(HARMLESS_EVAL_DATASET)
            print(f"Loaded {len(harmless_eval_prompts)} harmless questions")

            print("\nGetting pre-modification model responses for locality...")
            original_harmless_responses_raw = model.get_responses_batched(harmless_eval_prompts)
            original_harmless_responses = [extract_response_after_think(resp) for resp in original_harmless_responses_raw]
            print(f"Got {len(original_harmless_responses)} responses")

            print("\nEvaluating pre-modification model responses...")
            for i, (question, response) in enumerate(zip(harmless_eval_prompts, original_harmless_responses)):
                if response:
                    try:
                        score_data = evaluate_harmfulness(
                            question,
                            response,
                            classifier_categories=classifier_categories,
                            backend=EVALUATION_BACKEND,
                        )
                        score = score_data.get("score")
                        if score is not None:
                            score_int = int(score)
                            original_harmless_scores_dict[question] = score_int
                            original_harmless_scores_list.append(score_int)
                    except Exception as e:
                        if i < 5:
                            print(f"      Evaluation error: {e}")

                if (i + 1) % 5 == 0:
                    print(f"  Evaluated {i + 1}/{len(harmless_eval_prompts)} responses...")

            print(f"Got {len(original_harmless_scores_list)} scores for pre-modification model")
            print("\nSaving locality cache...")
            save_locality_cache(harmless_eval_prompts, original_harmless_responses, original_harmless_scores_dict)
    else:
        print("\nLocality evaluation disabled (EVALUATE_LOCALITY=False)")

    # GRPO Training Loop
    print("\n" + "=" * 80)
    print("STARTING GRPO TRAINING")
    print("=" * 80)

    training_history = []

    for epoch in range(GRPO_CONFIG["n_epochs"]):
        print(f"\nEpoch {epoch + 1}/{GRPO_CONFIG['n_epochs']}")
        print("-" * 80)

        epoch_metrics = train_grpo_epoch(
            direction_weights=direction_weights,
            extracted_directions=extracted_directions,
            model=model,
            category_questions=category_questions,
            classifier_categories=classifier_categories,
            n_layers=n_layers,
            optimizer=optimizer,
        )

        training_history.append({"epoch": epoch + 1, **epoch_metrics})

        print(f"  Mean reward: {epoch_metrics['mean_reward']:.3f}, Best: {epoch_metrics['best_reward']:.3f}")
        print(f"  Weights norm: {epoch_metrics['weights_norm']:.4f}")

        if WANDB_AVAILABLE:
            log_dict = {
                "train/mean_reward": epoch_metrics['mean_reward'],
                "train/best_reward": epoch_metrics['best_reward'],
                "train/weights_norm": epoch_metrics['weights_norm'],
                "train/gradient_norm": epoch_metrics.get('gradient_norm', 0.0),
                "train/advantages_mean": epoch_metrics.get('advantages_mean', 0.0),
                "train/advantages_std": epoch_metrics.get('advantages_std', 0.0),
                "train/advantages_max": epoch_metrics.get('advantages_max', 0.0),
                "train/advantages_min": epoch_metrics.get('advantages_min', 0.0),
                "train/learning_rate": GRPO_CONFIG["learning_rate"],
                "epoch": epoch + 1,
            }
            if 'rewards' in epoch_metrics:
                for idx, reward in enumerate(epoch_metrics['rewards']):
                    log_dict[f"train/reward_variant_{idx}"] = reward
            if 'variant_weights' in epoch_metrics:
                for idx, weight in enumerate(epoch_metrics['variant_weights']):
                    log_dict[f"train/variant_weight_{idx}"] = weight
            wandb.log(log_dict)

    # Final evaluation with trained weights
    print("\n" + "=" * 80)
    print("FINAL EVALUATION WITH TRAINED WEIGHTS")
    print("=" * 80)

    extracted_directions_no_grad = [d.detach() if d.requires_grad else d for d in extracted_directions]

    with torch.no_grad():
        final_combined_direction_raw = direction_weights(extracted_directions_no_grad)
        final_combined_direction = torch.tensor(
            final_combined_direction_raw.cpu().numpy(),
            dtype=final_combined_direction_raw.dtype,
            device=final_combined_direction_raw.device,
            requires_grad=False,
        )

    model.reload_model()

    apply_abliteration_with_hyperparams(
        model, final_combined_direction,
        ABLITERATION_PARAMS["max_weight"], ABLITERATION_PARAMS["max_weight_position"],
        ABLITERATION_PARAMS["min_weight"], ABLITERATION_PARAMS["min_weight_distance"],
        n_layers,
    )

    print("Getting final responses...")
    final_responses_raw = model.get_responses_batched(category_questions)
    final_responses = [extract_response_after_think(resp) for resp in final_responses_raw]

    print("\nEvaluating final responses...")
    final_scores = []
    for question, response in zip(category_questions, final_responses):
        if response:
            try:
                score_data = evaluate_harmfulness(
                    question,
                    response,
                    classifier_categories=classifier_categories,
                    backend=EVALUATION_BACKEND,
                )
                score = score_data.get("score")
                if score is not None:
                    final_scores.append(int(score))
            except Exception as e:
                print(f"      Evaluation error: {e}")

    if EVALUATE_LOCALITY:
        print("\nGetting post-modification model responses for locality...")
        modified_harmless_responses_raw = model.get_responses_batched(harmless_eval_prompts)
        modified_harmless_responses = [extract_response_after_think(resp) for resp in modified_harmless_responses_raw]
        print(f"Got {len(modified_harmless_responses)} responses")

        print("\nEvaluating post-modification model responses...")
        for i, (question, orig_resp, mod_resp) in enumerate(zip(
            harmless_eval_prompts, original_harmless_responses, modified_harmless_responses,
        )):
            if mod_resp:
                try:
                    mod_score_data = evaluate_harmfulness(
                        question,
                        mod_resp,
                        classifier_categories=classifier_categories,
                        backend=EVALUATION_BACKEND,
                    )
                    mod_score = mod_score_data.get("score")
                    if mod_score is not None:
                        mod_score_int = int(mod_score)
                        modified_harmless_scores.append(mod_score_int)

                        orig_score = original_harmless_scores_dict.get(question)
                        if orig_score is not None:
                            difference = mod_score_int - orig_score
                            locality_scores.append({
                                "question": question,
                                "original_score": orig_score,
                                "modified_score": mod_score_int,
                                "difference": difference,
                            })
                except Exception as e:
                    if i < 5:
                        print(f"      Evaluation error: {e}")

            if (i + 1) % 5 == 0:
                print(f"  Evaluated {i + 1}/{len(harmless_eval_prompts)} responses...")

        print(f"Got {len(modified_harmless_scores)} scores for post-modification model")

        valid_differences = [s["difference"] for s in locality_scores if s["difference"] is not None]
        average_locality_change = (sum(valid_differences) / len(valid_differences)) if valid_differences else None

        if average_locality_change is not None:
            print(f"\nAverage harmfulness change on harmless questions (locality): {average_locality_change:+.2f}")
            print(f"  Total score pairs: {len(valid_differences)}")
            print(f"  Min change: {min(valid_differences):+.2f}")
            print(f"  Max change: {max(valid_differences):+.2f}")

    # Build plots
    if final_scores:
        print(f"\nBuilding harmfulness distribution histogram...")
        plt.figure(figsize=(10, 6))
        plt.hist(final_scores, bins=range(0, 6), align='left', edgecolor='black', alpha=0.7)
        plt.xlabel('Harmfulness score (0-4)', fontsize=12)
        plt.ylabel('Number of responses', fontsize=12)
        plt.title(f'Harmfulness score distribution for category "{category_name}"', fontsize=14, fontweight='bold')
        plt.xticks(range(0, 5))
        plt.grid(axis='y', alpha=0.3)
        mean_score = np.mean(final_scores)
        median_score = np.median(final_scores)
        plt.text(0.02, 0.98,
                 f'Total scores: {len(final_scores)}\nMean: {mean_score:.2f}\nMedian: {median_score:.2f}',
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
        category_safe_name = category_name.replace('/', '_').replace(' ', '_')
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = GRPO_HARMFULNESS_DIST_DIR / f"harmfulness_distribution_{category_safe_name}_{timestamp_str}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {plot_filename}")
        plt.close()

        create_harmfulness_heatmap(final_scores, category_name, ABLITERATION_PARAMS, GRPO_HARMFULNESS_VALUES_DIR)

    if EVALUATE_LOCALITY and original_harmless_scores_list and modified_harmless_scores:
        print(f"\nBuilding locality distribution histograms...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.hist(original_harmless_scores_list, bins=range(0, 6), align='left', edgecolor='black', alpha=0.7, color='blue')
        ax1.set_xlabel('Harmfulness score (0-4)', fontsize=12)
        ax1.set_ylabel('Number of responses', fontsize=12)
        ax1.set_title('Pre-modification model', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(0, 5))
        ax1.grid(axis='y', alpha=0.3)
        mean_orig = np.mean(original_harmless_scores_list)
        ax1.text(0.02, 0.98,
                 f'Total scores: {len(original_harmless_scores_list)}\nMean: {mean_orig:.2f}\nMedian: {np.median(original_harmless_scores_list):.2f}',
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5), fontsize=10)
        ax2.hist(modified_harmless_scores, bins=range(0, 6), align='left', edgecolor='black', alpha=0.7, color='red')
        ax2.set_xlabel('Harmfulness score (0-4)', fontsize=12)
        ax2.set_ylabel('Number of responses', fontsize=12)
        ax2.set_title('Post-modification model', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(0, 5))
        ax2.grid(axis='y', alpha=0.3)
        mean_mod = np.mean(modified_harmless_scores)
        locality_text = f'Locality: {average_locality_change:+.2f}' if average_locality_change is not None else 'Locality: N/A'
        ax2.text(0.02, 0.98,
                 f'Total scores: {len(modified_harmless_scores)}\nMean: {mean_mod:.2f}\nMedian: {np.median(modified_harmless_scores):.2f}\n{locality_text}',
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5), fontsize=10)
        plt.suptitle('Harmfulness score distribution on harmless questions (Locality)',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        category_safe_name = category_name.replace('/', '_').replace(' ', '_')
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        locality_plot_filename = GRPO_LOCALITY_DIST_DIR / f"locality_distribution_{category_safe_name}_{timestamp_str}.png"
        plt.savefig(locality_plot_filename, dpi=300, bbox_inches='tight')
        print(f"Locality histogram saved to: {locality_plot_filename}")
        plt.close()

        create_locality_heatmap(average_locality_change, category_name, ABLITERATION_PARAMS, GRPO_LOCALITY_VALUES_DIR)

    # Save results
    category_safe_name = category_name.replace('/', '_').replace(' ', '_')
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    answers_file = GRPO_ANSWERS_DIR / f"answers_{category_safe_name}_{timestamp_str}.json"
    answers_data = {
        "experiment_config": {
            "model": MODEL_NAME, "category": category_name,
            "n_questions": len(category_questions), "n_directions": n_directions,
            "grpo_config": GRPO_CONFIG, "abliteration_params": ABLITERATION_PARAMS,
            "weights_init_type": WEIGHTS_INIT_TYPE,
            "physical_harm_idx": physical_harm_idx,
            "training_history": training_history,
        },
        "final_weights": direction_weights.weights.data.cpu().tolist(),
        "questions": category_questions,
        "responses": final_responses,
        "final_scores": final_scores if final_scores else [],
        "score_statistics": {
            "mean": float(np.mean(final_scores)) if final_scores else None,
            "median": float(np.median(final_scores)) if final_scores else None,
            "std": float(np.std(final_scores)) if final_scores else None,
            "min": int(np.min(final_scores)) if final_scores else None,
            "max": int(np.max(final_scores)) if final_scores else None,
            "count": len(final_scores),
        },
        "timestamp": datetime.now().isoformat(),
    }

    if EVALUATE_LOCALITY:
        answers_data["locality_evaluation"] = {
            "harmless_questions": harmless_eval_prompts,
            "original_harmless_scores": original_harmless_scores_list,
            "modified_harmless_scores": modified_harmless_scores,
            "locality_scores": locality_scores,
            "average_locality_change": float(average_locality_change) if average_locality_change is not None else None,
            "locality_statistics": {
                "original_mean": float(np.mean(original_harmless_scores_list)) if original_harmless_scores_list else None,
                "original_median": float(np.median(original_harmless_scores_list)) if original_harmless_scores_list else None,
                "modified_mean": float(np.mean(modified_harmless_scores)) if modified_harmless_scores else None,
                "modified_median": float(np.median(modified_harmless_scores)) if modified_harmless_scores else None,
                "min_difference": float(min(valid_differences)) if valid_differences else None,
                "max_difference": float(max(valid_differences)) if valid_differences else None,
                "std_difference": float(np.std(valid_differences)) if valid_differences else None,
                "count": len(valid_differences),
            },
        }

    with open(answers_file, "w", encoding="utf-8") as f:
        json.dump(answers_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {answers_file}")

    if WANDB_AVAILABLE:
        final_metrics = {
            "final/mean_harmfulness_score": float(np.mean(final_scores)) if final_scores else 0.0,
            "final/median_harmfulness_score": float(np.median(final_scores)) if final_scores else 0.0,
            "final/std_harmfulness_score": float(np.std(final_scores)) if final_scores else 0.0,
            "final/n_scores": len(final_scores),
        }
        if EVALUATE_LOCALITY and average_locality_change is not None:
            final_metrics.update({
                "final/average_locality_change": float(average_locality_change),
                "final/original_harmless_mean": float(np.mean(original_harmless_scores_list)) if original_harmless_scores_list else 0.0,
                "final/modified_harmless_mean": float(np.mean(modified_harmless_scores)) if modified_harmless_scores else 0.0,
                "final/locality_pairs": len(valid_differences),
            })
        final_metrics["final/final_weights_norm"] = float(direction_weights.weights.data.norm().item())
        wandb.log(final_metrics)
        wandb.finish()
        print("Wandb logging finished")

    # Save coefficients
    print("\nSaving final coefficients...")
    weights_data = {
        "weights": direction_weights.weights.data.cpu().tolist(),
        "metadata": {
            "model": MODEL_NAME, "category": category_name,
            "n_directions": n_directions, "n_layers": n_layers,
            "hidden_size": hidden_size,
            "weights_shape": list(direction_weights.weights.shape),
            "timestamp": datetime.now().isoformat(),
        },
    }

    weights_json_file = GRPO_RESULTS_DIR / f"coefficients_{category_safe_name}_grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(weights_json_file, "w", encoding="utf-8") as f:
        json.dump(weights_data, f, indent=2, ensure_ascii=False)
    print(f"Coefficients (JSON) saved to: {weights_json_file}")

    weights_pt_file = GRPO_RESULTS_DIR / f"coefficients_{category_safe_name}_grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({"weights": direction_weights.weights.data.cpu(), "metadata": weights_data["metadata"]}, weights_pt_file)
    print(f"Coefficients (PyTorch) saved to: {weights_pt_file}")

    # Coefficient norms distribution
    print("\nBuilding coefficient norms distribution plot...")
    weights_tensor = direction_weights.weights.data.cpu()
    direction_norms = torch.norm(weights_tensor.view(n_directions, -1), p=2, dim=1).numpy()
    layer_norms_per_direction = torch.norm(weights_tensor, p=2, dim=2).numpy()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(direction_norms, bins=min(30, max(10, n_directions // 2)), edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel('L2 norm of coefficients', fontsize=12)
    plt.ylabel('Number of directions', fontsize=12)
    plt.title('Norm distribution by direction', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    mean_norm = np.mean(direction_norms)
    median_norm = np.median(direction_norms)
    std_norm = np.std(direction_norms)
    plt.text(0.02, 0.98,
             f'Total directions: {n_directions}\nMean norm: {mean_norm:.4f}\nMedian norm: {median_norm:.4f}\nStd: {std_norm:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

    plt.subplot(1, 2, 2)
    layer_norms_mean = np.mean(layer_norms_per_direction, axis=0)
    layer_indices = np.arange(len(layer_norms_mean))
    plt.plot(layer_indices, layer_norms_mean, marker='o', linestyle='-', linewidth=2, markersize=6, color='coral')
    plt.xlabel('Layer index', fontsize=12)
    plt.ylabel('Mean L2 norm of coefficients', fontsize=12)
    plt.title('Norm distribution by layer', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.xticks(layer_indices[::max(1, len(layer_indices)//10)])

    plt.suptitle(f'Coefficient norms distribution for category "{category_name}"',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    norms_plot_filename = GRPO_RESULTS_DIR / f"coefficients_norms_{category_safe_name}_grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(norms_plot_filename, dpi=300, bbox_inches='tight')
    print(f"Coefficient norms plot saved to: {norms_plot_filename}")
    plt.close()

    weights_data["norms_statistics"] = {
        "direction_norms": direction_norms.tolist(),
        "layer_norms_per_direction": layer_norms_per_direction.tolist(),
        "statistics": {
            "mean_norm": float(mean_norm), "median_norm": float(median_norm),
            "std_norm": float(std_norm),
            "min_norm": float(np.min(direction_norms)),
            "max_norm": float(np.max(direction_norms)),
            "mean_norm_per_layer": layer_norms_mean.tolist(),
        },
    }
    with open(weights_json_file, "w", encoding="utf-8") as f:
        json.dump(weights_data, f, indent=2, ensure_ascii=False)
    print(f"Norms statistics added to coefficients file")

    print("=" * 80)


if __name__ == "__main__":
    main()
