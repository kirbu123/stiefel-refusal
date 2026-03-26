#!/usr/bin/env python3
"""
GRPO-IS baseline: GRPO with Importance Sampling for training direction weights.

Entry point: python -m baselines.graph_grpo
"""

import sys
from pathlib import Path

# Add project root and verl to path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_verl_path = Path(__file__).resolve().parent / "verl"
if str(_verl_path) not in sys.path:
    sys.path.insert(0, str(_verl_path))

import json
import random
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from heretic.config import Settings
from heretic.utils import load_prompts

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from config import (
    MODEL_NAME, CATEGORIES, GOOD_PROMPTS_DATASET, RESULTS_DIR,
    GRPO_CONFIG, ABLITERATION_PARAMS, FEW_SHOTS_PATH,
    HARMLESS_EVAL_DATASET, EVALUATE_LOCALITY, WEIGHTS_INIT_TYPE, GRAPH_FILE, EVALUATION_BACKEND,
    MMLU_CONFIG, DEBUG, get_method_results_dir,
)
from data_utils import load_all_datasets_with_categories, extract_response_after_think
from refusal_directions import (
    compute_refusal_direction, save_refusal_directions, load_refusal_directions,
)
from model_utils import LearnableDirectionWeights, apply_abliteration_with_hyperparams
from evaluate.judges import evaluate_harmfulness
from evaluate.mmlu import (
    build_mmlu_result,
    evaluate_model_on_mmlu,
    get_cached_or_evaluate_original_mmlu,
)

from baselines.graph_grpo.trainer import train_grpo_is_step


def main():
    print("=" * 80)
    print("GRPO-IS: GRPO with Importance Sampling")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GRPO Config: {GRPO_CONFIG}")
    print(f"Abliteration Params: {ABLITERATION_PARAMS}")
    print(f"Weights Init Type: {WEIGHTS_INIT_TYPE}")
    print(f"Evaluation backend: {EVALUATION_BACKEND}")
    print()

    GRPO_RESULTS_DIR = get_method_results_dir("graph_grpo")
    GRPO_ANSWERS_DIR = GRPO_RESULTS_DIR / "answers"
    GRPO_ANSWERS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    all_data = load_all_datasets_with_categories()
    data_by_category = {cat: [] for cat in CATEGORIES}
    for item in all_data:
        cat = item.get("category", "")
        if cat in data_by_category:
            data_by_category[cat].append(item)

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

    from heretic.model import Model
    model = Model(settings)
    n_layers = len(model.get_layers())
    print(f"Number of layers: {n_layers}")

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

    print("Loading good prompts...")
    good_prompts = load_prompts(GOOD_PROMPTS_DATASET)

    if not GRAPH_FILE.exists():
        print(f"ERROR: Graph file not found: {GRAPH_FILE}")
        return

    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        bad_tags = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(bad_tags)} tags from {GRAPH_FILE}")

    directions_file = RESULTS_DIR / f"refusal_directions_{MODEL_NAME.replace('/', '_')}.pt"
    if directions_file.exists():
        extracted_directions, loaded_tags = load_refusal_directions(directions_file, expected_tags=bad_tags)
        if loaded_tags != bad_tags:
            extracted_directions = []
            for tag_name in bad_tags:
                refusal_dir = compute_refusal_direction(model, [tag_name], good_prompts)
                extracted_directions.append(refusal_dir)
            save_refusal_directions(extracted_directions, bad_tags, MODEL_NAME, directions_file)
        else:
            device = next(model.get_layers()[0].parameters()).device
            extracted_directions = [d.to(device) for d in extracted_directions]
    else:
        extracted_directions = []
        for tag_name in bad_tags:
            refusal_dir = compute_refusal_direction(model, [tag_name], good_prompts)
            extracted_directions.append(refusal_dir)
        save_refusal_directions(extracted_directions, bad_tags, MODEL_NAME, directions_file)

    n_directions = len(extracted_directions)
    hidden_size = extracted_directions[0].shape[1]

    physical_harm_idx = 0
    for idx, tag in enumerate(bad_tags):
        if "physical harm" in tag.lower():
            physical_harm_idx = idx
            break

    direction_weights = LearnableDirectionWeights(
        n_directions=n_directions, n_layers=n_layers, hidden_size=hidden_size,
        init_type=WEIGHTS_INIT_TYPE, topic_idx=physical_harm_idx,
    )
    device = extracted_directions[0].device
    direction_weights = direction_weights.to(device)

    optimizer = torch.optim.Adam(direction_weights.parameters(), lr=GRPO_CONFIG["learning_rate"])

    category_name = "Physical harm"
    category_items = data_by_category.get(category_name, [])
    category_questions = [item.get("instruction", "") for item in category_items if item.get("instruction")]
    if DEBUG and category_questions:
        category_questions = category_questions[:1]
        print("DEBUG mode enabled: using 1 question from the category for fast smoke testing")
    print(f"Loaded {len(category_questions)} questions for category '{category_name}'")

    classifier_categories = []
    if EVALUATION_BACKEND == "local_llm_judge" and FEW_SHOTS_PATH.exists():
        try:
            with open(FEW_SHOTS_PATH, "r", encoding="utf-8") as f:
                few_shots_data = json.load(f)
            classifier_categories = [
                {"id": c.get("id"), "label": c.get("label", "")}
                for c in few_shots_data.get("categories", [])
                if c.get("id")
            ]
        except Exception:
            pass

    if WANDB_AVAILABLE:
        wandb.init(
            project="refusal_direction_grpo_is",
            name=f"grpo_is_{category_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": MODEL_NAME, "category": category_name,
                "n_directions": n_directions, "n_layers": n_layers,
                "grpo_config": GRPO_CONFIG, "abliteration_params": ABLITERATION_PARAMS,
            },
        )

    training_history = []
    print("\n" + "=" * 80)
    print("GRPO-IS TRAINING")
    print("=" * 80)

    for epoch in range(GRPO_CONFIG["n_epochs"]):
        print(f"\nEpoch {epoch + 1}/{GRPO_CONFIG['n_epochs']}")
        metrics = train_grpo_is_step(
            direction_weights=direction_weights,
            extracted_directions=extracted_directions,
            model=model,
            questions=category_questions,
            n_groups=GRPO_CONFIG["n_groups"],
            noise_scale=GRPO_CONFIG["noise_scale"],
            abliteration_params=ABLITERATION_PARAMS,
            optimizer=optimizer,
            classifier_categories=classifier_categories,
            n_layers=n_layers,
            ref_alpha=GRPO_CONFIG["ref_alpha"],
            is_clip_ratio=GRPO_CONFIG["is_clip_ratio"],
            clip_ratio=GRPO_CONFIG["clip_ratio"],
            loss_agg_mode=GRPO_CONFIG["loss_agg_mode"],
            backend=EVALUATION_BACKEND,
        )
        training_history.append({"epoch": epoch + 1, **metrics})
        print(f"  Mean reward: {metrics['mean_reward']:.3f}, Best: {metrics['best_reward']:.3f}")
        if WANDB_AVAILABLE:
            wandb.log({f"train/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}, step=epoch + 1)

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    with torch.no_grad():
        final_direction = direction_weights([d.detach() for d in extracted_directions])

    model.reload_model()
    apply_abliteration_with_hyperparams(
        model, final_direction,
        ABLITERATION_PARAMS["max_weight"] * GRPO_CONFIG["ref_alpha"],
        ABLITERATION_PARAMS["max_weight_position"],
        ABLITERATION_PARAMS["min_weight"] * GRPO_CONFIG["ref_alpha"],
        ABLITERATION_PARAMS["min_weight_distance"],
        n_layers,
    )

    mmlu_block = None
    if MMLU_CONFIG["enabled"] and original_mmlu_result is not None:
        print("Evaluating modified model on MMLU...")
        modified_mmlu_result = evaluate_model_on_mmlu(model, MMLU_CONFIG)
        mmlu_block = build_mmlu_result(
            config=MMLU_CONFIG,
            original_result=original_mmlu_result,
            modified_result=modified_mmlu_result,
            method_results_dir=GRPO_RESULTS_DIR,
            detail_prefix="mmlu_graph_grpo_final",
        )
        if mmlu_block is not None:
            print(
                f"MMLU accuracy original -> modified: "
                f"{mmlu_block['original']['accuracy']:.4f} -> {mmlu_block['modified']['accuracy']:.4f}"
            )

    final_responses_raw = model.get_responses_batched(category_questions)
    final_responses = [extract_response_after_think(r) for r in final_responses_raw]

    from baselines.graph_grpo.reward import compute_reward
    final_scores = [int(s) for s in compute_reward(category_questions, final_responses, classifier_categories, EVALUATION_BACKEND)]

    category_safe_name = category_name.replace("/", "_").replace(" ", "_")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    answers_data = {
        "experiment_config": {
            "model": MODEL_NAME, "category": category_name,
            "n_questions": len(category_questions), "n_directions": n_directions,
            "grpo_config": GRPO_CONFIG, "abliteration_params": ABLITERATION_PARAMS,
            "training_history": training_history,
        },
        "final_weights": direction_weights.weights.data.cpu().tolist(),
        "questions": category_questions,
        "responses": final_responses,
        "final_scores": final_scores,
        "score_statistics": {
            "mean": float(np.mean(final_scores)) if final_scores else None,
            "median": float(np.median(final_scores)) if final_scores else None,
            "count": len(final_scores),
        },
        "timestamp": datetime.now().isoformat(),
    }
    if mmlu_block is not None:
        answers_data["mmlu"] = mmlu_block

    answers_file = GRPO_ANSWERS_DIR / f"answers_{category_safe_name}_{timestamp_str}.json"
    with open(answers_file, "w", encoding="utf-8") as f:
        json.dump(answers_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {answers_file}")

    weights_pt_file = GRPO_RESULTS_DIR / f"coefficients_{category_safe_name}_{timestamp_str}.pt"
    torch.save({
        "weights": direction_weights.weights.data.cpu(),
        "metadata": {"model": MODEL_NAME, "n_directions": n_directions, "n_layers": n_layers},
    }, weights_pt_file)
    print(f"Coefficients saved to: {weights_pt_file}")

    if WANDB_AVAILABLE:
        wandb.finish()

    print("=" * 80)


if __name__ == "__main__":
    main()
