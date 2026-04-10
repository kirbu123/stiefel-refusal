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
import os
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    MODEL_NAME, MODEL_BATCH_SIZE, CATEGORIES, GOOD_PROMPTS_DATASET, RESULTS_DIR,
    GRPO_CONFIG, ABLITERATION_PARAMS, FEW_SHOTS_PATH,
    GRAPH_FILE, EVALUATION_BACKEND,
    MMLU_CONFIG, DEBUG, get_method_results_dir,
)
from dataset.load_dataset import load_dataset_split
from data_utils import load_datasets_with_categories, extract_response_after_think
from refusal_directions import (
    compute_refusal_direction, save_refusal_directions, load_refusal_directions,
)
from model_utils import LearnableDirectionWeights, apply_abliteration_with_hyperparams
from evaluate.mmlu import (
    build_mmlu_result,
    evaluate_model_on_mmlu,
    get_cached_or_evaluate_original_mmlu,
)
from benchmarks.integration import (
    build_benchmark_runner,
    get_benchmark_attack_success_rates,
)

from baselines.graph_grpo.optuna_optimizer import optimize_weights_with_optuna
from baselines.graph_grpo.trainer import train_grpo_is_step
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
    resolve_graph_grpo_reward_sign,
    resolve_graph_grpo_weights_mode,
    resolve_graph_grpo_weights_init_type,
    validate_graph_grpo_category_mode,
    validate_graph_grpo_category_dataset_source,
    validate_graph_grpo_category_filter,
    validate_graph_grpo_optimizer_compatibility,
    validate_graph_grpo_optimizer_method,
    validate_graph_grpo_optuna_sampler,
    validate_graph_grpo_optuna_weight_range,
    validate_graph_grpo_reward_sign,
    validate_graph_grpo_weights_mode,
    validate_graph_grpo_weights_init_type,
)


def _load_classifier_categories() -> List[Dict[str, str]]:
    classifier_categories: List[Dict[str, str]] = []
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
    return classifier_categories


def _evaluate_model_harmfulness(
    model,
    questions: List[str],
    classifier_categories: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Evaluate the current model state on the selected harmfulness batch."""
    from baselines.graph_grpo.reward import compute_reward

    responses_raw = model.get_responses_batched(questions)
    responses = []
    for raw in responses_raw:
        extracted = extract_response_after_think(raw)
        responses.append(extracted if extracted else raw.strip())
    scores = [int(score) for score in compute_reward(
        questions,
        responses,
        classifier_categories,
        EVALUATION_BACKEND,
    )]
    mean_harmfulness = float(np.mean(scores)) if scores else None
    return {
        "responses": responses,
        "scores": scores,
        "mean_harmfulness": mean_harmfulness,
    }


def _load_graph_grpo_category_items(
    category_dataset_source: str,
    category_name: str,
) -> List[Dict[str, Any]]:
    """Load the selected graph_grpo category from the configured dataset source."""
    if category_dataset_source == "jailbreakbench":
        all_data, _ = load_datasets_with_categories(
            category_dataset_source,
            category_name,
        )
        return all_data

    all_data, _ = load_datasets_with_categories(category_dataset_source)
    return [
        item for item in all_data
        if item.get("category", "") == category_name
    ]


def _load_harmful_split_questions(split: str) -> List[str]:
    """Load non-empty harmful instructions from the shared train/val/test split files."""
    questions = load_dataset_split(
        harmtype="harmful",
        split=split,
        instructions_only=True,
    )
    return [question.strip() for question in questions if isinstance(question, str) and question.strip()]


def _sample_fixed_questions(
    questions: List[str],
    sample_count: int,
    sample_seed: int,
) -> List[str]:
    """Deterministically sample a fixed prompt subset without disturbing global RNG state."""
    if sample_count >= len(questions):
        return list(questions)

    rng = random.Random(sample_seed)
    return rng.sample(questions, k=sample_count)


def _sample_training_questions(
    all_category_questions: List[str],
    debug_question_count: int,
) -> List[str]:
    """Sample the current training batch from the selected category dataset."""
    effective_question_count = min(MODEL_BATCH_SIZE, len(all_category_questions))
    if DEBUG:
        effective_question_count = min(effective_question_count, debug_question_count)

    if effective_question_count < len(all_category_questions):
        return random.sample(all_category_questions, k=effective_question_count)
    return list(all_category_questions)


def _build_directions_cache_path(
    results_dir: Path,
    model_name: str,
    category_mode: str,
    direction_count: int | None = None,
    prompt_seed: int | None = None,
) -> Path:
    """Keep single-category and all-category refusal-direction caches isolated."""
    model_safe = model_name.replace("/", "_")
    if category_mode == "single":
        return results_dir / f"refusal_directions_{model_safe}.pt"

    return results_dir / (
        f"refusal_directions_{model_safe}_all_categories_harmful_train_"
        f"n{direction_count}_seed{prompt_seed}.pt"
    )


def _load_or_compute_refusal_directions(
    model,
    direction_identifiers: List[str],
    good_prompts: List[str],
    directions_file: Path,
) -> List[torch.Tensor]:
    """Reuse cached refusal directions when identifiers match exactly, otherwise rebuild them."""
    if directions_file.exists():
        extracted_directions, loaded_tags = load_refusal_directions(
            directions_file,
            expected_tags=direction_identifiers,
        )
        if loaded_tags == direction_identifiers:
            device = next(model.get_layers()[0].parameters()).device
            return [direction.to(device) for direction in extracted_directions]

    extracted_directions = []
    for direction_identifier in direction_identifiers:
        refusal_dir = compute_refusal_direction(model, [direction_identifier], good_prompts)
        extracted_directions.append(refusal_dir)
    save_refusal_directions(extracted_directions, direction_identifiers, MODEL_NAME, directions_file)
    return extracted_directions


def _format_run_name_value(value: Any) -> str:
    """Format run-name scalars compactly and consistently."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _sanitize_run_name_part(value: Any) -> str:
    """Normalize free-form values into wandb-friendly run-name fragments."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown"


def _build_wandb_run_name(
    optimizer_method: str,
    category_mode: str,
    category_name: str,
    weights_mode: str,
    weights_init_type: str,
    grpo_config: Dict[str, Any],
    optuna_config: Dict[str, Any],
    model_name: str,
    results_root: Optional[Path] = None,
) -> str:
    """Build a readable wandb run name with common and optimizer-specific knobs."""
    parts = [
        _sanitize_run_name_part(optimizer_method),
        f"category_mode_{_sanitize_run_name_part(category_mode)}",
        _sanitize_run_name_part(category_name),
        f"weights_{_sanitize_run_name_part(weights_mode)}",
        f"init_{_sanitize_run_name_part(weights_init_type)}",
    ]

    if results_root is not None:
        root_parts = {p.lower() for p in results_root.parts}
        if "blocking" in root_parts:
            parts = [
                "blocking",
                f"model_{_sanitize_run_name_part(model_name)}",
            ] + parts

    if optimizer_method == "grpo":
        parts.extend([
            f"n_groups{_format_run_name_value(grpo_config['n_groups'])}",
            f"n_epochs{_format_run_name_value(grpo_config['n_epochs'])}",
            f"lr{_format_run_name_value(grpo_config['learning_rate'])}",
            f"noise{_format_run_name_value(grpo_config['noise_scale'])}",
            f"ref_alpha{_format_run_name_value(grpo_config['ref_alpha'])}",
            f"is_clip{_format_run_name_value(grpo_config['is_clip_ratio'])}",
            f"clip{_format_run_name_value(grpo_config['clip_ratio'])}",
            f"loss_{_sanitize_run_name_part(grpo_config['loss_agg_mode'])}",
            f"klcoef{_format_run_name_value(grpo_config['kl_loss_coef'])}",
        ])
    else:
        parts.extend([
            f"sampler_{_sanitize_run_name_part(optuna_config['sampler'])}",
            f"n_trials{_format_run_name_value(optuna_config['n_trials'])}",
            f"sampler_seed{_format_run_name_value(optuna_config['sampler_seed'])}",
            f"weight_min{_format_run_name_value(optuna_config['weight_min'])}",
            f"weight_max{_format_run_name_value(optuna_config['weight_max'])}",
            f"klcoef{_format_run_name_value(grpo_config['kl_loss_coef'])}",
        ])

    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return "_".join(parts)


def _define_model_state_eval_metrics(
    benchmark_names: List[str] | None = None,
) -> None:
    """Register shared wandb series so clean and best-value land on the same charts."""
    if not WANDB_AVAILABLE or not hasattr(wandb, "define_metric"):
        return

    wandb.define_metric("model_state_eval/point_index")
    wandb.define_metric(
        "model_state_eval/mmlu_score",
        step_metric="model_state_eval/point_index",
    )
    wandb.define_metric(
        "model_state_eval/harmfulness_on_full_dataset",
        step_metric="model_state_eval/point_index",
    )
    for benchmark_name in benchmark_names or []:
        wandb.define_metric(
            f"model_state_eval/{benchmark_name}_attack_success_rate",
            step_metric="model_state_eval/point_index",
        )


def _log_model_state_eval(
    prefixes: List[str],
    point_index: int,
    point_name: str,
    harmfulness_on_full_dataset: Optional[float],
    mmlu_score: Optional[float],
    benchmark_attack_success_rates: Optional[Dict[str, float]] = None,
) -> None:
    """Log scalar and shared-series wandb metrics for clean/best model comparisons."""
    if not WANDB_AVAILABLE:
        return

    payload: Dict[str, Any] = {
        "model_state_eval/point_index": point_index,
        "model_state_eval/point_name": point_name,
    }

    if harmfulness_on_full_dataset is not None:
        for prefix in prefixes:
            payload[f"{prefix}/harmfulness_on_full_dataset"] = harmfulness_on_full_dataset
            if prefix in {"clean_model", "best_value_model"}:
                payload[f"{prefix}/harmfulness"] = harmfulness_on_full_dataset
        payload["model_state_eval/harmfulness_on_full_dataset"] = harmfulness_on_full_dataset

    if mmlu_score is not None:
        for prefix in prefixes:
            payload[f"{prefix}/mmlu_score"] = mmlu_score
            if prefix in {"clean_model", "best_value_model"}:
                payload[f"{prefix}/mmlu_accuracy"] = mmlu_score
        payload["model_state_eval/mmlu_score"] = mmlu_score

    for benchmark_name, attack_success_rate in (benchmark_attack_success_rates or {}).items():
        for prefix in prefixes:
            payload[
                f"{prefix}/{benchmark_name}_attack_success_rate"
            ] = attack_success_rate
        payload[
            f"model_state_eval/{benchmark_name}_attack_success_rate"
        ] = attack_success_rate

    wandb.log(payload)


def _run_grpo_training(
    direction_weights: LearnableDirectionWeights,
    extracted_directions: List[torch.Tensor],
    model,
    all_category_questions: List[str],
    effective_noise_scale: float,
    classifier_categories: List[Dict[str, str]],
    n_layers: int,
    debug_question_count: int,
    reward_sign: float,
) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(direction_weights.parameters(), lr=GRPO_CONFIG["learning_rate"])
    training_history = []
    best_epoch = None
    best_train_batch_questions: List[str] = []
    best_train_batch_mean_reward: Optional[float] = None
    best_train_batch_mean_objective = float("-inf")
    best_train_batch_mean_harmfulness: Optional[float] = None
    best_train_batch_mean_kl: Optional[float] = None
    best_weights = direction_weights.weights.detach().cpu().clone()
    print("\n" + "=" * 80)
    print("GRPO-IS TRAINING")
    print("=" * 80)

    for epoch in range(GRPO_CONFIG["n_epochs"]):
        epoch_questions = _sample_training_questions(
            all_category_questions=all_category_questions,
            debug_question_count=debug_question_count,
        )
        print(f"\nEpoch {epoch + 1}/{GRPO_CONFIG['n_epochs']}")
        print(
            f"  Training on random batch of {len(epoch_questions)} question(s) "
            f"sampled from {len(all_category_questions)} available"
        )
        metrics = train_grpo_is_step(
            direction_weights=direction_weights,
            extracted_directions=extracted_directions,
            model=model,
            questions=epoch_questions,
            n_groups=GRPO_CONFIG["n_groups"],
            noise_scale=effective_noise_scale,
            abliteration_params=ABLITERATION_PARAMS,
            optimizer=optimizer,
            classifier_categories=classifier_categories,
            n_layers=n_layers,
            ref_alpha=GRPO_CONFIG["ref_alpha"],
            is_clip_ratio=GRPO_CONFIG["is_clip_ratio"],
            clip_ratio=GRPO_CONFIG["clip_ratio"],
            loss_agg_mode=GRPO_CONFIG["loss_agg_mode"],
            reward_sign=reward_sign,
            kl_loss_coef=GRPO_CONFIG["kl_loss_coef"],
            backend=EVALUATION_BACKEND,
        )
        training_history.append({
            "epoch": epoch + 1,
            "n_questions": len(epoch_questions),
            **metrics,
        })
        mean_harmfulness = metrics.get("mean_harmfulness")
        mean_kl = metrics.get("mean_kl")
        mean_objective = metrics.get("mean_objective")
        if mean_harmfulness is not None and mean_kl is not None and mean_objective is not None:
            print(
                f"  Mean objective: {mean_objective:.3f}, Mean reward: {metrics['mean_reward']:.3f}, "
                f"Best reward: {metrics['best_reward']:.3f} "
                f"(mean harmfulness={mean_harmfulness:.3f}, mean kl={mean_kl:.3f})"
            )
        else:
            print(f"  Mean reward: {metrics['mean_reward']:.3f}, Best: {metrics['best_reward']:.3f}")
        if metrics["mean_objective"] > best_train_batch_mean_objective:
            best_epoch = epoch + 1
            best_train_batch_mean_reward = float(metrics["mean_reward"])
            best_train_batch_mean_objective = float(metrics["mean_objective"])
            best_train_batch_mean_harmfulness = (
                float(mean_harmfulness) if mean_harmfulness is not None else None
            )
            best_train_batch_mean_kl = float(mean_kl) if mean_kl is not None else None
            best_train_batch_questions = list(epoch_questions)
            best_weights = direction_weights.weights.detach().cpu().clone()

        if WANDB_AVAILABLE:
            wandb.log({f"train/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}, step=epoch + 1)

    if best_epoch is None:
        raise ValueError("GRPO training finished without producing a best checkpoint.")

    return {
        "training_history": training_history,
        "optimization_history": [],
        "optuna_config": None,
        "best_epoch": best_epoch,
        "best_weights": best_weights,
        "best_train_batch_questions": best_train_batch_questions,
        "best_train_batch_mean_reward": best_train_batch_mean_reward,
        "best_train_batch_mean_objective": best_train_batch_mean_objective,
        "best_train_batch_mean_harmfulness": best_train_batch_mean_harmfulness,
        "best_train_batch_mean_kl": best_train_batch_mean_kl,
    }


def _final_evaluate(
    direction_weights: LearnableDirectionWeights,
    extracted_directions: List[torch.Tensor],
    model,
    category_questions: List[str],
    classifier_categories: List[Dict[str, str]],
    n_layers: int,
    original_mmlu_result: Optional[Dict[str, Any]],
    results_dir: Path,
    benchmark_runner=None,
    benchmark_run_label: str = "graph_grpo_final",
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    with torch.no_grad():
        final_direction = direction_weights([d.detach() for d in extracted_directions])

    model.reload_model()
    apply_abliteration_with_hyperparams(
        model,
        final_direction,
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
            method_results_dir=results_dir,
            detail_prefix="mmlu_graph_grpo_final",
        )
        if mmlu_block is not None:
            print(
                f"MMLU accuracy original -> modified: "
                f"{mmlu_block['original']['accuracy']:.4f} -> {mmlu_block['modified']['accuracy']:.4f}"
            )

    benchmark_results = {}
    if benchmark_runner is not None:
        print("Evaluating modified model on benchmarks...")
        benchmark_results = benchmark_runner.evaluate_modified(
            model,
            run_label=benchmark_run_label,
        )

    harmfulness_result = _evaluate_model_harmfulness(
        model=model,
        questions=category_questions,
        classifier_categories=classifier_categories,
    )

    return {
        "responses": harmfulness_result["responses"],
        "scores": harmfulness_result["scores"],
        "final_mean_harmfulness": harmfulness_result["mean_harmfulness"],
        "mmlu_block": mmlu_block,
        "benchmarks": benchmark_results,
    }


def main():
    category_mode = resolve_graph_grpo_category_mode(os.getenv("CATEGORY_MODE"))
    category_dataset_source = resolve_graph_grpo_category_dataset_source(
        os.getenv("CATEGORY_DATASET_SOURCE")
    )
    category_name = resolve_graph_grpo_category_filter(os.getenv("CATEGORY_FILTER"))
    all_categories_harmful_prompt_count = resolve_graph_grpo_all_categories_harmful_prompt_count(
        os.getenv("ALL_CATEGORIES_HARMFUL_PROMPT_COUNT")
    )
    all_categories_harmful_prompt_seed = resolve_graph_grpo_all_categories_harmful_prompt_seed(
        os.getenv("ALL_CATEGORIES_HARMFUL_PROMPT_SEED")
    )
    optimizer_method = resolve_graph_grpo_optimizer_method(os.getenv("OPTIMIZER_METHOD"))
    weights_mode = resolve_graph_grpo_weights_mode(os.getenv("WEIGHTS_MODE"))
    weights_init_type = resolve_graph_grpo_weights_init_type(os.getenv("WEIGHTS_INIT_TYPE"))
    debug_question_count = resolve_graph_grpo_debug_question_count(os.getenv("DEBUG_N_QUESTIONS"))
    debug_noise_scale = resolve_graph_grpo_debug_noise_scale(
        base_noise_scale=GRPO_CONFIG["noise_scale"],
        env_value=os.getenv("DEBUG_NOISE_SCALE"),
    )
    optuna_sampler = resolve_graph_grpo_optuna_sampler(os.getenv("OPTUNA_SAMPLER"))
    optuna_n_trials = resolve_graph_grpo_optuna_n_trials(os.getenv("OPTUNA_N_TRIALS"))
    optuna_sampler_seed = resolve_graph_grpo_optuna_sampler_seed(os.getenv("OPTUNA_SAMPLER_SEED"))
    optuna_weight_min = resolve_graph_grpo_optuna_weight_min(os.getenv("OPTUNA_WEIGHT_MIN"))
    optuna_weight_max = resolve_graph_grpo_optuna_weight_max(os.getenv("OPTUNA_WEIGHT_MAX"))
    reward_sign = resolve_graph_grpo_reward_sign(os.getenv("REWARD_SIGN"))
    effective_noise_scale = debug_noise_scale if DEBUG else GRPO_CONFIG["noise_scale"]

    validate_graph_grpo_category_mode(category_mode)
    validate_graph_grpo_category_dataset_source(category_dataset_source)
    validate_graph_grpo_category_filter(category_name, category_mode=category_mode)
    validate_graph_grpo_optimizer_method(optimizer_method)
    validate_graph_grpo_weights_mode(weights_mode)
    validate_graph_grpo_weights_init_type(weights_init_type, category_mode=category_mode)
    validate_graph_grpo_optimizer_compatibility(optimizer_method, weights_mode)
    validate_graph_grpo_optuna_sampler(optuna_sampler)
    validate_graph_grpo_optuna_weight_range(optuna_weight_min, optuna_weight_max)
    validate_graph_grpo_reward_sign(reward_sign)

    run_category_name = category_name if category_mode == "single" else "all_categories"
    category_display_name = (
        category_name if category_mode == "single" else "all categories"
    )

    optuna_config = {
        "sampler": optuna_sampler,
        "n_trials": optuna_n_trials,
        "sampler_seed": optuna_sampler_seed,
        "weight_min": optuna_weight_min,
        "weight_max": optuna_weight_max,
    }

    print("=" * 80)
    print("GRPO-IS: GRPO with Importance Sampling")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GRPO Config: {GRPO_CONFIG}")
    print(f"Abliteration Params: {ABLITERATION_PARAMS}")
    print(f"Optimizer Method: {optimizer_method}")
    print(f"Category mode: {category_mode}")
    print(f"Category dataset source: {category_dataset_source}")
    print(f"Category: {run_category_name}")
    print(f"Weights Mode: {weights_mode}")
    print(f"Weights Init Type: {weights_init_type}")
    print(f"Batch Size: {MODEL_BATCH_SIZE}")
    print(f"Evaluation backend: {EVALUATION_BACKEND}")
    print(f"Reward sign: {reward_sign:+g}")
    if category_mode == "all":
        print(
            "All-category prompt sampling: "
            f"count={all_categories_harmful_prompt_count}, "
            f"seed={all_categories_harmful_prompt_seed}"
        )
    if optimizer_method == "optuna":
        print(f"Optuna Config: {optuna_config}")
    print()

    GRPO_RESULTS_DIR = get_method_results_dir("graph_grpo")
    GRPO_ANSWERS_DIR = GRPO_RESULTS_DIR / "answers"
    GRPO_ANSWERS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    if category_mode == "single":
        category_items = _load_graph_grpo_category_items(
            category_dataset_source=category_dataset_source,
            category_name=category_name,
        )
        print(
            f"Loaded {len(category_items)} item(s) for category '{category_name}' "
            f"from source '{category_dataset_source}'"
        )
        all_category_questions = [
            item.get("instruction", "")
            for item in category_items
            if item.get("instruction")
        ]
        full_evaluation_questions = list(all_category_questions)
        direction_source = str(GRAPH_FILE)
        optimization_source = f"{category_dataset_source}:{category_name}"
        final_evaluation_source = optimization_source
        direction_prompt_count: int | None = None
        direction_prompt_seed: int | None = None
    else:
        all_category_questions = _load_harmful_split_questions("val")
        full_evaluation_questions = _load_harmful_split_questions("test")
        if not full_evaluation_questions:
            raise ValueError(
                "No full-evaluation questions found in harmful test split for CATEGORY_MODE='all'."
            )
        print(
            f"Loaded {len(all_category_questions)} harmful val question(s) for optimization"
        )
        print(
            f"Loaded {len(full_evaluation_questions)} harmful test question(s) for full evaluation"
        )
        direction_source = "dataset/splits/harmful_train.json"
        optimization_source = "dataset/splits/harmful_val.json"
        final_evaluation_source = "dataset/splits/harmful_test.json"
        direction_prompt_count = None
        direction_prompt_seed = all_categories_harmful_prompt_seed

    print("\nLoading model...")
    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]] if sys.argv else ["script"]
        settings = Settings(
            model=MODEL_NAME, batch_size=MODEL_BATCH_SIZE,
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

    if category_mode == "single":
        if not GRAPH_FILE.exists():
            print(f"ERROR: Graph file not found: {GRAPH_FILE}")
            return

        with open(GRAPH_FILE, "r", encoding="utf-8") as f:
            bad_tags = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(bad_tags)} tags from {GRAPH_FILE}")
        directions_file = _build_directions_cache_path(
            results_dir=RESULTS_DIR,
            model_name=MODEL_NAME,
            category_mode=category_mode,
        )
    else:
        harmful_train_questions = _load_harmful_split_questions("train")
        if not harmful_train_questions:
            raise ValueError(
                "No harmful train prompts found for CATEGORY_MODE='all'."
            )
        sampled_harmful_train_questions = _sample_fixed_questions(
            harmful_train_questions,
            min(all_categories_harmful_prompt_count, len(harmful_train_questions)),
            all_categories_harmful_prompt_seed,
        )
        direction_prompt_count = len(sampled_harmful_train_questions)
        bad_tags = sampled_harmful_train_questions
        print(
            f"Loaded {len(harmful_train_questions)} harmful train prompt(s); "
            f"sampled {len(sampled_harmful_train_questions)} for refusal directions"
        )
        directions_file = _build_directions_cache_path(
            results_dir=RESULTS_DIR,
            model_name=MODEL_NAME,
            category_mode=category_mode,
            direction_count=direction_prompt_count,
            prompt_seed=all_categories_harmful_prompt_seed,
        )

    extracted_directions = _load_or_compute_refusal_directions(
        model=model,
        direction_identifiers=bad_tags,
        good_prompts=good_prompts,
        directions_file=directions_file,
    )

    n_directions = len(extracted_directions)
    hidden_size = extracted_directions[0].shape[1]

    physical_harm_idx = 0
    if category_mode == "single":
        for idx, tag in enumerate(bad_tags):
            if "physical harm" in tag.lower():
                physical_harm_idx = idx
                break

    print(f"Using weights mode '{weights_mode}'")
    print(f"Using weights init type '{weights_init_type}'")
    print(f"Using optimizer method '{optimizer_method}'")
    if weights_init_type == "topic" and category_mode == "single":
        print(
            f"Topic init root index for graph tag 'Physical harm': "
            f"{physical_harm_idx} (tag='{bad_tags[physical_harm_idx]}')"
        )

    direction_weights = LearnableDirectionWeights(
        n_directions=n_directions, n_layers=n_layers, hidden_size=hidden_size,
        init_type=weights_init_type, topic_idx=physical_harm_idx, mode=weights_mode,
    )
    device = extracted_directions[0].device
    direction_weights = direction_weights.to(device)
    weights_shape = list(direction_weights.weights.shape)
    print(f"Trainable weights shape: {weights_shape}")

    if not all_category_questions:
        raise ValueError(
            f"No optimization questions found for category mode '{category_mode}'."
        )
    train_question_count = min(MODEL_BATCH_SIZE, len(all_category_questions))
    if DEBUG:
        train_question_count = min(train_question_count, debug_question_count)

    if category_mode == "single":
        print(
            f"Training batch size will be {train_question_count} question(s) sampled from "
            f"{len(all_category_questions)} available in category '{category_name}' "
            f"using batch_size={MODEL_BATCH_SIZE}"
        )
    else:
        print(
            f"Training batch size will be {train_question_count} question(s) sampled from "
            f"{len(all_category_questions)} available harmful val question(s) "
            f"using batch_size={MODEL_BATCH_SIZE}"
        )

    if DEBUG and train_question_count > 0:
        print(
            f"DEBUG mode enabled: using {train_question_count} question(s) from the optimization pool "
            f"and rollout noise_scale={effective_noise_scale:g} "
            f"(base noise_scale={GRPO_CONFIG['noise_scale']:g})"
        )
    print(
        f"Loaded {len(all_category_questions)} total optimization question(s) "
        f"and {len(full_evaluation_questions)} full-evaluation question(s)"
    )

    classifier_categories = _load_classifier_categories()
    benchmark_runner = build_benchmark_runner(
        method_results_dir=GRPO_RESULTS_DIR,
        model_name=MODEL_NAME,
        classifier_categories=classifier_categories,
    )

    if WANDB_AVAILABLE:
        wandb.init(
            project="weighted_refusal_direction",
            name=_build_wandb_run_name(
                optimizer_method=optimizer_method,
                category_mode=category_mode,
                category_name=run_category_name,
                weights_mode=weights_mode,
                weights_init_type=weights_init_type,
                grpo_config=GRPO_CONFIG,
                optuna_config=optuna_config,
                model_name=MODEL_NAME,
                results_root=RESULTS_DIR,
            ),
            config={
                "model": MODEL_NAME,
                "category": run_category_name,
                "category_mode": category_mode,
                "category_filter": category_name if category_mode == "single" else None,
                "category_dataset_source": category_dataset_source,
                "n_directions": n_directions, "n_layers": n_layers,
                "optimizer_method": optimizer_method,
                "weights_mode": weights_mode,
                "weights_init_type": weights_init_type,
                "weights_shape": weights_shape,
                "direction_source": direction_source,
                "optimization_source": optimization_source,
                "final_evaluation_source": final_evaluation_source,
                "direction_prompt_count": direction_prompt_count,
                "direction_prompt_seed": direction_prompt_seed,
                "grpo_config": GRPO_CONFIG, "abliteration_params": ABLITERATION_PARAMS,
                "optuna_config": optuna_config if optimizer_method == "optuna" else None,
                "reward_sign": reward_sign,
            },
        )
        _define_model_state_eval_metrics(
            list(benchmark_runner.enabled_benchmark_names()) if benchmark_runner is not None else []
        )

    print("\n" + "=" * 80)
    print("CLEAN MODEL EVALUATION")
    print("=" * 80)
    if category_mode == "single":
        print(
            f"Evaluating clean model on full dataset: {len(full_evaluation_questions)} "
            f"question(s); training will use batch of {train_question_count} question(s)"
        )
    else:
        print(
            f"Evaluating clean model on full harmful test split: "
            f"{len(full_evaluation_questions)} question(s); training will use val batches "
            f"of {train_question_count} question(s)"
        )
    model.reload_model()
    clean_harmfulness_result = _evaluate_model_harmfulness(
        model=model,
        questions=full_evaluation_questions,
        classifier_categories=classifier_categories,
    )
    clean_mean_harmfulness = clean_harmfulness_result["mean_harmfulness"]
    print(f"Clean model harmfulness: {clean_mean_harmfulness:.3f}" if clean_mean_harmfulness is not None else "Clean model harmfulness: n/a")

    clean_mmlu_score = None
    if original_mmlu_result is not None:
        clean_mmlu_score = original_mmlu_result["summary"]["accuracy"]

    clean_benchmark_summaries = {}
    if benchmark_runner is not None:
        print("Evaluating clean model on benchmarks...")
        clean_benchmark_summaries = benchmark_runner.prepare_original(model)
    clean_benchmark_attack_success_rates = {
        benchmark_name: float(summary["attack_success_rate"])
        for benchmark_name, summary in clean_benchmark_summaries.items()
        if summary.get("attack_success_rate") is not None
    }

    _log_model_state_eval(
        prefixes=["clean_model"],
        point_index=0,
        point_name="clean",
        harmfulness_on_full_dataset=clean_mean_harmfulness,
        mmlu_score=clean_mmlu_score,
        benchmark_attack_success_rates=clean_benchmark_attack_success_rates,
    )

    if optimizer_method == "grpo":
        optimization_result = _run_grpo_training(
            direction_weights=direction_weights,
            extracted_directions=extracted_directions,
            model=model,
            classifier_categories=classifier_categories,
            all_category_questions=all_category_questions,
            effective_noise_scale=effective_noise_scale,
            n_layers=n_layers,
            debug_question_count=debug_question_count,
            reward_sign=reward_sign,
        )
    else:
        print("\n" + "=" * 80)
        print("OPTUNA OPTIMIZATION")
        print("=" * 80)
        print(
            f"  Each Optuna trial will train on a random batch of {train_question_count} "
            f"question(s) sampled from {len(all_category_questions)} available"
        )
        optimization_result = optimize_weights_with_optuna(
            direction_weights=direction_weights,
            extracted_directions=extracted_directions,
            model=model,
            questions=all_category_questions,
            abliteration_params=ABLITERATION_PARAMS,
            classifier_categories=classifier_categories,
            n_layers=n_layers,
            ref_alpha=GRPO_CONFIG["ref_alpha"],
            reward_sign=reward_sign,
            kl_loss_coef=GRPO_CONFIG["kl_loss_coef"],
            n_trials=optuna_n_trials,
            sampler_name=optuna_sampler,
            sampler_seed=optuna_sampler_seed,
            weight_min=optuna_weight_min,
            weight_max=optuna_weight_max,
            question_sampler=lambda: _sample_training_questions(
                all_category_questions=all_category_questions,
                debug_question_count=debug_question_count,
            ),
            loss_agg_mode=GRPO_CONFIG["loss_agg_mode"],
            backend=EVALUATION_BACKEND,
        )
        optimization_result["training_history"] = []
        optimization_result["optuna_config"] = optuna_config
        print(
            f"Best Optuna trial: #{optimization_result['best_trial_number']} "
            f"with mean objective={optimization_result['best_value']:.3f} "
            f"(mean harmfulness={optimization_result.get('best_mean_harmfulness')}, "
            f"mean kl={optimization_result.get('best_mean_kl')})"
        )
        if WANDB_AVAILABLE:
            wandb.log(
                {
                    "optuna/best_trial_number": optimization_result["best_trial_number"],
                    "optuna/best_value": optimization_result["best_value"],
                    "optuna/best_objective": optimization_result.get("best_mean_objective", optimization_result["best_value"]),
                    "optuna/best_harmfulness": optimization_result.get("best_mean_harmfulness"),
                    "optuna/best_kl": optimization_result.get("best_mean_kl"),
                }
            )

    if category_mode == "all":
        evaluation_questions = full_evaluation_questions
    else:
        evaluation_questions = (
            optimization_result["best_trial_questions"]
            if optimizer_method == "optuna"
            else optimization_result["best_train_batch_questions"]
        )
    if optimizer_method == "grpo":
        with torch.no_grad():
            direction_weights.weights.data.copy_(
                optimization_result["best_weights"].to(
                    device=direction_weights.weights.device,
                    dtype=direction_weights.weights.dtype,
                )
            )

    category_safe_name = run_category_name.replace("/", "_").replace(" ", "_")
    final_result = _final_evaluate(
        direction_weights=direction_weights,
        extracted_directions=extracted_directions,
        model=model,
        category_questions=evaluation_questions,
        classifier_categories=classifier_categories,
        n_layers=n_layers,
        original_mmlu_result=original_mmlu_result,
        results_dir=GRPO_RESULTS_DIR,
        benchmark_runner=benchmark_runner,
        benchmark_run_label=f"graph_grpo_{category_safe_name}",
    )

    final_mean_harmfulness = final_result["final_mean_harmfulness"]
    if optimizer_method == "grpo":
        optimal_objective = optimization_result["best_train_batch_mean_objective"]
        optimal_harmfulness = optimization_result["best_train_batch_mean_harmfulness"]
        optimal_harmfulness_source = "best_train_batch_mean_objective"
        optimal_objective_source = "best_train_batch_mean_objective"
    else:
        best_mean_objective = optimization_result.get("best_mean_objective")
        best_mean_harmfulness = optimization_result.get("best_mean_harmfulness")
        optimal_objective = (
            best_mean_objective
            if best_mean_objective is not None
            else optimization_result["best_value"]
        )
        optimal_harmfulness = (
            best_mean_harmfulness
            if best_mean_harmfulness is not None
            else optimal_objective
        )
        optimal_harmfulness_source = "best_trial_mean_objective"
        optimal_objective_source = "best_trial_mean_objective"

    best_batch_model_harmfulness_result = _evaluate_model_harmfulness(
        model=model,
        questions=full_evaluation_questions,
        classifier_categories=classifier_categories,
    )
    best_batch_model_mean_harmfulness = best_batch_model_harmfulness_result["mean_harmfulness"]
    best_batch_model_mmlu_score = None
    if final_result["mmlu_block"] is not None:
        best_batch_model_mmlu_score = final_result["mmlu_block"]["modified"]["accuracy"]
    best_batch_model_benchmark_attack_success_rates = get_benchmark_attack_success_rates(
        final_result["benchmarks"],
    )

    best_model_prefixes = ["best_value_model"]
    if optimizer_method == "grpo":
        best_model_prefixes.append("best_batch_model")

    _log_model_state_eval(
        prefixes=best_model_prefixes,
        point_index=1,
        point_name="best_value",
        harmfulness_on_full_dataset=best_batch_model_mean_harmfulness,
        mmlu_score=best_batch_model_mmlu_score,
        benchmark_attack_success_rates=best_batch_model_benchmark_attack_success_rates,
    )

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    answers_data = {
        "experiment_config": {
            "model": MODEL_NAME,
            "category": run_category_name,
            "category_mode": category_mode,
            "category_filter": category_name if category_mode == "single" else None,
            "category_dataset_source": category_dataset_source,
            "direction_source": direction_source,
            "optimization_source": optimization_source,
            "final_evaluation_source": final_evaluation_source,
            "direction_prompt_count": direction_prompt_count,
            "direction_prompt_seed": direction_prompt_seed,
            "optimization_question_pool_size": len(all_category_questions),
            "full_evaluation_question_count": len(full_evaluation_questions),
            "n_questions": len(evaluation_questions), "n_directions": n_directions,
            "optimizer_method": optimizer_method,
            "reward_sign": reward_sign,
            "weights_mode": weights_mode, "weights_init_type": weights_init_type,
            "weights_shape": weights_shape,
            "grpo_config": GRPO_CONFIG, "abliteration_params": ABLITERATION_PARAMS,
            "training_history": optimization_result["training_history"],
            "optuna_config": optimization_result["optuna_config"],
        },
        "final_weights": direction_weights.weights.data.cpu().tolist(),
        "questions": evaluation_questions,
        "responses": final_result["responses"],
        "final_scores": final_result["scores"],
        "final_mean_harmfulness": final_mean_harmfulness,
        "optimal_objective": optimal_objective,
        "optimal_objective_source": optimal_objective_source,
        "optimal_harmfulness": optimal_harmfulness,
        "optimal_harmfulness_source": optimal_harmfulness_source,
        "score_statistics": {
            "mean": final_mean_harmfulness,
            "median": float(np.median(final_result["scores"])) if final_result["scores"] else None,
            "count": len(final_result["scores"]),
        },
        "timestamp": datetime.now().isoformat(),
    }
    if optimizer_method == "grpo":
        answers_data["experiment_config"]["best_grpo_epoch"] = optimization_result["best_epoch"]
        answers_data["experiment_config"]["best_train_batch_mean_reward"] = optimization_result["best_train_batch_mean_reward"]
        answers_data["experiment_config"]["best_train_batch_mean_objective"] = optimization_result["best_train_batch_mean_objective"]
        answers_data["experiment_config"]["best_train_batch_mean_harmfulness"] = optimization_result["best_train_batch_mean_harmfulness"]
        answers_data["experiment_config"]["best_train_batch_mean_kl"] = optimization_result["best_train_batch_mean_kl"]
        answers_data["experiment_config"]["best_train_batch_size"] = len(optimization_result["best_train_batch_questions"])
    else:
        answers_data["experiment_config"]["best_trial_mean_objective"] = optimization_result.get("best_mean_objective")
        answers_data["experiment_config"]["best_trial_mean_harmfulness"] = optimization_result.get("best_mean_harmfulness")
        answers_data["experiment_config"]["best_trial_mean_kl"] = optimization_result.get("best_mean_kl")
    if optimization_result["optimization_history"]:
        answers_data["optimization_history"] = optimization_result["optimization_history"]
    if final_result["mmlu_block"] is not None:
        answers_data["mmlu"] = final_result["mmlu_block"]
    if final_result["benchmarks"]:
        answers_data["benchmarks"] = final_result["benchmarks"]

    answers_file = GRPO_ANSWERS_DIR / f"answers_{category_safe_name}_{timestamp_str}.json"
    with open(answers_file, "w", encoding="utf-8") as f:
        json.dump(answers_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {answers_file}")

    weights_pt_file = GRPO_RESULTS_DIR / f"coefficients_{category_safe_name}_{timestamp_str}.pt"
    torch.save({
        "weights": direction_weights.weights.data.cpu(),
        "metadata": {
            "model": MODEL_NAME,
            "category": run_category_name,
            "category_mode": category_mode,
            "category_filter": category_name if category_mode == "single" else None,
            "category_dataset_source": category_dataset_source,
            "direction_source": direction_source,
            "optimization_source": optimization_source,
            "final_evaluation_source": final_evaluation_source,
            "direction_prompt_count": direction_prompt_count,
            "direction_prompt_seed": direction_prompt_seed,
            "n_directions": n_directions,
            "n_layers": n_layers,
            "optimizer_method": optimizer_method,
            "weights_mode": weights_mode,
            "weights_init_type": weights_init_type,
            "weights_shape": weights_shape,
        },
    }, weights_pt_file)
    print(f"Coefficients saved to: {weights_pt_file}")

    if WANDB_AVAILABLE:
        wandb.finish()

    print("=" * 80)


if __name__ == "__main__":
    main()
