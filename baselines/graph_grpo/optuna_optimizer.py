"""
Optuna-based scalar weight search for graph_grpo.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import torch
from optuna.samplers import (
    CmaEsSampler,
    GPSampler,
    QMCSampler,
    RandomSampler,
    TPESampler,
)

from data_utils import extract_response_after_think
from model_utils import LearnableDirectionWeights, apply_abliteration_with_hyperparams

from baselines.graph_grpo.reward import compute_reward


def create_optuna_sampler(sampler_name: str, sampler_seed: int) -> optuna.samplers.BaseSampler:
    """Build a supported Optuna sampler from a short config name."""
    if sampler_name == "tpe":
        return TPESampler(seed=sampler_seed)
    if sampler_name == "random":
        return RandomSampler(seed=sampler_seed)
    if sampler_name == "gp":
        return GPSampler(seed=sampler_seed)
    if sampler_name == "cmaes":
        try:
            return CmaEsSampler(seed=sampler_seed)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Optuna sampler 'cmaes' requires the optional Python package 'cmaes'. "
                "Install it with `pip install cmaes` or switch OPTUNA_SAMPLER to "
                "'tpe', 'random', 'gp', or 'qmc'."
            ) from exc
    if sampler_name == "qmc":
        return QMCSampler(seed=sampler_seed)
    raise ValueError(
        "Unsupported Optuna sampler "
        f"'{sampler_name}'. Expected one of: tpe, random, gp, cmaes, qmc."
    )


def _suggest_scalar_weights(
    trial: optuna.Trial,
    n_directions: int,
    weight_min: float,
    weight_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = [
        trial.suggest_float(f"weight_{direction_idx}", weight_min, weight_max)
        for direction_idx in range(n_directions)
    ]
    return torch.tensor(values, device=device, dtype=dtype)


def evaluate_scalar_weights(
    direction_weights: LearnableDirectionWeights,
    extracted_directions: List[torch.Tensor],
    scalar_weights: torch.Tensor,
    model,
    questions: List[str],
    abliteration_params: Dict[str, float],
    classifier_categories: List[Dict[str, Any]],
    n_layers: int,
    ref_alpha: float,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a single scalar coefficient vector on the current question batch."""
    with torch.no_grad():
        combined_direction = direction_weights.combine_with_weights(
            [direction.detach() for direction in extracted_directions],
            scalar_weights.detach(),
        )

    model.reload_model()
    apply_abliteration_with_hyperparams(
        model,
        combined_direction,
        abliteration_params["max_weight"] * ref_alpha,
        abliteration_params["max_weight_position"],
        abliteration_params["min_weight"] * ref_alpha,
        abliteration_params["min_weight_distance"],
        n_layers,
    )

    responses_raw = model.get_responses_batched(questions)
    responses = [extract_response_after_think(response) for response in responses_raw]
    scores = compute_reward(questions, responses, classifier_categories, backend)

    mean_reward = float(np.mean(scores)) if scores else 0.0
    best_reward = float(np.max(scores)) if scores else 0.0
    return {
        "weights": scalar_weights.detach().cpu().tolist(),
        "responses": responses,
        "scores": [float(score) for score in scores],
        "mean_reward": mean_reward,
        "best_reward": best_reward,
        "n_questions": len(questions),
    }


def optimize_scalar_weights_with_optuna(
    direction_weights: LearnableDirectionWeights,
    extracted_directions: List[torch.Tensor],
    model,
    questions: List[str],
    abliteration_params: Dict[str, float],
    classifier_categories: List[Dict[str, Any]],
    n_layers: int,
    ref_alpha: float,
    n_trials: int,
    sampler_name: str,
    sampler_seed: int,
    weight_min: float,
    weight_max: float,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Search scalar direction weights with Optuna and update the module in-place."""
    device = direction_weights.weights.device
    dtype = direction_weights.weights.dtype

    def objective(trial: optuna.Trial) -> float:
        scalar_weights = _suggest_scalar_weights(
            trial=trial,
            n_directions=direction_weights.n_directions,
            weight_min=weight_min,
            weight_max=weight_max,
            device=device,
            dtype=dtype,
        )
        trial_result = evaluate_scalar_weights(
            direction_weights=direction_weights,
            extracted_directions=extracted_directions,
            scalar_weights=scalar_weights,
            model=model,
            questions=questions,
            abliteration_params=abliteration_params,
            classifier_categories=classifier_categories,
            n_layers=n_layers,
            ref_alpha=ref_alpha,
            backend=backend,
        )
        trial.set_user_attr("weights", trial_result["weights"])
        trial.set_user_attr("mean_reward", trial_result["mean_reward"])
        trial.set_user_attr("best_reward", trial_result["best_reward"])
        trial.set_user_attr("n_questions", trial_result["n_questions"])
        return trial_result["mean_reward"]

    study = optuna.create_study(
        direction="maximize",
        sampler=create_optuna_sampler(sampler_name, sampler_seed),
    )
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_weights = torch.tensor(
        best_trial.user_attrs["weights"],
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        direction_weights.weights.copy_(best_weights)

    optimization_history = [
        {
            "trial_number": trial.number,
            "value": float(trial.value) if trial.value is not None else None,
            "weights": trial.user_attrs.get("weights"),
            "mean_reward": trial.user_attrs.get("mean_reward"),
            "best_reward": trial.user_attrs.get("best_reward"),
            "n_questions": trial.user_attrs.get("n_questions"),
            "state": trial.state.name,
        }
        for trial in study.trials
    ]

    return {
        "best_weights": best_trial.user_attrs["weights"],
        "best_value": float(best_trial.value),
        "best_trial_number": best_trial.number,
        "sampler_name": sampler_name,
        "optimization_history": optimization_history,
    }
