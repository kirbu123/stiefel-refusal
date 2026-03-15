"""
GRPO-IS training step using verl components.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Add verl to path before importing
_verl_path = Path(__file__).resolve().parent / "verl"
if str(_verl_path) not in sys.path:
    sys.path.insert(0, str(_verl_path))

from heretic.utils import empty_cache

from model_utils import LearnableDirectionWeights, apply_abliteration_with_hyperparams
from data_utils import extract_response_after_think

from baselines.graph_grpo.advantage import compute_grpo_is_advantage
from baselines.graph_grpo.reward import compute_reward
from baselines.graph_grpo.log_probs import compute_sequence_log_probs
from baselines.graph_grpo.hooks import register_abliteration_hooks, remove_hooks


def _make_actor_config(
    clip_ratio: float = 0.2,
    loss_agg_mode: str = "token-mean",
) -> SimpleNamespace:
    """Create minimal config for compute_policy_loss_vanilla."""
    config = SimpleNamespace()
    config.clip_ratio = clip_ratio
    config.clip_ratio_low = clip_ratio
    config.clip_ratio_high = clip_ratio
    config.clip_ratio_c = 3.0
    config.loss_agg_mode = loss_agg_mode
    config.global_batch_info = {}
    return config


def train_grpo_is_step(
    direction_weights: LearnableDirectionWeights,
    extracted_directions: List[torch.Tensor],
    model,
    questions: List[str],
    alphas: List[float],
    abliteration_params: Dict[str, float],
    optimizer: torch.optim.Optimizer,
    classifier_categories: List[Dict],
    n_layers: int,
    ref_alpha: float,
    is_clip_ratio: float,
    clip_ratio: float,
    loss_agg_mode: str,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One GRPO-IS training step.

    Args:
        direction_weights: Trainable P^{(l)}
        extracted_directions: Frozen refusal direction vectors
        model: Heretic model
        questions: Batch of queries
        alphas: Predefined alpha coefficients for M behavior policies
        abliteration_params: max_weight, max_weight_position, min_weight, min_weight_distance
        optimizer: Optimizer for direction_weights
        classifier_categories: For evaluate_harmfulness
        n_layers: Number of transformer layers
        ref_alpha: Reference alpha for policy
        is_clip_ratio: IS weight truncation threshold
        clip_ratio: PPO clip epsilon
        loss_agg_mode: Loss aggregation mode
        backend: Evaluation backend

    Returns:
        Dict of metrics
    """
    device = direction_weights.weights.device
    max_weight = abliteration_params["max_weight"]
    max_weight_position = abliteration_params["max_weight_position"]
    min_weight = abliteration_params["min_weight"]
    min_weight_distance = abliteration_params["min_weight_distance"]

    max_weight_pos_abs = max_weight_position * (n_layers - 1)
    min_weight_dist_abs = min_weight_distance * (n_layers - 1)

    all_responses = []
    all_rollout_log_probs = []
    all_masks = []

    extracted_directions_no_grad = [d.detach() if d.requires_grad else d for d in extracted_directions]

    # Step 1 & 2: Rollout from M behavior policies
    for alpha in alphas:
        with torch.no_grad():
            combined_direction = direction_weights(extracted_directions_no_grad)

        model.reload_model()
        scaled_max = max_weight * alpha
        scaled_min = min_weight * alpha
        apply_abliteration_with_hyperparams(
            model,
            combined_direction,
            scaled_max,
            max_weight_position,
            scaled_min,
            min_weight_distance,
            n_layers,
        )

        responses_raw = model.get_responses_batched(questions)
        responses = [extract_response_after_think(r) for r in responses_raw]
        all_responses.append(responses)

        rollout_lp, rollout_mask = compute_sequence_log_probs(model, questions, responses)
        all_rollout_log_probs.append(rollout_lp)
        all_masks.append(rollout_mask)
        empty_cache()

    # Step 3: Compute log pi_theta_old (base model)
    model.reload_model()
    all_old_log_probs = []
    for m in range(len(alphas)):
        old_lp, _ = compute_sequence_log_probs(model, questions, all_responses[m])
        all_old_log_probs.append(old_lp)
    empty_cache()

    # Flatten: we have M*N responses
    n_questions = len(questions)
    n_m = len(alphas)
    flat_responses = [r for m in range(n_m) for r in all_responses[m]]
    flat_questions = [q for m in range(n_m) for q in questions]
    question_indices = np.array([i for m in range(n_m) for i in range(n_questions)])

    # Stack and pad log probs to common max length
    max_len = max(p.shape[1] for p in all_rollout_log_probs)
    pad_fn = lambda t: torch.nn.functional.pad(t, (0, max_len - t.shape[1]), value=0.0)

    rollout_log_probs = torch.cat([pad_fn(p) for p in all_rollout_log_probs], dim=0)
    old_log_probs = torch.cat([pad_fn(p) for p in all_old_log_probs], dim=0)
    response_mask = torch.cat([pad_fn(m) for m in all_masks], dim=0)

    # Step 4: IS weights (verl)
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights

    log_ratio = old_log_probs - rollout_log_probs
    is_weights, is_metrics = compute_rollout_correction_weights(
        log_ratio=log_ratio,
        response_mask=response_mask,
        rollout_is="sequence",
        rollout_is_threshold=is_clip_ratio,
    )
    is_weights = is_weights.detach()

    # Step 5: Rewards
    scores = compute_reward(flat_questions, flat_responses, classifier_categories, backend)
    scores_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
    token_level_rewards = torch.zeros_like(rollout_log_probs, device=device)
    last_valid = (response_mask > 0).sum(dim=-1).long() - 1
    last_valid = torch.clamp(last_valid, min=0)
    token_level_rewards.scatter_(1, last_valid.unsqueeze(1), scores_tensor.unsqueeze(1).to(device))

    # Step 6: IS-corrected advantages
    advantages, returns = compute_grpo_is_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=question_indices,
        is_weights=is_weights,
    )
    advantages = advantages.detach()

    # Step 7: Policy loss (verl, differentiable)
    combined_direction = direction_weights(extracted_directions)
    handles = register_abliteration_hooks(
        model,
        combined_direction,
        ref_alpha,
        n_layers,
        max_weight,
        max_weight_position,
        min_weight,
        min_weight_distance,
    )

    log_probs, _ = compute_sequence_log_probs(model, flat_questions, flat_responses)
    max_lp_len = log_probs.shape[1]
    if max_lp_len < response_mask.shape[1]:
        log_probs = torch.nn.functional.pad(log_probs, (0, response_mask.shape[1] - max_lp_len), value=0.0)
    elif max_lp_len > response_mask.shape[1]:
        log_probs = log_probs[:, : response_mask.shape[1]]
    remove_hooks(handles)

    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    actor_config = _make_actor_config(clip_ratio=clip_ratio, loss_agg_mode=loss_agg_mode)
    loss, loss_metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_probs,
        log_prob=log_probs,
        advantages=advantages,
        response_mask=response_mask,
        rollout_is_weights=is_weights,
        config=actor_config,
    )

    # Step 8: Update
    optimizer.zero_grad()
    loss.backward()
    grad_norm = direction_weights.weights.grad.norm().item() if direction_weights.weights.grad is not None else 0.0
    optimizer.step()

    # Step 9: Off-policy metrics
    from verl.trainer.ppo.rollout_corr_helper import compute_offpolicy_metrics

    offpolicy_metrics = compute_offpolicy_metrics(
        old_log_prob=old_log_probs,
        rollout_log_prob=rollout_log_probs,
        response_mask=response_mask,
    )

    metrics = {
        "mean_reward": float(scores_tensor.mean().item()),
        "best_reward": float(scores_tensor.max().item()),
        "weights_norm": float(direction_weights.weights.data.norm().item()),
        "gradient_norm": grad_norm,
        **loss_metrics,
        **{f"offpolicy/{k}": v for k, v in offpolicy_metrics.items()},
        **{f"is/{k}": v for k, v in is_metrics.items()},
    }
    return metrics
