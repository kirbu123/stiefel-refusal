"""
GRPO training functions.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from heretic.model import Model
from heretic.utils import empty_cache

from model_utils import LearnableDirectionWeights, apply_abliteration_with_hyperparams
from data_utils import extract_response_after_think
from evaluate.judges import evaluate_harmfulness
from config import GRPO_CONFIG, ABLITERATION_PARAMS


def train_grpo_epoch(
    direction_weights: LearnableDirectionWeights,
    extracted_directions: List[torch.Tensor],
    model: Model,
    category_questions: List[str],
    classifier_categories: List[Dict],
    n_layers: int,
    optimizer: torch.optim.Optimizer,
) -> Dict:
    """
    Run one GRPO training epoch.

    Args:
        direction_weights: Learnable weights for directions
        extracted_directions: List of refusal directions
        model: Model to modify
        category_questions: Questions for evaluation
        classifier_categories: Classifier categories
        n_layers: Number of layers
        optimizer: Optimizer for weight updates

    Returns:
        Dict of epoch metrics
    """
    base_weights_cpu = direction_weights.weights.data.cpu().clone()
    device = direction_weights.weights.data.device

    group_rewards = []
    group_responses_list = []
    group_weights_cpu = []

    for variant_idx in range(GRPO_CONFIG["n_groups"]):
        print(f"  Variant {variant_idx + 1}/{GRPO_CONFIG['n_groups']}...")

        if variant_idx == 0:
            variant_weights_cpu = base_weights_cpu.clone()
        else:
            noise_cpu = torch.randn_like(base_weights_cpu) * GRPO_CONFIG["noise_scale"]
            variant_weights_cpu = (base_weights_cpu + noise_cpu).detach()
            del noise_cpu

        group_weights_cpu.append(variant_weights_cpu.clone())
        variant_weights = variant_weights_cpu.to(device)
        direction_weights.weights.data = variant_weights

        del variant_weights
        empty_cache()

        extracted_directions_no_grad = [d.detach() if d.requires_grad else d for d in extracted_directions]

        with torch.no_grad():
            combined_direction_raw = direction_weights(extracted_directions_no_grad)
            combined_direction = torch.tensor(
                combined_direction_raw.cpu().numpy(),
                dtype=combined_direction_raw.dtype,
                device=combined_direction_raw.device,
                requires_grad=False
            )

        model.reload_model()

        apply_abliteration_with_hyperparams(
            model,
            combined_direction,
            ABLITERATION_PARAMS["max_weight"],
            ABLITERATION_PARAMS["max_weight_position"],
            ABLITERATION_PARAMS["min_weight"],
            ABLITERATION_PARAMS["min_weight_distance"],
            n_layers
        )

        del combined_direction
        empty_cache()

        modified_responses_raw = model.get_responses_batched(category_questions)
        modified_responses = [extract_response_after_think(resp) for resp in modified_responses_raw]
        group_responses_list.append(modified_responses)

        del modified_responses_raw
        empty_cache()

        variant_rewards = []
        for question, response in zip(category_questions, modified_responses):
            if response:
                try:
                    score_data = evaluate_harmfulness(
                        question,
                        response,
                        classifier_categories=classifier_categories,
                    )
                    score = score_data.get("score")
                    if score is not None:
                        variant_rewards.append(float(score))
                except Exception as e:
                    print(f"      Evaluation error: {e}")

        avg_reward = np.mean(variant_rewards) if variant_rewards else 0.0
        group_rewards.append(avg_reward)
        print(f"    Mean reward: {avg_reward:.3f} (based on {len(variant_rewards)} scores)")

        empty_cache()

    group_rewards = np.array(group_rewards)
    advantages = group_rewards - group_rewards.mean()
    advantages_tensor = torch.tensor(advantages / GRPO_CONFIG["beta"], dtype=torch.float32, requires_grad=False)
    weights = torch.softmax(advantages_tensor, dim=0)

    print(f"  Variant weights: {weights.tolist()}")
    print(f"  Rewards: {group_rewards.tolist()}")
    print(f"  Advantages: {advantages.tolist()}")

    advantages_mean = float(advantages.mean())
    advantages_std = float(advantages.std())
    advantages_max = float(advantages.max())
    advantages_min = float(advantages.min())

    with torch.no_grad():
        current_weights_cpu = direction_weights.weights.data.cpu()
        update_step_cpu = torch.zeros_like(current_weights_cpu)

        for variant_idx, (variant_weights_cpu, weight) in enumerate(zip(group_weights_cpu, weights)):
            variant_diff_cpu = variant_weights_cpu - current_weights_cpu
            update_step_cpu.add_(variant_diff_cpu, alpha=float(weight))
            del variant_diff_cpu

        update_step = update_step_cpu.to(device)
        direction_weights.weights.data.add_(update_step, alpha=GRPO_CONFIG["learning_rate"])
        del update_step, update_step_cpu, current_weights_cpu
        empty_cache()

    optimizer.zero_grad()

    # GRPO loss for Gaussian policy:
    # log_prob(variant | params) = -||variant - params||^2 / (2 * sigma^2)
    # Loss = sum_i(advantage_i * ||variant_i - params||^2 / (2 * sigma^2))
    direction_weights.weights.requires_grad_(True)
    sigma = GRPO_CONFIG["noise_scale"]
    sigma_sq = sigma ** 2

    loss = torch.tensor(0.0, dtype=torch.float32, device=device)

    for variant_idx, variant_weights_cpu in enumerate(group_weights_cpu):
        variant_weights = variant_weights_cpu.to(device)
        advantage = advantages[variant_idx]

        diff = variant_weights - direction_weights.weights
        sq_dist = torch.sum(diff ** 2)

        loss = loss + advantage * sq_dist / (2 * sigma_sq)
        del variant_weights, diff

    loss.backward()

    total_grad_norm = 0.0
    if direction_weights.weights.grad is not None:
        total_grad_norm = direction_weights.weights.grad.norm().item()

    del loss
    empty_cache()
    optimizer.step()

    del group_weights_cpu, base_weights_cpu
    optimizer.zero_grad(set_to_none=True)
    empty_cache()

    return {
        "rewards": group_rewards.tolist(),
        "mean_reward": float(group_rewards.mean()),
        "best_reward": float(group_rewards.max()),
        "weights_norm": float(direction_weights.weights.data.norm().item()),
        "gradient_norm": total_grad_norm,
        "advantages_mean": advantages_mean,
        "advantages_std": advantages_std,
        "advantages_max": advantages_max,
        "advantages_min": advantages_min,
        "variant_weights": weights.tolist(),
    }
