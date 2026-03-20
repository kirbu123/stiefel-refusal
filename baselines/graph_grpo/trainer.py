"""
GRPO-IS training step using verl components.
"""

import sys
import time
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


class _ActorConfig(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def _make_actor_config(
    clip_ratio: float = 0.2,
    loss_agg_mode: str = "token-mean",
) -> _ActorConfig:
    """Create minimal config for compute_policy_loss_vanilla."""
    return _ActorConfig(
        clip_ratio=clip_ratio,
        clip_ratio_low=clip_ratio,
        clip_ratio_high=clip_ratio,
        clip_ratio_c=3.0,
        loss_agg_mode=loss_agg_mode,
        global_batch_info={},
    )


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

    step_t0 = time.time()
    print(f"  [Step 1-2] Rollout from {len(alphas)} behavior policies, {len(questions)} questions each")

    # Step 1 & 2: Rollout from M behavior policies
    for ai, alpha in enumerate(alphas):
        t0 = time.time()
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

        print(f"    Policy {ai+1}/{len(alphas)} (alpha={alpha:.2f}): generating responses...", end=" ", flush=True)
        responses_raw = model.get_responses_batched(questions)
        responses = [extract_response_after_think(r) for r in responses_raw]
        all_responses.append(responses)
        avg_resp_len = sum(len(r) for r in responses) / max(len(responses), 1)
        print(f"done ({len(responses)} responses, avg_len={avg_resp_len:.0f} chars)")

        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"    Policy {ai+1}/{len(alphas)}: computing rollout log-probs (GPU mem: {mem_gb:.2f} GB)...", end=" ", flush=True)
        with torch.no_grad():
            rollout_lp, rollout_mask = compute_sequence_log_probs(model, questions, responses)
        print(f"done (shape={list(rollout_lp.shape)}, mean_lp={rollout_lp.sum()/rollout_mask.sum():.4f})")

        all_rollout_log_probs.append(rollout_lp)
        all_masks.append(rollout_mask)
        empty_cache()
        print(f"    Policy {ai+1}/{len(alphas)} total: {time.time()-t0:.1f}s")

    print(f"  [Step 1-2] Rollout complete: {time.time()-step_t0:.1f}s")

    # Step 3: Compute log pi_theta_old (base model)
    t0 = time.time()
    print(f"  [Step 3] Computing old log-probs (base model) for {len(alphas)} response sets...")
    model.reload_model()
    all_old_log_probs = []
    for m in range(len(alphas)):
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"    Set {m+1}/{len(alphas)}: computing (GPU mem: {mem_gb:.2f} GB)...", end=" ", flush=True)
        with torch.no_grad():
            old_lp, _ = compute_sequence_log_probs(model, questions, all_responses[m])
        print(f"done (shape={list(old_lp.shape)})")
        all_old_log_probs.append(old_lp)
    empty_cache()
    print(f"  [Step 3] Old log-probs complete: {time.time()-t0:.1f}s")

    # Flatten: we have M*N responses
    n_questions = len(questions)
    n_m = len(alphas)
    flat_responses = [r for m in range(n_m) for r in all_responses[m]]
    flat_questions = [q for m in range(n_m) for q in questions]
    question_indices = np.array([i for m in range(n_m) for i in range(n_questions)])
    print(f"  [Flatten] {n_m} policies x {n_questions} questions = {len(flat_responses)} total responses")

    # Stack and pad log probs to common max length
    max_len = max(p.shape[1] for p in all_rollout_log_probs)
    pad_fn = lambda t: torch.nn.functional.pad(t, (0, max_len - t.shape[1]), value=0.0)

    rollout_log_probs = torch.cat([pad_fn(p) for p in all_rollout_log_probs], dim=0)
    old_log_probs = torch.cat([pad_fn(p) for p in all_old_log_probs], dim=0)
    response_mask = torch.cat([pad_fn(m) for m in all_masks], dim=0)
    print(f"  [Flatten] Padded to max_len={max_len}, shapes: rollout={list(rollout_log_probs.shape)}, old={list(old_log_probs.shape)}")

    # Step 4: IS weights (verl)
    t0 = time.time()
    print(f"  [Step 4] Computing IS weights (clip_ratio={is_clip_ratio})...", end=" ", flush=True)
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights

    log_ratio = old_log_probs - rollout_log_probs
    is_weights, is_metrics = compute_rollout_correction_weights(
        log_ratio=log_ratio,
        response_mask=response_mask,
        rollout_is="sequence",
        rollout_is_threshold=is_clip_ratio,
    )
    is_weights = is_weights.detach()
    seq_is = is_weights.max(dim=-1).values if is_weights.dim() > 1 else is_weights
    print(f"done ({time.time()-t0:.1f}s)")
    print(f"    IS weights: mean={seq_is.mean():.4f}, min={seq_is.min():.4f}, max={seq_is.max():.4f}")

    # Step 5: Rewards
    t0 = time.time()
    print(f"  [Step 5] Computing rewards for {len(flat_responses)} responses (backend={backend})...")
    scores = compute_reward(flat_questions, flat_responses, classifier_categories, backend)
    scores_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
    print(f"    Rewards: mean={scores_tensor.mean():.3f}, std={scores_tensor.std():.3f}, "
          f"min={scores_tensor.min():.1f}, max={scores_tensor.max():.1f} ({time.time()-t0:.1f}s)")
    token_level_rewards = torch.zeros(rollout_log_probs.shape, dtype=torch.float32, device=device)
    last_valid = (response_mask > 0).sum(dim=-1).long() - 1
    last_valid = torch.clamp(last_valid, min=0)
    token_level_rewards.scatter_(1, last_valid.unsqueeze(1), scores_tensor.unsqueeze(1).to(device))

    if backend == "llamaguard":
        try:
            from evaluate.evaluation_llamaguard import unload_llamaguard_evaluator
            unload_llamaguard_evaluator()
            print("    LlamaGuard unloaded to free GPU memory")
        except Exception:
            pass

    # Step 6: IS-corrected advantages
    t0 = time.time()
    print(f"  [Step 6] Computing IS-corrected advantages...", end=" ", flush=True)
    advantages, returns = compute_grpo_is_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=question_indices,
        is_weights=is_weights,
    )
    advantages = advantages.detach()
    valid_adv = advantages[response_mask > 0]
    print(f"done ({time.time()-t0:.1f}s)")
    print(f"    Advantages: mean={valid_adv.mean():.4f}, std={valid_adv.std():.4f}, "
          f"min={valid_adv.min():.4f}, max={valid_adv.max():.4f}")

    # Step 7+8: Policy loss with gradient accumulation (micro-batched)
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    t0 = time.time()
    print(f"  [Step 7+8] Policy loss + backward (ref_alpha={ref_alpha}, clip={clip_ratio})...")
    combined_direction = direction_weights(extracted_directions)
    actor_config = _make_actor_config(clip_ratio=clip_ratio, loss_agg_mode=loss_agg_mode)
    optimizer.zero_grad()

    n_total = len(flat_questions)
    micro_bs = 4
    empty_cache()
    total_valid_tokens = response_mask.sum().item()
    accumulated_loss = 0.0
    loss_metrics = {}

    target_len = response_mask.shape[1]

    for mb_start in range(0, n_total, micro_bs):
        mb_end = min(mb_start + micro_bs, n_total)
        mb_q = flat_questions[mb_start:mb_end]
        mb_r = flat_responses[mb_start:mb_end]
        mb_old_lp = old_log_probs[mb_start:mb_end]
        mb_adv = advantages[mb_start:mb_end]
        mb_mask = response_mask[mb_start:mb_end]
        mb_is = is_weights[mb_start:mb_end]

        handles = register_abliteration_hooks(
            model, combined_direction, ref_alpha, n_layers,
            max_weight, max_weight_position, min_weight, min_weight_distance,
        )
        mb_lp, _ = compute_sequence_log_probs(model, mb_q, mb_r)
        remove_hooks(handles)

        if mb_lp.shape[1] < target_len:
            mb_lp = torch.nn.functional.pad(mb_lp, (0, target_len - mb_lp.shape[1]), value=0.0)
        elif mb_lp.shape[1] > target_len:
            mb_lp = mb_lp[:, :target_len]

        mb_loss, mb_metrics = compute_policy_loss_vanilla(
            old_log_prob=mb_old_lp,
            log_prob=mb_lp,
            advantages=mb_adv,
            response_mask=mb_mask,
            rollout_is_weights=mb_is,
            config=actor_config,
        )

        mb_valid_tokens = mb_mask.sum().item()
        weight = mb_valid_tokens / total_valid_tokens if total_valid_tokens > 0 else 1.0 / n_total
        (mb_loss * weight).backward()
        accumulated_loss += mb_loss.item() * weight
        loss_metrics = mb_metrics

        del mb_lp, mb_loss
        empty_cache()

        if (mb_start // micro_bs) % 50 == 0:
            print(f"    micro-batch {mb_start//micro_bs + 1}/{(n_total + micro_bs - 1)//micro_bs}", flush=True)

    # #region agent log
    _mem = torch.cuda.memory_allocated() / 1e9
    _res = torch.cuda.memory_reserved() / 1e9
    with open(_lp, "a") as _f: _f.write(_json.dumps({"sessionId":"ebb21c","hypothesisId":"E","location":"trainer.py:after_grad_accum","message":"mem after gradient accumulation","data":{"accumulated_loss":round(accumulated_loss,6),"mem_alloc_gb":round(_mem,2),"mem_reserved_gb":round(_res,2),"has_grad":direction_weights.weights.grad is not None},"timestamp":int(_time.time()*1000)})+"\n")
    # #endregion

    grad_norm = direction_weights.weights.grad.norm().item() if direction_weights.weights.grad is not None else 0.0
    optimizer.step()
    print(f"    Loss: {accumulated_loss:.6f} ({time.time()-t0:.1f}s)")
    for k, v in loss_metrics.items():
        print(f"    {k}: {v}")
    print(f"    Grad norm: {grad_norm:.6f}")
    print(f"    Weights norm: {direction_weights.weights.data.norm().item():.6f}")
    print(f"    Weights range: [{direction_weights.weights.data.min().item():.4f}, {direction_weights.weights.data.max().item():.4f}]")

    # Step 9: Off-policy metrics
    print(f"  [Step 9] Computing off-policy metrics...", end=" ", flush=True)
    from verl.trainer.ppo.rollout_corr_helper import compute_offpolicy_metrics

    offpolicy_metrics = compute_offpolicy_metrics(
        old_log_prob=old_log_probs,
        rollout_log_prob=rollout_log_probs,
        response_mask=response_mask,
    )
    print(f"done")
    for k, v in offpolicy_metrics.items():
        print(f"    offpolicy/{k}: {v}")

    total_time = time.time() - step_t0
    print(f"  [TOTAL] Training step completed in {total_time:.1f}s")
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  [TOTAL] GPU memory allocated: {mem_gb:.2f} GB")

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
