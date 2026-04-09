"""
KL helpers for graph_grpo regularization.
"""

import sys
from pathlib import Path
from typing import Tuple

import torch

_verl_path = Path(__file__).resolve().parent / "verl"
if str(_verl_path) not in sys.path:
    sys.path.insert(0, str(_verl_path))

from verl.trainer.ppo.core_algos import agg_loss, kl_penalty


KL_LOSS_TYPE = "low_var_kl"


def compute_masked_mean_kl(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-token KL estimates and their masked aggregate."""
    kl_tensor = kl_penalty(
        logprob=log_prob,
        ref_logprob=ref_log_prob,
        kl_penalty=KL_LOSS_TYPE,
    )
    mean_kl = agg_loss(
        loss_mat=kl_tensor,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
    )
    return kl_tensor, mean_kl
