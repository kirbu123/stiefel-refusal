"""Tensor operations for low-rank additive subspace rotations."""

from __future__ import annotations

import torch
from torch import Tensor


def additive_subspace_rotation(
    x: Tensor,
    basis: Tensor,
    rotation: Tensor,
    *,
    use_transpose: bool = True,
) -> Tensor:
    """Apply ``x -> x + B(Q-I)B.T x`` to row-vector activations.

    ``use_transpose=True`` applies the column-vector transform with ``Q``.
    Setting it to ``False`` applies its inverse when ``Q`` is orthogonal.
    """
    identity = torch.eye(
        rotation.shape[-1],
        device=rotation.device,
        dtype=rotation.dtype,
    )
    delta_rotation = rotation.T - identity if use_transpose else rotation - identity
    return x + (x @ basis) @ delta_rotation @ basis.T


def dense_additive_rotation_matrix(basis: Tensor, rotation: Tensor) -> Tensor:
    """Compose the dense column-vector matrix ``I + B(Q-I)B.T``."""
    identity_k = torch.eye(
        rotation.shape[-1],
        device=rotation.device,
        dtype=rotation.dtype,
    )
    identity_d = torch.eye(
        basis.shape[-2],
        device=basis.device,
        dtype=basis.dtype,
    )
    return identity_d + basis @ (rotation - identity_k) @ basis.T
