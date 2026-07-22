"""Tensor operations for low-rank additive subspace rotations."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def select_activation_additive_layers(
    *,
    n_layers: int,
    num_opt_layers: int,
    best_layer: int,
    layer_scores: Sequence[float],
) -> set[int]:
    """Select active layers for ``activation_additive_rot``.

    ``nol=0`` means all layers. Positive values use the standard RDO
    middle-layer ranking and select exactly N layers (capped by the number of
    middle layers).
    """
    if n_layers <= 2:
        raise ValueError("Model must have at least 3 layers.")
    target = int(num_opt_layers)
    if target < 0:
        raise ValueError(f"num_opt_layers must be >= 0, got {num_opt_layers}")
    if target == 0 or target == n_layers:
        return set(range(n_layers))

    middle = list(range(1, n_layers - 1))
    if best_layer not in middle:
        raise ValueError(
            f"best_layer must be a middle layer in [1, {n_layers - 2}], got {best_layer}"
        )
    scores = list(layer_scores)
    if len(scores) != n_layers:
        scores = [0.0] * n_layers
    ranked = sorted(
        middle,
        key=lambda idx: (-float(scores[idx]), abs(idx - int(best_layer))),
    )
    ranked = [int(best_layer)] + [idx for idx in ranked if idx != int(best_layer)]
    return set(ranked[: min(target, len(middle))])


def select_baseline_ablation_layers(
    *,
    n_layers: int,
    num_opt_layers: int,
    best_layer: int,
    layer_scores: Sequence[float],
) -> set[int]:
    """Select baseline ablation layers with the shared RDO NOL policy.

    ``nol=0`` (or the effective all-layer count) preserves legacy baseline
    ablation on every layer. Positive values select ranked middle layers.
    """
    return select_activation_additive_layers(
        n_layers=n_layers,
        num_opt_layers=num_opt_layers,
        best_layer=best_layer,
        layer_scores=layer_scores,
    )


def infer_non_identity_layers(
    matrices: Tensor,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> set[int]:
    """Return indices whose dense square matrix is not identity."""
    if matrices.ndim != 3 or matrices.shape[-2] != matrices.shape[-1]:
        raise ValueError(
            "matrices must have shape (n_layers, dim, dim), got "
            f"{tuple(matrices.shape)}"
        )
    identity = torch.eye(
        matrices.shape[-1],
        device=matrices.device,
        dtype=matrices.dtype,
    )
    return {
        layer_idx
        for layer_idx, matrix in enumerate(matrices)
        if not torch.allclose(matrix, identity, atol=atol, rtol=rtol)
    }


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
