"""LoReFT/DiReFT operators and RDO adapters; importing this module loads no models.

ReFT (NeurIPS 2024), equations 2 and 3. Protect mode subtracts the learned
residual; it is a sign reversal, not in general the inverse affine map.
"""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal

from baselines.additive_rotation_ops import select_activation_additive_layers

REFT_MODES = frozenset({"loreft", "direft"})


class LoReFT(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        if not 1 <= rank <= dim:
            raise ValueError(f"rank must be in [1, {dim}], got {rank}")
        # Transposed R has orthonormal columns. Householder avoids a d x d
        # matrix exponential and use_trivialization=False avoids a dense base.
        self.basis = nn.Linear(rank, dim, bias=False)
        orthogonal(self.basis, "weight", orthogonal_map="householder", use_trivialization=False)
        self.W = nn.Parameter(self.R.detach().clone())
        self.b = nn.Parameter(torch.zeros(rank))

    @property
    def R(self):
        return self.basis.weight.T

    def residual(self, h):
        R = self.R.to(h)
        return (h @ (self.W.to(h) - R).T + self.b.to(h)) @ R

    def forward(self, h, sign=1):
        return h + sign * self.residual(h)


class DiReFT(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        if not 1 <= rank <= dim:
            raise ValueError(f"rank must be in [1, {dim}], got {rank}")
        initial = nn.Linear(dim, rank, bias=False).weight.detach()
        self.W1 = nn.Parameter(initial.clone())
        self.W2 = nn.Parameter(initial.clone())
        self.b = nn.Parameter(torch.zeros(rank))

    def residual(self, h):
        return (h @ self.W1.to(h).T + self.b.to(h)) @ self.W2.to(h)

    def forward(self, h, sign=1):
        return h + sign * self.residual(h)


class ReFTIntervention(nn.Module):
    """Shared layer-input adapter for nnsight and ordinary PyTorch CPU tests."""

    def __init__(self, module, dim, method, k_proj=35, num_opt_layers=0,
                 best_layer=None, layer_scores=None, device="cpu"):
        super().__init__()
        if method not in REFT_MODES:
            raise ValueError(f"Unknown ReFT method: {method}")
        # The live base model is not owned by this adapter or its optimizer.
        object.__setattr__(self, "module", module)
        self.method, self.dim, self.k_proj = method, dim, int(k_proj)
        n_layers = len(module.layers)
        scores = [0.0] * n_layers if layer_scores is None else list(layer_scores)
        if best_layer is None:
            best_layer = max(range(1, n_layers - 1), key=lambda i: scores[i])
        self._optimized_layer_idxs = select_activation_additive_layers(
            n_layers=n_layers, num_opt_layers=int(num_opt_layers),
            best_layer=int(best_layer), layer_scores=scores,
        )
        self.num_opt_layers = len(self._optimized_layer_idxs)
        self.n_layers = n_layers
        operator = LoReFT if method == "loreft" else DiReFT
        self.operators = nn.ModuleDict({
            str(i): operator(dim, self.k_proj) for i in sorted(self._optimized_layer_idxs)
        })
        self.register_buffer("_dummy_fn", torch.zeros(dim))
        self._guard_mode = "attack"
        self.to(device=device, dtype=torch.float32)

    @property
    def fn_vectors(self):
        return [self._dummy_fn]

    def transform(self, direction):
        return self._dummy_fn

    def stack_directions_for_log(self):
        return self._dummy_fn.detach().unsqueeze(0).cpu()

    def set_guard_mode(self, mode):
        if mode not in ("attack", "protect"):
            raise ValueError(f"Unsupported guard mode: {mode}")
        self._guard_mode = mode

    def apply_tensor(self, h, layer_idx, opposite=False):
        if layer_idx not in self._optimized_layer_idxs:
            return h
        if isinstance(h, tuple):
            return (self.apply_tensor(h[0], layer_idx, opposite),) + h[1:]
        sign = 1 if self._guard_mode == "attack" else -1
        return self.operators[str(layer_idx)](h, sign=-sign if opposite else sign)

    def forward(self, direction=None, best_layer=None):
        for i in sorted(self._optimized_layer_idxs):
            layer = self.module.layers[i]
            layer.input = self.apply_tensor(layer.input, i)

    def add(self, direction, alpha, layer_idx, best_layer=None):
        i = int(layer_idx)
        if i in self._optimized_layer_idxs:
            layer = self.module.layers[i]
            layer.input = self.apply_tensor(layer.input, i, opposite=True)

    def orthogonalize(self):
        # LoReFT's parametrization enforces orthogonality on every forward.
        pass

    def normalize(self):
        pass

    @contextmanager
    def hooks(self, opposite=False):
        """Apply the same intervention through eager PyTorch input hooks."""
        handles = []
        try:
            for i in sorted(self._optimized_layer_idxs):
                def hook(layer, inputs, index=i):
                    return (self.apply_tensor(inputs[0], index, opposite),) + inputs[1:]
                handles.append(self.module.layers[i].register_forward_pre_hook(hook))
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def export_checkpoint(self):
        return {
            "version": 1, "method": self.method, "dim": self.dim,
            "rank": self.k_proj, "n_layers": self.n_layers,
            "active_layer_indices": sorted(self._optimized_layer_idxs),
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
        }

    def load_checkpoint(self, payload):
        expected = (1, self.method, self.dim, self.k_proj, self.n_layers)
        actual = tuple(payload.get(k) for k in ("version", "method", "dim", "rank", "n_layers"))
        if actual != expected or payload.get("active_layer_indices") != sorted(self._optimized_layer_idxs):
            raise ValueError("ReFT checkpoint metadata does not match the intervention")
        self.load_state_dict(payload["state_dict"], strict=True)

    @classmethod
    def from_checkpoint(cls, module, payload, device="cpu"):
        active = payload["active_layer_indices"]
        n_layers = len(module.layers)
        if (not active or len(set(active)) != len(active)
                or any(type(i) is not int or not 0 <= i < n_layers for i in active)):
            raise ValueError("Invalid ReFT checkpoint active layers")
        # Reconstruct exact saved layers, independent of runtime layer selection.
        result = cls(module, payload["dim"], payload["method"], payload["rank"],
                     num_opt_layers=0, device=device)
        result.operators = nn.ModuleDict({str(i): result.operators[str(i)] for i in active})
        result._optimized_layer_idxs = set(active)
        result.num_opt_layers = len(active)
        result.load_checkpoint(payload)
        return result
