"""Upstream paper angular-steering (inference-time plane + fixed θ).

Extracts a 2D plane from SaladBench harmful/harmless activations using the
vendored angular-steering extract helpers, then applies the adaptive angular
rotation at generation time via nnsight layer.input edits.
"""

from __future__ import annotations

import gc
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_ANGULAR_ROOT = Path(__file__).resolve().parent / "src" / "angular-steering"
_PYTORCH_PURE = _ANGULAR_ROOT / "pytorch_pure"


def _load_pytorch_pure_module(name: str):
    """Load a module from baselines/src/angular-steering/pytorch_pure by file path."""
    path = _PYTORCH_PURE / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing angular-steering helper: {path}")
    # Ensure sibling imports (utils) resolve inside pytorch_pure.
    root_str = str(_PYTORCH_PURE)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    mod_name = f"_angular_steering_pytorch_pure_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def unwrap_hf_model(language_model) -> nn.Module:
    """Best-effort unwrap of nnsight LanguageModel to the underlying HF module."""
    # nnsight LanguageModel stores the HF PreTrainedModel on `_model`.
    candidate = getattr(language_model, "_model", None)
    if isinstance(candidate, nn.Module):
        return candidate
    candidate = getattr(language_model, "model", None)
    if candidate is None:
        raise RuntimeError("Could not unwrap HF model from LanguageModel for plane extraction")
    for inner in ("_module", "module", "_model"):
        inner_mod = getattr(candidate, inner, None)
        if isinstance(inner_mod, nn.Module):
            return inner_mod
    if isinstance(candidate, nn.Module):
        return candidate
    raise RuntimeError("Could not unwrap HF model from LanguageModel for plane extraction")


def _ensure_model_device_attr(hf_model: nn.Module) -> None:
    """Upstream extract uses ``model.device``; device_map models may lack it."""
    if hasattr(hf_model, "device"):
        return
    try:
        hf_model.device = next(hf_model.parameters()).device
    except StopIteration:
        hf_model.device = torch.device("cpu")


def extract_saladbench_plane(
    *,
    hf_model: nn.Module,
    tokenizer,
    harmful_instructions: list[str],
    harmless_instructions: list[str],
    strategy: str = "max_sim",
    n_samples: int = 512,
    batch_size: int = 8,
    positions: list[str] | None = None,
) -> dict[str, Any]:
    """Extract steering plane using upstream extract_activations / compute_steering_directions."""
    positions = positions or ["mid", "post"]
    if strategy not in ("max_sim", "max_norm"):
        raise ValueError(f"strategy must be max_sim or max_norm, got {strategy!r}")

    extract_mod = _load_pytorch_pure_module("extract_directions")
    _ensure_model_device_attr(hf_model)

    harmful = list(harmful_instructions)[:n_samples]
    harmless = list(harmless_instructions)[: min(n_samples, len(harmful))]
    harmful = harmful[: len(harmless)]

    num_layers = int(hf_model.config.num_hidden_layers)
    layers = list(range(num_layers))

    print(
        f"[paper_angular_steering] Extracting plane: "
        f"n_harmful={len(harmful)}, n_harmless={len(harmless)}, "
        f"strategy={strategy}, layers={num_layers}"
    )

    harmful_acts = extract_mod.extract_activations(
        hf_model, harmful, tokenizer, layers, positions, batch_size
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    harmless_acts = extract_mod.extract_activations(
        hf_model, harmless, tokenizer, layers, positions, batch_size
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    directions = extract_mod.compute_steering_directions(
        harmful_acts, harmless_acts, strategy
    )
    if strategy not in directions:
        raise RuntimeError(f"Strategy {strategy!r} missing from extract output keys={list(directions)}")

    cfg = directions[strategy]
    b1 = torch.as_tensor(cfg["first_direction"], dtype=torch.float32)
    b2 = torch.as_tensor(cfg["second_direction"], dtype=torch.float32)
    b1 = b1 / (b1.norm() + 1e-12)
    # Orthogonalize b2 against b1 at save time (runtime also re-orthonormalizes).
    b2 = b2 - torch.dot(b2, b1) * b1
    b2 = b2 / (b2.norm() + 1e-12)

    plane = {
        "b1": b1.cpu(),
        "b2": b2.cpu(),
        "layer": int(cfg["layer"]),
        "position": str(cfg["position"]),
        "strategy": strategy,
        "n_extract_samples": int(len(harmful)),
    }
    print(
        f"[paper_angular_steering] Selected layer={plane['layer']} "
        f"position={plane['position']} strategy={strategy}"
    )
    return plane


class PaperAngularSteering(nn.Module):
    """Fixed-plane angular steering with attack/protect θ switching."""

    def __init__(
        self,
        module,
        plane: dict[str, Any],
        *,
        attack_degree: float = 180.0,
        protect_degree: float = 0.0,
        adaptive_mode: int = 1,
        apply_all_layers: bool = True,
    ) -> None:
        super().__init__()
        self.module = module
        self.attack_degree = float(attack_degree) % 360.0
        self.protect_degree = float(protect_degree) % 360.0
        self.adaptive_mode = int(adaptive_mode)
        if self.adaptive_mode not in (0, 1):
            raise ValueError("adaptive_mode must be 0 or 1")
        self.apply_all_layers = bool(apply_all_layers)
        self._guard_mode = "attack"
        self.selected_layer = int(plane["layer"])
        self.position = str(plane.get("position", "mid"))

        b1 = torch.as_tensor(plane["b1"], dtype=torch.float32)
        b2 = torch.as_tensor(plane["b2"], dtype=torch.float32)
        b1 = b1 / (b1.norm() + 1e-12)
        b2 = b2 - torch.dot(b2, b1) * b1
        b2 = b2 / (b2.norm() + 1e-12)
        self.register_buffer("b1", b1)
        self.register_buffer("b2", b2)
        self._set_theta(self.attack_degree)

    def _set_theta(self, degree: float) -> None:
        theta = float(degree) % 360.0
        rad = float(np.deg2rad(theta))
        self.target_degree = theta
        self._cos_theta = float(np.cos(rad))
        self._sin_theta = float(np.sin(rad))

    def set_guard_mode(self, mode: str) -> None:
        mode = str(mode).strip().lower()
        if mode not in ("attack", "protect"):
            raise ValueError(f"Unsupported guard mode: {mode}")
        self._guard_mode = mode
        self._set_theta(self.attack_degree if mode == "attack" else self.protect_degree)

    def import_plane_checkpoint(self, plane: dict[str, Any]) -> None:
        """Load plane tensors from a checkpoint dict (eval reload)."""
        with torch.no_grad():
            b1 = torch.as_tensor(plane["b1"], dtype=torch.float32, device=self.b1.device)
            b2 = torch.as_tensor(plane["b2"], dtype=torch.float32, device=self.b2.device)
            b1 = b1 / (b1.norm() + 1e-12)
            b2 = b2 - torch.dot(b2, b1) * b1
            b2 = b2 / (b2.norm() + 1e-12)
            self.b1.copy_(b1)
            self.b2.copy_(b2)
        self.selected_layer = int(plane.get("layer", self.selected_layer))
        self.position = str(plane.get("position", self.position))
        if "attack_degree" in plane:
            self.attack_degree = float(plane["attack_degree"]) % 360.0
        if "protect_degree" in plane:
            self.protect_degree = float(plane["protect_degree"]) % 360.0
        if "adaptive_mode" in plane:
            self.adaptive_mode = int(plane["adaptive_mode"])
        self.set_guard_mode(self._guard_mode)

    def _angular_transform(self, x):
        if isinstance(x, tuple):
            if len(x) == 0:
                return x
            return (self._angular_transform(x[0]),) + x[1:]

        if not hasattr(x, "dtype") and hasattr(x, "__getitem__"):
            try:
                return (self._angular_transform(x[0]),) + tuple(x[1:])
            except Exception:
                return x

        b1 = self.b1
        b2 = self.b2
        try:
            b1 = b1.to(x)
            b2 = b2.to(x)
        except Exception:
            try:
                b1 = b1.to(device=x.device, dtype=x.dtype)
                b2 = b2.to(device=x.device, dtype=x.dtype)
            except Exception:
                pass

        proj1 = x @ b1
        proj2 = x @ b2
        px = proj1.unsqueeze(-1) * b1 + proj2.unsqueeze(-1) * b2
        scale = torch.linalg.vector_norm(px, dim=-1, keepdim=True)
        v_theta = self._cos_theta * b1 + self._sin_theta * b2
        steered = x - px + scale * v_theta

        if self.adaptive_mode == 0:
            return steered

        alignment = x @ b1
        mask = (alignment > 0).unsqueeze(-1)
        return x + mask.to(dtype=x.dtype) * (steered - x)

    def __call__(self, direction=None, best_layer: int = None):
        del direction, best_layer
        n_layers = len(self.module.layers)
        if self.apply_all_layers:
            layer_idxs = range(n_layers)
        else:
            layer_idxs = [max(0, min(self.selected_layer, n_layers - 1))]

        for layer_idx in layer_idxs:
            layer = self.module.layers[layer_idx]
            # Match RDO nnsight intervention site (layer residual input).
            layer.input = self._angular_transform(layer.input)

    def parameters(self):
        return []
