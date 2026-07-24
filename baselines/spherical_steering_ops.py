"""Upstream Spherical-Steering adapted for RDO refusal eval.

Extracts antipodal SaladBench prototypes (mu_harm / mu_safe), then applies
``spherical_geometric_logic`` at a selected layer via nnsight. Attack/protect
swap which prototype is the rotation target (mu_T).
"""

from __future__ import annotations

import gc
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from baselines.angular_steering_ops import _ensure_model_device_attr, unwrap_hf_model

_SPHERICAL_ROOT = Path(__file__).resolve().parent / "src" / "Spherical-Steering"


def _load_spherical_module():
    """Load vendored spherical_steering.py (core math only; no baukit required)."""
    path = _SPHERICAL_ROOT / "spherical_steering.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Spherical-Steering helper: {path}")
    root_str = str(_SPHERICAL_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    mod_name = "_spherical_steering_core"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _tokenize_batch(instructions: list[str], tokenizer, device: torch.device):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": ins}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for ins in instructions
        ]
    else:
        texts = list(instructions)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    return {k: v.to(device) for k, v in enc.items()}


def _extract_last_token_acts(
    hf_model: nn.Module,
    tokenizer,
    instructions: list[str],
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Return {layer_idx: (N, D)} last-token residual activations (layer output)."""
    _ensure_model_device_attr(hf_model)
    device = hf_model.device
    num_layers = int(hf_model.config.num_hidden_layers)
    cache: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}
    handles = []

    def _make_hook(layer_idx: int):
        def hook(_module, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            cache[layer_idx].append(h[:, -1, :].detach().float().cpu())
            return output

        return hook

    module_dict = dict(hf_model.named_modules())
    for layer_idx in range(num_layers):
        name = f"model.layers.{layer_idx}"
        if name not in module_dict:
            raise RuntimeError(
                f"Expected module {name!r} for spherical extract; "
                f"model may not be Llama-style"
            )
        handles.append(module_dict[name].register_forward_hook(_make_hook(layer_idx)))

    try:
        with torch.no_grad():
            for i in range(0, len(instructions), batch_size):
                batch = instructions[i : i + batch_size]
                inputs = _tokenize_batch(batch, tokenizer, device)
                _ = hf_model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return {idx: torch.cat(chunks, dim=0) for idx, chunks in cache.items() if chunks}


def extract_saladbench_prototypes(
    *,
    hf_model: nn.Module,
    tokenizer,
    harmful_instructions: list[str],
    harmless_instructions: list[str],
    n_samples: int = 512,
    batch_size: int = 8,
    kappa: float = 20.0,
    alpha: float = 0.7,
    beta: float = 0.1,
) -> dict[str, Any]:
    """Extract mu_harm / mu_safe and pick layer by max mean-diff norm."""
    harmful = list(harmful_instructions)[:n_samples]
    harmless = list(harmless_instructions)[: min(n_samples, len(harmful))]
    harmful = harmful[: len(harmless)]

    print(
        f"[paper_spherical_steering] Extracting prototypes: "
        f"n_harmful={len(harmful)}, n_harmless={len(harmless)}"
    )

    harmful_acts = _extract_last_token_acts(hf_model, tokenizer, harmful, batch_size)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    harmless_acts = _extract_last_token_acts(hf_model, tokenizer, harmless, batch_size)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_layer = None
    best_norm = -1.0
    best_mu_harm = None
    for layer_idx in sorted(harmful_acts.keys()):
        if layer_idx not in harmless_acts:
            continue
        h = harmful_acts[layer_idx]
        l = harmless_acts[layer_idx]
        h = h / (h.norm(dim=-1, keepdim=True) + 1e-12)
        l = l / (l.norm(dim=-1, keepdim=True) + 1e-12)
        diff = h.mean(dim=0) - l.mean(dim=0)
        nrm = float(diff.norm().item())
        if nrm > best_norm:
            best_norm = nrm
            best_layer = int(layer_idx)
            best_mu_harm = diff / (diff.norm() + 1e-12)

    if best_layer is None or best_mu_harm is None:
        raise RuntimeError("Failed to extract spherical prototypes from SaladBench")

    mu_harm = best_mu_harm.detach().float().cpu()
    mu_safe = (-mu_harm).clone()
    proto = {
        "mu_harm": mu_harm,
        "mu_safe": mu_safe,
        "layer": best_layer,
        "kappa": float(kappa),
        "alpha": float(alpha),
        "beta": float(beta),
        "n_extract_samples": int(len(harmful)),
        "diff_norm": float(best_norm),
    }
    print(
        f"[paper_spherical_steering] Selected layer={best_layer} "
        f"diff_norm={best_norm:.4f} kappa={kappa} alpha={alpha} beta={beta}"
    )
    return proto


def _spherical_geometric_logic_batched(x, mu_T, mu_H, kappa: float, alpha: float, beta: float):
    """Batched port of ``spherical_geometric_logic`` for tensors [..., D].

    nnsight Proxy-safe: no bitwise and, torch.where, torch.stack, or Python
    bool tests on activation values. Gate with float multiply/add like
    PaperAngularSteering (single comparison + soft theta gate).
    """
    try:
        x_f = x.float()
    except Exception:
        x_f = x
    try:
        mu_T = mu_T.to(dtype=x_f.dtype)
        mu_H = mu_H.to(dtype=x_f.dtype)
    except Exception:
        try:
            mu_T = mu_T.float()
            mu_H = mu_H.float()
        except Exception:
            pass

    orig_norm = torch.linalg.vector_norm(x_f, dim=-1, keepdim=True).clamp_min(1e-12)
    x_hat = x_f / orig_norm

    cos_T = (x_hat * mu_T).sum(dim=-1).clamp(-1, 1)
    cos_H = (x_hat * mu_H).sum(dim=-1).clamp(-1, 1)

    # 2-class softmax without torch.stack (Proxy-safe).
    logit_T = float(kappa) * cos_T
    logit_H = float(kappa) * cos_H
    e_T = torch.exp(logit_T)
    e_H = torch.exp(logit_H)
    denom_p = e_T + e_H
    p_T = e_T / denom_p
    p_H = e_H / denom_p
    delta = p_H - p_T

    denom = max(1.0 - float(beta), 1e-6)
    t = float(alpha) * (delta - float(beta)) / denom
    t = torch.clamp(t, 0.0, 1.0)

    theta = torch.acos(cos_T)
    # (delta > beta): one comparison, same pattern as angular steering.
    try:
        mask_dtype = x.dtype
    except Exception:
        mask_dtype = torch.float32
    active = (delta > float(beta)).to(dtype=mask_dtype)
    # Stand-in for (theta >= 1e-4) without a second Proxy bool op:
    # 0 at theta=0, 1 for theta >= 1e-4.
    theta_ok = torch.clamp(theta / 1e-4, 0.0, 1.0)
    gate = active * theta_ok
    t = t * gate

    theta_new = (1.0 - t) * theta
    sin_theta = torch.sin(theta).clamp_min(1e-6)
    u = (x_hat - cos_T.unsqueeze(-1) * mu_T) / sin_theta.unsqueeze(-1)
    x_new_hat = (
        torch.cos(theta_new).unsqueeze(-1) * mu_T
        + torch.sin(theta_new).unsqueeze(-1) * u
    )
    x_new = x_new_hat * orig_norm
    # Must identity-mask: with t=0 and tiny theta, SLERP collapses to mu_T.
    x_out = x_f + gate.unsqueeze(-1) * (x_new - x_f)
    try:
        return x_out.to(dtype=x.dtype)
    except Exception:
        return x_out


class PaperSphericalSteering(nn.Module):
    """Single-layer spherical steering with attack/protect prototype swap."""

    def __init__(
        self,
        module,
        prototype: dict[str, Any],
        *,
        kappa: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        self._guard_mode = "attack"
        self.selected_layer = int(prototype["layer"])
        self.kappa = float(prototype["kappa"] if kappa is None else kappa)
        self.alpha = float(prototype["alpha"] if alpha is None else alpha)
        self.beta = float(prototype["beta"] if beta is None else beta)

        mu_harm = torch.as_tensor(prototype["mu_harm"], dtype=torch.float32)
        mu_safe = torch.as_tensor(prototype["mu_safe"], dtype=torch.float32)
        mu_harm = mu_harm / (mu_harm.norm() + 1e-12)
        mu_safe = mu_safe / (mu_safe.norm() + 1e-12)
        self.register_buffer("mu_harm", mu_harm)
        self.register_buffer("mu_safe", mu_safe)
        # Ensure vendored module is importable (validates install path).
        _load_spherical_module()
        self._sync_active_prototypes()

    def _sync_active_prototypes(self) -> None:
        # Attack: rotate toward harmful when activation looks safe/refusal-like.
        # Protect: rotate toward safe when activation looks harmful-like.
        if self._guard_mode == "attack":
            self._mu_T = self.mu_harm
            self._mu_H = self.mu_safe
        else:
            self._mu_T = self.mu_safe
            self._mu_H = self.mu_harm

    def set_guard_mode(self, mode: str) -> None:
        mode = str(mode).strip().lower()
        if mode not in ("attack", "protect"):
            raise ValueError(f"Unsupported guard mode: {mode}")
        self._guard_mode = mode
        self._sync_active_prototypes()

    def import_prototype_checkpoint(self, prototype: dict[str, Any]) -> None:
        with torch.no_grad():
            mu_harm = torch.as_tensor(
                prototype["mu_harm"], dtype=torch.float32, device=self.mu_harm.device
            )
            mu_safe = torch.as_tensor(
                prototype["mu_safe"], dtype=torch.float32, device=self.mu_safe.device
            )
            mu_harm = mu_harm / (mu_harm.norm() + 1e-12)
            mu_safe = mu_safe / (mu_safe.norm() + 1e-12)
            self.mu_harm.copy_(mu_harm)
            self.mu_safe.copy_(mu_safe)
        self.selected_layer = int(prototype.get("layer", self.selected_layer))
        if "kappa" in prototype:
            self.kappa = float(prototype["kappa"])
        if "alpha" in prototype:
            self.alpha = float(prototype["alpha"])
        if "beta" in prototype:
            self.beta = float(prototype["beta"])
        self.set_guard_mode(self._guard_mode)

    def _spherical_transform(self, x):
        # Mirror PaperAngularSteering: no Python bool/ndim tests on nnsight Proxies.
        if isinstance(x, tuple):
            if len(x) == 0:
                return x
            return (self._spherical_transform(x[0]),) + x[1:]

        if not hasattr(x, "dtype") and hasattr(x, "__getitem__"):
            try:
                return (self._spherical_transform(x[0]),) + tuple(x[1:])
            except Exception:
                return x

        mu_T = self._mu_T
        mu_H = self._mu_H
        try:
            mu_T = mu_T.to(x)
            mu_H = mu_H.to(x)
        except Exception:
            try:
                mu_T = mu_T.to(device=x.device, dtype=torch.float32)
                mu_H = mu_H.to(device=x.device, dtype=torch.float32)
            except Exception:
                pass

        # Apply to the full activation tensor (all tokens). Under RDO's
        # intervene_every_step decode path this is typically last-token shaped;
        # keep this path free of Proxy-unsafe Python conditionals.
        return _spherical_geometric_logic_batched(
            x, mu_T, mu_H, self.kappa, self.alpha, self.beta
        )

    def __call__(self, direction=None, best_layer: int = None):
        del direction, best_layer
        n_layers = len(self.module.layers)
        layer_idx = max(0, min(int(self.selected_layer), n_layers - 1))
        layer = self.module.layers[layer_idx]
        layer.input = self._spherical_transform(layer.input)

    def parameters(self):
        return []
