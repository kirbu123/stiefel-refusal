#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
import seaborn as sns
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook", palette="deep")


def _torch_load(path: Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Backward compatibility with older torch versions.
        return torch.load(path, map_location=map_location)


def _parse_int_list(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_layer_training_info(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"best_layer", "n_layers", "num_opt_layers"} and value:
            data[key] = int(value)
        elif key in {"active_layer_indices", "non_optimized_layer_indices", "frozen_layer_indices"}:
            data[key] = _parse_int_list(value)
        else:
            data[key] = value
    return data


def _resolve_run_dir(run_arg: str) -> Path:
    p = Path(run_arg).expanduser().resolve()
    if p.exists():
        return p
    # Script lives under <repo>/baselines/scripts/, so parents[2] is repo root.
    root = Path(__file__).resolve().parents[2] / "results" / "rdo_refusal" / "tensorboard"
    candidate = root / run_arg
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not resolve run dir '{run_arg}'. "
        f"Tried '{p}' and '{candidate}'."
    )


def _select_checkpoint_file(checkpoints_dir: Path) -> Path:
    progress_dir = checkpoints_dir / "progress_checkpoints"
    if progress_dir.exists():
        candidates = sorted(progress_dir.glob("iters_*.pt"))
        if candidates:
            return candidates[-1]

    raise FileNotFoundError(
        f"No interpretation checkpoint found in '{checkpoints_dir}'. "
        "Expected 'active_layers.pt' or progress checkpoint files."
    )


def _layer_tensor(matrix_stack: torch.Tensor, layer_idx: int, stack_layers: list[int]) -> torch.Tensor:
    if matrix_stack.dim() != 3:
        raise ValueError(f"Expected 3D tensor [layers, *, *], got shape={tuple(matrix_stack.shape)}")

    if matrix_stack.shape[0] == len(stack_layers):
        try:
            pos = stack_layers.index(layer_idx)
        except ValueError as exc:
            raise ValueError(f"Layer {layer_idx} is not present in checkpoint layer index list.") from exc
        return matrix_stack[pos]

    if layer_idx < matrix_stack.shape[0]:
        return matrix_stack[layer_idx]

    raise ValueError(
        f"Cannot map layer {layer_idx} to tensor with first dim {matrix_stack.shape[0]} "
        f"and stack_layers={stack_layers}."
    )


def _plot_layer_matrices(layer_idx: int, proj_a: torch.Tensor, proj_b: torch.Tensor, out_file: Path) -> dict[str, float]:

    proj_a = proj_a.detach().cpu().float()
    proj_b = proj_b.detach().cpu().float()
    composed = proj_a @ proj_b
    singular_vals = torch.linalg.svdvals(composed)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes[0, 0]
    im = ax.imshow(proj_a.numpy(), aspect="auto", cmap="coolwarm")
    ax.set_title(f"Layer {layer_idx} proj_A")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    im = ax.imshow(proj_b.numpy(), aspect="auto", cmap="coolwarm")
    ax.set_title(f"Layer {layer_idx} proj_B")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    im = ax.imshow(composed.numpy(), aspect="auto", cmap="coolwarm")
    ax.set_title(f"Layer {layer_idx} composed M = proj_A @ proj_B")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    ax.plot(singular_vals.numpy(), marker="o")
    ax.set_title(f"Layer {layer_idx} singular values of composed M")
    ax.set_xlabel("index")
    ax.set_ylabel("sigma")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)

    return {
        "fro_norm_proj_A": float(torch.linalg.norm(proj_a).item()),
        "fro_norm_proj_B": float(torch.linalg.norm(proj_b).item()),
        "fro_norm_composed": float(torch.linalg.norm(composed).item()),
        "max_singular_value": float(singular_vals.max().item()),
        "mean_singular_value": float(singular_vals.mean().item()),
    }


def _progress_checkpoint_files(checkpoints_dir: Path) -> list[Path]:
    progress_dir = checkpoints_dir / "progress_checkpoints"
    if not progress_dir.exists():
        return []
    files = sorted(progress_dir.glob("iters_*.pt"))
    if files:
        return files
    return sorted(progress_dir.glob("opt_*.pt"))


def _infer_checkpoint_step(path: Path, payload: dict[str, Any]) -> int | None:
    for key in ("iters_steps", "iters_step", "opt_step", "step"):
        if key in payload:
            try:
                return int(payload[key])
            except Exception:
                pass
    m = re.search(r"(?:iters|opt)_(\d+)$", path.stem)
    if m:
        return int(m.group(1))
    return None


def _plot_progressive_m_norm_evolution(
    checkpoints_dir: Path,
    active_layers: list[int],
    out_file: Path,
) -> dict[str, int]:
    progress_files = _progress_checkpoint_files(checkpoints_dir)
    if not progress_files:
        return {"n_progress_checkpoints": 0, "n_layers_with_series": 0}

    xs: dict[int, list[int]] = {int(layer): [] for layer in active_layers}
    ys: dict[int, list[float]] = {int(layer): [] for layer in active_layers}

    for ckpt_file in progress_files:
        payload = _torch_load(ckpt_file, map_location="cpu")
        if not isinstance(payload, dict):
            continue
        if "proj_A" not in payload or "proj_B" not in payload:
            continue

        step = _infer_checkpoint_step(ckpt_file, payload)
        if step is None:
            continue

        proj_a_stack = payload["proj_A"]
        proj_b_stack = payload["proj_B"]
        if not torch.is_tensor(proj_a_stack) or not torch.is_tensor(proj_b_stack):
            continue

        stack_layers = payload.get("active_layer_indices", active_layers)
        if not isinstance(stack_layers, list):
            stack_layers = active_layers
        stack_layers = [int(i) for i in stack_layers]

        for layer in active_layers:
            try:
                la = _layer_tensor(proj_a_stack, int(layer), stack_layers).float()
                lb = _layer_tensor(proj_b_stack, int(layer), stack_layers).float()
            except Exception:
                continue
            composed = la @ lb
            xs[int(layer)].append(step)
            ys[int(layer)].append(float(torch.linalg.norm(composed).item()))

    fig, ax = plt.subplots(figsize=(12, 6))
    plotted_layers = 0
    for layer in active_layers:
        lx = xs[int(layer)]
        ly = ys[int(layer)]
        if not lx:
            continue
        plotted_layers += 1
        ax.plot(lx, ly, marker="o", linewidth=1.5, label=f"layer {int(layer)}")

    ax.set_title("Progressive checkpoint evolution: ||M||_F per active layer")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("Frobenius norm of M = proj_A @ proj_B")
    ax.grid(True, alpha=0.3)
    if plotted_layers > 0:
        ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)

    return {
        "n_progress_checkpoints": len(progress_files),
        "n_layers_with_series": plotted_layers,
    }


def _plot_progressive_rotation_angles(
    checkpoints_dir: Path,
    active_layers: list[int],
    out_file: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Plot progressive angle evolution for each active layer using one random vector v.

    For each checkpoint and layer:
      M = proj_A @ proj_B
      v_rot = M @ v
      angle = arccos( <v, v_rot> / (||v|| * ||v_rot||) )
    """
    progress_files = _progress_checkpoint_files(checkpoints_dir)
    if not progress_files:
        return {"n_progress_checkpoints": 0, "n_layers_with_series": 0}

    xs: dict[int, list[int]] = {int(layer): [] for layer in active_layers}
    ys_v_to_mv: dict[int, list[float]] = {int(layer): [] for layer in active_layers}
    ys_v_to_delta: dict[int, list[float]] = {int(layer): [] for layer in active_layers}
    ys_delta_to_mv: dict[int, list[float]] = {int(layer): [] for layer in active_layers}
    random_vectors: dict[int, torch.Tensor] = {}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    for ckpt_file in progress_files:
        payload = _torch_load(ckpt_file, map_location="cpu")
        if not isinstance(payload, dict):
            continue
        if "proj_A" not in payload or "proj_B" not in payload:
            continue

        step = _infer_checkpoint_step(ckpt_file, payload)
        if step is None:
            continue

        proj_a_stack = payload["proj_A"]
        proj_b_stack = payload["proj_B"]
        if not torch.is_tensor(proj_a_stack) or not torch.is_tensor(proj_b_stack):
            continue

        stack_layers = payload.get("active_layer_indices", active_layers)
        if not isinstance(stack_layers, list):
            stack_layers = active_layers
        stack_layers = [int(i) for i in stack_layers]

        for layer in active_layers:
            try:
                la = _layer_tensor(proj_a_stack, int(layer), stack_layers).float()
                lb = _layer_tensor(proj_b_stack, int(layer), stack_layers).float()
            except Exception:
                continue

            composed = la @ lb
            dim = int(composed.shape[1])
            if int(layer) not in random_vectors:
                v = torch.randn(dim, generator=generator, dtype=torch.float32)
                v = v / (torch.linalg.norm(v) + 1e-12)
                random_vectors[int(layer)] = v
            v = random_vectors[int(layer)]

            v_rot = composed @ v
            v_delta = v - v_rot
            norm_v = float(torch.linalg.norm(v).item())
            norm_rot = float(torch.linalg.norm(v_rot).item())
            norm_delta = float(torch.linalg.norm(v_delta).item())
            if norm_v <= 0 or norm_rot <= 0 or norm_delta <= 0:
                continue
            cos_v_mv = float(torch.dot(v, v_rot).item() / (norm_v * norm_rot + 1e-12))
            cos_v_delta = float(torch.dot(v, v_delta).item() / (norm_v * norm_delta + 1e-12))
            cos_delta_mv = float(torch.dot(v_delta, v_rot).item() / (norm_delta * norm_rot + 1e-12))
            cos_v_mv = max(-1.0, min(1.0, cos_v_mv))
            cos_v_delta = max(-1.0, min(1.0, cos_v_delta))
            cos_delta_mv = max(-1.0, min(1.0, cos_delta_mv))
            angle_v_mv = float(torch.arccos(torch.tensor(cos_v_mv)).item())
            angle_v_delta = float(torch.arccos(torch.tensor(cos_v_delta)).item())
            angle_delta_mv = float(torch.arccos(torch.tensor(cos_delta_mv)).item())

            xs[int(layer)].append(int(step))
            ys_v_to_mv[int(layer)].append(angle_v_mv)
            ys_v_to_delta[int(layer)].append(angle_v_delta)
            ys_delta_to_mv[int(layer)].append(angle_delta_mv)

    fig, ax = plt.subplots(figsize=(12, 6))
    plotted_layers = 0
    for layer in active_layers:
        lx = xs[int(layer)]
        ly = ys_v_to_mv[int(layer)]
        if not lx:
            continue
        plotted_layers += 1
        pairs = sorted(zip(lx, ly), key=lambda t: t[0])
        sx = [p[0] for p in pairs]
        sy = [p[1] for p in pairs]
        ax.plot(sx, sy, marker="o", linewidth=1.5, label=f"layer {int(layer)}")

    ax.set_title("Progressive checkpoint evolution: angle(v, Mv) per active layer")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("angle in radians")
    ax.grid(True, alpha=0.3)
    if plotted_layers > 0:
        ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)

    separate_out_file = out_file.with_name(f"{out_file.stem}_three_vector_angles.png")
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharex=False, sharey=False)
    angle_specs = [
        ("angle(v, Mv)", ys_v_to_mv),
        ("angle(v, v - Mv)", ys_v_to_delta),
        ("angle(v - Mv, Mv)", ys_delta_to_mv),
    ]
    for ax_i, (title, ydict) in zip(axes, angle_specs):
        local_plotted = 0
        for layer in active_layers:
            lx = xs[int(layer)]
            ly = ydict[int(layer)]
            if not lx:
                continue
            local_plotted += 1
            pairs = sorted(zip(lx, ly), key=lambda t: t[0])
            sx = [p[0] for p in pairs]
            sy = [p[1] for p in pairs]
            ax_i.plot(sx, sy, marker="o", linewidth=1.4, label=f"layer {int(layer)}")
        ax_i.set_title(f"Progressive {title}")
        ax_i.set_xlabel("checkpoint step")
        ax_i.set_ylabel("angle in radians")
        ax_i.grid(True, alpha=0.3)
        if local_plotted > 0:
            ax_i.legend(loc="best", fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(separate_out_file, dpi=180)
    plt.close(fig)

    return {
        "n_progress_checkpoints": len(progress_files),
        "n_layers_with_series": plotted_layers,
        "separate_three_vector_plot": str(separate_out_file),
    }


def _infer_model_id(hparams: dict[str, Any], run_dir: Path) -> str:
    model_id = hparams.get("model_id")
    if isinstance(model_id, str) and model_id.strip():
        return model_id.strip()
    model_name = hparams.get("model")
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip().split("/")[-1]
    m = re.search(r"(?:basic|rdo)_rdo_(.+?)_shtiefel", run_dir.name)
    if m:
        return m.group(1)
    raise ValueError("Could not infer model_id from hparams or run directory name.")


def _load_refusal_targets(
    repo_root: Path,
    model_id: str,
    split_name: str,
) -> tuple[list[str], list[str]]:
    targets_dir = repo_root / "results" / "rdo_refusal" / "rdo" / model_id / split_name / "targets"
    harmful_path = targets_dir / "harmful_targets.json"
    harmless_path = targets_dir / "harmless_targets.json"
    if not harmful_path.exists() or not harmless_path.exists():
        raise FileNotFoundError(
            f"Missing target files for model='{model_id}' split='{split_name}' under '{targets_dir}'."
        )
    harmful_raw = json.loads(harmful_path.read_text(encoding="utf-8"))
    harmless_raw = json.loads(harmless_path.read_text(encoding="utf-8"))

    harmful_texts: list[str] = []
    harmless_texts: list[str] = []

    if isinstance(harmful_raw, list):
        for item in harmful_raw:
            if not isinstance(item, dict):
                continue
            prompt = str(item.get("prompt") or "").strip()
            ablation = str(item.get("ablation") or "").strip()
            text = f"{prompt} {ablation}".strip()
            if text:
                harmful_texts.append(text)

    if isinstance(harmless_raw, list):
        for item in harmless_raw:
            if not isinstance(item, dict):
                continue
            prompt = str(item.get("prompt") or "").strip()
            retain = str(item.get("retain") or "").strip()
            addition = str(item.get("addition") or "").strip()
            text = f"{prompt} {retain}".strip() if retain else f"{prompt} {addition}".strip()
            if text:
                harmless_texts.append(text)

    if not harmful_texts or not harmless_texts:
        raise ValueError(
            f"Loaded empty harmful/harmless texts from '{targets_dir}'. "
            f"harmful={len(harmful_texts)}, harmless={len(harmless_texts)}"
        )
    return harmful_texts, harmless_texts


def _mean_hidden_state_from_texts(
    texts: list[str],
    model_name: str,
    layer_idx: int,
    token_pos: int,
    *,
    batch_size: int = 8,
    max_length: int = 512,
) -> torch.Tensor:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not texts:
        raise ValueError("texts cannot be empty.")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    sum_vec: torch.Tensor | None = None
    n = 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = model(**encoded, output_hidden_states=True)
            hidden_states = out.hidden_states
            if hidden_states is None:
                raise ValueError("Model forward did not return hidden_states.")
            if layer_idx < 0 or layer_idx >= len(hidden_states):
                raise ValueError(
                    f"Invalid layer_idx={layer_idx} for hidden_states length={len(hidden_states)}."
                )
            layer_hidden = hidden_states[layer_idx].detach().float().cpu()
            attn_mask = encoded["attention_mask"].detach().cpu()
            seq_lens = attn_mask.sum(dim=1).long()

            for b in range(layer_hidden.shape[0]):
                seq_len = int(seq_lens[b].item())
                if seq_len <= 0:
                    continue
                if token_pos >= 0:
                    pos_idx = min(token_pos, seq_len - 1)
                else:
                    pos_idx = max(0, seq_len + token_pos)
                vec = layer_hidden[b, pos_idx, :]
                if sum_vec is None:
                    sum_vec = torch.zeros_like(vec)
                sum_vec += vec
                n += 1

    if sum_vec is None or n == 0:
        raise ValueError("No valid hidden states extracted from target texts.")
    return sum_vec / float(n)


def _plot_progressive_refusal_andgles(
    run_dir: Path,
    checkpoints_dir: Path,
    hparams: dict[str, Any],
    active_layers: list[int],
    out_file: Path,
    pca_out_file: Path,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    model_id = _infer_model_id(hparams, run_dir)
    split_name = str(hparams.get("splits", "saladbench"))
    model_name = str(hparams.get("model", model_id))
    harmful_texts, harmless_texts = _load_refusal_targets(repo_root, model_id, split_name)

    direction_dir = repo_root / "results" / "rdo_refusal" / "dim" / model_id
    direction_path = direction_dir / "direction.pt"
    direction_meta_path = direction_dir / "direction_metadata.json"
    if not direction_path.exists():
        raise FileNotFoundError(f"Missing refusal direction file: '{direction_path}'.")
    if not direction_meta_path.exists():
        raise FileNotFoundError(f"Missing direction metadata file: '{direction_meta_path}'.")

    direction_meta = json.loads(direction_meta_path.read_text(encoding="utf-8"))
    direction_layer = int(direction_meta.get("layer", active_layers[0]))
    direction_pos = int(direction_meta.get("pos", -1))
    refusal_direction = _torch_load(direction_path, map_location="cpu").detach().float().flatten()

    target_layer = direction_layer if direction_layer in [int(x) for x in active_layers] else int(active_layers[0])
    x_hf = _mean_hidden_state_from_texts(
        harmful_texts,
        model_name=model_name,
        layer_idx=target_layer,
        token_pos=direction_pos,
    )
    x_harmless = _mean_hidden_state_from_texts(
        harmless_texts,
        model_name=model_name,
        layer_idx=target_layer,
        token_pos=direction_pos,
    )

    dim = int(x_hf.numel())
    if refusal_direction.numel() != dim:
        raise ValueError(
            f"Refusal direction dimension mismatch: direction={refusal_direction.numel()} vs x_hf={dim}."
        )
    x_hf_minus_refusal = x_hf - refusal_direction

    progress_files = _progress_checkpoint_files(checkpoints_dir)
    if not progress_files:
        return {
            "n_progress_checkpoints": 0,
            "n_points_plotted": 0,
            "model_id": model_id,
            "split_name": split_name,
        }

    steps: list[int] = []
    angle_hf_to_hf_minus_refusal: list[float] = []
    angle_hf_to_rot: list[float] = []
    angle_hf_minus_refusal_to_rot: list[float] = []
    rotated_vectors: list[torch.Tensor] = []

    norm_hf = float(torch.linalg.norm(x_hf).item())
    norm_hf_minus_refusal = float(torch.linalg.norm(x_hf_minus_refusal).item())
    if norm_hf <= 0 or norm_hf_minus_refusal <= 0:
        raise ValueError("Invalid zero-norm vector encountered for x_hf or x_hf - refusal_direction.")

    for ckpt_file in progress_files:
        payload = _torch_load(ckpt_file, map_location="cpu")
        if not isinstance(payload, dict):
            continue
        if "proj_A" not in payload or "proj_B" not in payload:
            continue
        step = _infer_checkpoint_step(ckpt_file, payload)
        if step is None:
            continue

        proj_a_stack = payload["proj_A"]
        proj_b_stack = payload["proj_B"]
        if not torch.is_tensor(proj_a_stack) or not torch.is_tensor(proj_b_stack):
            continue

        stack_layers = payload.get("active_layer_indices", active_layers)
        if not isinstance(stack_layers, list):
            stack_layers = active_layers
        stack_layers = [int(i) for i in stack_layers]
        try:
            la = _layer_tensor(proj_a_stack, int(target_layer), stack_layers).float()
            lb = _layer_tensor(proj_b_stack, int(target_layer), stack_layers).float()
        except Exception:
            continue
        composed = la @ lb
        x_rot = composed @ x_hf
        norm_rot = float(torch.linalg.norm(x_rot).item())
        if norm_rot <= 0:
            continue

        cos_hf_hf_minus = float(torch.dot(x_hf, x_hf_minus_refusal).item() / (norm_hf * norm_hf_minus_refusal + 1e-12))
        cos_hf_rot = float(torch.dot(x_hf, x_rot).item() / (norm_hf * norm_rot + 1e-12))
        cos_hf_minus_rot = float(
            torch.dot(x_hf_minus_refusal, x_rot).item() / (norm_hf_minus_refusal * norm_rot + 1e-12)
        )
        cos_hf_hf_minus = max(-1.0, min(1.0, cos_hf_hf_minus))
        cos_hf_rot = max(-1.0, min(1.0, cos_hf_rot))
        cos_hf_minus_rot = max(-1.0, min(1.0, cos_hf_minus_rot))

        steps.append(int(step))
        angle_hf_to_hf_minus_refusal.append(float(torch.arccos(torch.tensor(cos_hf_hf_minus)).item()))
        angle_hf_to_rot.append(float(torch.arccos(torch.tensor(cos_hf_rot)).item()))
        angle_hf_minus_refusal_to_rot.append(float(torch.arccos(torch.tensor(cos_hf_minus_rot)).item()))
        rotated_vectors.append(x_rot.detach().float().cpu())

    if not steps:
        return {
            "n_progress_checkpoints": len(progress_files),
            "n_points_plotted": 0,
            "model_id": model_id,
            "split_name": split_name,
        }

    pairs = sorted(
        zip(steps, angle_hf_to_hf_minus_refusal, angle_hf_to_rot, angle_hf_minus_refusal_to_rot, rotated_vectors),
        key=lambda t: t[0],
    )
    steps = [p[0] for p in pairs]
    angle_hf_to_hf_minus_refusal = [p[1] for p in pairs]
    angle_hf_to_rot = [p[2] for p in pairs]
    angle_hf_minus_refusal_to_rot = [p[3] for p in pairs]
    rotated_vectors = [p[4] for p in pairs]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, angle_hf_to_hf_minus_refusal, marker="o", linewidth=1.5, label="angle(x_hf, x_hf - refusal_direction)")
    ax.plot(steps, angle_hf_to_rot, marker="o", linewidth=1.5, label="angle(x_hf, M @ x_hf)")
    ax.plot(
        steps,
        angle_hf_minus_refusal_to_rot,
        marker="o",
        linewidth=1.5,
        label="angle(x_hf - refusal_direction, M @ x_hf)",
    )
    ax.set_title(f"Progressive refusal angles (layer={target_layer}, pos={direction_pos})")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("angle in radians")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)

    all_vectors = [x_hf, x_harmless, x_hf_minus_refusal] + rotated_vectors
    labels = ["x_hf", "x_harmless", "x_hf_minus_refusal"] + [f"M@x_hf@{s}" for s in steps]
    mat = torch.stack([v.detach().float().cpu() for v in all_vectors], dim=0)
    mat_centered = mat - mat.mean(dim=0, keepdim=True)
    q = min(3, mat_centered.shape[0], mat_centered.shape[1])
    if q < 3:
        pca_coords = torch.zeros((mat_centered.shape[0], 3), dtype=torch.float32)
        pca_coords[:, :q] = mat_centered[:, :q]
    else:
        _, _, v = torch.pca_lowrank(mat_centered, q=3)
        pca_coords = mat_centered @ v[:, :3]

    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(111, projection="3d")
    coords = pca_coords.numpy()
    ax3d.scatter(coords[0, 0], coords[0, 1], coords[0, 2], c="red", marker="*", s=180, label="x_hf")
    ax3d.scatter(coords[1, 0], coords[1, 1], coords[1, 2], c="green", marker="*", s=180, label="x_harmless")
    ax3d.scatter(coords[2, 0], coords[2, 1], coords[2, 2], c="black", marker="x", s=110, label="x_hf - refusal")

    start = 3
    end = start + len(rotated_vectors)
    if end > start:
        rot = coords[start:end]
        ax3d.plot(rot[:, 0], rot[:, 1], rot[:, 2], color="tab:blue", alpha=0.6, linewidth=1.5)
        ax3d.scatter(rot[:, 0], rot[:, 1], rot[:, 2], c=range(len(rot)), cmap="Blues", s=55, label="M@x_hf trajectory")
        ax3d.text(rot[0, 0], rot[0, 1], rot[0, 2], f"step={steps[0]}")
        ax3d.text(rot[-1, 0], rot[-1, 1], rot[-1, 2], f"step={steps[-1]}")

    ax3d.set_title("3D PCA relation: x_hf, x_hf-refusal, harmless mean, and M@x_hf")
    ax3d.set_xlabel("PC1")
    ax3d.set_ylabel("PC2")
    ax3d.set_zlabel("PC3")
    ax3d.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(pca_out_file, dpi=180)
    plt.close(fig)

    return {
        "n_progress_checkpoints": len(progress_files),
        "n_points_plotted": len(steps),
        "target_layer": target_layer,
        "target_position": direction_pos,
        "model_id": model_id,
        "split_name": split_name,
        "n_harmful_targets": len(harmful_texts),
        "n_harmless_targets": len(harmless_texts),
        "pca_labels": labels,
    }


def _plot_all_progressive_layer_matrices(
    checkpoints_dir: Path,
    active_layers: list[int],
    outputs_dir: Path,
) -> dict[str, int]:
    """
    Generate layer interpretation plots for every progressive checkpoint.
    Output cardinality is up to: len(active_layers) x len(progress_checkpoints).
    """
    progress_files = _progress_checkpoint_files(checkpoints_dir)
    if not progress_files:
        return {"n_progress_checkpoints": 0, "n_layer_checkpoint_plots": 0}

    per_ckpt_dir = outputs_dir / "progressive_layer_plots"
    per_ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_plots = 0

    for ckpt_file in progress_files:
        payload = _torch_load(ckpt_file, map_location="cpu")
        if not isinstance(payload, dict):
            continue
        if "proj_A" not in payload or "proj_B" not in payload:
            continue

        step = _infer_checkpoint_step(ckpt_file, payload)
        if step is None:
            continue

        proj_a_stack = payload["proj_A"]
        proj_b_stack = payload["proj_B"]
        if not torch.is_tensor(proj_a_stack) or not torch.is_tensor(proj_b_stack):
            continue

        stack_layers = payload.get("active_layer_indices", active_layers)
        if not isinstance(stack_layers, list):
            stack_layers = active_layers
        stack_layers = [int(i) for i in stack_layers]

        for layer_idx in active_layers:
            try:
                la = _layer_tensor(proj_a_stack, int(layer_idx), stack_layers)
                lb = _layer_tensor(proj_b_stack, int(layer_idx), stack_layers)
            except Exception:
                continue
            out_file = per_ckpt_dir / f"step_{int(step):08d}_layer_{int(layer_idx):03d}_proj_interp.png"
            _plot_layer_matrices(int(layer_idx), la, lb, out_file)
            n_plots += 1

    return {
        "n_progress_checkpoints": len(progress_files),
        "n_layer_checkpoint_plots": n_plots,
    }


def _plot_diag_and_sorted_distributions(
    layer_idx: int,
    step: int,
    proj_a: torch.Tensor,
    proj_b: torch.Tensor,
    out_file: Path,
) -> None:
    proj_a = proj_a.detach().cpu().float()
    proj_b = proj_b.detach().cpu().float()
    composed = proj_a @ proj_b

    diag_vals = torch.diagonal(composed).numpy()
    sorted_proj_a = torch.sort(proj_a.reshape(-1))[0].numpy()
    sorted_proj_b = torch.sort(proj_b.reshape(-1))[0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(diag_vals, bins=80, alpha=0.75, color="tab:blue")
    ax.set_title(f"Layer {layer_idx} step {step}: diag(M) distribution")
    ax.set_xlabel("diag(M) value")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(sorted_proj_a, linewidth=1.2, label="sorted proj_A values")
    ax.plot(sorted_proj_b, linewidth=1.2, label="sorted proj_B values")
    ax.set_title(f"Layer {layer_idx} step {step}: sorted value distributions")
    ax.set_xlabel("sorted index")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def _plot_all_progressive_diag_distributions(
    checkpoints_dir: Path,
    active_layers: list[int],
    outputs_dir: Path,
) -> dict[str, int]:
    progress_files = _progress_checkpoint_files(checkpoints_dir)
    if not progress_files:
        return {"n_progress_checkpoints": 0, "n_diag_distribution_plots": 0}

    per_ckpt_dir = outputs_dir / "progressive_diag_distributions"
    per_ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_plots = 0

    for ckpt_file in progress_files:
        payload = _torch_load(ckpt_file, map_location="cpu")
        if not isinstance(payload, dict):
            continue
        if "proj_A" not in payload or "proj_B" not in payload:
            continue

        step = _infer_checkpoint_step(ckpt_file, payload)
        if step is None:
            continue

        proj_a_stack = payload["proj_A"]
        proj_b_stack = payload["proj_B"]
        if not torch.is_tensor(proj_a_stack) or not torch.is_tensor(proj_b_stack):
            continue

        stack_layers = payload.get("active_layer_indices", active_layers)
        if not isinstance(stack_layers, list):
            stack_layers = active_layers
        stack_layers = [int(i) for i in stack_layers]

        for layer_idx in active_layers:
            try:
                la = _layer_tensor(proj_a_stack, int(layer_idx), stack_layers)
                lb = _layer_tensor(proj_b_stack, int(layer_idx), stack_layers)
            except Exception:
                continue
            out_file = per_ckpt_dir / f"step_{int(step):08d}_layer_{int(layer_idx):03d}_diag_sorted_dist.png"
            _plot_diag_and_sorted_distributions(int(layer_idx), int(step), la, lb, out_file)
            n_plots += 1

    return {
        "n_progress_checkpoints": len(progress_files),
        "n_diag_distribution_plots": n_plots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpret rotation checkpoints for RDO runs.")
    parser.add_argument(
        "run_subdir",
        type=str,
        help=(
            "TensorBoard run subdir name or full path. "
            "Example: basic_rdo_DeepSeek-R1-Distill-Qwen-7B_shtiefel_proj_rot_nol0_f0f0a75a6e90"
        ),
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_subdir)
    checkpoints_dir = run_dir / "checkpoints"
    outputs_dir = run_dir / "rotation_interpretation"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    info_path = checkpoints_dir / "layer_training_info.txt"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing '{info_path}'.")
    layer_info = _parse_layer_training_info(info_path)
    active_layers = layer_info.get("active_layer_indices", [])
    if not active_layers:
        raise ValueError("No active_layer_indices found in layer_training_info.txt")

    hparams_path = run_dir / "hparams.json"
    if not hparams_path.exists():
        raise FileNotFoundError(f"Missing '{hparams_path}'.")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    direction_mode = hparams.get("direction_mode")
    assert direction_mode == "shtiefel_proj_rot", (
        f"NotImplemented for direction_mode='{direction_mode}'. "
        "Only 'shtiefel_proj_rot' is supported."
    )

    checkpoint_file = _select_checkpoint_file(checkpoints_dir)
    payload = _torch_load(checkpoint_file, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint '{checkpoint_file}' is not a dict payload.")
    if "proj_A" not in payload or "proj_B" not in payload:
        raise ValueError(f"Checkpoint '{checkpoint_file}' does not contain both 'proj_A' and 'proj_B'.")

    proj_a_stack = payload["proj_A"]
    proj_b_stack = payload["proj_B"]
    if not torch.is_tensor(proj_a_stack) or not torch.is_tensor(proj_b_stack):
        raise ValueError("Expected tensor values for proj_A and proj_B.")

    stack_layers = payload.get("active_layer_indices", active_layers)
    if not isinstance(stack_layers, list):
        stack_layers = active_layers
    stack_layers = [int(i) for i in stack_layers]

    layer_stats: dict[str, dict[str, float]] = {}
    for layer_idx in active_layers:
        la = _layer_tensor(proj_a_stack, layer_idx, stack_layers)
        lb = _layer_tensor(proj_b_stack, layer_idx, stack_layers)
        out_file = outputs_dir / f"layer_{layer_idx:03d}_proj_interp.png"
        layer_stats[str(layer_idx)] = _plot_layer_matrices(layer_idx, la, lb, out_file)

    evolution_plot = outputs_dir / "progressive_m_norm_evolution.png"
    evolution_info = _plot_progressive_m_norm_evolution(
        checkpoints_dir=checkpoints_dir,
        active_layers=active_layers,
        out_file=evolution_plot,
    )
    angle_evolution_plot = outputs_dir / "progressive_rotation_angles.png"
    angle_evolution_info = _plot_progressive_rotation_angles(
        checkpoints_dir=checkpoints_dir,
        active_layers=active_layers,
        out_file=angle_evolution_plot,
    )
    refusal_angle_plot = outputs_dir / "progressive_refusal_angles.png"
    refusal_pca_plot = outputs_dir / "progressive_refusal_pca_3d.png"
    refusal_angle_info = _plot_progressive_refusal_andgles(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        hparams=hparams,
        active_layers=active_layers,
        out_file=refusal_angle_plot,
        pca_out_file=refusal_pca_plot,
    )
    # progressive_plots_info = _plot_all_progressive_layer_matrices(
    #     checkpoints_dir=checkpoints_dir,
    #     active_layers=active_layers,
    #     outputs_dir=outputs_dir,
    # )
    progressive_plots_info = {
        "enabled": False,
        "n_progress_checkpoints": 0,
        "n_layer_checkpoint_plots": 0,
    }
    progressive_diag_dist_info = _plot_all_progressive_diag_distributions(
        checkpoints_dir=checkpoints_dir,
        active_layers=active_layers,
        outputs_dir=outputs_dir,
    )

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_used": str(checkpoint_file),
        "direction_mode": direction_mode,
        "active_layer_indices": active_layers,
        "layer_stats": layer_stats,
        "progressive_norm_evolution_plot": str(evolution_plot),
        "progressive_norm_evolution_info": evolution_info,
        "progressive_rotation_angles_plot": str(angle_evolution_plot),
        "progressive_rotation_angles_info": angle_evolution_info,
        "progressive_refusal_angles_plot": str(refusal_angle_plot),
        "progressive_refusal_pca_plot": str(refusal_pca_plot),
        "progressive_refusal_angles_info": refusal_angle_info,
        "progressive_layer_plots_dir": str(outputs_dir / "progressive_layer_plots"),
        "progressive_layer_plots_info": progressive_plots_info,
        "progressive_diag_distributions_dir": str(outputs_dir / "progressive_diag_distributions"),
        "progressive_diag_distributions_info": progressive_diag_dist_info,
    }
    (outputs_dir / "rotation_interpret_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Saved interpretation outputs to: {outputs_dir}")


if __name__ == "__main__":
    main()
