#!/usr/bin/env python3
"""Compare start/end best-layer activations across RDO runs as vectors from origin.

Fits an uncentered 3D PCA (SVD) so plot origin (0,0,0) corresponds to the zero
activation. Draws only:
  - start / final dots per experiment
  - vectors from origin to those dots
  - DIM refusal direction from origin
No shared-start alignment.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIM_ROOT = PROJECT_ROOT / "results" / "rdo_refusal" / "dim"
ACTIVATION_SUBDIR = Path("checkpoints") / "tmp_best_layer_activations"

EXP_COLORS = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
)

_DIRECTION_MODE_DISPLAY = {
    "baseline": "RDO",
    "activation_additive_rot": "Cayley Steering",
    "shtiefel_additive_rot": "Stiefel Rotation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot uncentered 3D PCA of start/end activations from multiple "
            "experiment logdirs as origin vectors, plus DIM refusal direction."
        )
    )
    parser.add_argument(
        "--exp-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Experiment TensorBoard / result logdirs containing activation dumps",
    )
    parser.add_argument(
        "--dim-root",
        type=Path,
        default=DEFAULT_DIM_ROOT,
        help="DIM root containing <model_id>/direction.pt and direction_metadata.json",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Optional model id under --dim-root; inferred from hparams if omitted",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path (default: "
            "<first-exp>/checkpoints/activation_pca_vector_compare_3d.png)"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG resolution",
    )
    parser.add_argument(
        "--direction-scale",
        type=float,
        default=None,
        help=(
            "Refusal arrow length in PC coordinates. "
            "Default: 0.28 * median ||start/end|| in PC space (compact direction cue)"
        ),
    )
    return parser.parse_args()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_snapshots(exp_dir: Path) -> tuple[list[torch.Tensor], list[int]]:
    dump_dir = exp_dir / ACTIVATION_SUBDIR
    if not dump_dir.is_dir():
        raise FileNotFoundError(
            f"Missing activation dumps for {exp_dir}: expected {dump_dir}. "
            "Re-run with --eval_activation_pca --keep_activation_pca_dumps"
        )
    points: list[torch.Tensor] = []
    steps: list[int] = []
    for path in sorted(dump_dir.glob("step_*.pt")):
        payload = _torch_load(path)
        if isinstance(payload, dict):
            activation = payload.get("activation")
            step = int(payload.get("step", len(steps)))
        else:
            activation = payload
            step = len(steps)
        if activation is None:
            continue
        points.append(torch.as_tensor(activation).float().reshape(-1))
        steps.append(step)
    if len(points) < 2:
        raise ValueError(f"{dump_dir} has fewer than 2 snapshots ({len(points)})")
    return points, steps


def _read_hparams(exp_dir: Path) -> dict[str, Any]:
    path = exp_dir / "hparams.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_model_id(exp_dirs: list[Path], explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates: list[str] = []
    for exp_dir in exp_dirs:
        hp = _read_hparams(exp_dir)
        for key in ("model_id", "model"):
            value = hp.get(key)
            if value:
                text = str(value).strip()
                candidates.append(text.rsplit("/", 1)[-1])
                break
        else:
            match = re.search(r"basic_rdo_([^_]+(?:-[^_]+)*)_", exp_dir.name)
            if match:
                candidates.append(match.group(1))
    if not candidates:
        raise ValueError("Could not infer model id; pass --model-id explicitly")
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise ValueError(
            f"Conflicting model ids across experiments: {unique}; pass --model-id"
        )
    return unique[0]


def _pretty_direction_label(mode: str) -> str:
    mode = str(mode).strip()
    return _DIRECTION_MODE_DISPLAY.get(mode, mode)


def _exp_label(exp_dir: Path) -> str:
    hp = _read_hparams(exp_dir)
    mode = str(hp.get("direction_mode") or "").strip()
    if mode:
        return _pretty_direction_label(mode)
    for token in (
        "activation_additive_rot",
        "shtiefel_additive_rot",
        "activation_rot",
        "shtiefel_proj_rot",
        "shtiefel_rot",
        "baseline",
        "paper_angular_steering",
        "paper_spherical_steering",
    ):
        if token in exp_dir.name:
            return _pretty_direction_label(token)
    return exp_dir.name


def _load_refusal_direction(dim_root: Path, model_id: str) -> tuple[torch.Tensor, int]:
    model_dir = dim_root / model_id
    direction_path = model_dir / "direction.pt"
    metadata_path = model_dir / "direction_metadata.json"
    if not direction_path.is_file():
        raise FileNotFoundError(f"Missing DIM direction: {direction_path}")
    direction = torch.as_tensor(_torch_load(direction_path)).float().reshape(-1)
    best_layer = -1
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        best_layer = int(metadata.get("layer", -1))
    if direction.norm() <= 0:
        raise ValueError(f"DIM refusal direction has zero norm: {direction_path}")
    return direction, best_layer


def _project_uncentered_pca(
    matrices: list[torch.Tensor],
) -> tuple[np.ndarray, torch.Tensor]:
    """Uncentered SVD PCA so zero activation maps to plot origin.

    Returns (N,3) projections and basis (D,3).
    """
    mat = torch.stack(matrices, dim=0).float()
    q = min(3, mat.shape[0], mat.shape[1])
    # Economy SVD: mat ≈ U @ diag(S) @ Vh, Vh rows are right singular vectors.
    _, _, vh = torch.linalg.svd(mat, full_matrices=False)
    basis = vh[:q].T.contiguous()
    proj = (mat @ basis).numpy()
    if proj.shape[1] < 3:
        pad = np.zeros((proj.shape[0], 3 - proj.shape[1]), dtype=proj.dtype)
        proj = np.concatenate([proj, pad], axis=1)
    if basis.shape[1] < 3:
        pad = torch.zeros(basis.shape[0], 3 - basis.shape[1], dtype=basis.dtype)
        basis = torch.cat([basis, pad], dim=1)
    return proj, basis


def _project_point(point: torch.Tensor, basis: torch.Tensor) -> np.ndarray:
    vec = point.float().reshape(-1) @ basis
    out = vec.detach().cpu().numpy().reshape(-1)
    if out.shape[0] < 3:
        out = np.concatenate([out, np.zeros(3 - out.shape[0], dtype=out.dtype)])
    return out[:3]


def _draw_origin_vector(
    ax,
    xyz: np.ndarray,
    *,
    color: str,
    lw: float = 1.0,
    tip_frac: float = 0.10,
) -> None:
    """Thin shaft from origin + a small tip arrow (avoids huge 3D quiver heads)."""
    xyz = np.asarray(xyz, dtype=float).reshape(3)
    ax.plot(
        [0.0, float(xyz[0])],
        [0.0, float(xyz[1])],
        [0.0, float(xyz[2])],
        color=color,
        lw=lw,
        alpha=0.9,
        zorder=2,
    )
    tip_frac = float(np.clip(tip_frac, 0.02, 0.4))
    base = (1.0 - tip_frac) * xyz
    tip = tip_frac * xyz
    ax.quiver(
        float(base[0]),
        float(base[1]),
        float(base[2]),
        float(tip[0]),
        float(tip[1]),
        float(tip[2]),
        color=color,
        linewidth=lw,
        arrow_length_ratio=0.45,
        normalize=False,
        zorder=3,
    )


def plot_vector_compare(
    *,
    endpoints: list[dict[str, Any]],
    refusal_direction: torch.Tensor,
    output: Path,
    dpi: int,
    direction_scale: float | None,
) -> None:
    starts = [item["start"] for item in endpoints]
    ends = [item["end"] for item in endpoints]
    fit_points = starts + ends

    proj_all, basis = _project_uncentered_pca(fit_points)
    n_exp = len(endpoints)
    start_xyz = proj_all[:n_exp]
    end_xyz = proj_all[n_exp:]

    pc_norms = [
        float(np.linalg.norm(xyz))
        for xyz in list(start_xyz) + list(end_xyz)
        if float(np.linalg.norm(xyz)) > 0
    ]
    median_pc = float(np.median(pc_norms)) if pc_norms else 1.0

    # Compact refusal: unit direction in PC space, length << activation vectors.
    unit_dir = refusal_direction / refusal_direction.norm().clamp_min(1e-12)
    refusal_pc = _project_point(unit_dir, basis)
    refusal_norm = float(np.linalg.norm(refusal_pc))
    if direction_scale is not None:
        refusal_len = float(direction_scale)
    else:
        refusal_len = 0.28 * median_pc
    if refusal_norm > 1e-12:
        refusal_xyz = refusal_pc / refusal_norm * refusal_len
    else:
        refusal_xyz = np.array([refusal_len, 0.0, 0.0], dtype=float)

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # Origin marker only (no legend entry).
    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        c=["#222222"],
        s=40,
        marker="o",
        depthshade=True,
        zorder=5,
        edgecolors="k",
        linewidths=0.4,
    )

    for index, item in enumerate(endpoints):
        color = EXP_COLORS[index % len(EXP_COLORS)]
        label = item["label"]
        s_xyz = start_xyz[index]
        e_xyz = end_xyz[index]

        _draw_origin_vector(ax, s_xyz, color=color, lw=1.0, tip_frac=0.09)
        _draw_origin_vector(ax, e_xyz, color=color, lw=1.15, tip_frac=0.09)

        ax.scatter(
            [s_xyz[0]],
            [s_xyz[1]],
            [s_xyz[2]],
            c=[color],
            s=65,
            marker="o",
            depthshade=True,
            zorder=4,
            edgecolors="k",
            linewidths=0.4,
            label=f"{label} (start)",
        )
        ax.scatter(
            [e_xyz[0]],
            [e_xyz[1]],
            [e_xyz[2]],
            c=[color],
            s=70,
            marker="^",
            depthshade=True,
            zorder=4,
            edgecolors="k",
            linewidths=0.4,
            label=f"{label} (end)",
        )

    _draw_origin_vector(ax, refusal_xyz, color="#111111", lw=1.4, tip_frac=0.18)
    # Proxy artist so legend shows refusal without a huge quiver glyph.
    ax.plot([], [], [], color="#111111", lw=1.4, label="refusal direction")

    ax.set_xlabel("PC1", fontweight="bold")
    ax.set_ylabel("PC2", fontweight="bold")
    ax.set_zlabel("PC3", fontweight="bold")
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    plt.setp(ax.get_zticklabels(), fontweight="bold")
    ax.legend(loc="best", prop={"weight": "bold", "size": 8})
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[activation_pca_vector_compare] wrote {output}")


def main() -> int:
    args = parse_args()
    exp_dirs = [path.resolve() for path in args.exp_dirs]
    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            raise FileNotFoundError(f"Experiment dir does not exist: {exp_dir}")

    model_id = _infer_model_id(exp_dirs, args.model_id)
    refusal_direction, best_layer = _load_refusal_direction(
        args.dim_root.resolve(),
        model_id,
    )

    endpoints: list[dict[str, Any]] = []
    for exp_dir in exp_dirs:
        points, _steps = _load_snapshots(exp_dir)
        if points[0].numel() != refusal_direction.numel():
            raise ValueError(
                f"Dim mismatch for {exp_dir}: activation {points[0].shape} vs "
                f"refusal direction {tuple(refusal_direction.shape)}"
            )
        endpoints.append(
            {
                "exp_dir": str(exp_dir),
                "label": _exp_label(exp_dir),
                "start": points[0],
                "end": points[-1],
            }
        )

    output = (
        args.output.resolve()
        if args.output is not None
        else exp_dirs[0] / "checkpoints" / "activation_pca_vector_compare_3d.png"
    )
    plot_vector_compare(
        endpoints=endpoints,
        refusal_direction=refusal_direction,
        output=output,
        dpi=args.dpi,
        direction_scale=args.direction_scale,
    )
    print(
        f"[activation_pca_vector_compare] model_id={model_id} best_layer={best_layer} "
        f"exps={len(endpoints)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
