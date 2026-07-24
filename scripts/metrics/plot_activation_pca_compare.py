#!/usr/bin/env python3
"""Compare best-layer activation PCA trajectories across RDO runs + DIM refusal direction."""

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
    "#1f77b4",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#d62728",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one shared 3D PCA of best-layer activation dumps from multiple "
            "experiment logdirs, with a single shared green start and one DIM "
            "refusal-direction arrow."
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
        help="Output PNG path (default: <first-exp>/checkpoints/activation_pca_compare_3d.png)",
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
            "Arrow length in activation space as a multiple of unit refusal direction. "
            "Default: median ||end-start|| across experiments"
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


_DIRECTION_MODE_DISPLAY = {
    "baseline": "RDO",
    "activation_additive_rot": "Cayley Steering",
    "shtiefel_additive_rot": "Stiefel Rotation",
}


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


def _project_pca(
    matrices: list[torch.Tensor],
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Fit PCA on stacked rows; return (T,3) projections, mean, V[:, :3]."""
    mat = torch.stack(matrices, dim=0).float()
    mean = mat.mean(dim=0)
    centered = mat - mean
    q = min(3, centered.shape[0], centered.shape[1])
    _, _, V = torch.pca_lowrank(centered, q=q)
    basis = V[:, :q]
    proj = (centered @ basis).numpy()
    if proj.shape[1] < 3:
        pad = np.zeros((proj.shape[0], 3 - proj.shape[1]), dtype=proj.dtype)
        proj = np.concatenate([proj, pad], axis=1)
    if basis.shape[1] < 3:
        pad = torch.zeros(basis.shape[0], 3 - basis.shape[1], dtype=basis.dtype)
        basis = torch.cat([basis, pad], dim=1)
    return proj, mean, basis


def _project_point(point: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> np.ndarray:
    vec = (point.float().reshape(-1) - mean) @ basis
    out = vec.detach().cpu().numpy().reshape(-1)
    if out.shape[0] < 3:
        out = np.concatenate([out, np.zeros(3 - out.shape[0], dtype=out.dtype)])
    return out[:3]


def plot_compare(
    *,
    trajectories: list[dict[str, Any]],
    shared_start: torch.Tensor,
    refusal_direction: torch.Tensor,
    output: Path,
    dpi: int,
    direction_scale: float | None,
) -> None:
    # Replace each trajectory start with the shared green start so all begins coincide.
    aligned: list[list[torch.Tensor]] = []
    for traj in trajectories:
        points = list(traj["points"])
        points[0] = shared_start.clone()
        aligned.append(points)

    all_points = [point for points in aligned for point in points]
    displacements = [
        float((points[-1] - points[0]).norm().item())
        for points in aligned
        if len(points) >= 2
    ]
    scale = float(direction_scale) if direction_scale is not None else (
        float(np.median(displacements)) if displacements else 1.0
    )
    if scale <= 0:
        scale = 1.0
    unit_dir = refusal_direction / refusal_direction.norm().clamp_min(1e-12)
    refusal_end = shared_start + scale * unit_dir

    # Fit PCA on activations only (not the refusal tip), then project everything.
    proj_all, mean, basis = _project_pca(all_points)
    start_xyz = _project_point(shared_start, mean, basis)
    refusal_xyz = _project_point(refusal_end, mean, basis)

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    offset = 0
    for index, (traj, points) in enumerate(zip(trajectories, aligned)):
        n = len(points)
        xyz = proj_all[offset : offset + n]
        offset += n
        color = EXP_COLORS[index % len(EXP_COLORS)]
        label = traj["label"]
        ax.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            "-",
            color=color,
            lw=1.4,
            alpha=0.85,
            label=label,
            zorder=2,
        )
        if n > 2:
            ax.scatter(
                xyz[1:-1, 0],
                xyz[1:-1, 1],
                xyz[1:-1, 2],
                c=[color],
                s=18,
                alpha=0.75,
                depthshade=True,
                zorder=3,
            )
        ax.scatter(
            [xyz[-1, 0]],
            [xyz[-1, 1]],
            [xyz[-1, 2]],
            c=[color],
            s=55,
            marker="^",
            depthshade=True,
            zorder=4,
            edgecolors="k",
            linewidths=0.4,
        )

    ax.scatter(
        [start_xyz[0]],
        [start_xyz[1]],
        [start_xyz[2]],
        c=["green"],
        s=90,
        marker="o",
        depthshade=True,
        zorder=5,
        edgecolors="k",
        linewidths=0.5,
        label="shared start",
    )
    ax.quiver(
        start_xyz[0],
        start_xyz[1],
        start_xyz[2],
        refusal_xyz[0] - start_xyz[0],
        refusal_xyz[1] - start_xyz[1],
        refusal_xyz[2] - start_xyz[2],
        color="#111111",
        linewidth=2.0,
        arrow_length_ratio=0.12,
        label="refusal direction",
        zorder=6,
    )

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
    print(f"[activation_pca_compare] wrote {output}")


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

    trajectories: list[dict[str, Any]] = []
    starts: list[torch.Tensor] = []
    for exp_dir in exp_dirs:
        points, steps = _load_snapshots(exp_dir)
        if points[0].numel() != refusal_direction.numel():
            raise ValueError(
                f"Dim mismatch for {exp_dir}: activation {points[0].shape} vs "
                f"refusal direction {tuple(refusal_direction.shape)}"
            )
        trajectories.append(
            {
                "exp_dir": str(exp_dir),
                "label": _exp_label(exp_dir),
                "points": points,
                "steps": steps,
            }
        )
        starts.append(points[0])

    shared_start = torch.stack(starts, dim=0).mean(dim=0)
    output = (
        args.output.resolve()
        if args.output is not None
        else exp_dirs[0] / "checkpoints" / "activation_pca_compare_3d.png"
    )
    plot_compare(
        trajectories=trajectories,
        shared_start=shared_start,
        refusal_direction=refusal_direction,
        output=output,
        dpi=args.dpi,
        direction_scale=args.direction_scale,
    )
    print(
        f"[activation_pca_compare] model_id={model_id} best_layer={best_layer} "
        f"exps={len(trajectories)} dumps kept under "
        f".../checkpoints/{ACTIVATION_SUBDIR.name}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
