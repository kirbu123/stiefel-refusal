#!/usr/bin/env python3
"""
Generate consolidated experiment tables and num_opt_layers ablation plots
from RDO refusal experiment directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency
    sns = None


MAJOR_GUARD_METRICS = ("n", "mean_score", "mean_unsafe_probability", "pct_unsafe")
VOLATILE_FAMILY_KEYS = {
    "num_opt_layers",
    "result_path",
    "result_root",
    "add_layer",
    "alpha",
    "cone_dim",
    "optimizer_name",
}
VOLATILE_IM_FAMILY_KEYS = VOLATILE_FAMILY_KEYS | {"init_mode"}
VOLATILE_PRR_FAMILY_KEYS = {
    "proj_reduce_ratio",
    "result_path",
    "result_root",
    "add_layer",
    "alpha",
    "cone_dim",
    "optimizer_name",
}
EVAL_FILE_RE = re.compile(r"eval_metrics_(\d{8}_\d{6})\.(csv|json)$")
MIN_ABLATION_POINTS = 3
INIT_MODE_DISPLAY = {
    "random": "Random",
    "diag_permutation": "Diagonal",
    "ab_orthogonal": "Orthogonal",
}
BACKEND_DISPLAY = {
    "llamaguard": "Llama-Guard",
    "qwen3guard": "Qwen3-Guard",
}
RDO_LINE_COLOR = "#E30B5C"
PROTECT_BASELINE_COLOR = "purple"
ATTACK_OURS_COLOR = "#E49B0F"  # Cambridge
PROTECT_OURS_COLOR = "#40B5AD"  # Verdigris
INIT_MODE_BAR_COLORS = {
    "Orthogonal": "#40B5AD",  # Verdigris
    "Diagonal": "#E49B0F",  # Cambridge
    "Random": "#0BDA51",  # Malachite
}
ANGULAR_STEERING_PROTECT_BASELINES = {
    "llamaguard": 0.03,
    "qwen3guard": 0.045,
}


@dataclass
class RunDiagnostics:
    run_dir: str
    status: str
    reason: str
    metrics_file: str | None = None
    metrics_type: str | None = None


def _setup_plot_style() -> None:
    if sns is not None:
        sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["axes.labelweight"] = "semibold"
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11


def _safe_value(v: Any) -> Any:
    # Convert numpy/pandas scalar values (e.g. int64/float64) to native Python scalars.
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)


def _sanitize_token(text: Any) -> str:
    token = str(text)
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", token)
    token = token.strip("._-")
    return token or "na"


def _apply_axis_text_style(ax: plt.Axes) -> None:
    ax.xaxis.label.set_fontweight("semibold")
    ax.yaxis.label.set_fontweight("semibold")
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("semibold")


def _apply_prr_r_xtick_override(
    ax: plt.Axes,
    ticks: list[float],
    plot_prefix: str,
    family_hash: str,
) -> None:
    # Hardcoded styling rule requested by user:
    # for PRR r-plots only, hide tick label "3" when tick "2" exists.
    if plot_prefix != "r" or not str(family_hash).endswith("_prr"):
        return

    has_two = any(math.isclose(float(t), 2.0, rel_tol=0.0, abs_tol=1e-9) for t in ticks)
    has_three = any(math.isclose(float(t), 3.0, rel_tol=0.0, abs_tol=1e-9) for t in ticks)
    if not (has_two and has_three):
        return

    labels: list[str] = []
    for t in ticks:
        value = float(t)
        if math.isclose(value, 3.0, rel_tol=0.0, abs_tol=1e-9):
            labels.append("")
        elif value.is_integer():
            labels.append(str(int(value)))
        else:
            labels.append(f"{value:g}")
    ax.set_xticklabels(labels)


def _phase_series_color(phase: str) -> str | None:
    if phase == "refined_attack":
        return PROTECT_OURS_COLOR
    if phase == "refined_protect":
        return ATTACK_OURS_COLOR
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RDO experiment tables and num_opt_layers ablation plots."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        type=str,
        help="Ablation directory containing experiment subdirectories.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=str,
        help="Output directory for plots and LaTeX tables.",
    )
    parser.add_argument(
        "--tight",
        action="store_true",
        help="If set, family plots include only pct_unsafe metrics.",
    )
    return parser.parse_args()


def _discover_run_dirs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted({p.parent for p in input_dir.rglob("hparams.json")})


def _choose_latest_eval_file(run_dir: Path) -> tuple[Path | None, str | None]:
    candidates: list[tuple[str, Path, str]] = []
    for p in run_dir.glob("eval_metrics_*.*"):
        match = EVAL_FILE_RE.search(p.name)
        if not match:
            continue
        ts = match.group(1)
        ext = match.group(2)
        candidates.append((ts, p, ext))
    if not candidates:
        return None, None

    csv_candidates = [x for x in candidates if x[2] == "csv"]
    source = csv_candidates if csv_candidates else candidates
    source.sort(key=lambda x: (x[0], x[1].name))
    latest = source[-1]
    return latest[1], latest[2]


def _rows_from_eval_csv(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    required = {"benchmark", "backend", "group", "phase", "metric", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    records = df.to_dict(orient="records")
    canonical_keys = {
        (
            str(row.get("benchmark", "")),
            str(row.get("backend", "")),
            str(row.get("group", "")),
            str(row.get("phase", "")),
            str(row.get("metric", "")),
        )
        for row in records
        if str(row.get("phase", "")) not in ("refined", "delta")
    }
    normalized: list[dict[str, Any]] = []
    for row in records:
        phase = str(row.get("phase", ""))
        if phase == "refined":
            row = {**row, "phase": "refined_attack"}
        elif phase == "delta":
            row = {**row, "phase": "delta_attack"}
        key = (
            str(row.get("benchmark", "")),
            str(row.get("backend", "")),
            str(row.get("group", "")),
            str(row.get("phase", "")),
            str(row.get("metric", "")),
        )
        if phase in ("refined", "delta") and key in canonical_keys:
            continue
        normalized.append(row)
    return normalized


def _locality_rows_from_payload(locality_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark, block in locality_metrics.items():
        for phase in ("initial", "refined_attack", "refined_protect"):
            source_phase = phase
            if phase == "refined_attack" and not block.get(phase):
                source_phase = "refined"
            for metric, value in (block.get(source_phase) or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    rows.append(
                        {
                            "benchmark": benchmark,
                            "backend": "",
                            "group": "",
                            "phase": phase,
                            "metric": metric,
                            "value": value,
                        }
                    )
        metric_name = block.get("metric_name")
        for mode_name in ("attack", "protect"):
            delta = block.get(f"delta_{mode_name}")
            if mode_name == "attack" and delta is None:
                delta = block.get("delta")
            if metric_name and delta is not None:
                rows.append(
                    {
                        "benchmark": benchmark,
                        "backend": "",
                        "group": "",
                        "phase": f"delta_{mode_name}",
                        "metric": metric_name,
                        "value": delta,
                    }
                )
    return rows


def _rows_from_eval_json(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows: list[dict[str, Any]] = []
    guard_metrics = payload.get("guard_metrics") or {}
    for backend, backend_block in guard_metrics.items():
        for group in ("harmful", "harmless"):
            split_block = (backend_block or {}).get(group) or {}
            for phase in ("initial", "refined_attack", "refined_protect"):
                source_phase = phase
                if phase == "refined_attack" and not split_block.get(phase):
                    source_phase = "refined"
                stats = split_block.get(source_phase) or {}
                stats = stats or {}
                for metric in MAJOR_GUARD_METRICS:
                    value = stats.get(metric)
                    if value is None:
                        continue
                    rows.append(
                        {
                            "benchmark": "guard",
                            "backend": backend,
                            "group": group,
                            "phase": phase,
                            "metric": metric,
                            "value": value,
                        }
                    )

    # Backward-compatible fallback format.
    if payload.get("llamaguard") and "llamaguard" not in guard_metrics:
        block = payload.get("llamaguard") or {}
        for group in ("harmful", "harmless"):
            split_block = block.get(group) or {}
            for phase in ("initial", "refined_attack", "refined_protect"):
                source_phase = phase
                if phase == "refined_attack" and not split_block.get(phase):
                    source_phase = "refined"
                stats = split_block.get(source_phase) or {}
                for metric in MAJOR_GUARD_METRICS:
                    value = stats.get(metric)
                    if value is None:
                        continue
                    rows.append(
                        {
                            "benchmark": "guard",
                            "backend": "llamaguard",
                            "group": group,
                            "phase": phase,
                            "metric": metric,
                            "value": value,
                        }
                    )

    mmlu = payload.get("mmlu") or {}
    for phase in ("initial", "refined_attack", "refined_protect"):
        source_phase = phase
        if phase == "refined_attack" and not mmlu.get(phase):
            source_phase = "refined"
        phase_stats = mmlu.get(source_phase) or {}
        for metric in ("accuracy", "correct", "total", "invalid_predictions"):
            value = phase_stats.get(metric)
            if value is None:
                continue
            rows.append(
                {
                    "benchmark": "mmlu",
                    "backend": "",
                    "group": "",
                    "phase": phase,
                    "metric": metric,
                    "value": value,
                }
            )
    for mode_name in ("attack", "protect"):
        delta_accuracy = mmlu.get(f"delta_accuracy_{mode_name}")
        delta_metrics = mmlu.get(f"delta_metrics_{mode_name}") or {}
        if mode_name == "attack":
            if delta_accuracy is None:
                delta_accuracy = mmlu.get("delta_accuracy")
            if not delta_metrics:
                delta_metrics = mmlu.get("delta_metrics") or {}
        if delta_accuracy is not None:
            rows.append(
                {
                    "benchmark": "mmlu",
                    "backend": "",
                    "group": "",
                    "phase": f"delta_{mode_name}",
                    "metric": "accuracy",
                    "value": delta_accuracy,
                }
            )
        for metric, value in delta_metrics.items():
            if value is not None:
                rows.append(
                    {
                        "benchmark": "mmlu",
                        "backend": "",
                        "group": "",
                        "phase": f"delta_{mode_name}",
                        "metric": metric,
                        "value": value,
                    }
                )

    rows.extend(_locality_rows_from_payload(payload.get("locality_metrics") or {}))
    return rows


def _metric_col_name(metric_row: dict[str, Any]) -> str | None:
    benchmark = str(metric_row.get("benchmark", "")).strip().lower()
    phase = _sanitize_token(metric_row.get("phase", "na"))
    metric = _sanitize_token(metric_row.get("metric", "na"))
    if benchmark == "guard":
        backend = _sanitize_token(metric_row.get("backend", "na"))
        group = _sanitize_token(metric_row.get("group", "na"))
        return f"guard__{backend}__{group}__{phase}__{metric}"
    if benchmark == "mmlu":
        return f"mmlu__{phase}__{metric}"
    if benchmark in ("ppl", "arc_easy", "arc_challenge", "gsm8k"):
        return f"{benchmark}__{phase}__{metric}"
    return None


def _family_payload(hparams: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in sorted(hparams.keys()):
        if key in VOLATILE_FAMILY_KEYS:
            continue
        payload[key] = _safe_value(hparams[key])
    return payload


def _im_family_payload(hparams: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in sorted(hparams.keys()):
        if key in VOLATILE_IM_FAMILY_KEYS:
            continue
        payload[key] = _safe_value(hparams[key])
    return payload


def _family_descriptor(payload: dict[str, Any]) -> str:
    keys = (
        "model_id",
        "direction_mode",
        "init_mode",
        "orth_method",
        "proj_reduce_ratio",
        "llamaguard_data",
        "eval_split",
    )
    parts: list[str] = []
    for key in keys:
        if key in payload:
            parts.append(f"{key}={payload[key]}")
    return ", ".join(parts) if parts else "default"


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return float(v)
    try:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    except (TypeError, ValueError):
        return None


def _load_run(run_dir: Path) -> tuple[dict[str, Any] | None, RunDiagnostics]:
    hparams_path = run_dir / "hparams.json"
    if not hparams_path.exists():
        return None, RunDiagnostics(str(run_dir), "skipped", "missing hparams.json")

    try:
        with open(hparams_path, "r", encoding="utf-8") as f:
            hparams = json.load(f)
    except Exception as e:  # pragma: no cover - filesystem/JSON corruption path
        return None, RunDiagnostics(str(run_dir), "skipped", f"hparams parse error: {e}")

    metrics_path, metrics_type = _choose_latest_eval_file(run_dir)
    if metrics_path is None or metrics_type is None:
        return None, RunDiagnostics(str(run_dir), "skipped", "missing eval_metrics_*.csv/json")

    try:
        metric_rows = (
            _rows_from_eval_csv(metrics_path)
            if metrics_type == "csv"
            else _rows_from_eval_json(metrics_path)
        )
    except Exception as e:  # pragma: no cover - malformed files
        return None, RunDiagnostics(
            str(run_dir),
            "skipped",
            f"metrics parse error: {e}",
            metrics_file=str(metrics_path),
            metrics_type=metrics_type,
        )

    record: dict[str, Any] = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "metrics_file": str(metrics_path),
        "metrics_type": metrics_type,
    }
    for k, v in hparams.items():
        record[f"hp__{k}"] = _safe_value(v)

    for row in metric_rows:
        col = _metric_col_name(row)
        if col is None:
            continue
        val = _coerce_float(row.get("value"))
        if val is None:
            continue
        record[col] = val

    family_payload = _family_payload(hparams)
    family_json = json.dumps(family_payload, sort_keys=True, ensure_ascii=False)
    record["family_hash"] = hashlib.sha1(family_json.encode("utf-8")).hexdigest()[:10]
    record["family_descriptor"] = _family_descriptor(family_payload)
    record["num_opt_layers"] = _coerce_float(hparams.get("num_opt_layers"))
    record["_family_json"] = family_json
    im_family_payload = _im_family_payload(hparams)
    im_family_json = json.dumps(im_family_payload, sort_keys=True, ensure_ascii=False)
    record["im_family_hash"] = hashlib.sha1(im_family_json.encode("utf-8")).hexdigest()[:10]
    record["_im_family_json"] = im_family_json

    return record, RunDiagnostics(
        run_dir=str(run_dir),
        status="loaded",
        reason="ok",
        metrics_file=str(metrics_path),
        metrics_type=metrics_type,
    )


def _write_full_tables(df: pd.DataFrame, output_dir: Path) -> None:
    out_csv = output_dir / "all_experiments.csv"
    out_tex = output_dir / "all_experiments.tex"
    ordered = df.sort_values(["family_hash", "num_opt_layers", "run_name"], na_position="last")
    ordered.to_csv(out_csv, index=False)
    ordered.to_latex(out_tex, index=False, float_format="%.6g", escape=True)


def _parse_guard_col(col: str) -> tuple[str, str, str, str] | None:
    if not col.startswith("guard__"):
        return None
    parts = col.split("__", 4)
    if len(parts) != 5:
        return None
    _, backend, group, phase, metric = parts
    return backend, group, phase, metric


def _guard_col_name(backend: str, group: str, phase: str, metric: str) -> str:
    return (
        f"guard__{_sanitize_token(backend)}__{_sanitize_token(group)}__"
        f"{_sanitize_token(phase)}__{_sanitize_token(metric)}"
    )


def _phase_display_label(phase: str) -> str:
    # Requested label unification for refined attack/protect plots.
    if phase in {"refined_attack", "refined_protect"}:
        return "Stiefel Rotation"
    return phase


def _swapped_joint_label(phase: str) -> str:
    # Logs are attack/protect-permuted: display swapped labels on joint plots.
    if phase == "refined_attack":
        return "Stiefel Rotation (protect ours)"
    if phase == "refined_protect":
        return "Stiefel Rotation (attack ours)"
    return _phase_display_label(phase)


def _display_x_label(raw_label: str, x_col: str) -> str:
    if x_col == "hp__init_mode":
        return INIT_MODE_DISPLAY.get(raw_label, raw_label)
    return raw_label


def _backend_display_name(backend: str) -> str:
    return BACKEND_DISPLAY.get(backend, backend)


def _angular_steering_protect_baseline(
    backend: str, phase: str, metric: str
) -> float | None:
    # Requested manual protect baseline override.
    if phase != "refined_protect" or metric != "pct_unsafe":
        return None
    return ANGULAR_STEERING_PROTECT_BASELINES.get(backend)


def _plot_series(
    family_df: pd.DataFrame,
    metric_col: str,
    backend: str,
    group: str,
    phase: str,
    metric: str,
    family_hash: str,
    family_dir: Path,
    initial_metric_value: float | None,
    baseline_metric_value: float | None,
    x_col: str = "num_opt_layers",
    x_label: str = "Number of layers",
    series_label: str | None = None,
    plot_kind: str = "line",
    plot_prefix: str = "num_opt_layers",
    reduce_baseline_line: bool = False,
    show_rdo_line: bool = True,
) -> bool:
    subset = family_df[[x_col, metric_col]].dropna().copy()
    numeric_x = pd.to_numeric(subset[x_col], errors="coerce")
    x_positions: list[int] | None = None
    if numeric_x.notna().all():
        subset = subset.assign(_x_numeric=numeric_x)
        subset = subset.sort_values("_x_numeric")
        subset = subset.drop_duplicates(subset=["_x_numeric"], keep="last")
    else:
        subset = subset.assign(_x_label=subset[x_col].astype(str))
        subset = subset.sort_values("_x_label")
        subset = subset.drop_duplicates(subset=["_x_label"], keep="last")
    if len(subset) < MIN_ABLATION_POINTS:
        return False
    if numeric_x.notna().all():
        if subset["_x_numeric"].nunique() < MIN_ABLATION_POINTS:
            return False
    elif subset["_x_label"].nunique() < MIN_ABLATION_POINTS:
        return False

    base = (
        f"{_sanitize_token(plot_prefix)}__{_sanitize_token(backend)}__{_sanitize_token(group)}__"
        f"{_sanitize_token(phase)}__{_sanitize_token(metric)}__family_{family_hash}"
    )
    plots_dir = family_dir / "plots"
    tables_dir = family_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    y = subset[metric_col].astype(float).tolist()
    if numeric_x.notna().all():
        x = subset["_x_numeric"].astype(float).tolist()
        x_labels = None
    else:
        x_labels = [
            _display_x_label(lbl, x_col) for lbl in subset["_x_label"].astype(str).tolist()
        ]
        x_positions = list(range(len(x_labels)))
        x = x_positions

    fig, ax = plt.subplots()
    if plot_kind == "bar":
        if x_labels is not None:
            if x_col == "hp__init_mode":
                colors = [INIT_MODE_BAR_COLORS.get(str(lbl), "steelblue") for lbl in x_labels]
            elif sns is not None:
                colors = sns.color_palette("tab10", n_colors=len(x))
            else:
                colors = [plt.cm.tab10(i % 10) for i in range(len(x))]
            for i, (xv, yv, lbl) in enumerate(zip(x, y, x_labels)):
                ax.bar(
                    xv,
                    yv,
                    width=0.62,
                    color=colors[i],
                    edgecolor="black",
                    alpha=0.9,
                    label=str(lbl),
                )
        else:
            ax.bar(
                x,
                y,
                width=0.62,
                color="steelblue",
                edgecolor="black",
                alpha=0.9,
                label=series_label or _phase_display_label(phase),
            )
    else:
        line_color = _phase_series_color(phase)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.8,
            color=line_color,
            label=series_label or _phase_display_label(phase),
        )
    if initial_metric_value is not None:
        ax.axhline(
            y=initial_metric_value,
            linestyle="--",
            linewidth=1.7,
            color="black",
            alpha=0.9,
            label="initial model",
        )
    protect_override = _angular_steering_protect_baseline(
        backend=backend, phase=phase, metric=metric
    )
    baseline_value_to_plot = (
        protect_override if protect_override is not None else baseline_metric_value
    )
    if baseline_value_to_plot is not None and show_rdo_line:
        baseline_label = "RDO"
        baseline_color = RDO_LINE_COLOR
        if protect_override is not None:
            baseline_label = "Angular Steering (protect baseline)"
            baseline_color = PROTECT_BASELINE_COLOR
        ax.axhline(
            y=baseline_value_to_plot,
            linestyle="-.",
            linewidth=1.2 if reduce_baseline_line else 1.8,
            color=baseline_color,
            alpha=0.5 if reduce_baseline_line else 0.9,
            label=baseline_label,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{_backend_display_name(backend)} score")
    if x_labels is not None:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=20, ha="right")
    else:
        numeric_ticks = sorted(set(x))
        ax.set_xticks(numeric_ticks)
        _apply_prr_r_xtick_override(
            ax=ax,
            ticks=numeric_ticks,
            plot_prefix=plot_prefix,
            family_hash=family_hash,
        )
    _apply_axis_text_style(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(plots_dir / f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)

    table_df = subset.rename(columns={metric_col: "metric_value"})
    direction_mode = family_df["hp__direction_mode"].iloc[0] if "hp__direction_mode" in family_df.columns else None
    orth_method = family_df["hp__orth_method"].iloc[0] if "hp__orth_method" in family_df.columns else None
    init_mode = family_df["hp__init_mode"].iloc[0] if "hp__init_mode" in family_df.columns else None
    proj_reduce_ratio = (
        family_df["hp__proj_reduce_ratio"].iloc[0] if "hp__proj_reduce_ratio" in family_df.columns else None
    )
    param_values = (
        f"direction_mode={direction_mode}, "
        f"orth_method={orth_method}, "
        f"init_mode={init_mode}, "
        f"proj_reduce_ratio={proj_reduce_ratio}"
    )
    table_df.insert(0, "family_hash", family_hash)
    table_df.insert(1, "family_param_values", param_values)
    table_df.insert(2, "direction_mode", direction_mode)
    table_df.insert(3, "orth_method", orth_method)
    table_df.insert(4, "init_mode", init_mode)
    table_df.insert(5, "proj_reduce_ratio", proj_reduce_ratio)
    table_df.insert(6, "backend", backend)
    table_df.insert(7, "group", group)
    table_df.insert(8, "phase", phase)
    table_df.insert(9, "metric", metric)
    table_df.to_csv(tables_dir / f"{base}.csv", index=False)
    table_df.to_latex(
        tables_dir / f"{base}.tex",
        index=False,
        float_format="%.6g",
        escape=True,
        caption=f"{plot_prefix} ablation for {backend}/{group}/{phase}/{metric}",
        label=f"tab:{base[:80]}",
    )
    return True


def _plot_attack_protect_joint(
    family_df: pd.DataFrame,
    backend: str,
    group: str,
    metric: str,
    family_hash: str,
    family_dir: Path,
    initial_metric_value: float | None,
    baseline_attack_value: float | None,
    baseline_protect_value: float | None,
    x_col: str,
    x_label: str,
    plot_prefix: str,
    reduce_baseline_line: bool = False,
    show_rdo_lines: bool = True,
) -> bool:
    attack_col = _guard_col_name(backend=backend, group=group, phase="refined_attack", metric=metric)
    protect_col = _guard_col_name(backend=backend, group=group, phase="refined_protect", metric=metric)
    if attack_col not in family_df.columns or protect_col not in family_df.columns:
        return False

    subset = family_df[[x_col, attack_col, protect_col]].dropna(subset=[x_col]).copy()
    numeric_x = pd.to_numeric(subset[x_col], errors="coerce")
    x_positions: list[int] | None = None
    if numeric_x.notna().all():
        subset = subset.assign(_x_numeric=numeric_x)
        subset = subset.sort_values("_x_numeric")
        subset = subset.drop_duplicates(subset=["_x_numeric"], keep="last")
        if subset["_x_numeric"].nunique() < MIN_ABLATION_POINTS:
            return False
        x_vals = subset["_x_numeric"].astype(float).tolist()
        x_ticks = sorted(set(x_vals))
        x_labels = None
    else:
        subset = subset.assign(_x_label=subset[x_col].astype(str))
        subset = subset.sort_values("_x_label")
        subset = subset.drop_duplicates(subset=["_x_label"], keep="last")
        if subset["_x_label"].nunique() < MIN_ABLATION_POINTS:
            return False
        x_labels = [_display_x_label(lbl, x_col) for lbl in subset["_x_label"].astype(str).tolist()]
        x_positions = list(range(len(x_labels)))
        x_vals = x_positions
        x_ticks = x_positions

    y_attack = pd.to_numeric(subset[attack_col], errors="coerce").tolist()
    y_protect = pd.to_numeric(subset[protect_col], errors="coerce").tolist()

    base = (
        f"{_sanitize_token(plot_prefix)}__{_sanitize_token(backend)}__{_sanitize_token(group)}__"
        f"joint_attack_protect__{_sanitize_token(metric)}__family_{family_hash}"
    )
    plots_dir = family_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(
        x_vals,
        y_attack,
        marker="o",
        linewidth=2.0,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.8,
        color=PROTECT_OURS_COLOR,
        label=_swapped_joint_label("refined_attack"),
    )
    ax.plot(
        x_vals,
        y_protect,
        marker="o",
        linewidth=2.0,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.8,
        color=ATTACK_OURS_COLOR,
        label=_swapped_joint_label("refined_protect"),
    )
    if initial_metric_value is not None:
        ax.axhline(
            y=initial_metric_value,
            linestyle="--",
            linewidth=1.7,
            color="black",
            alpha=0.9,
            label="Initial model",
        )
    if baseline_attack_value is not None and show_rdo_lines:
        ax.axhline(
            y=baseline_attack_value,
            linestyle="-.",
            linewidth=1.2 if reduce_baseline_line else 1.8,
            color=RDO_LINE_COLOR,
            alpha=0.5 if reduce_baseline_line else 0.9,
            label="RDO (attack baseline)",
        )
    if baseline_protect_value is not None and show_rdo_lines:
        protect_override = _angular_steering_protect_baseline(
            backend=backend, phase="refined_protect", metric=metric
        )
        baseline_protect_value_to_plot = (
            protect_override if protect_override is not None else baseline_protect_value
        )
        baseline_protect_label = "RDO (protect baseline)"
        if protect_override is not None:
            baseline_protect_label = "Angular Steering (protect baseline)"
        ax.axhline(
            y=baseline_protect_value_to_plot,
            linestyle=":",
            linewidth=1.2 if reduce_baseline_line else 1.8,
            color=PROTECT_BASELINE_COLOR,
            alpha=0.5 if reduce_baseline_line else 0.9,
            label=baseline_protect_label,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{_backend_display_name(backend)} score")
    ax.set_xticks(x_ticks)
    if x_labels is not None:
        ax.set_xticklabels(x_labels, rotation=20, ha="right")
    else:
        _apply_prr_r_xtick_override(
            ax=ax,
            ticks=x_ticks,
            plot_prefix=plot_prefix,
            family_hash=family_hash,
        )
    _apply_axis_text_style(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(plots_dir / f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def _generate_joint_attack_protect_plots(
    family_df: pd.DataFrame,
    family_hash: str,
    family_dir: Path,
    baseline_metric_lookup: dict[str, float],
    guard_cols: list[str] | None,
    x_col: str,
    x_label: str,
    plot_prefix: str,
    reduce_baseline_line: bool = False,
    show_rdo_lines: bool = True,
) -> int:
    guard_cols = (
        guard_cols
        if guard_cols is not None
        else [c for c in family_df.columns if c.startswith("guard__")]
    )
    guard_cols = [c for c in guard_cols if c in family_df.columns]
    parsed_cols = [(_parse_guard_col(c), c) for c in guard_cols]
    combo_keys: set[tuple[str, str, str]] = set()
    for parsed, _ in parsed_cols:
        if parsed is None:
            continue
        backend, group, phase, metric = parsed
        if phase in {"refined_attack", "refined_protect"}:
            combo_keys.add((backend, group, metric))

    generated_files = 0
    for backend, group, metric in sorted(combo_keys):
        initial_col = _guard_col_name(backend=backend, group=group, phase="initial", metric=metric)
        initial_metric_value = None
        if initial_col in family_df.columns:
            initial_vals = pd.to_numeric(family_df[initial_col], errors="coerce").dropna()
            if not initial_vals.empty:
                initial_metric_value = float(initial_vals.iloc[0])

        attack_col = _guard_col_name(backend=backend, group=group, phase="refined_attack", metric=metric)
        protect_col = _guard_col_name(backend=backend, group=group, phase="refined_protect", metric=metric)
        # Logs are attack/protect-permuted on joint plots; keep baseline values
        # aligned to displayed labels by swapping lookup columns too.
        baseline_attack = baseline_metric_lookup.get(protect_col)
        baseline_protect = baseline_metric_lookup.get(attack_col)
        if _plot_attack_protect_joint(
            family_df=family_df,
            backend=backend,
            group=group,
            metric=metric,
            family_hash=family_hash,
            family_dir=family_dir,
            initial_metric_value=initial_metric_value,
            baseline_attack_value=baseline_attack,
            baseline_protect_value=baseline_protect,
            x_col=x_col,
            x_label=x_label,
            plot_prefix=plot_prefix,
            reduce_baseline_line=reduce_baseline_line,
            show_rdo_lines=show_rdo_lines,
        ):
            generated_files += 2
    return generated_files


def _ensure_family_dir_and_manifest(output_dir: Path, family_hash: str, family_json: str) -> Path:
    family_dir = output_dir / "families" / f"family_{_sanitize_token(family_hash)}"
    family_dir.mkdir(parents=True, exist_ok=True)

    try:
        family_params = json.loads(family_json)
    except Exception:
        family_params = {"raw_family_json": family_json}

    manifest = {
        "family_hash": family_hash,
        "family_parameters": family_params,
    }
    with open(family_dir / "family_parameters.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return family_dir


def _ensure_im_family_dir_and_manifest(
    output_dir: Path, im_family_hash: str, im_family_json: str, init_modes: list[str]
) -> Path:
    family_dir = output_dir / "families" / f"family_{_sanitize_token(im_family_hash)}_im"
    family_dir.mkdir(parents=True, exist_ok=True)

    try:
        family_params = json.loads(im_family_json)
    except Exception:
        family_params = {"raw_family_json": im_family_json}

    manifest = {
        "family_hash": im_family_hash,
        "family_type": "init_mode_ablation",
        "family_parameters": family_params,
        "varying_parameter": "init_mode",
        "init_modes": sorted(set(init_modes)),
    }
    with open(family_dir / "family_parameters.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return family_dir


def _ensure_prr_family_dir_and_manifest(
    output_dir: Path,
    prr_family_hash: str,
    prr_family_json: str,
    project_ratios: list[float],
    r_values: list[int],
) -> Path:
    family_dir = output_dir / "families" / f"family_{_sanitize_token(prr_family_hash)}_prr"
    family_dir.mkdir(parents=True, exist_ok=True)

    try:
        family_params = json.loads(prr_family_json)
    except Exception:
        family_params = {"raw_family_json": prr_family_json}

    manifest = {
        "family_hash": prr_family_hash,
        "family_type": "prr_ablation",
        "family_parameters": family_params,
        "varying_parameter": "proj_reduce_ratio",
        "varying_parameter_label": "project ratio",
        "project_ratios": sorted(set(project_ratios)),
        "derived_parameter": "r=int(3500/proj_reduce_ratio)",
        "r_values": sorted(set(r_values)),
    }
    with open(family_dir / "family_parameters.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return family_dir


def _build_baseline_metric_lookup(df: pd.DataFrame, guard_cols: list[str]) -> dict[str, float]:
    baseline_mask = pd.Series(False, index=df.index)
    if "run_name" in df.columns:
        baseline_mask = baseline_mask | df["run_name"].astype(str).str.contains("baseline", case=False, na=False)
    if "run_dir" in df.columns:
        baseline_mask = baseline_mask | df["run_dir"].astype(str).str.contains("baseline", case=False, na=False)
    if "hp__direction_mode" in df.columns:
        baseline_mask = baseline_mask | (df["hp__direction_mode"].astype(str).str.lower() == "baseline")

    baseline_df = df[baseline_mask].copy()
    lookup: dict[str, float] = {}
    if baseline_df.empty:
        return lookup

    for col in guard_cols:
        values = pd.to_numeric(baseline_df[col], errors="coerce").dropna() if col in baseline_df.columns else pd.Series(dtype=float)
        if values.empty:
            continue
        lookup[col] = float(values.mean())
    return lookup


def _generate_init_mode_boundary_plots(
    df: pd.DataFrame,
    output_dir: Path,
    baseline_metric_lookup: dict[str, float],
    tight: bool = False,
) -> tuple[int, int]:
    # Include all guard metrics for init_mode ablations.
    guard_cols = [c for c in df.columns if c.startswith("guard__")]
    if tight:
        guard_cols = [c for c in guard_cols if c.endswith("__pct_unsafe")]
    if not guard_cols:
        return 0, 0

    hp_cols = [c for c in df.columns if c.startswith("hp__")]
    exclude_hp = {f"hp__{k}" for k in VOLATILE_IM_FAMILY_KEYS}
    group_cols = [c for c in hp_cols if c not in exclude_hp]
    if not group_cols:
        return 0, 0

    working_df = df.copy()
    if "hp__direction_mode" in working_df.columns:
        working_df = working_df[
            working_df["hp__direction_mode"].astype(str).str.lower().ne("baseline")
        ].copy()
    if working_df.empty:
        return 0, 0

    plot_count = 0
    table_count = 0
    for group_key, fam_df in working_df.groupby(group_cols, dropna=False):
        if "hp__init_mode" not in fam_df.columns:
            continue
        init_modes = fam_df["hp__init_mode"].astype(str).dropna().unique().tolist()
        if len(init_modes) < MIN_ABLATION_POINTS:
            continue

        if isinstance(group_key, tuple):
            group_values = group_key
        else:
            group_values = (group_key,)
        family_params = {col[4:]: _safe_value(val) for col, val in zip(group_cols, group_values)}
        im_family_json = json.dumps(family_params, sort_keys=True, ensure_ascii=False)
        im_family_hash = hashlib.sha1(im_family_json.encode("utf-8")).hexdigest()[:10]

        family_dir = _ensure_im_family_dir_and_manifest(
            output_dir=output_dir,
            im_family_hash=str(im_family_hash),
            im_family_json=im_family_json,
            init_modes=init_modes,
        )
        for metric_col in sorted(guard_cols):
            parsed = _parse_guard_col(metric_col)
            if parsed is None:
                continue
            backend, group, phase, metric = parsed
            if phase == "initial":
                continue
            initial_col = _guard_col_name(backend=backend, group=group, phase="initial", metric=metric)
            initial_metric_value = None
            if initial_col in fam_df.columns:
                initial_vals = pd.to_numeric(fam_df[initial_col], errors="coerce").dropna()
                if not initial_vals.empty:
                    initial_metric_value = float(initial_vals.iloc[0])
            baseline_metric_value = baseline_metric_lookup.get(metric_col)
            generated = _plot_series(
                family_df=fam_df,
                metric_col=metric_col,
                backend=backend,
                group=group,
                phase=phase,
                metric=metric,
                family_hash=f"{im_family_hash}_im",
                family_dir=family_dir,
                initial_metric_value=initial_metric_value,
                baseline_metric_value=baseline_metric_value,
                x_col="hp__init_mode",
                x_label="Init mode",
                series_label=None,
                plot_kind="bar",
                reduce_baseline_line=True,
                show_rdo_line=False,
            )
            if generated:
                plot_count += 2
                table_count += 2
        plot_count += _generate_joint_attack_protect_plots(
            family_df=fam_df,
            family_hash=f"{im_family_hash}_im",
            family_dir=family_dir,
            baseline_metric_lookup=baseline_metric_lookup,
            guard_cols=guard_cols,
            x_col="hp__init_mode",
            x_label="Init mode",
            plot_prefix="init_mode",
            reduce_baseline_line=True,
            show_rdo_lines=False,
        )
    return plot_count, table_count


def _generate_prr_boundary_plots(
    df: pd.DataFrame,
    output_dir: Path,
    baseline_metric_lookup: dict[str, float],
    tight: bool = False,
) -> tuple[int, int]:
    guard_cols = [c for c in df.columns if c.startswith("guard__")]
    if tight:
        guard_cols = [c for c in guard_cols if c.endswith("__pct_unsafe")]
    if not guard_cols:
        return 0, 0

    hp_cols = [c for c in df.columns if c.startswith("hp__")]
    exclude_hp = {f"hp__{k}" for k in VOLATILE_PRR_FAMILY_KEYS}
    group_cols = [c for c in hp_cols if c not in exclude_hp]
    if not group_cols or "hp__proj_reduce_ratio" not in df.columns:
        return 0, 0

    working_df = df.copy()
    if "hp__direction_mode" in working_df.columns:
        working_df = working_df[
            working_df["hp__direction_mode"].astype(str).str.lower().ne("baseline")
        ].copy()
    if working_df.empty:
        return 0, 0

    working_df["hp__proj_reduce_ratio"] = pd.to_numeric(
        working_df["hp__proj_reduce_ratio"], errors="coerce"
    )
    working_df = working_df.dropna(subset=["hp__proj_reduce_ratio"]).copy()
    if working_df.empty:
        return 0, 0

    working_df["hp__r"] = working_df["hp__proj_reduce_ratio"].apply(
        lambda prr: int(3500.0 / prr) if prr and prr > 0 else math.nan
    )

    plot_count = 0
    table_count = 0
    for group_key, fam_df in working_df.groupby(group_cols, dropna=False):
        unique_prr = pd.to_numeric(fam_df["hp__proj_reduce_ratio"], errors="coerce").dropna()
        if unique_prr.nunique() < MIN_ABLATION_POINTS:
            continue

        if isinstance(group_key, tuple):
            group_values = group_key
        else:
            group_values = (group_key,)
        family_params = {col[4:]: _safe_value(val) for col, val in zip(group_cols, group_values)}
        prr_family_json = json.dumps(family_params, sort_keys=True, ensure_ascii=False)
        prr_family_hash = hashlib.sha1(prr_family_json.encode("utf-8")).hexdigest()[:10]

        r_values = pd.to_numeric(fam_df["hp__r"], errors="coerce").dropna().astype(int).tolist()
        family_dir = _ensure_prr_family_dir_and_manifest(
            output_dir=output_dir,
            prr_family_hash=str(prr_family_hash),
            prr_family_json=prr_family_json,
            project_ratios=unique_prr.astype(float).tolist(),
            r_values=r_values,
        )
        for metric_col in sorted(guard_cols):
            parsed = _parse_guard_col(metric_col)
            if parsed is None:
                continue
            backend, group, phase, metric = parsed
            if phase == "initial":
                continue
            initial_col = _guard_col_name(backend=backend, group=group, phase="initial", metric=metric)
            initial_metric_value = None
            if initial_col in fam_df.columns:
                initial_vals = pd.to_numeric(fam_df[initial_col], errors="coerce").dropna()
                if not initial_vals.empty:
                    initial_metric_value = float(initial_vals.iloc[0])
            baseline_metric_value = baseline_metric_lookup.get(metric_col)

            generated = _plot_series(
                family_df=fam_df,
                metric_col=metric_col,
                backend=backend,
                group=group,
                phase=phase,
                metric=metric,
                family_hash=f"{prr_family_hash}_prr",
                family_dir=family_dir,
                initial_metric_value=initial_metric_value,
                baseline_metric_value=baseline_metric_value,
                x_col="hp__proj_reduce_ratio",
                x_label="project ratio",
                series_label=None,
                plot_kind="line",
                plot_prefix="project_ratio",
            )
            if generated:
                plot_count += 2
                table_count += 2

            generated = _plot_series(
                family_df=fam_df,
                metric_col=metric_col,
                backend=backend,
                group=group,
                phase=phase,
                metric=metric,
                family_hash=f"{prr_family_hash}_prr",
                family_dir=family_dir,
                initial_metric_value=initial_metric_value,
                baseline_metric_value=baseline_metric_value,
                x_col="hp__r",
                x_label="n",
                series_label=None,
                plot_kind="line",
                plot_prefix="r",
            )
            if generated:
                plot_count += 2
                table_count += 2
        plot_count += _generate_joint_attack_protect_plots(
            family_df=fam_df,
            family_hash=f"{prr_family_hash}_prr",
            family_dir=family_dir,
            baseline_metric_lookup=baseline_metric_lookup,
            guard_cols=guard_cols,
            x_col="hp__proj_reduce_ratio",
            x_label="project ratio",
            plot_prefix="project_ratio",
        )
        plot_count += _generate_joint_attack_protect_plots(
            family_df=fam_df,
            family_hash=f"{prr_family_hash}_prr",
            family_dir=family_dir,
            baseline_metric_lookup=baseline_metric_lookup,
            guard_cols=guard_cols,
            x_col="hp__r",
            x_label="n",
            plot_prefix="r",
        )
    return plot_count, table_count


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    families_dir = output_dir / "families"
    meta_dir = output_dir / "meta"

    output_dir.mkdir(parents=True, exist_ok=True)
    families_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    _setup_plot_style()
    run_dirs = _discover_run_dirs(input_dir)

    rows: list[dict[str, Any]] = []
    diagnostics: list[RunDiagnostics] = []
    for run_dir in run_dirs:
        row, diag = _load_run(run_dir)
        diagnostics.append(diag)
        if row is not None:
            rows.append(row)

    if not rows:
        diagnostics_payload = [d.__dict__ for d in diagnostics]
        with open(meta_dir / "run_diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(diagnostics_payload, f, indent=2, ensure_ascii=False)
        raise RuntimeError("No valid runs loaded from input_dir.")

    df = pd.DataFrame(rows)
    if "num_opt_layers" in df.columns:
        df["num_opt_layers"] = pd.to_numeric(df["num_opt_layers"], errors="coerce")

    _write_full_tables(df, output_dir)

    guard_cols = [c for c in df.columns if c.startswith("guard__")]
    if args.tight:
        guard_cols = [c for c in guard_cols if c.endswith("__pct_unsafe")]
    baseline_metric_lookup = _build_baseline_metric_lookup(df, guard_cols)
    plot_count = 0
    table_count = 0
    for metric_col in sorted(guard_cols):
        parsed = _parse_guard_col(metric_col)
        if parsed is None:
            continue
        backend, group, phase, metric = parsed
        if phase == "initial":
            continue
        initial_col = _guard_col_name(backend=backend, group=group, phase="initial", metric=metric)
        for family_hash, family_df in df.groupby("family_hash", dropna=False):
            family_json = str(family_df["_family_json"].iloc[0]) if "_family_json" in family_df.columns else "{}"
            family_dir = _ensure_family_dir_and_manifest(
                output_dir=output_dir,
                family_hash=str(family_hash),
                family_json=family_json,
            )
            initial_metric_value = None
            if initial_col in family_df.columns:
                initial_vals = pd.to_numeric(family_df[initial_col], errors="coerce").dropna()
                if not initial_vals.empty:
                    initial_metric_value = float(initial_vals.iloc[0])
            baseline_metric_value = baseline_metric_lookup.get(metric_col)
            generated = _plot_series(
                family_df=family_df,
                metric_col=metric_col,
                backend=backend,
                group=group,
                phase=phase,
                metric=metric,
                family_hash=str(family_hash),
                family_dir=family_dir,
                initial_metric_value=initial_metric_value,
                baseline_metric_value=baseline_metric_value,
            )
            if generated:
                plot_count += 2  # PNG + PDF
                table_count += 2  # CSV + TEX

    for family_hash, family_df in df.groupby("family_hash", dropna=False):
        family_json = str(family_df["_family_json"].iloc[0]) if "_family_json" in family_df.columns else "{}"
        family_dir = _ensure_family_dir_and_manifest(
            output_dir=output_dir,
            family_hash=str(family_hash),
            family_json=family_json,
        )
        plot_count += _generate_joint_attack_protect_plots(
            family_df=family_df,
            family_hash=str(family_hash),
            family_dir=family_dir,
            baseline_metric_lookup=baseline_metric_lookup,
            guard_cols=guard_cols,
            x_col="num_opt_layers",
            x_label="Number of layers",
            plot_prefix="num_opt_layers",
        )

    im_plot_count, im_table_count = _generate_init_mode_boundary_plots(
        df=df,
        output_dir=output_dir,
        baseline_metric_lookup=baseline_metric_lookup,
        tight=args.tight,
    )
    plot_count += im_plot_count
    table_count += im_table_count

    prr_plot_count, prr_table_count = _generate_prr_boundary_plots(
        df=df,
        output_dir=output_dir,
        baseline_metric_lookup=baseline_metric_lookup,
        tight=args.tight,
    )
    plot_count += prr_plot_count
    table_count += prr_table_count

    diagnostics_payload = [d.__dict__ for d in diagnostics]
    with open(meta_dir / "run_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics_payload, f, indent=2, ensure_ascii=False)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "runs_discovered": len(run_dirs),
        "runs_loaded": len(rows),
        "runs_skipped": len(run_dirs) - len(rows),
        "guard_metric_columns": len(guard_cols),
        "plot_files_generated": plot_count,
        "plot_table_files_generated": table_count,
    }
    with open(meta_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("RDO research report generation complete")
    print("=" * 72)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Full table CSV: {output_dir / 'all_experiments.csv'}")
    print(f"Full table TEX: {output_dir / 'all_experiments.tex'}")
    print(f"Families dir: {families_dir}")
    print(f"Diagnostics: {meta_dir / 'run_diagnostics.json'}")


if __name__ == "__main__":
    main()
