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
    "add_layer",
    "alpha",
    "cone_dim",
    "optimizer_name",
}
VOLATILE_IM_FAMILY_KEYS = VOLATILE_FAMILY_KEYS | {"init_mode"}
EVAL_FILE_RE = re.compile(r"eval_metrics_(\d{8}_\d{6})\.(csv|json)$")


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
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11


def _safe_value(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def _sanitize_token(text: Any) -> str:
    token = str(text)
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", token)
    token = token.strip("._-")
    return token or "na"


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
    return df.to_dict(orient="records")


def _rows_from_eval_json(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows: list[dict[str, Any]] = []
    guard_metrics = payload.get("guard_metrics") or {}
    for backend, backend_block in guard_metrics.items():
        for group in ("harmful", "harmless"):
            split_block = (backend_block or {}).get(group) or {}
            for phase, stats in split_block.items():
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
            for phase in ("initial", "refined"):
                stats = split_block.get(phase) or {}
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
    for phase in ("initial", "refined"):
        phase_stats = mmlu.get(phase) or {}
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
    if mmlu.get("delta_accuracy") is not None:
        rows.append(
            {
                "benchmark": "mmlu",
                "backend": "",
                "group": "",
                "phase": "delta",
                "metric": "accuracy",
                "value": mmlu.get("delta_accuracy"),
            }
        )
    for metric, value in (mmlu.get("delta_metrics") or {}).items():
        if value is None:
            continue
        rows.append(
            {
                "benchmark": "mmlu",
                "backend": "",
                "group": "",
                "phase": "delta",
                "metric": metric,
                "value": value,
            }
        )
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
) -> bool:
    subset = family_df[[x_col, metric_col]].dropna().copy()
    subset = subset.sort_values(x_col)
    subset = subset.drop_duplicates(subset=[x_col], keep="last")
    if len(subset) < 2:
        return False
    if subset[x_col].nunique() < 2:
        return False

    base = (
        f"num_opt_layers__{_sanitize_token(backend)}__{_sanitize_token(group)}__"
        f"{_sanitize_token(phase)}__{_sanitize_token(metric)}__family_{family_hash}"
    )
    plots_dir = family_dir / "plots"
    tables_dir = family_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    y = subset[metric_col].astype(float).tolist()
    if x_col != "num_opt_layers":
        x_labels = subset[x_col].astype(str).tolist()
        x_positions = list(range(len(x_labels)))
        x = x_positions
    else:
        x = subset[x_col].astype(float).tolist()
        x_labels = None

    fig, ax = plt.subplots()
    ax.plot(
        x,
        y,
        marker="o",
        linewidth=2.0,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label=series_label or f"{phase}",
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
    if baseline_metric_value is not None:
        ax.axhline(
            y=baseline_metric_value,
            linestyle="-.",
            linewidth=1.8,
            color="red",
            alpha=0.9,
            label="rdo",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{backend} score")
    if x_labels is not None:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=20, ha="right")
    else:
        ax.set_xticks(sorted(set(x)))
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
        caption=f"num_opt_layers ablation for {backend}/{group}/{phase}/{metric}",
        label=f"tab:{base[:80]}",
    )
    return True


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
) -> tuple[int, int]:
    # Boundary metric is interpreted as guard mean_score.
    guard_cols = [c for c in df.columns if c.startswith("guard__") and c.endswith("__mean_score")]
    if not guard_cols or "im_family_hash" not in df.columns:
        return 0, 0

    plot_count = 0
    table_count = 0
    for im_family_hash, fam_df in df.groupby("im_family_hash", dropna=False):
        if "hp__init_mode" not in fam_df.columns:
            continue
        init_modes = fam_df["hp__init_mode"].astype(str).dropna().unique().tolist()
        if len(init_modes) < 2:
            continue
        im_family_json = str(fam_df["_im_family_json"].iloc[0]) if "_im_family_json" in fam_df.columns else "{}"
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
                series_label=f"{phase} boundary",
            )
            if generated:
                plot_count += 2
                table_count += 2
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

    im_plot_count, im_table_count = _generate_init_mode_boundary_plots(
        df=df,
        output_dir=output_dir,
        baseline_metric_lookup=baseline_metric_lookup,
    )
    plot_count += im_plot_count
    table_count += im_table_count

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
