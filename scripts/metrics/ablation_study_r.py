#!/usr/bin/env python3
"""Generate r-ablation tables and plots from RDO TensorBoard run directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization.research import (  # noqa: E402
    VOLATILE_PRR_FAMILY_KEYS,
    _choose_latest_eval_file,
    _rows_from_eval_csv,
    _rows_from_eval_json,
)


PHASES = ("initial", "refined_attack", "refined_protect")
GUARD_METRICS = ("pct_unsafe", "mean_score", "mean_unsafe_probability")
CAPABILITY_METRICS = {
    "mmlu": "accuracy",
    "ppl": "perplexity",
    "arc_easy": "accuracy",
    "arc_challenge": "accuracy",
    "gsm8k": "exact_match",
}
PHASE_LABELS = {
    "initial": "Initial",
    "refined_attack": "Refined attack",
    "refined_protect": "Refined protect",
}
PHASE_COLORS = {
    "initial": "#333333",
    "refined_attack": "#d95f02",
    "refined_protect": "#1b9e77",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot guard, MMLU, PPL, ARC, and GSM8K metrics against "
            "r = numerator / proj_reduce_ratio."
        )
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Directory containing RDO run subdirectories with hparams.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "rdo_refusal" / "analysis" / "ablation_study_r",
        help="Output directory for tables, plots, and diagnostics",
    )
    parser.add_argument(
        "--r-numerator",
        type=float,
        default=3500.0,
        help="Numerator in r = numerator / proj_reduce_ratio (default: 3500)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _family_payload(hparams: dict[str, Any]) -> dict[str, Any]:
    excluded = set(VOLATILE_PRR_FAMILY_KEYS)
    return {
        key: _json_safe(value)
        for key, value in sorted(hparams.items())
        if key not in excluded
    }


def _family_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _sanitize(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    return text.strip("_") or "na"


def _clean_field(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().lower()


def _is_plotted_metric(row: dict[str, Any]) -> bool:
    benchmark = _clean_field(row.get("benchmark", ""))
    metric = _clean_field(row.get("metric", ""))
    phase = _clean_field(row.get("phase", ""))
    if phase not in PHASES:
        return False
    if benchmark == "guard":
        return metric in GUARD_METRICS
    return CAPABILITY_METRICS.get(benchmark) == metric


def _paired_json_payload(eval_path: Path) -> dict[str, Any]:
    json_path = eval_path if eval_path.suffix == ".json" else eval_path.with_suffix(".json")
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_experiments(
    experiment_dir: Path,
    r_numerator: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")

    run_dirs = sorted({path.parent for path in experiment_dir.rglob("hparams.json")})
    records: list[dict[str, Any]] = []
    families: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {
        "experiment_dir": str(experiment_dir.resolve()),
        "runs_discovered": len(run_dirs),
        "runs_loaded": 0,
        "runs_skipped": [],
        "guard_errors": [],
    }

    for run_dir in run_dirs:
        try:
            hparams = json.loads((run_dir / "hparams.json").read_text(encoding="utf-8"))
            prr = float(hparams.get("proj_reduce_ratio"))
            if not math.isfinite(prr) or prr <= 0:
                raise ValueError("missing or invalid proj_reduce_ratio")
            eval_path, eval_format = _choose_latest_eval_file(run_dir)
            if eval_path is None or eval_format is None:
                raise FileNotFoundError("no eval_metrics file")
            metric_rows = (
                _rows_from_eval_csv(eval_path)
                if eval_format == "csv"
                else _rows_from_eval_json(eval_path)
            )
            payload = _family_payload(hparams)
            family_id = _family_id(payload)
            families.setdefault(family_id, payload)
            r_value = r_numerator / prr

            kept = 0
            for row in metric_rows:
                if not _is_plotted_metric(row):
                    continue
                try:
                    value = float(row["value"])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                records.append(
                    {
                        "family_id": family_id,
                        "run_dir": str(run_dir),
                        "eval_file": str(eval_path),
                        "proj_reduce_ratio": prr,
                        "r": r_value,
                        "benchmark": _clean_field(row.get("benchmark", "")),
                        "backend": _clean_field(row.get("backend", "")),
                        "group": _clean_field(row.get("group", "")),
                        "phase": _clean_field(row.get("phase", "")),
                        "metric": _clean_field(row.get("metric", "")),
                        "value": value,
                    }
                )
                kept += 1

            if kept == 0:
                raise ValueError("eval file has no supported tri-mode metrics")

            diagnostics["runs_loaded"] += 1
            eval_payload = _paired_json_payload(eval_path)
            for backend, error in (eval_payload.get("guard_errors") or {}).items():
                diagnostics["guard_errors"].append(
                    {
                        "run_dir": str(run_dir),
                        "backend": backend,
                        "error": error,
                    }
                )
        except Exception as exc:
            diagnostics["runs_skipped"].append(
                {
                    "run_dir": str(run_dir),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    return pd.DataFrame.from_records(records), families, diagnostics


def aggregate_metrics(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    group_columns = [
        "family_id",
        "proj_reduce_ratio",
        "r",
        "benchmark",
        "backend",
        "group",
        "phase",
        "metric",
    ]
    aggregated = (
        records.groupby(group_columns, dropna=False)["value"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    aggregated["std"] = aggregated["std"].fillna(0.0)
    return aggregated.sort_values(["family_id", "benchmark", "backend", "group", "metric", "r", "phase"])


def _series_title(benchmark: str, backend: str, group: str, metric: str) -> str:
    if benchmark == "guard":
        return f"{backend} {group}: {metric.replace('_', ' ')}"
    return f"{benchmark.replace('_', ' ').upper()}: {metric.replace('_', ' ')}"


def _series_ylabel(benchmark: str, metric: str) -> str:
    if metric in ("accuracy", "exact_match", "pct_unsafe"):
        return metric.replace("_", " ").title()
    if metric == "perplexity":
        return "Perplexity (lower is better)"
    if metric == "mean_unsafe_probability":
        return "Mean unsafe probability"
    if metric == "mean_score":
        return "Mean guard score"
    return metric.replace("_", " ").title()


def plot_family(
    family_df: pd.DataFrame,
    family_dir: Path,
    *,
    dpi: int,
    r_numerator: float,
) -> list[str]:
    plot_dir = family_dir / "plots"
    table_dir = family_dir / "tables"
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    family_df.to_csv(table_dir / "aggregated_metrics.csv", index=False)

    generated: list[str] = []
    series_columns = ["benchmark", "backend", "group", "metric"]
    for series_key, series_df in family_df.groupby(series_columns, dropna=False):
        benchmark, backend, group, metric = [str(value) for value in series_key]
        available_phases = [
            phase for phase in PHASES if not series_df[series_df["phase"] == phase].empty
        ]
        if not available_phases:
            continue

        fig, axis = plt.subplots(figsize=(7.2, 4.6))
        for phase in available_phases:
            phase_df = series_df[series_df["phase"] == phase].sort_values("r")
            x_values = phase_df["r"].to_numpy(dtype=float)
            means = phase_df["mean"].to_numpy(dtype=float)
            stds = phase_df["std"].to_numpy(dtype=float)
            axis.plot(
                x_values,
                means,
                marker="o",
                linewidth=2,
                color=PHASE_COLORS[phase],
                label=PHASE_LABELS[phase],
            )
            if (stds > 0).any():
                axis.fill_between(
                    x_values,
                    means - stds,
                    means + stds,
                    color=PHASE_COLORS[phase],
                    alpha=0.18,
                    linewidth=0,
                )

        axis.set_xlabel(
            rf"$r = {r_numerator:g} / \mathrm{{proj\_reduce\_ratio}}$"
        )
        axis.set_ylabel(_series_ylabel(benchmark, metric))
        axis.set_title(_series_title(benchmark, backend, group, metric))
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
        fig.tight_layout()

        stem_parts = [benchmark]
        if backend:
            stem_parts.append(backend)
        if group:
            stem_parts.append(group)
        stem_parts.append(metric)
        stem = "__".join(_sanitize(part) for part in stem_parts)
        for extension in ("png", "pdf"):
            output_path = plot_dir / f"{stem}.{extension}"
            save_kwargs = {"bbox_inches": "tight"}
            if extension == "png":
                save_kwargs["dpi"] = dpi
            fig.savefig(output_path, **save_kwargs)
            generated.append(str(output_path))
        plt.close(fig)

    return generated


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, families, diagnostics = load_experiments(
        args.experiment_dir.resolve(),
        args.r_numerator,
    )
    if records.empty:
        raise RuntimeError("No supported metrics were found in the experiment directory")
    aggregated = aggregate_metrics(records)
    records.to_csv(output_dir / "run_metrics.csv", index=False)
    aggregated.to_csv(output_dir / "aggregated_metrics.csv", index=False)

    generated_plots: list[str] = []
    family_summaries = []
    for family_id, family_df in aggregated.groupby("family_id"):
        family_dir = output_dir / "families" / f"family_{family_id}"
        family_dir.mkdir(parents=True, exist_ok=True)
        parameters = families[str(family_id)]
        (family_dir / "family_parameters.json").write_text(
            json.dumps(parameters, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        family_plots = plot_family(
            family_df,
            family_dir,
            dpi=args.dpi,
            r_numerator=args.r_numerator,
        )
        generated_plots.extend(family_plots)
        family_summaries.append(
            {
                "family_id": str(family_id),
                "r_values": sorted(float(value) for value in family_df["r"].unique()),
                "proj_reduce_ratios": sorted(
                    float(value) for value in family_df["proj_reduce_ratio"].unique()
                ),
                "plots": len(family_plots),
            }
        )

    diagnostics["metric_rows"] = len(records)
    diagnostics["aggregated_rows"] = len(aggregated)
    diagnostics["families"] = len(family_summaries)
    (output_dir / "run_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    summary = {
        "r_formula": f"r = {args.r_numerator:g} / proj_reduce_ratio",
        "families": family_summaries,
        "generated_plot_files": generated_plots,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        f"Loaded {diagnostics['runs_loaded']}/{diagnostics['runs_discovered']} runs; "
        f"generated {len(generated_plots)} plot files in {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
