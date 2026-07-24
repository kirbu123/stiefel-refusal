#!/usr/bin/env python3
"""Generate num_opt_layers ablation tables and plots from RDO runs."""

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
from scripts.metrics.ablation_study_r import (  # noqa: E402
    DEFAULT_RDO_EXP_LIST,
    RDO_PHASE_COLORS,
    ROTATION_LABELS,
    ROTATION_MARKERS,
    ROTATION_PHASE_COLORS,
    load_rdo_references,
    require_baseline_references_for_family,
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
    "initial": "Initial model",
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
            "Plot guard, MMLU, PPL, ARC, and GSM8K metrics against the "
            "effective number of optimized layers (nol)."
        )
    )
    parser.add_argument(
        "-a",
        "--activation-experiment",
        type=Path,
        help="activation_additive_rot experiment-family directory",
    )
    parser.add_argument(
        "-s",
        "--stiefel-experiment",
        type=Path,
        help="shtiefel_additive_rot experiment-family directory",
    )
    parser.add_argument(
        "-b",
        "--baseline-experiment",
        type=Path,
        help=(
            "baseline experiment-family directory; when provided, plot only "
            "this baseline NOL sweep (no RDO/angular/spherical hlines)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "rdo_refusal"
            / "analysis"
            / "ablation_study_nol"
        ),
        help="Output directory for tables, plots, and diagnostics",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    parser.add_argument(
        "--rdo-exp-list",
        type=Path,
        default=DEFAULT_RDO_EXP_LIST,
        help=(
            "YAML/TXT map of RDO / angular-steering / spherical-steering "
            "baselines used for horizontal reference lines"
        ),
    )
    args = parser.parse_args()
    if args.baseline_experiment is not None:
        if args.activation_experiment is not None or args.stiefel_experiment is not None:
            parser.error("-b/--baseline-experiment cannot be combined with -a or -s")
    elif args.activation_experiment is None or args.stiefel_experiment is None:
        parser.error("provide either -b, or both -a and -s")
    return args


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _family_payload(hparams: dict[str, Any]) -> dict[str, Any]:
    # The x parameter and its nol=0 marker must not split one ablation family.
    excluded = set(VOLATILE_PRR_FAMILY_KEYS) | {
        "num_opt_layers",
        "optimize_all_layers",
        "clear_ckpts",
    }
    return {
        key: _json_safe(value)
        for key, value in sorted(hparams.items())
        if key not in excluded
        and not str(key).startswith(("angular_", "spherical_"))
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


def _effective_nol(run_dir: Path, hparams: dict[str, Any]) -> tuple[int, str]:
    """Read effective nol, with artifact fallback for older nol=0 runs."""
    value = hparams.get("num_opt_layers")
    if value is None:
        raise ValueError("missing num_opt_layers")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"invalid num_opt_layers: {value!r}")
    nol = int(numeric)
    if nol > 0:
        return nol, "hparams"

    active_path = run_dir / "checkpoints" / "active_layers.pt"
    if active_path.exists():
        try:
            import torch

            artifact = torch.load(active_path, map_location="cpu")
            active = artifact.get("active_layer_indices") if isinstance(artifact, dict) else None
            if active:
                return len(set(int(index) for index in active)), "active_layers.pt"
        except Exception:
            pass
    raise ValueError(
        "num_opt_layers=0 has no effective layer count; expected a current "
        "hparams.json or checkpoints/active_layers.pt"
    )


def load_experiments(
    experiment_dir: Path,
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
            nol, nol_source = _effective_nol(run_dir, hparams)
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

            kept = 0
            for row in metric_rows:
                if not _is_plotted_metric(row):
                    continue
                try:
                    metric_value = float(row["value"])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(metric_value):
                    continue
                records.append(
                    {
                        "family_id": family_id,
                        "run_dir": str(run_dir),
                        "eval_file": str(eval_path),
                        "num_opt_layers": nol,
                        "num_opt_layers_source": nol_source,
                        "benchmark": _clean_field(row.get("benchmark", "")),
                        "backend": _clean_field(row.get("backend", "")),
                        "group": _clean_field(row.get("group", "")),
                        "phase": _clean_field(row.get("phase", "")),
                        "metric": _clean_field(row.get("metric", "")),
                        "value": metric_value,
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


def aggregate_metrics(
    records: pd.DataFrame,
    *,
    reject_duplicates: bool = False,
) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    group_columns = [
        "family_id",
        "num_opt_layers",
        "benchmark",
        "backend",
        "group",
        "phase",
        "metric",
    ]
    if "rotation_family" in records.columns:
        group_columns.insert(1, "rotation_family")
    if reject_duplicates:
        duplicate_counts = records.groupby(group_columns, dropna=False).size()
        duplicates = duplicate_counts[duplicate_counts > 1]
        if not duplicates.empty:
            first_key = duplicates.index[0]
            raise ValueError(
                "Multiple experiments have identical ablation settings; "
                f"first duplicate key={first_key!r}, count={int(duplicates.iloc[0])}. "
                "Remove duplicate runs instead of averaging them."
            )
    aggregated = (
        records.groupby(group_columns, dropna=False)["value"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    aggregated["std"] = aggregated["std"].fillna(0.0)
    return aggregated.sort_values(
        [
            "family_id",
            "benchmark",
            "backend",
            "group",
            "metric",
            "num_opt_layers",
            "phase",
        ]
    )


def _series_title(benchmark: str, backend: str, group: str, metric: str) -> str:
    if benchmark == "guard":
        return f"{backend} {group}: {metric.replace('_', ' ')}"
    return f"{benchmark.replace('_', ' ').upper()}: {metric.replace('_', ' ')}"


def _series_ylabel(benchmark: str, backend: str, metric: str) -> str:
    if benchmark == "guard":
        return {
            "llamaguard": "Llama-Guard score",
            "qwen3guard": "Qwen-Guard score",
            "wildguard": "Wild-Guard score",
        }.get(backend, "Guard score")
    if metric in ("accuracy", "exact_match", "pct_unsafe"):
        return metric.replace("_", " ").title()
    if metric == "perplexity":
        return "Perplexity (lower is better)"
    if metric == "mean_unsafe_probability":
        return "Mean unsafe probability"
    return metric.replace("_", " ").title()


def plot_family(
    family_df: pd.DataFrame,
    family_dir: Path,
    *,
    dpi: int,
    rdo_reference: dict[str, Any] | None = None,
    baseline_references: list[dict[str, Any]] | None = None,
) -> list[str]:
    plot_dir = family_dir / "plots"
    table_dir = family_dir / "tables"
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    family_df.to_csv(table_dir / "aggregated_metrics.csv", index=False)

    if baseline_references is None:
        baseline_references = []
        if rdo_reference is not None:
            baseline_references = [
                {
                    **rdo_reference,
                    "kind": rdo_reference.get("kind", "rdo"),
                    "swap_phases": True,
                    "label_prefix": "RDO",
                    "colors": RDO_PHASE_COLORS,
                }
            ]

    generated: list[str] = []
    available_families = set(family_df["rotation_family"].astype(str))
    rotation_families = tuple(
        family
        for family in ("activation", "stiefel", "baseline")
        if family in available_families
    )
    initial_family = (
        "activation" if "activation" in available_families else rotation_families[0]
    )
    series_columns = ["benchmark", "backend", "group", "metric"]
    for series_key, series_df in family_df.groupby(series_columns, dropna=False):
        benchmark, backend, group, metric = [str(value) for value in series_key]
        fig, axis = plt.subplots(figsize=(7.2, 4.6))
        x_ticks: set[int] = set()
        initial_df = series_df[
            (series_df["phase"] == "initial")
            & (series_df["rotation_family"] == initial_family)
        ].sort_values("num_opt_layers")
        if initial_df.empty:
            initial_df = series_df[series_df["phase"] == "initial"].sort_values(
                "num_opt_layers"
            )
        if not initial_df.empty:
            initial_df = initial_df.drop_duplicates(subset=["num_opt_layers"])
            initial_x = initial_df["num_opt_layers"].to_numpy(dtype=int)
            x_ticks.update(int(value) for value in initial_x)
            axis.plot(
                initial_x,
                initial_df["mean"].to_numpy(dtype=float),
                linewidth=2,
                linestyle=":",
                color=PHASE_COLORS["initial"],
                label=PHASE_LABELS["initial"],
            )

        for rotation_family in rotation_families:
            for phase in ("refined_attack", "refined_protect"):
                phase_df = series_df[
                    (series_df["phase"] == phase)
                    & (series_df["rotation_family"] == rotation_family)
                ].sort_values("num_opt_layers")
                if phase_df.empty:
                    continue
                x_values = phase_df["num_opt_layers"].to_numpy(dtype=int)
                means = phase_df["mean"].to_numpy(dtype=float)
                stds = phase_df["std"].to_numpy(dtype=float)
                x_ticks.update(int(value) for value in x_values)
                if rotation_family == "baseline":
                    curve_color = PHASE_COLORS[phase]
                    marker = "D"
                    rotation_label = "Baseline RDO"
                else:
                    curve_color = ROTATION_PHASE_COLORS[(rotation_family, phase)]
                    marker = ROTATION_MARKERS[rotation_family]
                    rotation_label = ROTATION_LABELS.get(
                        rotation_family,
                        rotation_family.replace("_", " ").title(),
                    )
                axis.plot(
                    x_values,
                    means,
                    marker=marker,
                    linewidth=2,
                    linestyle="-",
                    color=curve_color,
                    label=f"{rotation_label} {phase.removeprefix('refined_')}",
                )
                if (stds > 0).any():
                    axis.fill_between(
                        x_values,
                        means - stds,
                        means + stds,
                        color=curve_color,
                        alpha=0.12,
                        linewidth=0,
                    )

        for reference in baseline_references:
            metric_lookup = reference["metrics"]
            colors = reference.get("colors") or RDO_PHASE_COLORS
            label_prefix = str(reference.get("label_prefix") or "RDO")
            if reference.get("swap_phases", False):
                phase_pairs = (
                    ("refined_attack", "refined_protect"),
                    ("refined_protect", "refined_attack"),
                )
            else:
                phase_pairs = (
                    ("refined_attack", "refined_attack"),
                    ("refined_protect", "refined_protect"),
                )
            for source_phase, display_phase in phase_pairs:
                reference_value = metric_lookup.get(
                    (benchmark, backend, group, metric, source_phase)
                )
                if reference_value is None:
                    continue
                phase_name = display_phase.removeprefix("refined_")
                axis.axhline(
                    reference_value,
                    color=colors[display_phase],
                    linestyle=":",
                    linewidth=2.2,
                    label=f"{label_prefix} {phase_name}",
                    zorder=1,
                )

        axis.set_xlabel("Number of optimized layers", fontweight="bold")
        axis.set_ylabel(
            _series_ylabel(benchmark, backend, metric),
            fontweight="bold",
        )
        axis.set_title(
            _series_title(benchmark, backend, group, metric),
            fontweight="bold",
        )
        axis.set_xticks(sorted(x_ticks))
        axis.grid(True, alpha=0.25)
        for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
            tick_label.set_fontweight("bold")
        axis.legend(frameon=False, prop={"weight": "bold"})
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
            save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
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

    baseline_only = args.baseline_experiment is not None
    if baseline_only:
        experiment_specs = (("baseline", "baseline", args.baseline_experiment),)
    else:
        experiment_specs = (
            ("activation", "activation_additive_rot", args.activation_experiment),
            ("stiefel", "shtiefel_additive_rot", args.stiefel_experiment),
        )
    record_frames: list[pd.DataFrame] = []
    parameter_sets: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {
        "mode": "baseline_only" if baseline_only else "rotation_comparison",
        "experiments": {},
        "runs_discovered": 0,
        "runs_loaded": 0,
        "runs_skipped": [],
        "guard_errors": [],
    }
    for rotation_family, expected_mode, experiment_dir in experiment_specs:
        family_records, families, family_diagnostics = load_experiments(
            experiment_dir.resolve()
        )
        if len(families) != 1:
            raise ValueError(
                f"{rotation_family} experiment must contain exactly one parameter "
                f"family, found {len(families)}"
            )
        parameters = next(iter(families.values()))
        if _clean_field(parameters.get("direction_mode")) != expected_mode:
            raise ValueError(
                f"{rotation_family} experiment has direction_mode="
                f"{parameters.get('direction_mode')!r}, expected {expected_mode!r}"
            )
        family_records = family_records.copy()
        family_records["rotation_family"] = rotation_family
        record_frames.append(family_records)
        parameter_sets[rotation_family] = parameters
        diagnostics["experiments"][rotation_family] = family_diagnostics
        diagnostics["runs_discovered"] += family_diagnostics["runs_discovered"]
        diagnostics["runs_loaded"] += family_diagnostics["runs_loaded"]
        diagnostics["runs_skipped"].extend(family_diagnostics["runs_skipped"])
        diagnostics["guard_errors"].extend(family_diagnostics["guard_errors"])

    records = pd.concat(record_frames, ignore_index=True)
    comparison_id = _family_id(parameter_sets)
    records["family_id"] = comparison_id
    if records.empty:
        raise RuntimeError("No supported metrics were found in the experiment directories")
    # Baseline-only directories may contain repeated runs for the same nol.
    # Treat those as replicates and report their mean/std; paired rotation
    # comparisons remain strict so accidentally duplicated runs are not hidden.
    aggregated = aggregate_metrics(records, reject_duplicates=not baseline_only)
    diagnostics["duplicate_policy"] = (
        "average_replicates" if baseline_only else "reject"
    )
    records.to_csv(output_dir / "run_metrics.csv", index=False)
    aggregated.to_csv(output_dir / "aggregated_metrics.csv", index=False)

    family_dir = output_dir / "families" / f"family_{comparison_id}"
    family_dir.mkdir(parents=True, exist_ok=True)
    baseline_references: list[dict[str, Any]] = []
    rdo_reference = None
    if not baseline_only:
        rdo_references = load_rdo_references(args.rdo_exp_list.resolve())
        baseline_references = require_baseline_references_for_family(
            parameter_sets["activation"],
            rdo_references,
        )
        rdo_reference = next(
            (item for item in baseline_references if item.get("kind") == "rdo"),
            None,
        )
    (family_dir / "family_parameters.json").write_text(
        json.dumps(parameter_sets, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    generated_plots = plot_family(
        aggregated,
        family_dir,
        dpi=args.dpi,
        baseline_references=baseline_references,
    )
    family_summaries = [
        {
            "family_id": comparison_id,
            "num_opt_layers_values": sorted(
                int(value) for value in aggregated["num_opt_layers"].unique()
            ),
            "plots": len(generated_plots),
            "rdo_reference": (
                rdo_reference["experiment_dir"]
                if rdo_reference is not None
                else None
            ),
            "baseline_references": [
                {
                    "kind": item.get("kind"),
                    "model_key": item.get("model_key"),
                    "experiment_dir": item.get("experiment_dir"),
                    "swap_phases": item.get("swap_phases"),
                }
                for item in baseline_references
            ],
        }
    ]

    diagnostics["metric_rows"] = len(records)
    diagnostics["aggregated_rows"] = len(aggregated)
    diagnostics["families"] = len(family_summaries)
    (output_dir / "run_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    summary = {
        "x_parameter": "num_opt_layers",
        "mode": diagnostics["mode"],
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
