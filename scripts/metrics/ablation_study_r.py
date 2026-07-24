#!/usr/bin/env python3
"""Generate k_proj ablation tables and plots from RDO TensorBoard runs."""

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
    "initial": "Initial model",
    "refined_attack": "Refined attack",
    "refined_protect": "Refined protect",
}
PHASE_COLORS = {
    "initial": "#333333",
    "refined_attack": "#d95f02",
    "refined_protect": "#1b9e77",
}
RDO_PHASE_LABELS = {
    "refined_attack": "RDO attack",
    "refined_protect": "RDO protect",
}
RDO_PHASE_COLORS = {
    "refined_attack": "#b2182b",
    "refined_protect": "#ef8a62",
}
BASELINE_KIND_STYLES = {
    "rdo": {
        "label_prefix": "RDO",
        "swap_phases": True,
        "colors": {
            "refined_attack": "#b2182b",
            "refined_protect": "#ef8a62",
        },
    },
    "angular-steering": {
        "label_prefix": "Angular Steering",
        "swap_phases": False,
        "colors": {
            "refined_attack": "#542788",
            "refined_protect": "#998ec3",
        },
    },
    "spherical-steering": {
        "label_prefix": "Spherical Steering",
        # Stored attack/protect are swapped, same as RDO.
        "swap_phases": True,
        "colors": {
            "refined_attack": "#01665e",
            "refined_protect": "#5ab4ac",
        },
    },
}
ROTATION_LABELS = {
    "activation": "Cayley Rotation",
    "stiefel": "Stiefel Rotation",
}
ROTATION_MARKERS = {
    "activation": "o",
    "stiefel": "s",
}
ROTATION_PHASE_COLORS = {
    ("activation", "refined_attack"): "#d95f02",
    ("activation", "refined_protect"): "#1b9e77",
    ("stiefel", "refined_attack"): "#377eb8",
    ("stiefel", "refined_protect"): "#984ea3",
}
DEFAULT_RDO_EXP_LIST = (
    PROJECT_ROOT
    / "results"
    / "rdo_refusal"
    / "AAAI-results"
    / "rdo_exp_list.yaml"
)
RDO_MODEL_KEY_MARKERS = (
    ("qwen-1.5b-ease", ("qwen2.5-1.5b-instruct-ease", "qwen2.5-1.5b", "ease")),
    ("falcon-7b-base", ("falcon3-7b-base", "falcon3")),
    ("deepseek", ("deepseek",)),
    ("olmo", ("olmo",)),
    ("qwen", ("qwen",)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot guard, MMLU, PPL, ARC, and GSM8K metrics against the "
            "direct low-rank projection width k_proj."
        )
    )
    parser.add_argument(
        "-a",
        "--activation-experiment",
        type=Path,
        required=True,
        help="activation_additive_rot experiment-family directory",
    )
    parser.add_argument(
        "-s",
        "--stiefel-experiment",
        type=Path,
        required=True,
        help="shtiefel_additive_rot experiment-family directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "rdo_refusal" / "analysis" / "ablation_study_k_proj",
        help="Output directory for tables, plots, and diagnostics",
    )
    parser.add_argument(
        "--legacy-projection-dim",
        type=float,
        default=3500.0,
        help=(
            "Projection dimension used only to derive k_proj for legacy runs "
            "that store proj_reduce_ratio (default: 3500)"
        ),
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    parser.add_argument(
        "--rdo-exp-list",
        type=Path,
        default=DEFAULT_RDO_EXP_LIST,
        help=(
            "Baseline experiment map (.yaml preferred, or legacy .txt). "
            "YAML may include rdo, angular-steering, and spherical-steering entries"
        ),
    )
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
    excluded = set(VOLATILE_PRR_FAMILY_KEYS) | {"k_proj", "clear_ckpts"}
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


def _model_identifier_variants(value: Any) -> set[str]:
    """Return comparable full-repository and basename model identifiers."""
    cleaned = _clean_field(value)
    if not cleaned:
        return set()
    return {cleaned, cleaned.rsplit("/", 1)[-1]}


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


def _metrics_lookup_from_experiment(experiment_dir: Path) -> dict[tuple[str, str, str, str, str], float]:
    eval_path, eval_format = _choose_latest_eval_file(experiment_dir)
    if eval_path is None or eval_format is None:
        raise FileNotFoundError(f"No eval_metrics file in {experiment_dir}")
    metric_rows = (
        _rows_from_eval_csv(eval_path)
        if eval_format == "csv"
        else _rows_from_eval_json(eval_path)
    )
    lookup: dict[tuple[str, str, str, str, str], float] = {}
    for row in metric_rows:
        phase = _clean_field(row.get("phase", ""))
        if phase not in ("refined_attack", "refined_protect") or not _is_plotted_metric(row):
            continue
        try:
            value = float(row["value"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        lookup[
            (
                _clean_field(row.get("benchmark", "")),
                _clean_field(row.get("backend", "")),
                _clean_field(row.get("group", "")),
                _clean_field(row.get("metric", "")),
                phase,
            )
        ] = value
    return lookup


def _resolve_experiment_dir(path_text: str, map_path: Path) -> Path:
    experiment_dir = Path(path_text.strip()).expanduser()
    if not experiment_dir.is_absolute():
        experiment_dir = (map_path.parent / experiment_dir).resolve()
    return experiment_dir


def _infer_model_key(hparams: dict[str, Any], preferred_keys: list[str] | None = None) -> str | None:
    identifiers = set()
    for value in (hparams.get("model"), hparams.get("model_id")):
        identifiers.update(_model_identifier_variants(value))
    joined = " ".join(sorted(identifiers))
    preferred = preferred_keys or [key for key, _ in RDO_MODEL_KEY_MARKERS]
    for model_key, markers in RDO_MODEL_KEY_MARKERS:
        if model_key not in preferred:
            continue
        if any(marker in joined for marker in markers):
            return model_key
    for model_key in preferred:
        if model_key in joined:
            return model_key
    return None


def _build_baseline_reference(
    *,
    kind: str,
    model_key: str,
    experiment_dir: Path,
) -> dict[str, Any]:
    style = BASELINE_KIND_STYLES[kind]
    hparams_path = experiment_dir / "hparams.json"
    if not hparams_path.is_file():
        raise FileNotFoundError(f"Missing hparams for {kind} baseline: {hparams_path}")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    identifiers = {_clean_field(model_key)}
    for value in (hparams.get("model"), hparams.get("model_id")):
        identifiers.update(_model_identifier_variants(value))
    return {
        "kind": kind,
        "model_key": model_key,
        "experiment_dir": str(experiment_dir),
        "identifiers": {value for value in identifiers if value},
        "metrics": _metrics_lookup_from_experiment(experiment_dir),
        "swap_phases": bool(style["swap_phases"]),
        "label_prefix": str(style["label_prefix"]),
        "colors": dict(style["colors"]),
    }


def _load_references_from_root(
    *,
    kind: str,
    root: Path,
    model_keys: list[str],
) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing {kind} experiment root: {root}")
    references: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (run_dir / "hparams.json").is_file():
            continue
        hparams = json.loads((run_dir / "hparams.json").read_text(encoding="utf-8"))
        model_key = _infer_model_key(hparams, preferred_keys=model_keys)
        if model_key is None:
            continue
        references.append(
            _build_baseline_reference(
                kind=kind,
                model_key=model_key,
                experiment_dir=run_dir,
            )
        )
    return references


def load_rdo_references(map_path: Path) -> list[dict[str, Any]]:
    """Load RDO / angular / spherical baseline metric lookups from YAML or legacy TXT."""
    if not map_path.is_file():
        raise FileNotFoundError(f"Baseline experiment map does not exist: {map_path}")

    references: list[dict[str, Any]] = []
    suffix = map_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(map_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid baseline map YAML: {map_path}")

        rdo_entries = payload.get("rdo") or {}
        if not isinstance(rdo_entries, dict) or not rdo_entries:
            raise ValueError(f"YAML map {map_path} must contain a non-empty 'rdo' mapping")
        model_keys = [str(key) for key in rdo_entries]
        for model_key, experiment_text in rdo_entries.items():
            references.append(
                _build_baseline_reference(
                    kind="rdo",
                    model_key=str(model_key),
                    experiment_dir=_resolve_experiment_dir(str(experiment_text), map_path),
                )
            )

        for kind in ("angular-steering", "spherical-steering"):
            section = payload.get(kind) or {}
            if not section:
                continue
            if not isinstance(section, dict) or "root" not in section:
                raise ValueError(
                    f"YAML map section {kind!r} must be a mapping with a 'root' path"
                )
            references.extend(
                _load_references_from_root(
                    kind=kind,
                    root=_resolve_experiment_dir(str(section["root"]), map_path),
                    model_keys=model_keys,
                )
            )
        return references

    # Legacy TXT: one `model: path` line per RDO baseline.
    for line_number, raw_line in enumerate(
        map_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        model_key, separator, experiment_text = line.partition(":")
        if not separator or not model_key.strip() or not experiment_text.strip():
            raise ValueError(
                f"Invalid RDO map entry at {map_path}:{line_number}; "
                "expected 'model: experiment_path'"
            )
        references.append(
            _build_baseline_reference(
                kind="rdo",
                model_key=model_key.strip(),
                experiment_dir=_resolve_experiment_dir(experiment_text, map_path),
            )
        )
    return references


def rdo_reference_for_family(
    family_parameters: dict[str, Any],
    references: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Match a family to an RDO run by identifier, then specific map-key markers."""
    matched = baseline_references_for_family(family_parameters, references)
    for reference in matched:
        if reference.get("kind") == "rdo":
            return reference
    return None


def baseline_references_for_family(
    family_parameters: dict[str, Any],
    references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return all mapped baselines (RDO/angular/spherical) for one experiment family."""
    family_identifiers = set()
    for value in (
        family_parameters.get("model"),
        family_parameters.get("model_id"),
    ):
        family_identifiers.update(_model_identifier_variants(value))

    matched: list[dict[str, Any]] = []
    seen_kinds: set[str] = set()

    for reference in references:
        if family_identifiers & set(reference["identifiers"]):
            kind = str(reference.get("kind", "rdo"))
            if kind in seen_kinds:
                continue
            matched.append(reference)
            seen_kinds.add(kind)

    if len(seen_kinds) < len(BASELINE_KIND_STYLES):
        joined = " ".join(sorted(family_identifiers))
        resolved_key = None
        for model_key, markers in RDO_MODEL_KEY_MARKERS:
            if any(marker in joined for marker in markers):
                resolved_key = model_key
                break
        if resolved_key is not None:
            for reference in references:
                kind = str(reference.get("kind", "rdo"))
                if kind in seen_kinds:
                    continue
                if _clean_field(reference["model_key"]) != resolved_key:
                    continue
                matched.append(reference)
                seen_kinds.add(kind)

    # Preserve stable plotting order.
    order = {kind: index for index, kind in enumerate(BASELINE_KIND_STYLES)}
    matched.sort(key=lambda item: order.get(str(item.get("kind")), 99))
    return matched


def require_rdo_reference_for_family(
    family_parameters: dict[str, Any],
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the mapped RDO baseline or fail instead of silently omitting it."""
    reference = rdo_reference_for_family(family_parameters, references)
    if reference is not None:
        return reference
    model = family_parameters.get("model") or family_parameters.get("model_id")
    available_keys = ", ".join(
        str(item["model_key"])
        for item in references
        if item.get("kind") == "rdo"
    )
    raise ValueError(
        f"No RDO baseline experiment matched model {model!r}; "
        f"available map keys: {available_keys}"
    )


def require_baseline_references_for_family(
    family_parameters: dict[str, Any],
    references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Require an RDO match, then attach any angular/spherical baselines for the model."""
    matched = baseline_references_for_family(family_parameters, references)
    if any(item.get("kind") == "rdo" for item in matched):
        return matched
    require_rdo_reference_for_family(family_parameters, references)
    return matched


def load_experiments(
    experiment_dir: Path,
    legacy_projection_dim: float = 3500.0,
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
            k_proj_source = "k_proj"
            legacy_prr = None
            if hparams.get("k_proj") is not None:
                k_proj = float(hparams["k_proj"])
            elif hparams.get("proj_reduce_ratio") is not None:
                legacy_prr = float(hparams["proj_reduce_ratio"])
                if not math.isfinite(legacy_prr) or legacy_prr <= 0:
                    raise ValueError("invalid legacy proj_reduce_ratio")
                k_proj = legacy_projection_dim / legacy_prr
                k_proj_source = "legacy_proj_reduce_ratio"
            else:
                raise ValueError("missing k_proj")
            if not math.isfinite(k_proj) or k_proj <= 0:
                raise ValueError("invalid k_proj")
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
                        "k_proj": k_proj,
                        "k_proj_source": k_proj_source,
                        "legacy_proj_reduce_ratio": legacy_prr,
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


def aggregate_metrics(
    records: pd.DataFrame,
    *,
    reject_duplicates: bool = False,
) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    group_columns = [
        "family_id",
        "k_proj",
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
        ["family_id", "benchmark", "backend", "group", "metric", "k_proj", "phase"]
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
            # Backward-compatible single RDO reference.
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
    series_columns = ["benchmark", "backend", "group", "metric"]
    for series_key, series_df in family_df.groupby(series_columns, dropna=False):
        benchmark, backend, group, metric = [str(value) for value in series_key]
        fig, axis = plt.subplots(figsize=(7.2, 4.6))
        initial_df = series_df[
            (series_df["phase"] == "initial")
            & (series_df["rotation_family"] == "activation")
        ].sort_values("k_proj")
        if initial_df.empty:
            initial_df = series_df[series_df["phase"] == "initial"].sort_values(
                "k_proj"
            )
        if not initial_df.empty:
            initial_df = initial_df.drop_duplicates(subset=["k_proj"])
            axis.plot(
                initial_df["k_proj"].to_numpy(dtype=float),
                initial_df["mean"].to_numpy(dtype=float),
                linewidth=2,
                linestyle=":",
                color=PHASE_COLORS["initial"],
                label=PHASE_LABELS["initial"],
            )

        for rotation_family in ("activation", "stiefel"):
            for phase in ("refined_attack", "refined_protect"):
                phase_df = series_df[
                    (series_df["phase"] == phase)
                    & (series_df["rotation_family"] == rotation_family)
                ].sort_values("k_proj")
                if phase_df.empty:
                    continue
                x_values = phase_df["k_proj"].to_numpy(dtype=float)
                means = phase_df["mean"].to_numpy(dtype=float)
                stds = phase_df["std"].to_numpy(dtype=float)
                curve_color = ROTATION_PHASE_COLORS[(rotation_family, phase)]
                axis.plot(
                    x_values,
                    means,
                    marker=ROTATION_MARKERS[rotation_family],
                    linewidth=2,
                    linestyle="-",
                    color=curve_color,
                    label=f"{ROTATION_LABELS[rotation_family]} {phase.removeprefix('refined_')}",
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

        axis.set_xlabel("n", fontweight="bold")
        axis.set_ylabel(
            _series_ylabel(benchmark, backend, metric),
            fontweight="bold",
        )
        axis.grid(True, alpha=0.25)
        for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
            tick_label.set_fontweight("bold")
        n_legend = len(axis.get_legend_handles_labels()[0])
        axis.legend(
            frameon=False,
            prop={"weight": "bold", "size": 8},
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=max(1, min(4, n_legend)),
            columnspacing=0.9,
            handlelength=1.4,
            handletextpad=0.35,
            borderaxespad=0.0,
        )
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

    experiment_specs = (
        ("activation", "activation_additive_rot", args.activation_experiment),
        ("stiefel", "shtiefel_additive_rot", args.stiefel_experiment),
    )
    record_frames: list[pd.DataFrame] = []
    parameter_sets: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {
        "experiments": {},
        "runs_discovered": 0,
        "runs_loaded": 0,
        "runs_skipped": [],
        "guard_errors": [],
    }
    for rotation_family, expected_mode, experiment_dir in experiment_specs:
        family_records, families, family_diagnostics = load_experiments(
            experiment_dir.resolve(),
            args.legacy_projection_dim,
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
    rdo_references = load_rdo_references(args.rdo_exp_list.resolve())
    if records.empty:
        raise RuntimeError("No supported metrics were found in the experiment directories")
    aggregated = aggregate_metrics(records, reject_duplicates=True)
    records.to_csv(output_dir / "run_metrics.csv", index=False)
    aggregated.to_csv(output_dir / "aggregated_metrics.csv", index=False)

    family_dir = output_dir / "families" / f"family_{comparison_id}"
    family_dir.mkdir(parents=True, exist_ok=True)
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
            "k_proj_values": sorted(
                float(value) for value in aggregated["k_proj"].unique()
            ),
            "plots": len(generated_plots),
            "rdo_reference": (
                rdo_reference["experiment_dir"] if rdo_reference is not None else None
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
        "x_parameter": "k_proj",
        "legacy_conversion": (
            f"k_proj = {args.legacy_projection_dim:g} / proj_reduce_ratio"
        ),
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
