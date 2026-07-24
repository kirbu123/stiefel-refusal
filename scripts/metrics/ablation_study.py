#!/usr/bin/env python3
"""Run NOL and k_proj ablation plots for every model under AAAI result roots."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = Path(__file__).resolve().parent
NOL_SCRIPT = METRICS_DIR / "ablation_study_nol.py"
K_PROJ_SCRIPT = METRICS_DIR / "ablation_study_r.py"

DEFAULT_NOL_ROOT = (
    PROJECT_ROOT / "results" / "rdo_refusal" / "AAAI-results" / "nol-ablations"
)
DEFAULT_R_ROOT = (
    PROJECT_ROOT / "results" / "rdo_refusal" / "AAAI-results" / "r-ablations"
)
DEFAULT_BASELINE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "rdo_refusal"
    / "AAAI-results"
    / "baseline"
    / "nol-ablations"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "rdo_refusal" / "analysis"
DEFAULT_RDO_EXP_LIST = (
    PROJECT_ROOT / "results" / "rdo_refusal" / "AAAI-results" / "rdo_exp_list.yaml"
)

ACTIVATION_DIR = "activation_additive_rot"
STIEFEL_DIR = "shtiefel_additive_rot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover model directories under NOL and r-ablation roots and run "
            "ablation_study_nol.py / ablation_study_r.py for each pair of "
            "activation_additive_rot and shtiefel_additive_rot experiment trees. "
            "Optionally also plot baseline RDO NOL ablations via -b."
        )
    )
    parser.add_argument(
        "-n",
        "--nol-root",
        type=Path,
        default=DEFAULT_NOL_ROOT,
        help="Root directory of NOL ablation experiments",
    )
    parser.add_argument(
        "-r",
        "--r-root",
        type=Path,
        default=DEFAULT_R_ROOT,
        help="Root directory of k_proj / r ablation experiments",
    )
    parser.add_argument(
        "-b",
        "--baseline-root",
        type=Path,
        default=None,
        help=(
            "Optional root of baseline RDO NOL ablations "
            f"(e.g. {DEFAULT_BASELINE_ROOT}). When set, writes plots under "
            "<output-dir>/baseline/<model>/"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Output root. Plots are written under "
            "<output-dir>/<model>/ablation_study_nol, "
            "<output-dir>/<model>/ablation_study_k_proj, and "
            "(with -b) <output-dir>/baseline/<model>"
        ),
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    parser.add_argument(
        "--rdo-exp-list",
        type=Path,
        default=DEFAULT_RDO_EXP_LIST,
        help=(
            "YAML/TXT map of RDO / angular-steering / spherical-steering "
            "baselines passed through to child scripts"
        ),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used to launch the child ablation scripts",
    )
    return parser.parse_args()


def _discover_models(root: Path) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment root does not exist: {root}")
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def _require_mode_dirs(model_dir: Path) -> tuple[Path, Path]:
    activation = model_dir / ACTIVATION_DIR
    stiefel = model_dir / STIEFEL_DIR
    missing = [str(path) for path in (activation, stiefel) if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Model directory {model_dir} is missing required mode dirs: "
            + ", ".join(missing)
        )
    return activation, stiefel


def _run_script(
    *,
    python_bin: Path,
    script: Path,
    activation: Path,
    stiefel: Path,
    output_dir: Path,
    dpi: int,
    rdo_exp_list: Path | None,
) -> None:
    command = [
        str(python_bin),
        str(script),
        "-a",
        str(activation),
        "-s",
        str(stiefel),
        "--output-dir",
        str(output_dir),
        "--dpi",
        str(dpi),
    ]
    if rdo_exp_list is not None:
        command.extend(["--rdo-exp-list", str(rdo_exp_list)])

    print(f"[ablation_study] Running: {' '.join(command)}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{script.name} failed with exit code {completed.returncode} "
            f"(activation={activation}, stiefel={stiefel}, output={output_dir})"
        )


def _run_baseline_script(
    *,
    python_bin: Path,
    script: Path,
    baseline_experiment: Path,
    output_dir: Path,
    dpi: int,
) -> None:
    command = [
        str(python_bin),
        str(script),
        "-b",
        str(baseline_experiment),
        "--output-dir",
        str(output_dir),
        "--dpi",
        str(dpi),
    ]
    print(f"[ablation_study] Running: {' '.join(command)}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{script.name} failed with exit code {completed.returncode} "
            f"(baseline={baseline_experiment}, output={output_dir})"
        )


def main() -> int:
    args = parse_args()
    nol_root = args.nol_root.resolve()
    r_root = args.r_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rdo_exp_list = args.rdo_exp_list.resolve() if args.rdo_exp_list is not None else None
    baseline_root = (
        args.baseline_root.resolve() if args.baseline_root is not None else None
    )

    if not NOL_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing NOL script: {NOL_SCRIPT}")
    if not K_PROJ_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing k_proj script: {K_PROJ_SCRIPT}")

    nol_models = _discover_models(nol_root)
    r_models = _discover_models(r_root)
    baseline_models = _discover_models(baseline_root) if baseline_root is not None else []
    all_models = sorted(set(nol_models) | set(r_models) | set(baseline_models))
    if not all_models:
        roots = [str(nol_root), str(r_root)]
        if baseline_root is not None:
            roots.append(str(baseline_root))
        raise RuntimeError("No model directories found under " + " or ".join(roots))

    summary: list[dict[str, object]] = []
    for model_name in all_models:
        model_summary: dict[str, object] = {"model": model_name, "jobs": []}
        model_output = output_root / model_name

        if model_name in nol_models:
            activation, stiefel = _require_mode_dirs(nol_root / model_name)
            nol_output = model_output / "ablation_study_nol"
            _run_script(
                python_bin=args.python,
                script=NOL_SCRIPT,
                activation=activation,
                stiefel=stiefel,
                output_dir=nol_output,
                dpi=args.dpi,
                rdo_exp_list=rdo_exp_list,
            )
            model_summary["jobs"].append(
                {
                    "kind": "nol",
                    "activation": str(activation),
                    "stiefel": str(stiefel),
                    "output_dir": str(nol_output),
                }
            )

        if model_name in r_models:
            activation, stiefel = _require_mode_dirs(r_root / model_name)
            k_proj_output = model_output / "ablation_study_k_proj"
            _run_script(
                python_bin=args.python,
                script=K_PROJ_SCRIPT,
                activation=activation,
                stiefel=stiefel,
                output_dir=k_proj_output,
                dpi=args.dpi,
                rdo_exp_list=rdo_exp_list,
            )
            model_summary["jobs"].append(
                {
                    "kind": "k_proj",
                    "activation": str(activation),
                    "stiefel": str(stiefel),
                    "output_dir": str(k_proj_output),
                }
            )

        if model_name in baseline_models:
            assert baseline_root is not None
            baseline_experiment = baseline_root / model_name
            baseline_output = output_root / "baseline" / model_name
            _run_baseline_script(
                python_bin=args.python,
                script=NOL_SCRIPT,
                baseline_experiment=baseline_experiment,
                output_dir=baseline_output,
                dpi=args.dpi,
            )
            model_summary["jobs"].append(
                {
                    "kind": "baseline_nol",
                    "baseline": str(baseline_experiment),
                    "output_dir": str(baseline_output),
                }
            )

        summary.append(model_summary)

    print(
        f"[ablation_study] Completed {len(summary)} models; "
        f"outputs under {output_root}"
    )
    for item in summary:
        jobs = ", ".join(str(job["kind"]) for job in item["jobs"])
        print(f"  - {item['model']}: {jobs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
