#!/usr/bin/env python3
"""
Script to build plots from model answer files.

Usage:
    python draw_graphics.py <path_to_answers_file>

The script infers baseline type from the file path and saves plots to the corresponding directories.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# Project root for evaluate module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization.plots import (
    plot_harmfulness_heatmap,
    plot_locality_heatmap,
    plot_harmfulness_distribution,
    plot_locality_distribution,
    generate_plot_filename,
)


# Supported baseline types
BASELINE_TYPES = [
    "graph_average",
    "topic_ablation", 
    "tag_ablation",
    "graph_grpo",
    "full_question",
]


def detect_baseline_type(file_path: Path) -> Optional[str]:
    """Infer baseline type from file path."""
    path_str = str(file_path.resolve())
    for baseline_type in BASELINE_TYPES:
        if f"/{baseline_type}/" in path_str or f"\\{baseline_type}\\" in path_str:
            return baseline_type
    return None


def get_output_directories(file_path: Path, baseline_type: str) -> Dict[str, Path]:
    """Return paths to directories for saving plots."""
    path_str = str(file_path.resolve())
    baseline_idx = path_str.find(f"/{baseline_type}/")
    if baseline_idx == -1:
        baseline_idx = path_str.find(f"\\{baseline_type}\\")
    if baseline_idx == -1:
        raise ValueError(f"Could not find baseline directory '{baseline_type}' in path: {file_path}")

    baseline_root = Path(path_str[:baseline_idx]) / baseline_type
    dirs = {
        "harmfulness_distribution": baseline_root / "harmfulness" / "distribution_plots",
        "harmfulness_heatmap": baseline_root / "harmfulness" / "heatmap_plots",
        "locality_distribution": baseline_root / "locality" / "distribution_plots",
        "locality_heatmap": baseline_root / "locality" / "heatmap_plots",
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_answers_file(file_path: Path) -> Dict:
    """Load a JSON answers file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fill_evaluations_if_empty(data: Dict, file_path: Path, evaluator: Any) -> Dict:
    """Run LlamaGuard evaluation when evaluations are missing, then persist."""
    harmful = data.get("harmful_questions", {})
    if harmful.get("original_evaluations") and harmful.get("modified_evaluations"):
        return data

    try:
        from evaluate.evaluate_answers import process_answers_format
    except ImportError as e:
        print(f"WARNING: Cannot run evaluation (missing evaluate module): {e}")
        return data

    print("  Evaluations empty, running LlamaGuard evaluation...")
    data = process_answers_format(data, evaluator)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Evaluations saved to {file_path}")
    except Exception as e:
        print(f"  WARNING: Could not save evaluations to file: {e}")
    return data


def extract_scores(data: Dict) -> Dict[str, List[int]]:
    """Extract scores from a single-answer-file format."""
    scores = {
        "original_harmful": [],
        "modified_harmful": [],
        "original_harmless": [],
        "modified_harmless": [],
        "locality_differences": [],
    }

    harmful_data = data.get("harmful_questions", {})
    original_evals = harmful_data.get("original_evaluations", [])
    modified_evals = harmful_data.get("modified_evaluations", [])

    def _is_llamaguard_eval(e):
        exp = e.get("explanation") or ""
        return str(exp).strip().startswith("LlamaGuard:")

    for eval_item in original_evals:
        if _is_llamaguard_eval(eval_item):
            continue
        score = eval_item.get("score")
        if score is not None:
            scores["original_harmful"].append(int(score))

    for eval_item in modified_evals:
        if _is_llamaguard_eval(eval_item):
            continue
        score = eval_item.get("score")
        if score is not None:
            scores["modified_harmful"].append(int(score))

    harmless_data = data.get("harmless_questions", {})
    for loc_item in harmless_data.get("locality_scores", []):
        orig_score = loc_item.get("original_score")
        mod_score = loc_item.get("modified_score")
        diff = loc_item.get("difference")
        if orig_score is not None:
            scores["original_harmless"].append(int(orig_score))
        if mod_score is not None:
            scores["modified_harmless"].append(int(mod_score))
        if diff is not None:
            scores["locality_differences"].append(int(diff))

    return scores


# ---------------------------------------------------------------------------
# File-format detection helpers
# ---------------------------------------------------------------------------

def is_combined_results_file(data: Dict) -> bool:
    return "results" in data and "experiment_config" in data


def is_grpo_format_file(data: Dict) -> bool:
    return (
        "category" in data
        and "questions" in data
        and "responses" in data
        and "scores" in data
        and "experiment_info" not in data
    )


def parse_hyperparams_from_filename(filename: str) -> Dict[str, float]:
    """Parse hyperparameters embedded in a filename."""
    import re
    hyperparams = {}
    patterns = {
        "max_weight": r"max_weight=([0-9.]+)",
        "max_weight_position": r"max_weight_position=([0-9.]+)",
        "min_weight": r"min_weight=([0-9.]+)",
        "min_weight_distance": r"min_weight_distance=([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, filename)
        if match:
            hyperparams[key] = float(match.group(1))
    return hyperparams


def extract_scores_grpo(data: Dict) -> Dict[str, List[int]]:
    """Extract scores from GRPO format file (modified only)."""
    scores = {
        "original_harmful": [],
        "modified_harmful": [],
        "original_harmless": [],
        "modified_harmless": [],
        "locality_differences": [],
    }
    raw_scores = data.get("scores", [])
    scores["modified_harmful"] = [int(s) for s in raw_scores if s is not None]
    return scores


def extract_scores_from_combined_result(result_data: Dict) -> Dict[str, List[int]]:
    """Extract scores from one entry of a combined-results file."""
    scores = {
        "original_harmful": [],
        "modified_harmful": [],
        "original_harmless": [],
        "modified_harmless": [],
        "locality_differences": [],
    }
    scores["original_harmful"] = [int(s) for s in result_data.get("original_scores", []) if s is not None]
    scores["modified_harmful"] = [int(s) for s in result_data.get("modified_scores", []) if s is not None]

    for loc_item in result_data.get("locality_scores", []):
        orig_score = loc_item.get("original_score")
        mod_score = loc_item.get("modified_score")
        diff = loc_item.get("difference")
        if orig_score is not None:
            scores["original_harmless"].append(int(orig_score))
        if mod_score is not None:
            scores["modified_harmless"].append(int(mod_score))
        if diff is not None:
            scores["locality_differences"].append(int(diff))
    return scores


# ---------------------------------------------------------------------------
# Processing pipelines (data loading -> shared plot functions)
# ---------------------------------------------------------------------------

def process_answers_files(
    file_paths: List[Path],
    evaluator: Any = None,
    evaluator_name: str = "",
) -> None:
    """Top-level entry point: detect format and dispatch."""
    if not file_paths:
        print("ERROR: No files provided")
        sys.exit(1)

    print("=" * 80)
    print(f"Processing {len(file_paths)} file(s)")
    print("=" * 80)

    baseline_type = detect_baseline_type(file_paths[0])
    if baseline_type is None:
        print(f"ERROR: Could not determine baseline type from path: {file_paths[0]}")
        print(f"Supported types: {BASELINE_TYPES}")
        sys.exit(1)

    for file_path in file_paths[1:]:
        if detect_baseline_type(file_path) != baseline_type:
            print("ERROR: Files must be from the same baseline type")
            print(f"  First file: {baseline_type}")
            print(f"  File {file_path}: {detect_baseline_type(file_path)}")
            sys.exit(1)

    print(f"Baseline type: {baseline_type}")

    output_dirs = get_output_directories(file_paths[0], baseline_type)
    print("Output directories:")
    for name, path in output_dirs.items():
        print(f"  {name}: {path}")

    first_data = load_answers_file(file_paths[0])

    if is_combined_results_file(first_data):
        process_combined_results_files(file_paths, output_dirs, baseline_type, evaluator_name)
    elif is_grpo_format_file(first_data):
        process_grpo_format_files(file_paths, output_dirs, baseline_type, evaluator_name)
    else:
        process_single_answer_files(file_paths, output_dirs, baseline_type, evaluator, evaluator_name)

    print("\n" + "=" * 80)
    print("Processing completed!")
    print("=" * 80)


def process_combined_results_files(
    file_paths: List[Path],
    output_dirs: Dict[str, Path],
    baseline_type: str,
    evaluator_name: str = "",
) -> None:
    """Process combined results files (tag_ablation format)."""
    all_heatmap_harm = []
    all_heatmap_loc = []
    category_name = None

    for file_idx, file_path in enumerate(file_paths, 1):
        print(f"\n{'='*80}")
        print(f"File {file_idx}/{len(file_paths)}: {file_path.name}")
        print(f"{'='*80}")

        data = load_answers_file(file_path)
        experiment_config = data.get("experiment_config", {})
        tag_name = experiment_config.get("tag", "Unknown")
        if category_name is None:
            category_name = tag_name

        print(f"Tag: {tag_name}")
        print(f"Total hyperparameter combinations: {experiment_config.get('total_combinations', 0)}")

        results = data.get("results", {})
        for result_idx, (param_key, result_data) in enumerate(results.items(), 1):
            hyperparams = result_data.get("hyperparameters", {})
            print(f"\n  [{result_idx}/{len(results)}] {param_key}")
            print(f"    Hyperparameters: {hyperparams}")

            scores = extract_scores_from_combined_result(result_data)
            print(f"    Scores: orig_harm={len(scores['original_harmful'])}, "
                  f"mod_harm={len(scores['modified_harmful'])}, "
                  f"orig_harmless={len(scores['original_harmless'])}, "
                  f"mod_harmless={len(scores['modified_harmless'])}, "
                  f"loc_diff={len(scores['locality_differences'])}")

            if scores["original_harmful"] or scores["modified_harmful"]:
                fname = generate_plot_filename("harmfulness_distribution", tag_name, evaluator_name, ext=".png")
                plot_harmfulness_distribution(
                    scores["original_harmful"], scores["modified_harmful"],
                    output_dirs["harmfulness_distribution"] / fname,
                    evaluator_name=evaluator_name, method_name=baseline_type,
                    category_name=tag_name, hyperparams=hyperparams,
                )

            if scores["original_harmless"] or scores["modified_harmless"]:
                fname = generate_plot_filename("locality_distribution", tag_name, evaluator_name, ext=".png")
                plot_locality_distribution(
                    scores["original_harmless"], scores["modified_harmless"],
                    scores["locality_differences"],
                    output_dirs["locality_distribution"] / fname,
                    evaluator_name=evaluator_name, method_name=baseline_type,
                    category_name=tag_name, hyperparams=hyperparams,
                )

            if scores["modified_harmful"]:
                all_heatmap_harm.append({
                    "max_weight": hyperparams.get("max_weight"),
                    "min_weight": hyperparams.get("min_weight"),
                    "mean_modified_score": np.mean(scores["modified_harmful"]),
                })

            avg_loc = result_data.get("average_locality_change")
            if avg_loc is not None:
                all_heatmap_loc.append({
                    "max_weight": hyperparams.get("max_weight"),
                    "min_weight": hyperparams.get("min_weight"),
                    "average_locality_change": avg_loc,
                })

    print(f"\n{'='*80}")
    print("Building combined heatmap grids...")
    print(f"{'='*80}")

    if all_heatmap_harm:
        fname = generate_plot_filename("harmfulness_heatmap", category_name, evaluator_name)
        plot_harmfulness_heatmap(
            all_heatmap_harm,
            output_dirs["harmfulness_heatmap"] / fname,
            evaluator_name=evaluator_name, method_name=baseline_type,
            category_name=category_name,
        )

    if all_heatmap_loc:
        fname = generate_plot_filename("locality_heatmap", category_name, evaluator_name)
        plot_locality_heatmap(
            all_heatmap_loc,
            output_dirs["locality_heatmap"] / fname,
            evaluator_name=evaluator_name, method_name=baseline_type,
            category_name=category_name,
        )


def process_single_answer_files(
    file_paths: List[Path],
    output_dirs: Dict[str, Path],
    baseline_type: str,
    evaluator: Any = None,
    evaluator_name: str = "",
) -> None:
    """Process single answer files (graph_average, topic_ablation format)."""
    all_heatmap_harm = []
    all_heatmap_loc = []
    category_name = None

    for file_idx, file_path in enumerate(file_paths, 1):
        print(f"\n{'='*80}")
        print(f"File {file_idx}/{len(file_paths)}: {file_path.name}")
        print(f"{'='*80}")

        data = load_answers_file(file_path)
        if evaluator:
            data = _fill_evaluations_if_empty(data, file_path, evaluator)

        experiment_info = data.get("experiment_info", {})
        file_category_name = experiment_info.get("category", "Unknown")
        hyperparams = experiment_info.get("hyperparameters", {})

        if category_name is None:
            category_name = file_category_name
        if file_category_name != category_name:
            print(f"WARNING: Category mismatch: {file_category_name} != {category_name}")

        print(f"Category: {file_category_name}")
        print(f"Hyperparameters: {hyperparams}")

        scores = extract_scores(data)
        print(f"Scores: orig_harm={len(scores['original_harmful'])}, "
              f"mod_harm={len(scores['modified_harmful'])}, "
              f"orig_harmless={len(scores['original_harmless'])}, "
              f"mod_harmless={len(scores['modified_harmless'])}, "
              f"loc_diff={len(scores['locality_differences'])}")

        if not scores["original_harmful"] and not scores["modified_harmful"] and not evaluator:
            print("  NOTE: No scores found. Run with --evaluate-if-empty to run LlamaGuard evaluation.")

        if scores["original_harmful"] or scores["modified_harmful"]:
            fname = generate_plot_filename("harmfulness_distribution", file_category_name, evaluator_name, ext=".png")
            plot_harmfulness_distribution(
                scores["original_harmful"], scores["modified_harmful"],
                output_dirs["harmfulness_distribution"] / fname,
                evaluator_name=evaluator_name, method_name=baseline_type,
                category_name=file_category_name, hyperparams=hyperparams,
            )

        if scores["original_harmless"] or scores["modified_harmless"]:
            fname = generate_plot_filename("locality_distribution", file_category_name, evaluator_name, ext=".png")
            plot_locality_distribution(
                scores["original_harmless"], scores["modified_harmless"],
                scores["locality_differences"],
                output_dirs["locality_distribution"] / fname,
                evaluator_name=evaluator_name, method_name=baseline_type,
                category_name=file_category_name, hyperparams=hyperparams,
            )

        if scores["modified_harmful"]:
            all_heatmap_harm.append({
                "max_weight": hyperparams.get("max_weight"),
                "min_weight": hyperparams.get("min_weight"),
                "mean_modified_score": np.mean(scores["modified_harmful"]),
            })

        if scores["locality_differences"]:
            all_heatmap_loc.append({
                "max_weight": hyperparams.get("max_weight"),
                "min_weight": hyperparams.get("min_weight"),
                "average_locality_change": np.mean(scores["locality_differences"]),
            })

    print(f"\n{'='*80}")
    print("Building combined heatmap grids...")
    print(f"{'='*80}")

    if all_heatmap_harm:
        fname = generate_plot_filename("harmfulness_heatmap", category_name, evaluator_name)
        plot_harmfulness_heatmap(
            all_heatmap_harm,
            output_dirs["harmfulness_heatmap"] / fname,
            evaluator_name=evaluator_name, method_name=baseline_type,
            category_name=category_name,
        )

    if all_heatmap_loc:
        fname = generate_plot_filename("locality_heatmap", category_name, evaluator_name)
        plot_locality_heatmap(
            all_heatmap_loc,
            output_dirs["locality_heatmap"] / fname,
            evaluator_name=evaluator_name, method_name=baseline_type,
            category_name=category_name,
        )


def process_grpo_format_files(
    file_paths: List[Path],
    output_dirs: Dict[str, Path],
    baseline_type: str,
    evaluator_name: str = "",
) -> None:
    """Process GRPO format files."""
    all_heatmap_harm = []
    category_name = None

    for file_idx, file_path in enumerate(file_paths, 1):
        print(f"\n{'='*80}")
        print(f"File {file_idx}/{len(file_paths)}: {file_path.name}")
        print(f"{'='*80}")

        data = load_answers_file(file_path)
        file_category_name = data.get("category", "Unknown")
        if category_name is None:
            category_name = file_category_name
        if file_category_name != category_name:
            print(f"WARNING: Category mismatch: {file_category_name} != {category_name}")

        hyperparams = parse_hyperparams_from_filename(file_path.name)
        if not hyperparams:
            exp_config = data.get("experiment_config", {})
            abliteration_params = exp_config.get("abliteration_params", {})
            if abliteration_params:
                hyperparams = abliteration_params

        print(f"Category: {file_category_name}")
        print(f"Hyperparameters: {hyperparams}")

        scores = extract_scores_grpo(data)
        print(f"  Harmful (modified): {len(scores['modified_harmful'])}")

        if scores["modified_harmful"]:
            fname = generate_plot_filename("harmfulness_distribution", file_category_name, evaluator_name, ext=".png")
            plot_harmfulness_distribution(
                [], scores["modified_harmful"],
                output_dirs["harmfulness_distribution"] / fname,
                evaluator_name=evaluator_name, method_name=baseline_type,
                category_name=file_category_name, hyperparams=hyperparams,
            )

            mean_score = np.mean(scores["modified_harmful"])
            all_heatmap_harm.append({
                "max_weight": hyperparams.get("max_weight"),
                "min_weight": hyperparams.get("min_weight"),
                "mean_modified_score": mean_score,
            })

    print(f"\n{'='*80}")
    print("Building GRPO heatmap...")
    print(f"{'='*80}")

    if all_heatmap_harm:
        for item in all_heatmap_harm:
            print(f"  max_w={item['max_weight']}, min_w={item['min_weight']}, "
                  f"mean={item['mean_modified_score']:.2f}")
        fname = generate_plot_filename("harmfulness_heatmap", category_name or "GRPO", evaluator_name)
        plot_harmfulness_heatmap(
            all_heatmap_harm,
            output_dirs["harmfulness_heatmap"] / fname,
            evaluator_name=evaluator_name, method_name=baseline_type,
            category_name=category_name or "GRPO",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(paths=None):
    parser = argparse.ArgumentParser(
        description="Build plots from answer files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python draw_graphics.py results/baselines/graph_average/answers/*.json
  python draw_graphics.py --evaluator-name deepseek results/graph_average/answers/*.json
        """,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--evaluate-if-empty", action="store_true",
        help="Run LlamaGuard evaluation when evaluations are empty",
    )
    parser.add_argument(
        "--evaluator-name", type=str, default="",
        help="Name of the evaluator (e.g. llamaguard, deepseek) for plot titles/filenames",
    )

    args = parser.parse_args()

    file_paths = [Path(__file__).parent.parent / Path(fp) for fp in paths]

    for file_path in file_paths:
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            sys.exit(1)
        if not file_path.is_file():
            print(f"ERROR: Path is not a file: {file_path}")
            sys.exit(1)
        if file_path.suffix.lower() != ".json":
            print(f"WARNING: File does not have .json extension: {file_path}")

    evaluator = None
    if args.evaluate_if_empty:
        try:
            from evaluate.evaluation_llamaguard import LlamaGuardEvaluator
            print("Loading LlamaGuard evaluator...")
            evaluator = LlamaGuardEvaluator()
            print("Evaluator ready.")
        except Exception as e:
            print(f"WARNING: Could not load evaluator: {e}")

    process_answers_files(file_paths, evaluator, evaluator_name=args.evaluator_name)


if __name__ == "__main__":
    paths = [
        "results/graph_average/answers/graph_average_physical_harm_deepinfra_1.json",
        "results/graph_average/answers/graph_average_physical_harm_deepinfra_2.json",
        "results/graph_average/answers/graph_average_physical_harm_deepinfra_3.json",
        "results/graph_average/answers/graph_average_physical_harm_deepinfra_4.json"
    ]
    main(paths)
