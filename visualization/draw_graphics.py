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
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Supported baseline types
BASELINE_TYPES = [
    "graph_average",
    "topic_ablation", 
    "tag_ablation",
    "graph_grpo",
    "full_question",
]


def detect_baseline_type(file_path: Path) -> Optional[str]:
    """
    Infer baseline type from file path.
    
    Args:
        file_path: Path to answers file
        
    Returns:
        Baseline name or None if not detected
    """
    path_str = str(file_path.resolve())
    
    for baseline_type in BASELINE_TYPES:
        if f"/{baseline_type}/" in path_str or f"\\{baseline_type}\\" in path_str:
            return baseline_type
    
    return None


def get_output_directories(file_path: Path, baseline_type: str) -> Dict[str, Path]:
    """
    Return paths to directories for saving plots.
    
    Args:
        file_path: Path to answers file
        baseline_type: Baseline type
        
    Returns:
        Dict of directory paths
    """
    # Find baselines root
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
    
    # Create directories if needed
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def load_answers_file(file_path: Path) -> Dict:
    """
    Load answers file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Loaded data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_scores(data: Dict) -> Dict[str, List[int]]:
    """
    Extract scores from answers file data.
    
    Args:
        data: File data
        
    Returns:
        Dict of scores
    """
    scores = {
        "original_harmful": [],
        "modified_harmful": [],
        "original_harmless": [],
        "modified_harmless": [],
        "locality_differences": [],
    }
    
    # Extract scores for harmful questions
    harmful_data = data.get("harmful_questions", {})
    
    # Format 1: original_evaluations / modified_evaluations (lists of dict)
    original_evals = harmful_data.get("original_evaluations", [])
    modified_evals = harmful_data.get("modified_evaluations", [])
    
    for eval_item in original_evals:
        score = eval_item.get("score")
        if score is not None:
            scores["original_harmful"].append(int(score))
    
    for eval_item in modified_evals:
        score = eval_item.get("score")
        if score is not None:
            scores["modified_harmful"].append(int(score))
    
    # Extract scores for harmless questions (locality)
    harmless_data = data.get("harmless_questions", {})
    locality_scores = harmless_data.get("locality_scores", [])
    
    for loc_item in locality_scores:
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


def create_harmfulness_distribution_plot(
    original_scores: List[int],
    modified_scores: List[int],
    category_name: str,
    hyperparams: Dict,
    output_path: Path
) -> None:
    """
    Create harmfulness score distribution plot.
    
    Args:
        original_scores: Scores for original responses
        modified_scores: Scores for modified responses
        category_name: Category name
        hyperparams: Experiment hyperparameters
        output_path: Path to save plot
    """
    if not original_scores and not modified_scores:
        print("No data for harmfulness distribution plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot for original responses
    if original_scores:
        axes[0].hist(original_scores, bins=range(0, 6), align='left', 
                     edgecolor='black', alpha=0.7, color='blue')
        mean_orig = np.mean(original_scores)
        median_orig = np.median(original_scores)
        axes[0].text(0.02, 0.98, 
                    f'Total scores: {len(original_scores)}\n'
                    f'Mean: {mean_orig:.2f}\n'
                    f'Median: {median_orig:.2f}',
                    transform=axes[0].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                    fontsize=10)
    axes[0].set_xlabel('Harmfulness score (0-4)', fontsize=12)
    axes[0].set_ylabel('Number of responses', fontsize=12)
    axes[0].set_title('Original responses (pre-modification)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(0, 5))
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot for modified responses
    if modified_scores:
        axes[1].hist(modified_scores, bins=range(0, 6), align='left', 
                     edgecolor='black', alpha=0.7, color='red')
        mean_mod = np.mean(modified_scores)
        median_mod = np.median(modified_scores)
        
        if original_scores:
            change = mean_mod - np.mean(original_scores)
            change_text = f'\nChange: {change:+.2f}'
        else:
            change_text = ''
        
        axes[1].text(0.02, 0.98, 
                    f'Total scores: {len(modified_scores)}\n'
                    f'Mean: {mean_mod:.2f}\n'
                    f'Median: {median_mod:.2f}{change_text}',
                    transform=axes[1].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
                    fontsize=10)
    axes[1].set_xlabel('Harmfulness score (0-4)', fontsize=12)
    axes[1].set_ylabel('Number of responses', fontsize=12)
    axes[1].set_title('Modified responses (post-modification)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(0, 5))
    axes[1].grid(axis='y', alpha=0.3)
    
    hyperparam_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items()])
    plt.suptitle(
        f'Harmfulness score distribution for category "{category_name}"\n'
        f'Hyperparameters: {hyperparam_str}',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Harmfulness distribution plot saved to: {output_path}")
    plt.close()


def create_locality_distribution_plot(
    original_scores: List[int],
    modified_scores: List[int],
    differences: List[int],
    category_name: str,
    hyperparams: Dict,
    output_path: Path
) -> None:
    """
    Create locality score distribution plot.
    
    Args:
        original_scores: Scores for original responses on harmless questions
        modified_scores: Scores for modified responses on harmless questions
        differences: Score differences (modified - original)
        category_name: Category name
        hyperparams: Experiment hyperparameters
        output_path: Path to save plot
    """
    if not original_scores and not modified_scores:
        print("No data for locality distribution plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot for original scores (harmless questions)
    if original_scores:
        axes[0].hist(original_scores, bins=range(0, 6), align='left', 
                     edgecolor='black', alpha=0.7, color='blue')
        mean_orig = np.mean(original_scores)
        median_orig = np.median(original_scores)
        axes[0].text(0.02, 0.98, 
                    f'Total scores: {len(original_scores)}\n'
                    f'Mean: {mean_orig:.2f}\n'
                    f'Median: {median_orig:.2f}',
                    transform=axes[0].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                    fontsize=10)
    axes[0].set_xlabel('Harmfulness score (0-4)', fontsize=12)
    axes[0].set_ylabel('Number of responses', fontsize=12)
    axes[0].set_title('Original responses (harmless questions)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(0, 5))
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot for modified scores (harmless questions)
    if modified_scores:
        axes[1].hist(modified_scores, bins=range(0, 6), align='left', 
                     edgecolor='black', alpha=0.7, color='red')
        mean_mod = np.mean(modified_scores)
        median_mod = np.median(modified_scores)
        avg_change = np.mean(differences) if differences else 0
        axes[1].text(0.02, 0.98, 
                    f'Total scores: {len(modified_scores)}\n'
                    f'Mean: {mean_mod:.2f}\n'
                    f'Median: {median_mod:.2f}\n'
                    f'Average change: {avg_change:+.2f}',
                    transform=axes[1].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
                    fontsize=10)
    axes[1].set_xlabel('Harmfulness score (0-4)', fontsize=12)
    axes[1].set_ylabel('Number of responses', fontsize=12)
    axes[1].set_title('Modified responses (harmless questions)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(0, 5))
    axes[1].grid(axis='y', alpha=0.3)
    
    # Plot distribution of changes (locality differences)
    if differences:
        axes[2].hist(differences, bins=range(-4, 6), align='left', 
                     edgecolor='black', alpha=0.7, color='green')
        axes[2].axvline(x=0, color='black', linestyle='--', linewidth=2, label='No change')
        mean_diff = np.mean(differences)
        median_diff = np.median(differences)
        axes[2].text(0.02, 0.98, 
                    f'Total pairs: {len(differences)}\n'
                    f'Mean change: {mean_diff:+.2f}\n'
                    f'Median change: {median_diff:+.2f}',
                    transform=axes[2].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                    fontsize=10)
        axes[2].legend()
    axes[2].set_xlabel('Score change (modified - original)', fontsize=12)
    axes[2].set_ylabel('Number of responses', fontsize=12)
    axes[2].set_title('Change distribution (Locality)', fontsize=14, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    hyperparam_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items()])
    plt.suptitle(
        f'Locality score distribution for category "{category_name}"\n'
        f'Hyperparameters: {hyperparam_str}',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Locality distribution plot saved to: {output_path}")
    plt.close()


def create_harmfulness_heatmap_grid(
    heatmap_data: List[Dict],
    category_name: str,
    output_path: Path
) -> None:
    """
    Create a single heatmap grid for harmfulness scores from multiple files.
    
    Creates a 2D heatmap where:
    - Rows: min_weight values
    - Columns: max_weight values
    - Cell values: mean harmfulness scores
    
    Args:
        heatmap_data: List of dicts with keys: max_weight, max_weight_position,
                     min_weight, min_weight_distance, mean_modified_score
        category_name: Category name
        output_path: Path to save the heatmap
    """
    if not heatmap_data:
        print("No data for harmfulness heatmap")
        return

    df = pd.DataFrame(heatmap_data)

    # Create pivot table: rows = min_weight, columns = max_weight
    pivot_data = df.pivot_table(
        values="mean_modified_score",
        index="min_weight",
        columns="max_weight",
        aggfunc="mean"
    )

    if pivot_data.empty:
        print("No data for harmfulness heatmap (empty pivot)")
        return

    # Sort index and columns
    pivot_data = pivot_data.sort_index(ascending=True)
    pivot_data = pivot_data.reindex(sorted(pivot_data.columns), axis=1)

    # Create figure with single heatmap
    n_rows, n_cols = pivot_data.shape
    fig_width = max(4, n_cols * 1.5 + 1.5)  # Extra space for colorbar
    fig_height = max(4, n_rows * 1.5)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create heatmap with colorbar on the right
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=4,
        cbar=True,
        cbar_kws={"label": "Mean Harmfulness Score (0-4)"},
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"fontsize": 12}
    )

    ax.set_xlabel("max_weight", fontsize=12)
    ax.set_ylabel("min_weight", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Harmfulness heatmap grid saved to: {output_path}")
    plt.close()


def create_locality_heatmap_grid(
    heatmap_data: List[Dict],
    category_name: str,
    output_path: Path
) -> None:
    """
    Create a single heatmap grid for locality scores from multiple files.
    
    Creates a 2D heatmap where:
    - Rows: min_weight values
    - Columns: max_weight values
    - Cell values: average locality change
    
    Args:
        heatmap_data: List of dicts with keys: max_weight, max_weight_position,
                     min_weight, min_weight_distance, average_locality_change
        category_name: Category name
        output_path: Path to save the heatmap
    """
    if not heatmap_data:
        print("No data for locality heatmap")
        return

    df = pd.DataFrame(heatmap_data)

    # Create pivot table: rows = min_weight, columns = max_weight
    pivot_data = df.pivot_table(
        values="average_locality_change",
        index="min_weight",
        columns="max_weight",
        aggfunc="mean"
    )

    if pivot_data.empty:
        print("No data for locality heatmap (empty pivot)")
        return

    # Sort index and columns
    pivot_data = pivot_data.sort_index(ascending=True)
    pivot_data = pivot_data.reindex(sorted(pivot_data.columns), axis=1)

    # Create figure with single heatmap
    n_rows, n_cols = pivot_data.shape
    fig_width = max(4, n_cols * 1.5 + 1.5)  # Extra space for colorbar
    fig_height = max(4, n_rows * 1.5)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create heatmap with colorbar on the right
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        cbar=True,
        cbar_kws={"label": "Average Locality Change"},
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"fontsize": 12}
    )

    ax.set_xlabel("max_weight", fontsize=12)
    ax.set_ylabel("min_weight", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Locality heatmap grid saved to: {output_path}")
    plt.close()


def is_combined_results_file(data: Dict) -> bool:
    """
    Check if file contains combined results (multiple hyperparameter sets in one file).
    
    Combined results files have structure:
    {
        "experiment_config": {...},
        "results": {
            "param_key1": {...},
            "param_key2": {...},
            ...
        }
    }
    
    Single answer files have structure:
    {
        "experiment_info": {...},
        "harmful_questions": {...},
        "harmless_questions": {...}
    }
    """
    return "results" in data and "experiment_config" in data


def is_grpo_format_file(data: Dict) -> bool:
    """
    Check if file is in GRPO format.
    
    GRPO format files have structure:
    {
        "category": "...",
        "questions": [...],
        "responses": [...],
        "scores": [...],
        "timestamp": "..."
    }
    """
    return (
        "category" in data 
        and "questions" in data 
        and "responses" in data 
        and "scores" in data
        and "experiment_info" not in data
    )


def parse_hyperparams_from_filename(filename: str) -> Dict[str, float]:
    """
    Parse hyperparameters from filename.
    
    Expected format: answers_Category_max_weight=X_&max_weight_position=Y_&min_weight=Z_&min_weight_distance=W_timestamp.json
    
    Args:
        filename: Filename to parse
        
    Returns:
        Dict with hyperparameters (may be empty if not found)
    """
    import re
    
    hyperparams = {}
    
    # Pattern for hyperparameters in filename
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
    """
    Extract scores from GRPO format file.
    
    GRPO files only contain scores for the modified model's responses to harmful questions.
    No original scores or locality data available.
    
    Args:
        data: Data from GRPO format file
        
    Returns:
        Dict with extracted scores (only modified_harmful will be populated)
    """
    scores = {
        "original_harmful": [],
        "modified_harmful": [],
        "original_harmless": [],
        "modified_harmless": [],
        "locality_differences": [],
    }
    
    # GRPO format: scores are for the model's responses to harmful questions
    raw_scores = data.get("scores", [])
    scores["modified_harmful"] = [int(s) for s in raw_scores if s is not None]
    
    return scores


def extract_scores_from_combined_result(result_data: Dict) -> Dict[str, List[int]]:
    """
    Extract scores from a single result entry in combined results file.
    
    Args:
        result_data: Single result dict with keys like modified_scores, 
                    original_scores, locality_scores, etc.
    
    Returns:
        Dict with extracted scores
    """
    scores = {
        "original_harmful": [],
        "modified_harmful": [],
        "original_harmless": [],
        "modified_harmless": [],
        "locality_differences": [],
    }
    
    # Extract harmful question scores
    original_scores = result_data.get("original_scores", [])
    modified_scores = result_data.get("modified_scores", [])
    
    scores["original_harmful"] = [int(s) for s in original_scores if s is not None]
    scores["modified_harmful"] = [int(s) for s in modified_scores if s is not None]
    
    # Extract locality scores
    locality_scores = result_data.get("locality_scores", [])
    for loc_item in locality_scores:
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


def generate_output_filename(
    category_name: str, param_key: str, plot_type: str, ext: str = ".png"
) -> str:
    """
    Generate output filename for a plot.

    Args:
        category_name: Category name
        param_key: Parameter key
        plot_type: Plot type
        ext: File extension (e.g. .png, .pdf)

    Returns:
        Filename string
    """
    category_safe = category_name.replace('/', '_').replace(' ', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{plot_type}_{category_safe}_{timestamp}{ext}"


def process_answers_files(file_paths: List[Path]) -> None:
    """
    Process answer files and build plots.
    
    Supports two file formats:
    1. Single answer files (one hyperparameter set per file) - used by graph_average, topic_ablation
    2. Combined results files (multiple hyperparameter sets in one file) - used by tag_ablation
    
    For each hyperparameter set: build distribution plots.
    For all sets: build one combined heatmap grid.
    
    Args:
        file_paths: List of paths to answer files
    """
    if not file_paths:
        print("ERROR: No files provided")
        sys.exit(1)
    
    print("=" * 80)
    print(f"Processing {len(file_paths)} file(s)")
    print("=" * 80)
    
    # Determine baseline type from first file
    baseline_type = detect_baseline_type(file_paths[0])
    if baseline_type is None:
        print(f"ERROR: Could not determine baseline type from path: {file_paths[0]}")
        print(f"Supported types: {BASELINE_TYPES}")
        sys.exit(1)
    
    # Verify all files are from the same baseline type
    for file_path in file_paths[1:]:
        if detect_baseline_type(file_path) != baseline_type:
            print(f"ERROR: Files must be from the same baseline type")
            print(f"  First file: {baseline_type}")
            print(f"  File {file_path}: {detect_baseline_type(file_path)}")
            sys.exit(1)
    
    print(f"Baseline type: {baseline_type}")
    
    # Get output directories
    output_dirs = get_output_directories(file_paths[0], baseline_type)
    print(f"Output directories:")
    for name, path in output_dirs.items():
        print(f"  {name}: {path}")
    
    # Check file format and process accordingly
    first_data = load_answers_file(file_paths[0])
    
    if is_combined_results_file(first_data):
        # Process combined results file (tag_ablation format)
        process_combined_results_files(file_paths, output_dirs, baseline_type)
    elif is_grpo_format_file(first_data):
        # Process GRPO format files
        process_grpo_format_files(file_paths, output_dirs, baseline_type)
    else:
        # Process single answer files (graph_average, topic_ablation format)
        process_single_answer_files(file_paths, output_dirs, baseline_type)
    
    print("\n" + "=" * 80)
    print("Processing completed!")
    print("=" * 80)


def process_combined_results_files(
    file_paths: List[Path],
    output_dirs: Dict[str, Path],
    baseline_type: str
) -> None:
    """
    Process combined results files (tag_ablation format).
    Each file contains results for multiple hyperparameter sets.
    """
    all_heatmap_data_harmfulness = []
    all_heatmap_data_locality = []
    category_name = None
    
    for file_idx, file_path in enumerate(file_paths, 1):
        print(f"\n{'='*80}")
        print(f"File {file_idx}/{len(file_paths)}: {file_path.name}")
        print(f"{'='*80}")
        
        # Load data
        data = load_answers_file(file_path)
        
        # Extract experiment config
        experiment_config = data.get("experiment_config", {})
        tag_name = experiment_config.get("tag", "Unknown")
        
        # Use tag as category name
        if category_name is None:
            category_name = tag_name
        
        print(f"Tag: {tag_name}")
        print(f"Total hyperparameter combinations: {experiment_config.get('total_combinations', 0)}")
        
        # Process each result (each hyperparameter set)
        results = data.get("results", {})
        
        for result_idx, (param_key, result_data) in enumerate(results.items(), 1):
            hyperparams = result_data.get("hyperparameters", {})
            
            print(f"\n  [{result_idx}/{len(results)}] {param_key}")
            print(f"    Hyperparameters: {hyperparams}")
            
            # Extract scores
            scores = extract_scores_from_combined_result(result_data)
            
            print(f"    Extracted scores:")
            print(f"      Harmful (original): {len(scores['original_harmful'])}")
            print(f"      Harmful (modified): {len(scores['modified_harmful'])}")
            print(f"      Harmless (original): {len(scores['original_harmless'])}")
            print(f"      Harmless (modified): {len(scores['modified_harmless'])}")
            print(f"      Locality differences: {len(scores['locality_differences'])}")
            
            # Build distribution plots for this hyperparameter set
            print(f"    Building distribution plots...")
            
            # 1. Harmfulness distribution
            if scores['original_harmful'] or scores['modified_harmful']:
                filename = generate_output_filename(
                    tag_name, param_key, "harmfulness_distribution"
                )
                output_path = output_dirs["harmfulness_distribution"] / filename
                create_harmfulness_distribution_plot(
                    scores['original_harmful'],
                    scores['modified_harmful'],
                    tag_name,
                    hyperparams,
                    output_path
                )
            
            # 2. Locality distribution
            if scores['original_harmless'] or scores['modified_harmless']:
                filename = generate_output_filename(
                    tag_name, param_key, "locality_distribution"
                )
                output_path = output_dirs["locality_distribution"] / filename
                create_locality_distribution_plot(
                    scores['original_harmless'],
                    scores['modified_harmless'],
                    scores['locality_differences'],
                    tag_name,
                    hyperparams,
                    output_path
                )
            
            # Collect data for heatmaps
            if scores['modified_harmful']:
                mean_score = np.mean(scores['modified_harmful'])
                all_heatmap_data_harmfulness.append({
                    "max_weight": hyperparams.get("max_weight"),
                    "max_weight_position": hyperparams.get("max_weight_position"),
                    "min_weight": hyperparams.get("min_weight"),
                    "min_weight_distance": hyperparams.get("min_weight_distance"),
                    "mean_modified_score": mean_score,
                })
            
            # Get average locality change directly from result_data
            avg_locality_change = result_data.get("average_locality_change")
            if avg_locality_change is not None:
                all_heatmap_data_locality.append({
                    "max_weight": hyperparams.get("max_weight"),
                    "max_weight_position": hyperparams.get("max_weight_position"),
                    "min_weight": hyperparams.get("min_weight"),
                    "min_weight_distance": hyperparams.get("min_weight_distance"),
                    "average_locality_change": avg_locality_change,
                })
    
    # Build combined heatmaps
    print(f"\n{'='*80}")
    print("Building combined heatmap grids...")
    print(f"{'='*80}")
    
    # Harmfulness heatmap grid
    if all_heatmap_data_harmfulness:
        filename = generate_output_filename(
            category_name, "combined", "harmfulness_heatmap", ext=".pdf"
        )
        output_path = output_dirs["harmfulness_heatmap"] / filename
        create_harmfulness_heatmap_grid(
            all_heatmap_data_harmfulness,
            category_name,
            output_path
        )
    
    # Locality heatmap grid
    if all_heatmap_data_locality:
        filename = generate_output_filename(
            category_name, "combined", "locality_heatmap", ext=".pdf"
        )
        output_path = output_dirs["locality_heatmap"] / filename
        create_locality_heatmap_grid(
            all_heatmap_data_locality,
            category_name,
            output_path
        )


def process_single_answer_files(
    file_paths: List[Path],
    output_dirs: Dict[str, Path],
    baseline_type: str
) -> None:
    """
    Process single answer files (graph_average, topic_ablation format).
    Each file contains results for one hyperparameter set.
    """
    all_heatmap_data_harmfulness = []
    all_heatmap_data_locality = []
    category_name = None
    
    for file_idx, file_path in enumerate(file_paths, 1):
        print(f"\n{'='*80}")
        print(f"File {file_idx}/{len(file_paths)}: {file_path.name}")
        print(f"{'='*80}")
        
        # Load data
        data = load_answers_file(file_path)
        
        # Extract experiment info
        experiment_info = data.get("experiment_info", {})
        file_category_name = experiment_info.get("category", "Unknown")
        hyperparams = experiment_info.get("hyperparameters", {})
        param_key = experiment_info.get("param_key", "unknown")
        
        # Set category name from first file
        if category_name is None:
            category_name = file_category_name
        
        # Verify all files are from the same category
        if file_category_name != category_name:
            print(f"WARNING: Category mismatch: {file_category_name} != {category_name}")
        
        print(f"Category: {file_category_name}")
        print(f"Hyperparameters: {hyperparams}")
        
        # Extract scores
        scores = extract_scores(data)
        
        print(f"Extracted scores:")
        print(f"  Harmful (original): {len(scores['original_harmful'])}")
        print(f"  Harmful (modified): {len(scores['modified_harmful'])}")
        print(f"  Harmless (original): {len(scores['original_harmless'])}")
        print(f"  Harmless (modified): {len(scores['modified_harmless'])}")
        print(f"  Locality differences: {len(scores['locality_differences'])}")
        
        # Build distribution plots for this file
        print(f"\nBuilding distribution plots for file {file_idx}...")
        
        # 1. Harmfulness distribution
        if scores['original_harmful'] or scores['modified_harmful']:
            filename = generate_output_filename(
                file_category_name, param_key, "harmfulness_distribution"
            )
            output_path = output_dirs["harmfulness_distribution"] / filename
            create_harmfulness_distribution_plot(
                scores['original_harmful'],
                scores['modified_harmful'],
                file_category_name,
                hyperparams,
                output_path
            )
        
        # 2. Locality distribution
        if scores['original_harmless'] or scores['modified_harmless']:
            filename = generate_output_filename(
                file_category_name, param_key, "locality_distribution"
            )
            output_path = output_dirs["locality_distribution"] / filename
            create_locality_distribution_plot(
                scores['original_harmless'],
                scores['modified_harmless'],
                scores['locality_differences'],
                file_category_name,
                hyperparams,
                output_path
            )
        
        # Collect data for heatmaps
        if scores['modified_harmful']:
            mean_score = np.mean(scores['modified_harmful'])
            all_heatmap_data_harmfulness.append({
                "max_weight": hyperparams.get("max_weight"),
                "max_weight_position": hyperparams.get("max_weight_position"),
                "min_weight": hyperparams.get("min_weight"),
                "min_weight_distance": hyperparams.get("min_weight_distance"),
                "mean_modified_score": mean_score,
            })
        
        if scores['locality_differences']:
            avg_change = np.mean(scores['locality_differences'])
            all_heatmap_data_locality.append({
                "max_weight": hyperparams.get("max_weight"),
                "max_weight_position": hyperparams.get("max_weight_position"),
                "min_weight": hyperparams.get("min_weight"),
                "min_weight_distance": hyperparams.get("min_weight_distance"),
                "average_locality_change": avg_change,
            })
    
    # Build combined heatmaps
    print(f"\n{'='*80}")
    print("Building combined heatmap grids...")
    print(f"{'='*80}")
    
    # Harmfulness heatmap grid
    if all_heatmap_data_harmfulness:
        filename = generate_output_filename(
            category_name, "combined", "harmfulness_heatmap", ext=".pdf"
        )
        output_path = output_dirs["harmfulness_heatmap"] / filename
        create_harmfulness_heatmap_grid(
            all_heatmap_data_harmfulness,
            category_name,
            output_path
        )
    
    # Locality heatmap grid
    if all_heatmap_data_locality:
        filename = generate_output_filename(
            category_name, "combined", "locality_heatmap", ext=".pdf"
        )
        output_path = output_dirs["locality_heatmap"] / filename
        create_locality_heatmap_grid(
            all_heatmap_data_locality,
            category_name,
            output_path
        )


def process_grpo_format_files(
    file_paths: List[Path],
    output_dirs: Dict[str, Path],
    baseline_type: str
) -> None:
    """
    Process GRPO format files.
    
    GRPO format files contain only harmfulness scores for the model's responses.
    No original scores or locality data available.
    """
    all_heatmap_data_harmfulness = []
    category_name = None
    
    for file_idx, file_path in enumerate(file_paths, 1):
        print(f"\n{'='*80}")
        print(f"File {file_idx}/{len(file_paths)}: {file_path.name}")
        print(f"{'='*80}")
        
        # Load data
        data = load_answers_file(file_path)
        
        # Extract info from GRPO format
        file_category_name = data.get("category", "Unknown")
        timestamp = data.get("timestamp", "")
        num_questions = len(data.get("questions", []))
        num_responses = len(data.get("responses", []))
        
        # Set category name from first file
        if category_name is None:
            category_name = file_category_name
        
        # Verify all files are from the same category
        if file_category_name != category_name:
            print(f"WARNING: Category mismatch: {file_category_name} != {category_name}")
        
        # Try to parse hyperparameters from filename or data
        hyperparams = parse_hyperparams_from_filename(file_path.name)
        
        # If not in filename, try to get from data (experiment_config.abliteration_params)
        if not hyperparams:
            exp_config = data.get("experiment_config", {})
            abliteration_params = exp_config.get("abliteration_params", {})
            if abliteration_params:
                hyperparams = abliteration_params
        
        print(f"Category: {file_category_name}")
        print(f"Timestamp: {timestamp}")
        print(f"Questions: {num_questions}, Responses: {num_responses}")
        print(f"Hyperparameters: {hyperparams}")
        
        # Extract scores using GRPO-specific function
        scores = extract_scores_grpo(data)
        
        print(f"Extracted scores:")
        print(f"  Harmful (modified): {len(scores['modified_harmful'])}")
        
        # Build distribution plot for this file
        print(f"\nBuilding distribution plots for file {file_idx}...")
        
        # Harmfulness distribution (only modified scores for GRPO)
        if scores['modified_harmful']:
            # Extract param_key from filename (e.g., "answers_Physical_harm_20260125_172522.json")
            param_key = file_path.stem.replace("answers_", "")
            
            filename = generate_output_filename(
                file_category_name, param_key, "harmfulness_distribution"
            )
            output_path = output_dirs["harmfulness_distribution"] / filename
            
            # Create distribution plot with only modified scores
            create_grpo_distribution_plot(
                scores['modified_harmful'],
                file_category_name,
                output_path
            )
            
            # Collect data for heatmap
            mean_score = np.mean(scores['modified_harmful'])
            all_heatmap_data_harmfulness.append({
                "file": file_path.name,
                "category": file_category_name,
                "mean_score": mean_score,
                "mean_modified_score": mean_score,  # For compatibility with standard heatmap
                "num_samples": len(scores['modified_harmful']),
                "max_weight": hyperparams.get("max_weight"),
                "min_weight": hyperparams.get("min_weight"),
                "max_weight_position": hyperparams.get("max_weight_position"),
                "min_weight_distance": hyperparams.get("min_weight_distance"),
            })
    
    # Build heatmap
    print(f"\n{'='*80}")
    print("Building GRPO heatmap...")
    print(f"{'='*80}")
    
    if all_heatmap_data_harmfulness:
        # Print summary
        for item in all_heatmap_data_harmfulness:
            print(f"  {item['file']}: mean={item['mean_score']:.2f}, n={item['num_samples']}")
        
        # Create heatmap
        filename = generate_output_filename(
            category_name or "GRPO", "summary", "harmfulness_heatmap", ext=".pdf"
        )
        output_path = output_dirs["harmfulness_heatmap"] / filename
        create_grpo_heatmap(
            all_heatmap_data_harmfulness,
            category_name or "GRPO",
            output_path
        )


def create_grpo_distribution_plot(
    scores: List[int],
    category_name: str,
    output_path: Path
) -> None:
    """
    Create distribution plot for GRPO format (only modified scores).
    
    Args:
        scores: List of harmfulness scores (1-5)
        category_name: Name of the category
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    ax.hist(scores, bins=bins, edgecolor='black', alpha=0.7, color='steelblue', label='GRPO Model')
    
    # Add statistics
    mean_score = np.mean(scores)
    ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
    
    # Labels and title
    ax.set_xlabel('Harmfulness Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Harmfulness Distribution - {category_name}\n(GRPO Model)', fontsize=14)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text with statistics
    stats_text = f'N = {len(scores)}\nMean = {mean_score:.2f}\nStd = {np.std(scores):.2f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_grpo_heatmap(
    heatmap_data: List[Dict],
    category_name: str,
    output_path: Path
) -> None:
    """
    Create a heatmap for GRPO results.
    
    If hyperparameters (max_weight, min_weight) are available, creates a standard
    heatmap grid like other methods. Otherwise creates a summary heatmap.
    
    Args:
        heatmap_data: List of dicts with keys: file, category, mean_score, num_samples,
                     and optionally max_weight, min_weight, etc.
        category_name: Overall category name
        output_path: Path to save the heatmap
    """
    if not heatmap_data:
        print("No data for GRPO heatmap")
        return
    
    df = pd.DataFrame(heatmap_data)
    
    # Check if we have hyperparameters for standard heatmap
    has_hyperparams = (
        df['max_weight'].notna().any() and 
        df['min_weight'].notna().any()
    )
    
    if has_hyperparams:
        # Create standard heatmap with min_weight/max_weight axes (like other methods)
        pivot_data = df.pivot_table(
            values="mean_modified_score",
            index="min_weight",
            columns="max_weight",
            aggfunc="mean"
        )
        
        if pivot_data.empty:
            print("No data for GRPO heatmap (empty pivot)")
            return
        
        # Sort index and columns
        pivot_data = pivot_data.sort_index(ascending=True)
        pivot_data = pivot_data.reindex(sorted(pivot_data.columns), axis=1)
        
        # Create figure with single heatmap
        n_rows, n_cols = pivot_data.shape
        fig_width = max(4, n_cols * 1.5 + 1.5)
        fig_height = max(4, n_rows * 1.5)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=4,
            cbar=True,
            cbar_kws={"label": "Mean Harmfulness Score (0-4)"},
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"fontsize": 12}
        )
        
        ax.set_xlabel("max_weight", fontsize=12)
        ax.set_ylabel("min_weight", fontsize=12)
        
    else:
        # No hyperparameters - create a simple 1x1 heatmap with default values
        fig, ax = plt.subplots(figsize=(6, 4))
        
        mean_score = df['mean_score'].mean()
        
        # Create 1x1 heatmap
        data_matrix = np.array([[mean_score]])
        
        sns.heatmap(
            data_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=4,
            cbar=True,
            cbar_kws={"label": "Mean Harmfulness Score (0-4)"},
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"fontsize": 14, "fontweight": "bold"},
            xticklabels=["3.0"],  # max_weight
            yticklabels=["1.0"]   # min_weight
        )
        
        ax.set_xlabel("max_weight", fontsize=12)
        ax.set_ylabel("min_weight", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"GRPO heatmap saved to: {output_path}")
    plt.close()


def main(paths = None):
    parser = argparse.ArgumentParser(
        description="Build plots from answer files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python draw_graphics.py results/baselines/topic_ablation/answers/answers_Physical_harm_...json
  
  # Multiple files (distribution plots per file, combined heatmap grid)
  python draw_graphics.py file1.json file2.json file3.json
  
  # Using wildcards (bash)
  python draw_graphics.py results/baselines/graph_average/answers/*.json

The script automatically determines baseline type from the path and saves plots
to corresponding subdirectories (harmfulness/distribution_plots, 
harmfulness/heatmap_plots, locality/distribution_plots, locality/heatmap_plots).

For multiple files:
- Each file gets its own distribution plots
- All files are combined into one heatmap grid
        """
    )
    
    # parser.add_argument(
    #     "file_paths",
    #     type=str,
    #     nargs="+",
    #     help="Path(s) to answer files (JSON). Multiple files will create combined heatmaps."
    # )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    # if paths is not None:
    #     args.file_paths = paths
    
    file_paths = [ Path(__file__).parent.parent / Path(fp) for fp in paths]
    
    # Check all files exist
    for file_path in file_paths:
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            sys.exit(1)
        
        if not file_path.is_file():
            print(f"ERROR: Path is not a file: {file_path}")
            sys.exit(1)
        
        if not file_path.suffix.lower() == ".json":
            print(f"WARNING: File does not have .json extension: {file_path}")
    
    # Process files
    process_answers_files(file_paths)


if __name__ == "__main__":
    paths = [
        "graph_experiments/results/baselines/graph_grpo/answers/answers_Physical_harm_20260125_172522.json"
    ]
    main(paths)
