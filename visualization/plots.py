"""
Shared plotting functions for harmfulness heatmaps, locality heatmaps,
and score distribution histograms.

This is the single canonical copy -- baselines, evaluation runners,
and draw_graphics.py all import from here.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_title(
    plot_label: str,
    evaluator_name: str = "",
    method_name: str = "",
    category_name: str = "",
) -> str:
    """Compose a multi-line plot title from metadata fields."""
    parts = [p for p in [evaluator_name, method_name, category_name] if p]
    subtitle = " | ".join(parts)
    if subtitle:
        return f"{plot_label}\n{subtitle}"
    return plot_label


def generate_plot_filename(
    plot_type: str,
    category_name: str,
    evaluator_name: str = "",
    ext: str = ".pdf",
) -> str:
    """Build a timestamped filename, optionally prefixed with the evaluator."""
    prefix = f"{evaluator_name}_" if evaluator_name else ""
    category_safe = category_name.replace("/", "_").replace(" ", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{plot_type}_{category_safe}_{ts}{ext}"


# ---------------------------------------------------------------------------
# Harmfulness heatmap
# ---------------------------------------------------------------------------

def plot_harmfulness_heatmap(
    heatmap_data: List[Dict],
    output_path: Path,
    evaluator_name: str = "",
    method_name: str = "",
    category_name: str = "",
) -> None:
    """Create a 2-D heatmap of mean harmfulness scores.

    Args:
        heatmap_data: List of dicts, each with at least ``max_weight``,
            ``min_weight``, and ``mean_modified_score``.
        output_path: Where to save the figure (PDF recommended).
        evaluator_name: E.g. "llamaguard", "deepseek".
        method_name: E.g. "graph_average", "tag_ablation".
        category_name: E.g. "Physical harm".
    """
    if not heatmap_data:
        print("No data for harmfulness heatmap")
        return

    df = pd.DataFrame(heatmap_data)

    pivot_data = df.pivot_table(
        values="mean_modified_score",
        index="min_weight",
        columns="max_weight",
        aggfunc="mean",
    )

    if pivot_data.empty:
        print("No data for harmfulness heatmap (empty pivot)")
        return

    pivot_data = pivot_data.sort_index(ascending=True)
    pivot_data = pivot_data.reindex(sorted(pivot_data.columns), axis=1)

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
        annot_kws={"fontsize": 12},
    )

    ax.set_xlabel("max_weight", fontsize=12)
    ax.set_ylabel("min_weight", fontsize=12)

    title = _build_title("Harmfulness Heatmap", evaluator_name, method_name, category_name)
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Harmfulness heatmap saved to: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Locality heatmap
# ---------------------------------------------------------------------------

def plot_locality_heatmap(
    heatmap_data: List[Dict],
    output_path: Path,
    evaluator_name: str = "",
    method_name: str = "",
    category_name: str = "",
) -> None:
    """Create a 2-D heatmap of average locality change.

    Args:
        heatmap_data: List of dicts, each with at least ``max_weight``,
            ``min_weight``, and ``average_locality_change``.
        output_path: Where to save the figure (PDF recommended).
        evaluator_name: E.g. "llamaguard", "deepseek".
        method_name: E.g. "graph_average", "tag_ablation".
        category_name: E.g. "Physical harm".
    """
    if not heatmap_data:
        print("No data for locality heatmap")
        return

    df = pd.DataFrame(heatmap_data)

    pivot_data = df.pivot_table(
        values="average_locality_change",
        index="min_weight",
        columns="max_weight",
        aggfunc="mean",
    )

    if pivot_data.empty:
        print("No data for locality heatmap (empty pivot)")
        return

    pivot_data = pivot_data.sort_index(ascending=True)
    pivot_data = pivot_data.reindex(sorted(pivot_data.columns), axis=1)

    n_rows, n_cols = pivot_data.shape
    fig_width = max(4, n_cols * 1.5 + 1.5)
    fig_height = max(4, n_rows * 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

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
        annot_kws={"fontsize": 12},
    )

    ax.set_xlabel("max_weight", fontsize=12)
    ax.set_ylabel("min_weight", fontsize=12)

    title = _build_title("Locality Heatmap", evaluator_name, method_name, category_name)
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Locality heatmap saved to: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Harmfulness distribution
# ---------------------------------------------------------------------------

def plot_harmfulness_distribution(
    original_scores: List[int],
    modified_scores: List[int],
    output_path: Path,
    evaluator_name: str = "",
    method_name: str = "",
    category_name: str = "",
    hyperparams: Optional[Dict] = None,
) -> None:
    """Create a side-by-side histogram of original vs modified harmfulness scores.

    Args:
        original_scores: Scores for original (unmodified) responses.
        modified_scores: Scores for modified responses.
        output_path: Where to save the figure.
        evaluator_name: E.g. "llamaguard", "deepseek".
        method_name: E.g. "graph_average".
        category_name: E.g. "Physical harm".
        hyperparams: Optional dict of hyperparameters to show in the title.
    """
    if not original_scores and not modified_scores:
        print("No data for harmfulness distribution plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if original_scores:
        axes[0].hist(
            original_scores, bins=range(0, 6), align="left",
            edgecolor="black", alpha=0.7, color="blue",
        )
        mean_orig = np.mean(original_scores)
        median_orig = np.median(original_scores)
        axes[0].text(
            0.02, 0.98,
            f"Total scores: {len(original_scores)}\n"
            f"Mean: {mean_orig:.2f}\n"
            f"Median: {median_orig:.2f}",
            transform=axes[0].transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
            fontsize=10,
        )
    axes[0].set_xlabel("Harmfulness score (0-4)", fontsize=12)
    axes[0].set_ylabel("Number of responses", fontsize=12)
    axes[0].set_title("Original responses (pre-modification)", fontsize=14, fontweight="bold")
    axes[0].set_xticks(range(0, 5))
    axes[0].grid(axis="y", alpha=0.3)

    if modified_scores:
        axes[1].hist(
            modified_scores, bins=range(0, 6), align="left",
            edgecolor="black", alpha=0.7, color="red",
        )
        mean_mod = np.mean(modified_scores)
        median_mod = np.median(modified_scores)
        change_text = ""
        if original_scores:
            change = mean_mod - np.mean(original_scores)
            change_text = f"\nChange: {change:+.2f}"
        axes[1].text(
            0.02, 0.98,
            f"Total scores: {len(modified_scores)}\n"
            f"Mean: {mean_mod:.2f}\n"
            f"Median: {median_mod:.2f}{change_text}",
            transform=axes[1].transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
            fontsize=10,
        )
    axes[1].set_xlabel("Harmfulness score (0-4)", fontsize=12)
    axes[1].set_ylabel("Number of responses", fontsize=12)
    axes[1].set_title("Modified responses (post-modification)", fontsize=14, fontweight="bold")
    axes[1].set_xticks(range(0, 5))
    axes[1].grid(axis="y", alpha=0.3)

    title = _build_title("Harmfulness Score Distribution", evaluator_name, method_name, category_name)
    if hyperparams:
        hp_str = ", ".join(f"{k}={v}" for k, v in hyperparams.items())
        title += f"\nHyperparameters: {hp_str}"
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Harmfulness distribution plot saved to: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Locality distribution
# ---------------------------------------------------------------------------

def plot_locality_distribution(
    original_scores: List[int],
    modified_scores: List[int],
    differences: List[int],
    output_path: Path,
    evaluator_name: str = "",
    method_name: str = "",
    category_name: str = "",
    hyperparams: Optional[Dict] = None,
) -> None:
    """Create a 3-panel histogram: original, modified, and difference (locality).

    Args:
        original_scores: Scores for original responses on harmless questions.
        modified_scores: Scores for modified responses on harmless questions.
        differences: Score differences (modified - original).
        output_path: Where to save the figure.
        evaluator_name: E.g. "llamaguard", "deepseek".
        method_name: E.g. "graph_average".
        category_name: E.g. "Physical harm".
        hyperparams: Optional dict of hyperparameters to show in the title.
    """
    if not original_scores and not modified_scores:
        print("No data for locality distribution plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    if original_scores:
        axes[0].hist(
            original_scores, bins=range(0, 6), align="left",
            edgecolor="black", alpha=0.7, color="blue",
        )
        mean_orig = np.mean(original_scores)
        median_orig = np.median(original_scores)
        axes[0].text(
            0.02, 0.98,
            f"Total scores: {len(original_scores)}\n"
            f"Mean: {mean_orig:.2f}\n"
            f"Median: {median_orig:.2f}",
            transform=axes[0].transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
            fontsize=10,
        )
    axes[0].set_xlabel("Harmfulness score (0-4)", fontsize=12)
    axes[0].set_ylabel("Number of responses", fontsize=12)
    axes[0].set_title("Original responses (harmless questions)", fontsize=14, fontweight="bold")
    axes[0].set_xticks(range(0, 5))
    axes[0].grid(axis="y", alpha=0.3)

    if modified_scores:
        axes[1].hist(
            modified_scores, bins=range(0, 6), align="left",
            edgecolor="black", alpha=0.7, color="red",
        )
        mean_mod = np.mean(modified_scores)
        median_mod = np.median(modified_scores)
        avg_change = np.mean(differences) if differences else 0
        axes[1].text(
            0.02, 0.98,
            f"Total scores: {len(modified_scores)}\n"
            f"Mean: {mean_mod:.2f}\n"
            f"Median: {median_mod:.2f}\n"
            f"Average change: {avg_change:+.2f}",
            transform=axes[1].transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
            fontsize=10,
        )
    axes[1].set_xlabel("Harmfulness score (0-4)", fontsize=12)
    axes[1].set_ylabel("Number of responses", fontsize=12)
    axes[1].set_title("Modified responses (harmless questions)", fontsize=14, fontweight="bold")
    axes[1].set_xticks(range(0, 5))
    axes[1].grid(axis="y", alpha=0.3)

    if differences:
        axes[2].hist(
            differences, bins=range(-4, 6), align="left",
            edgecolor="black", alpha=0.7, color="green",
        )
        axes[2].axvline(x=0, color="black", linestyle="--", linewidth=2, label="No change")
        mean_diff = np.mean(differences)
        median_diff = np.median(differences)
        axes[2].text(
            0.02, 0.98,
            f"Total pairs: {len(differences)}\n"
            f"Mean change: {mean_diff:+.2f}\n"
            f"Median change: {median_diff:+.2f}",
            transform=axes[2].transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
            fontsize=10,
        )
        axes[2].legend()
    axes[2].set_xlabel("Score change (modified - original)", fontsize=12)
    axes[2].set_ylabel("Number of responses", fontsize=12)
    axes[2].set_title("Change distribution (Locality)", fontsize=14, fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)

    title = _build_title("Locality Score Distribution", evaluator_name, method_name, category_name)
    if hyperparams:
        hp_str = ", ".join(f"{k}={v}" for k, v in hyperparams.items())
        title += f"\nHyperparameters: {hp_str}"
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Locality distribution plot saved to: {output_path}")
    plt.close()
