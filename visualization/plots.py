"""
Shared plotting functions for harmfulness heatmaps, locality heatmaps,
and score distribution histograms.

This is the single canonical copy -- baselines and evaluation runners
import from here instead of duplicating the code.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Harmfulness heatmap
# ---------------------------------------------------------------------------

def create_harmfulness_heatmap(
    results_data: Dict,
    category_name: str,
    output_dir: Path,
) -> None:
    """Create heatmap for harmfulness scores grouped by hyperparameters.

    ``results_data`` is expected to have a ``"results"`` key mapping
    param_key -> result_dict, where each result_dict contains
    ``"hyperparameters"`` and ``"modified_scores"``.
    """
    heatmap_data = []

    for param_key, result_data in results_data.get("results", {}).items():
        hyperparams = result_data.get("hyperparameters", {})
        modified_scores = result_data.get("modified_scores", [])
        if not modified_scores:
            continue
        mean_modified = np.mean(modified_scores)
        heatmap_data.append({
            "max_weight": hyperparams.get("max_weight"),
            "max_weight_position": hyperparams.get("max_weight_position"),
            "min_weight": hyperparams.get("min_weight"),
            "min_weight_distance": hyperparams.get("min_weight_distance"),
            "mean_modified_score": mean_modified,
            "param_key": param_key,
        })

    if not heatmap_data:
        print("No data for harmfulness heatmap")
        return

    df = pd.DataFrame(heatmap_data)

    unique_min_weights = sorted(df["min_weight"].unique())
    unique_min_distances = sorted(df["min_weight_distance"].unique())

    n_plots = len(unique_min_weights) * len(unique_min_distances)
    n_cols = min(2, len(unique_min_distances))
    n_rows = max(1, (n_plots + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = np.asarray(axes).flatten()

    plot_idx = 0
    for min_w in unique_min_weights:
        for min_dist in unique_min_distances:
            if plot_idx >= len(axes):
                break
            ax = axes[plot_idx]
            subset = df[(df["min_weight"] == min_w) & (df["min_weight_distance"] == min_dist)]

            if len(subset) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
                ax.axis("off")
                plot_idx += 1
                continue

            pivot_data = subset.pivot_table(
                values="mean_modified_score",
                index="max_weight_position",
                columns="max_weight",
                aggfunc="mean",
            )
            if pivot_data.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
                ax.axis("off")
                plot_idx += 1
                continue

            sns.heatmap(
                pivot_data, annot=True, fmt=".1f", cmap="RdYlGn_r",
                vmin=0, vmax=4,
                cbar_kws={"label": "Harmfulness Score (0-4)"},
                ax=ax, linewidths=0.5, linecolor="gray",
            )
            ax.set_title(
                f"min_weight={min_w}, min_weight_distance={min_dist}\n"
                f"(Refusal Level: 0=Garbled, 1=Safe, 4=Harmful)",
                fontsize=11, fontweight="bold",
            )
            ax.set_xlabel("max_weight", fontsize=10)
            ax.set_ylabel("max_weight_position", fontsize=10)
            plot_idx += 1

    for idx in range(plot_idx, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(
        f"Heatmap: Harmfulness Scores by hyperparameters\nCategory: {category_name}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    category_safe = category_name.replace("/", "_").replace(" ", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = output_dir / f"harmfulness_heatmap_{category_safe}_{ts}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved to: {plot_filename}")
    plt.close()


# ---------------------------------------------------------------------------
# Locality heatmap
# ---------------------------------------------------------------------------

def create_locality_heatmap(
    results_data: Dict,
    category_name: str,
    output_dir: Path,
) -> None:
    """Create heatmap for locality scores grouped by hyperparameters."""
    heatmap_data = []

    for param_key, result_data in results_data.get("results", {}).items():
        hyperparams = result_data.get("hyperparameters", {})
        avg_locality_change = result_data.get("average_locality_change")
        if avg_locality_change is None:
            continue
        heatmap_data.append({
            "max_weight": hyperparams.get("max_weight"),
            "max_weight_position": hyperparams.get("max_weight_position"),
            "min_weight": hyperparams.get("min_weight"),
            "min_weight_distance": hyperparams.get("min_weight_distance"),
            "average_locality_change": avg_locality_change,
            "param_key": param_key,
        })

    if not heatmap_data:
        print("No data for locality heatmap")
        return

    df = pd.DataFrame(heatmap_data)

    unique_min_weights = sorted(df["min_weight"].unique())
    unique_min_distances = sorted(df["min_weight_distance"].unique())

    n_plots = len(unique_min_weights) * len(unique_min_distances)
    n_cols = min(2, len(unique_min_distances))
    n_rows = max(1, (n_plots + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = np.asarray(axes).flatten()

    plot_idx = 0
    for min_w in unique_min_weights:
        for min_dist in unique_min_distances:
            if plot_idx >= len(axes):
                break
            ax = axes[plot_idx]
            subset = df[(df["min_weight"] == min_w) & (df["min_weight_distance"] == min_dist)]

            if len(subset) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
                ax.axis("off")
                plot_idx += 1
                continue

            pivot_data = subset.pivot_table(
                values="average_locality_change",
                index="max_weight_position",
                columns="max_weight",
                aggfunc="mean",
            )
            if pivot_data.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
                ax.axis("off")
                plot_idx += 1
                continue

            sns.heatmap(
                pivot_data, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0,
                cbar_kws={"label": "Average Locality Change"},
                ax=ax, linewidths=0.5, linecolor="gray",
            )
            ax.set_title(
                f"min_weight={min_w}, min_weight_distance={min_dist}\n"
                f"(Locality: positive = degradation)",
                fontsize=11, fontweight="bold",
            )
            ax.set_xlabel("max_weight", fontsize=10)
            ax.set_ylabel("max_weight_position", fontsize=10)
            plot_idx += 1

    for idx in range(plot_idx, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(
        f"Heatmap: Locality Scores by hyperparameters\nCategory: {category_name}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    category_safe = category_name.replace("/", "_").replace(" ", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = output_dir / f"locality_heatmap_{category_safe}_{ts}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Locality heatmap saved to: {plot_filename}")
    plt.close()


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def create_distribution_plots(
    original_scores: List[int],
    modified_scores: List[int],
    locality_scores: List[Dict],
    category_name: str,
    harmfulness_output_dir: Path,
    locality_output_dir: Path,
) -> None:
    """Create score distribution histograms (harmfulness + locality)."""
    category_safe = category_name.replace("/", "_").replace(" ", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Harmfulness distribution ---
    if original_scores and modified_scores:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.hist(original_scores, bins=range(0, 6), align="left",
                 edgecolor="black", alpha=0.7, color="blue")
        ax1.set_xlabel("Harmfulness score (0-4)", fontsize=12)
        ax1.set_ylabel("Number of responses", fontsize=12)
        ax1.set_title("Original responses (pre-modification)", fontsize=14, fontweight="bold")
        ax1.set_xticks(range(0, 5))
        ax1.grid(axis="y", alpha=0.3)
        mean_orig = np.mean(original_scores)
        median_orig = np.median(original_scores)
        ax1.text(0.02, 0.98,
                 f"Total scores: {len(original_scores)}\nMean: {mean_orig:.2f}\nMedian: {median_orig:.2f}",
                 transform=ax1.transAxes, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5), fontsize=10)

        ax2.hist(modified_scores, bins=range(0, 6), align="left",
                 edgecolor="black", alpha=0.7, color="red")
        ax2.set_xlabel("Harmfulness score (0-4)", fontsize=12)
        ax2.set_ylabel("Number of responses", fontsize=12)
        ax2.set_title("Modified responses (post-modification)", fontsize=14, fontweight="bold")
        ax2.set_xticks(range(0, 5))
        ax2.grid(axis="y", alpha=0.3)
        mean_mod = np.mean(modified_scores)
        median_mod = np.median(modified_scores)
        ax2.text(0.02, 0.98,
                 f"Total scores: {len(modified_scores)}\nMean: {mean_mod:.2f}\nMedian: {median_mod:.2f}",
                 transform=ax2.transAxes, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5), fontsize=10)

        plt.suptitle(
            f'Harmfulness score distribution for category "{category_name}"',
            fontsize=16, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plot_filename = harmfulness_output_dir / f"harmfulness_distribution_{category_safe}_{ts}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"Harmfulness distribution plot saved to: {plot_filename}")
        plt.close()

    # --- Locality distribution ---
    if locality_scores:
        valid_differences = [s["difference"] for s in locality_scores if s["difference"] is not None]
        original_locality = [s["original_score"] for s in locality_scores]
        modified_locality = [s["modified_score"] for s in locality_scores]

        if valid_differences:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

            ax1.hist(original_locality, bins=range(0, 6), align="left",
                     edgecolor="black", alpha=0.7, color="blue")
            ax1.set_xlabel("Harmfulness score (0-4)", fontsize=12)
            ax1.set_ylabel("Number of responses", fontsize=12)
            ax1.set_title("Original responses (harmless questions)", fontsize=14, fontweight="bold")
            ax1.set_xticks(range(0, 5))
            ax1.grid(axis="y", alpha=0.3)
            ax1.text(0.02, 0.98,
                     f"Total scores: {len(original_locality)}\nMean: {np.mean(original_locality):.2f}\n"
                     f"Median: {np.median(original_locality):.2f}",
                     transform=ax1.transAxes, verticalalignment="top",
                     bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5), fontsize=10)

            ax2.hist(modified_locality, bins=range(0, 6), align="left",
                     edgecolor="black", alpha=0.7, color="red")
            ax2.set_xlabel("Harmfulness score (0-4)", fontsize=12)
            ax2.set_ylabel("Number of responses", fontsize=12)
            ax2.set_title("Modified responses (harmless questions)", fontsize=14, fontweight="bold")
            ax2.set_xticks(range(0, 5))
            ax2.grid(axis="y", alpha=0.3)
            avg_change = np.mean(valid_differences)
            ax2.text(0.02, 0.98,
                     f"Total scores: {len(modified_locality)}\nMean: {np.mean(modified_locality):.2f}\n"
                     f"Median: {np.median(modified_locality):.2f}\nAverage change: {avg_change:+.2f}",
                     transform=ax2.transAxes, verticalalignment="top",
                     bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5), fontsize=10)

            ax3.hist(valid_differences, bins=range(-4, 6), align="left",
                     edgecolor="black", alpha=0.7, color="green")
            ax3.set_xlabel("Score change (modified - original)", fontsize=12)
            ax3.set_ylabel("Number of responses", fontsize=12)
            ax3.set_title("Change distribution (Locality)", fontsize=14, fontweight="bold")
            ax3.axvline(x=0, color="black", linestyle="--", linewidth=2, label="No change")
            ax3.grid(axis="y", alpha=0.3)
            ax3.legend()
            ax3.text(0.02, 0.98,
                     f"Total pairs: {len(valid_differences)}\nMean change: {np.mean(valid_differences):+.2f}\n"
                     f"Median change: {np.median(valid_differences):+.2f}",
                     transform=ax3.transAxes, verticalalignment="top",
                     bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5), fontsize=10)

            plt.suptitle(
                f'Locality score distribution for category "{category_name}"',
                fontsize=16, fontweight="bold", y=1.02,
            )
            plt.tight_layout()
            plot_filename = locality_output_dir / f"locality_distribution_{category_safe}_{ts}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            print(f"Locality distribution plot saved to: {plot_filename}")
            plt.close()
