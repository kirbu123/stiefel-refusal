"""Visualization helpers for RDO experiment analysis."""

try:
    from .plots import (
        plot_harmfulness_heatmap,
        plot_locality_heatmap,
        plot_harmfulness_distribution,
        plot_locality_distribution,
        generate_plot_filename,
    )
except ModuleNotFoundError:
    # Optional legacy plotting helpers; research.py does not depend on them.
    pass
