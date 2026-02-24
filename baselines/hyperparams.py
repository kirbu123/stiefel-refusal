"""
Hyperparameters for grid-search abliteration experiments.

Each value is a list of floats to iterate over. Override via environment
variables using comma-separated values, e.g.:
    export GRID_MAX_WEIGHT="2.0,2.5,3.0"
"""

import os


def _parse_float_list(env_var: str, default: list[float]) -> list[float]:
    val = os.getenv(env_var)
    if val is None:
        return default
    return [float(x.strip()) for x in val.split(",")]


HYPERPARAMS = {
    "max_weight": _parse_float_list("GRID_MAX_WEIGHT", [2.5, 3.0]),
    "max_weight_position": _parse_float_list("GRID_MAX_WEIGHT_POSITION", [0.7]),
    "min_weight": _parse_float_list("GRID_MIN_WEIGHT", [0.0, 1.0]),
    "min_weight_distance": _parse_float_list("GRID_MIN_WEIGHT_DISTANCE", [0.3]),
}
