"""
Runtime-only configuration helpers for graph_grpo.
"""


def resolve_graph_grpo_optimizer_method(env_value: str | None) -> str:
    """Default graph_grpo to the existing GRPO trainer unless overridden."""
    if env_value is None:
        return "grpo"

    normalized = env_value.strip()
    if not normalized:
        return "grpo"
    return normalized


def validate_graph_grpo_optimizer_method(method: str) -> None:
    """Validate graph_grpo optimizer selection."""
    if method not in {"grpo", "optuna"}:
        raise ValueError(
            f"graph_grpo only supports OPTIMIZER_METHOD in {{'grpo', 'optuna'}}, got '{method}'."
        )


def validate_graph_grpo_optimizer_compatibility(method: str, weights_mode: str) -> None:
    """Ensure the selected optimizer supports the requested weight parameterization."""
    if method == "optuna" and weights_mode != "scalar":
        raise ValueError(
            "graph_grpo currently supports OPTIMIZER_METHOD='optuna' only with "
            "WEIGHTS_MODE='scalar'."
        )


def resolve_graph_grpo_weights_mode(env_value: str | None) -> str:
    """Default graph_grpo to scalar-per-direction weights when env is unset."""
    if env_value is None:
        return "scalar"

    normalized = env_value.strip()
    if not normalized:
        return "scalar"
    return normalized


def validate_graph_grpo_weights_mode(mode: str) -> None:
    """Validate graph_grpo weight parameterization."""
    if mode not in {"scalar", "dense"}:
        raise ValueError(
            f"graph_grpo only supports WEIGHTS_MODE in {{'scalar', 'dense'}}, got '{mode}'."
        )


def resolve_graph_grpo_debug_question_count(
    env_value: str | None,
    default: int = 4,
) -> int:
    """Use a small but non-degenerate question count for debug smoke runs."""
    if env_value is None:
        return default

    normalized = env_value.strip()
    if not normalized:
        return default

    value = int(normalized)
    if value < 1:
        raise ValueError(f"DEBUG_N_QUESTIONS must be >= 1, got {value}")
    return value


def resolve_graph_grpo_debug_noise_scale(
    base_noise_scale: float,
    env_value: str | None,
    default: float = 0.02,
) -> float:
    """Keep debug rollout noise small enough for a useful local training signal."""
    if env_value is None or not env_value.strip():
        return min(base_noise_scale, default)

    value = float(env_value.strip())
    if value < 0:
        raise ValueError(f"DEBUG_NOISE_SCALE must be >= 0, got {value}")
    return value


def resolve_graph_grpo_weights_init_type(env_value: str | None) -> str:
    """Default graph_grpo to a nonzero initialization when env is unset."""
    if env_value is None:
        return "average"

    normalized = env_value.strip()
    if not normalized:
        return "average"
    return normalized


def validate_graph_grpo_weights_init_type(init_type: str) -> None:
    """Reject zero init because the current differentiable path cannot leave it."""
    if init_type == "zero":
        raise ValueError(
            "graph_grpo does not support WEIGHTS_INIT_TYPE='zero': "
            "the current differentiable intervention path has a dead start at zero, "
            "so training cannot move off that point. Use 'average' (default) or 'topic'."
        )


def resolve_graph_grpo_optuna_n_trials(
    env_value: str | None,
    default: int = 50,
) -> int:
    """Resolve the Optuna trial budget."""
    if env_value is None or not env_value.strip():
        return default

    value = int(env_value.strip())
    if value < 1:
        raise ValueError(f"OPTUNA_N_TRIALS must be >= 1, got {value}")
    return value


def resolve_graph_grpo_optuna_sampler_seed(
    env_value: str | None,
    default: int = 42,
) -> int:
    """Resolve the Optuna sampler seed."""
    if env_value is None or not env_value.strip():
        return default
    return int(env_value.strip())


def resolve_graph_grpo_optuna_sampler(
    env_value: str | None,
    default: str = "tpe",
) -> str:
    """Resolve the Optuna sampler name."""
    if env_value is None:
        return default

    normalized = env_value.strip()
    if not normalized:
        return default
    return normalized


def validate_graph_grpo_optuna_sampler(sampler: str) -> None:
    """Validate the supported Optuna sampler choices."""
    if sampler not in {"tpe", "random", "gp", "cmaes", "qmc"}:
        raise ValueError(
            "graph_grpo only supports OPTUNA_SAMPLER in "
            "{'tpe', 'random', 'gp', 'cmaes', 'qmc'}, "
            f"got '{sampler}'."
        )


def resolve_graph_grpo_optuna_weight_min(
    env_value: str | None,
    default: float = -2.0,
) -> float:
    """Resolve the lower search bound for scalar Optuna coefficients."""
    if env_value is None or not env_value.strip():
        return default
    return float(env_value.strip())


def resolve_graph_grpo_optuna_weight_max(
    env_value: str | None,
    default: float = 2.0,
) -> float:
    """Resolve the upper search bound for scalar Optuna coefficients."""
    if env_value is None or not env_value.strip():
        return default
    return float(env_value.strip())


def validate_graph_grpo_optuna_weight_range(weight_min: float, weight_max: float) -> None:
    """Reject inverted scalar Optuna search ranges."""
    if weight_min > weight_max:
        raise ValueError(
            f"OPTUNA weight range must satisfy OPTUNA_WEIGHT_MIN <= OPTUNA_WEIGHT_MAX, "
            f"got {weight_min} > {weight_max}."
        )
