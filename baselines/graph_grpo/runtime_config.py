"""
Runtime-only configuration helpers for graph_grpo.
"""


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
