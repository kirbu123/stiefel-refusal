"""
Runtime-only configuration helpers for graph_grpo.
"""


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
