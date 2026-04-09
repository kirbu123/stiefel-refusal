"""
Helpers for reading benchmark settings from project config/env state.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


DEFAULT_JAILBREAKBENCH_CONFIG = {
    "judge_mode": "project",
    "max_samples": 100,
    "categories": [],
    "harmful_score_threshold": 3,
}


def normalize_enabled_benchmarks(raw_value: Any) -> tuple[str, ...]:
    """Normalize BENCHMARKS_ENABLED from env/config into a stable tuple."""
    if raw_value is None:
        return ()

    if isinstance(raw_value, str):
        items = [part.strip() for part in raw_value.split(",")]
    elif isinstance(raw_value, Iterable):
        items = [str(part).strip() for part in raw_value]
    else:
        items = [str(raw_value).strip()]

    normalized: list[str] = []
    for item in items:
        if not item:
            continue
        key = item.lower()
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized)


def normalize_jailbreakbench_config(
    raw_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize benchmark-specific settings into serializable values."""
    config = dict(DEFAULT_JAILBREAKBENCH_CONFIG)
    if raw_config:
        config.update(dict(raw_config))

    judge_mode = str(config.get("judge_mode", "project")).strip().lower() or "project"

    raw_max_samples = config.get("max_samples", 100)
    max_samples = int(raw_max_samples) if raw_max_samples is not None else 100

    raw_categories = config.get("categories", [])
    if isinstance(raw_categories, str):
        categories = [part.strip() for part in raw_categories.split(",") if part.strip()]
    else:
        categories = [str(part).strip() for part in raw_categories or [] if str(part).strip()]

    harmful_score_threshold = int(config.get("harmful_score_threshold", 3))

    return {
        "judge_mode": judge_mode,
        "max_samples": max_samples,
        "categories": categories,
        "harmful_score_threshold": harmful_score_threshold,
    }


def get_enabled_benchmark_configs() -> dict[str, dict[str, Any]]:
    """
    Resolve enabled benchmark configs from the project's config module.

    Defaults to no enabled benchmarks when the config module does not expose
    benchmark-related attributes (useful for tests that stub a minimal config).
    """
    import config as project_config

    enabled = normalize_enabled_benchmarks(
        getattr(project_config, "BENCHMARKS_ENABLED", ())
    )
    if not enabled:
        return {}

    configs: dict[str, dict[str, Any]] = {}

    if "jailbreakbench" in enabled:
        raw_jbb_config = getattr(project_config, "JAILBREAKBENCH_CONFIG", {})
        configs["jailbreakbench"] = normalize_jailbreakbench_config(raw_jbb_config)

    for benchmark_name in enabled:
        configs.setdefault(benchmark_name, {})

    return configs
