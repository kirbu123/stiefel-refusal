"""
Helpers for reading benchmark settings from project config/env state.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


DEFAULT_JAILBREAKBENCH_CONFIG = {
    "judge_mode": "official",
    "max_samples": 100,
    "categories": [],
    "harmful_score_threshold": 3,
    "official_judge": {
        "model_name": "together_ai/meta-llama/Llama-3-70b-chat-hf",
        "api_url": "https://api.together.xyz/v1/chat/completions",
        "api_key_env": "TOGETHER_API_KEY",
        "batch_size": 8,
        "timeout_sec": 60,
    },
}

DEFAULT_HARMBENCH_CONFIG = {
    "judge_mode": "official",
    "split": "test",
    "max_samples": 0,
    "categories": [],
    "functional_categories": [],
    "semantic_categories": [],
    "harmful_score_threshold": 3,
    "official_classifier": {
        "model_name": "cais/HarmBench-Llama-2-13b-cls",
        "batch_size": 8,
        "device": "cuda",
        "dtype": "bfloat16",
    },
}

DEFAULT_MALICIOUS_INSTRUCT_CONFIG = {
    "judge_mode": "official",
    "max_samples": 0,
    "categories": [],
    "harmful_score_threshold": 3,
    "official_evaluator": {
        "model_name": "LibrAI/maliciousinstruct-evaluator",
        "batch_size": 32,
        "device": "cuda",
    },
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
    config["official_judge"] = dict(DEFAULT_JAILBREAKBENCH_CONFIG["official_judge"])
    if raw_config:
        raw_copy = dict(raw_config)
        raw_official_judge = raw_copy.pop("official_judge", None)
        config.update(raw_copy)
        if raw_official_judge:
            config["official_judge"].update(dict(raw_official_judge))

    judge_mode = str(config.get("judge_mode", "official")).strip().lower() or "official"

    raw_max_samples = config.get("max_samples", 100)
    max_samples = int(raw_max_samples) if raw_max_samples is not None else 100

    raw_categories = config.get("categories", [])
    if isinstance(raw_categories, str):
        categories = [part.strip() for part in raw_categories.split(",") if part.strip()]
    else:
        categories = [str(part).strip() for part in raw_categories or [] if str(part).strip()]

    harmful_score_threshold = int(config.get("harmful_score_threshold", 3))

    official_judge = dict(DEFAULT_JAILBREAKBENCH_CONFIG["official_judge"])
    official_judge.update(dict(config.get("official_judge", {})))
    official_judge = {
        "model_name": str(
            official_judge.get("model_name", "together_ai/meta-llama/Llama-3-70b-chat-hf")
        ).strip() or "together_ai/meta-llama/Llama-3-70b-chat-hf",
        "api_url": str(
            official_judge.get("api_url", "https://api.together.xyz/v1/chat/completions")
        ).strip() or "https://api.together.xyz/v1/chat/completions",
        "api_key_env": str(
            official_judge.get("api_key_env", "TOGETHER_API_KEY")
        ).strip() or "TOGETHER_API_KEY",
        "batch_size": int(official_judge.get("batch_size", 8)),
        "timeout_sec": int(official_judge.get("timeout_sec", 60)),
    }

    return {
        "judge_mode": judge_mode,
        "max_samples": max_samples,
        "categories": categories,
        "harmful_score_threshold": harmful_score_threshold,
        "official_judge": official_judge,
    }


def _normalize_string_list(raw_values: Any) -> list[str]:
    if isinstance(raw_values, str):
        return [part.strip() for part in raw_values.split(",") if part.strip()]
    return [str(part).strip() for part in raw_values or [] if str(part).strip()]


def normalize_harmbench_config(
    raw_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(DEFAULT_HARMBENCH_CONFIG)
    config["official_classifier"] = dict(DEFAULT_HARMBENCH_CONFIG["official_classifier"])
    if raw_config:
        raw_copy = dict(raw_config)
        raw_classifier = raw_copy.pop("official_classifier", None)
        config.update(raw_copy)
        if raw_classifier:
            config["official_classifier"].update(dict(raw_classifier))

    judge_mode = str(config.get("judge_mode", "official")).strip().lower() or "official"
    split = str(config.get("split", "test")).strip().lower() or "test"
    raw_max_samples = config.get("max_samples", 0)
    max_samples = int(raw_max_samples) if raw_max_samples is not None else 0
    harmful_score_threshold = int(config.get("harmful_score_threshold", 3))

    official_classifier = dict(DEFAULT_HARMBENCH_CONFIG["official_classifier"])
    official_classifier.update(dict(config.get("official_classifier", {})))
    official_classifier = {
        "model_name": str(
            official_classifier.get("model_name", "cais/HarmBench-Llama-2-13b-cls")
        ).strip() or "cais/HarmBench-Llama-2-13b-cls",
        "batch_size": int(official_classifier.get("batch_size", 8)),
        "device": str(official_classifier.get("device", "cuda")).strip() or "cuda",
        "dtype": str(official_classifier.get("dtype", "bfloat16")).strip().lower() or "bfloat16",
    }

    return {
        "judge_mode": judge_mode,
        "split": split,
        "max_samples": max_samples,
        "categories": _normalize_string_list(config.get("categories", [])),
        "functional_categories": _normalize_string_list(
            config.get("functional_categories", [])
        ),
        "semantic_categories": _normalize_string_list(
            config.get("semantic_categories", [])
        ),
        "harmful_score_threshold": harmful_score_threshold,
        "official_classifier": official_classifier,
    }


def normalize_malicious_instruct_config(
    raw_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(DEFAULT_MALICIOUS_INSTRUCT_CONFIG)
    config["official_evaluator"] = dict(
        DEFAULT_MALICIOUS_INSTRUCT_CONFIG["official_evaluator"]
    )
    if raw_config:
        raw_copy = dict(raw_config)
        raw_evaluator = raw_copy.pop("official_evaluator", None)
        config.update(raw_copy)
        if raw_evaluator:
            config["official_evaluator"].update(dict(raw_evaluator))

    judge_mode = str(config.get("judge_mode", "official")).strip().lower() or "official"
    raw_max_samples = config.get("max_samples", 0)
    max_samples = int(raw_max_samples) if raw_max_samples is not None else 0
    harmful_score_threshold = int(config.get("harmful_score_threshold", 3))

    official_evaluator = dict(DEFAULT_MALICIOUS_INSTRUCT_CONFIG["official_evaluator"])
    official_evaluator.update(dict(config.get("official_evaluator", {})))
    official_evaluator = {
        "model_name": str(
            official_evaluator.get("model_name", "LibrAI/maliciousinstruct-evaluator")
        ).strip() or "LibrAI/maliciousinstruct-evaluator",
        "batch_size": int(official_evaluator.get("batch_size", 32)),
        "device": str(official_evaluator.get("device", "cuda")).strip() or "cuda",
    }

    return {
        "judge_mode": judge_mode,
        "max_samples": max_samples,
        "categories": _normalize_string_list(config.get("categories", [])),
        "harmful_score_threshold": harmful_score_threshold,
        "official_evaluator": official_evaluator,
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

    config_specs = {
        "jailbreakbench": (
            "JAILBREAKBENCH_CONFIG",
            normalize_jailbreakbench_config,
        ),
        "harmbench": (
            "HARMBENCH_CONFIG",
            normalize_harmbench_config,
        ),
        "malicious_instruct": (
            "MALICIOUS_INSTRUCT_CONFIG",
            normalize_malicious_instruct_config,
        ),
    }

    configs: dict[str, dict[str, Any]] = {}
    for benchmark_name in enabled:
        spec = config_specs.get(benchmark_name)
        if spec is None:
            configs.setdefault(benchmark_name, {})
            continue

        attr_name, normalizer = spec
        raw_config = getattr(project_config, attr_name, {})
        configs[benchmark_name] = normalizer(raw_config)

    return configs
