"""
Shared helpers for binary jailbreak-style benchmark summaries.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_bucket_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(records)
    jailbroken_count = sum(1 for record in records if record.get("jailbroken"))
    scored_values = [
        int(record["score"])
        for record in records
        if record.get("score") is not None
    ]
    attack_success_rate = (jailbroken_count / total) if total else 0.0
    mean_score = (
        sum(scored_values) / len(scored_values)
        if scored_values else None
    )
    return {
        "attack_success_rate": attack_success_rate,
        "n_prompts": total,
        "jailbroken_count": jailbroken_count,
        "mean_score": mean_score,
        "n_scored": len(scored_values),
    }


def summarize_binary_records(
    records: Sequence[Mapping[str, Any]],
    *,
    bucket_fields: Mapping[str, str],
) -> dict[str, Any]:
    summary = build_bucket_summary(records)

    for bucket_key, record_field in bucket_fields.items():
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            bucket_value = str(record.get(record_field, "")).strip()
            grouped.setdefault(bucket_value, []).append(record)

        summary[bucket_key] = {
            key: build_bucket_summary(bucket)
            for key, bucket in sorted(grouped.items())
        }

    return summary


def strip_bucket_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "attack_success_rate": summary.get("attack_success_rate", 0.0),
        "n_prompts": summary.get("n_prompts", 0),
        "jailbroken_count": summary.get("jailbroken_count", 0),
        "mean_score": summary.get("mean_score"),
        "n_scored": summary.get("n_scored", 0),
    }


def build_bucket_comparison(
    original_buckets: Mapping[str, Mapping[str, Any]],
    modified_buckets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    keys = sorted(set(original_buckets) | set(modified_buckets))
    comparison: dict[str, Any] = {}
    for key in keys:
        original = strip_bucket_summary(original_buckets.get(key, {}))
        modified = strip_bucket_summary(modified_buckets.get(key, {}))
        comparison[key] = {
            "original": original,
            "modified": modified,
            "delta_attack_success_rate": (
                float(modified["attack_success_rate"])
                - float(original["attack_success_rate"])
            ),
        }
    return comparison


def delta_attack_success_rate(
    original_summary: Mapping[str, Any],
    modified_summary: Mapping[str, Any],
) -> float:
    original_rate = float(original_summary.get("attack_success_rate", 0.0))
    modified_rate = float(modified_summary.get("attack_success_rate", 0.0))
    return modified_rate - original_rate


def build_binary_result_block(
    *,
    original_summary: Mapping[str, Any],
    modified_summary: Mapping[str, Any],
    config: Mapping[str, Any],
    original_details_file: str,
    modified_details_file: str,
    bucket_keys: Sequence[str],
) -> dict[str, Any]:
    block = {
        "original": strip_bucket_summary(original_summary),
        "modified": strip_bucket_summary(modified_summary),
        "delta_attack_success_rate": delta_attack_success_rate(
            original_summary,
            modified_summary,
        ),
        "config": dict(config),
        "details_file": {
            "original": original_details_file,
            "modified": modified_details_file,
        },
    }

    for bucket_key in bucket_keys:
        block[bucket_key] = build_bucket_comparison(
            original_summary.get(bucket_key, {}),
            modified_summary.get(bucket_key, {}),
        )

    return block
