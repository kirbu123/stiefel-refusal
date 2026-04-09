"""
Thin integration helpers for baseline entrypoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .runner import BenchmarkRunner


def build_benchmark_runner(
    *,
    method_results_dir: Path,
    model_name: str,
    classifier_categories: Sequence[Mapping[str, Any]],
) -> BenchmarkRunner | None:
    runner = BenchmarkRunner(
        method_results_dir=method_results_dir,
        model_name=model_name,
        classifier_categories=classifier_categories,
    )
    if not runner.has_enabled_benchmarks():
        return None
    return runner


def get_benchmark_attack_success_rate(
    benchmark_results: Mapping[str, Any] | None,
    benchmark_name: str,
) -> float | None:
    if not benchmark_results:
        return None
    benchmark_block = benchmark_results.get(benchmark_name, {})
    modified = benchmark_block.get("modified", {})
    attack_success_rate = modified.get("attack_success_rate")
    if attack_success_rate is None:
        return None
    return float(attack_success_rate)
