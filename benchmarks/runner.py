"""
Generic benchmark runner with original-result caching.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import BenchmarkIO
from .registry import BenchmarkRegistry, get_default_registry
from .settings import get_enabled_benchmark_configs


class BenchmarkRunner:
    """Evaluate enabled benchmarks on clean and edited model states."""

    def __init__(
        self,
        *,
        method_results_dir: Path,
        model_name: str,
        classifier_categories: Sequence[Mapping[str, Any]],
        registry: BenchmarkRegistry | None = None,
        benchmark_io: BenchmarkIO | None = None,
        enabled_benchmark_configs: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        self.method_results_dir = Path(method_results_dir)
        self.model_name = model_name
        self.classifier_categories = list(classifier_categories)
        self.registry = registry or get_default_registry()
        self.io = benchmark_io or BenchmarkIO(self.method_results_dir)

        raw_configs = (
            dict(enabled_benchmark_configs)
            if enabled_benchmark_configs is not None
            else get_enabled_benchmark_configs()
        )

        self._configs: dict[str, dict[str, Any]] = {}
        for benchmark_name, raw_config in raw_configs.items():
            definition = self.registry.get(benchmark_name)
            self._configs[benchmark_name] = definition.normalize_config(raw_config)

        self._original_results: dict[str, dict[str, Any]] = {}

    def has_enabled_benchmarks(self) -> bool:
        return bool(self._configs)

    def enabled_benchmark_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._configs))

    def prepare_original(self, model: Any) -> dict[str, dict[str, Any]]:
        """
        Ensure original benchmark results are available and return their summaries.
        """
        summaries: dict[str, dict[str, Any]] = {}
        for benchmark_name, config in self._configs.items():
            evaluation = self._load_or_compute_original(
                benchmark_name=benchmark_name,
                model=model,
                config=config,
            )
            summaries[benchmark_name] = dict(evaluation["summary"])
        return summaries

    def evaluate_modified(
        self,
        model: Any,
        *,
        run_label: str,
    ) -> dict[str, dict[str, Any]]:
        """
        Evaluate edited model state against all enabled benchmarks.
        """
        blocks: dict[str, dict[str, Any]] = {}
        for benchmark_name, config in self._configs.items():
            definition = self.registry.get(benchmark_name)
            original = self._load_or_compute_original(
                benchmark_name=benchmark_name,
                model=model,
                config=config,
            )
            modified = self._compute_modified(
                benchmark_name=benchmark_name,
                definition=definition,
                model=model,
                config=config,
                run_label=run_label,
            )
            blocks[benchmark_name] = definition.build_result_block(
                original_summary=original["summary"],
                modified_summary=modified["summary"],
                config=config,
                original_details_file=original["details_file"],
                modified_details_file=modified["details_file"],
            )
        return blocks

    def _load_or_compute_original(
        self,
        *,
        benchmark_name: str,
        model: Any,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        if benchmark_name in self._original_results:
            return self._original_results[benchmark_name]

        definition = self.registry.get(benchmark_name)
        cache_path = self.io.original_cache_path(
            benchmark_name,
            model_name=self.model_name,
            config=config,
        )

        if cache_path.exists():
            payload = self.io.load_json(cache_path)
            evaluation = {
                "summary": payload.get("summary", {}),
                "details_file": self.io.to_project_relative(cache_path),
            }
            self._original_results[benchmark_name] = evaluation
            return evaluation

        payload = definition.evaluate_model(
            model,
            config=config,
            classifier_categories=self.classifier_categories,
        )
        cache_payload = {
            "benchmark": benchmark_name,
            "config": config,
            "model_name": self.model_name,
            "summary": payload["summary"],
            "details": payload["details"],
            "saved_at": datetime.now().isoformat(),
        }
        self.io.save_json(cache_path, cache_payload)
        evaluation = {
            "summary": payload["summary"],
            "details_file": self.io.to_project_relative(cache_path),
        }
        self._original_results[benchmark_name] = evaluation
        return evaluation

    def _compute_modified(
        self,
        *,
        benchmark_name: str,
        definition,
        model: Any,
        config: dict[str, Any],
        run_label: str,
    ) -> dict[str, Any]:
        payload = definition.evaluate_model(
            model,
            config=config,
            classifier_categories=self.classifier_categories,
        )
        details_path = self.io.modified_details_path(
            benchmark_name,
            run_label=run_label,
        )
        details_payload = {
            "benchmark": benchmark_name,
            "config": config,
            "model_name": self.model_name,
            "summary": payload["summary"],
            "details": payload["details"],
            "saved_at": datetime.now().isoformat(),
        }
        self.io.save_json(details_path, details_payload)
        return {
            "summary": payload["summary"],
            "details_file": self.io.to_project_relative(details_path),
        }
