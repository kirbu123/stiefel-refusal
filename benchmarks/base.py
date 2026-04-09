"""
Abstract interfaces for model benchmarks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence


class BenchmarkDefinition(ABC):
    """Interface that every benchmark implementation must satisfy."""

    name: str

    @abstractmethod
    def normalize_config(
        self,
        raw_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Normalize and validate benchmark-specific config."""

    @abstractmethod
    def evaluate_model(
        self,
        model: Any,
        *,
        config: dict[str, Any],
        classifier_categories: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """
        Evaluate the current model state on the benchmark.

        Returns:
            Dict with keys:
              - summary: benchmark summary used in top-level results
              - details: full per-sample payload persisted to JSON
        """

    @abstractmethod
    def build_result_block(
        self,
        *,
        original_summary: dict[str, Any],
        modified_summary: dict[str, Any],
        config: dict[str, Any],
        original_details_file: str,
        modified_details_file: str,
    ) -> dict[str, Any]:
        """Build the final benchmark block stored in answers JSON."""
