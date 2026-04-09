"""
Registry of available benchmark definitions.
"""

from __future__ import annotations

from functools import lru_cache

from .base import BenchmarkDefinition


class BenchmarkRegistry:
    """Name -> benchmark definition registry."""

    def __init__(self):
        self._definitions: dict[str, BenchmarkDefinition] = {}

    def register(self, definition: BenchmarkDefinition) -> None:
        name = definition.name.strip().lower()
        if not name:
            raise ValueError("Benchmark definition name must be non-empty.")
        self._definitions[name] = definition

    def get(self, benchmark_name: str) -> BenchmarkDefinition:
        key = benchmark_name.strip().lower()
        if key not in self._definitions:
            supported = ", ".join(sorted(self._definitions))
            raise ValueError(
                f"Unknown benchmark '{benchmark_name}'. "
                f"Supported benchmarks: {supported}"
            )
        return self._definitions[key]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))


@lru_cache(maxsize=1)
def get_default_registry() -> BenchmarkRegistry:
    from .harmbench import HarmBenchBenchmark
    from .jailbreakbench import JailbreakBenchBenchmark

    registry = BenchmarkRegistry()
    registry.register(HarmBenchBenchmark())
    registry.register(JailbreakBenchBenchmark())
    return registry
