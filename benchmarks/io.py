"""
I/O helpers for benchmark caching and details persistence.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _resolve_project_root() -> Path:
    try:
        from config import PROJECT_ROOT

        return Path(PROJECT_ROOT)
    except Exception:
        return Path.cwd()


class BenchmarkIO:
    """Filesystem helpers for benchmark outputs."""

    def __init__(
        self,
        method_results_dir: Path,
        *,
        project_root: Path | None = None,
    ):
        self.method_results_dir = Path(method_results_dir)
        self.project_root = Path(project_root) if project_root is not None else _resolve_project_root()

    def benchmark_dir(self, benchmark_name: str) -> Path:
        return self.method_results_dir / "benchmarks" / benchmark_name

    def cache_dir(self, benchmark_name: str) -> Path:
        return self.benchmark_dir(benchmark_name) / "cache"

    def details_dir(self, benchmark_name: str) -> Path:
        return self.benchmark_dir(benchmark_name) / "details"

    def original_cache_path(
        self,
        benchmark_name: str,
        *,
        model_name: str,
        config: dict[str, Any],
    ) -> Path:
        model_hash = self._hash_string(model_name)
        config_hash = self._hash_json(config)
        return self.cache_dir(benchmark_name) / f"original_{model_hash}_{config_hash}.json"

    def modified_details_path(
        self,
        benchmark_name: str,
        *,
        run_label: str,
    ) -> Path:
        safe_label = self._sanitize_filename(run_label)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.details_dir(benchmark_name) / f"{safe_label}_{timestamp}.json"

    def load_json(self, path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def to_project_relative(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.project_root.resolve()))
        except ValueError:
            return str(path)

    def _hash_string(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]

    def _hash_json(self, value: Any) -> str:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def _sanitize_filename(self, value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized or "run"
