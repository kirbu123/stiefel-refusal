"""
JailbreakBench harmful-split benchmark definition.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_utils import (
    SUPPORTED_JAILBREAKBENCH_CATEGORIES,
    extract_response_after_think,
)
from evaluate.judges import evaluate_harmfulness

from .base import BenchmarkDefinition
from .settings import normalize_jailbreakbench_config


def _resolve_dataset_path() -> Path:
    from config import PROJECT_ROOT

    return Path(PROJECT_ROOT) / "dataset" / "raw" / "jailbreakbench.csv"


class BaseJailbreakBenchJudge(ABC):
    """Judge adapter for benchmark-specific success criteria."""

    @abstractmethod
    def evaluate(self, prompt: str, response: str) -> dict[str, Any]:
        """
        Return benchmark verdict and raw evaluation.

        Returns:
            Dict with keys:
              - score
              - jailbroken
              - evaluation
        """


class ProjectJailbreakJudge(BaseJailbreakBenchJudge):
    """Bridge from project harmfulness evaluators to JailbreakBench verdicts."""

    def __init__(
        self,
        *,
        harmful_score_threshold: int,
        classifier_categories: Sequence[Mapping[str, Any]],
    ):
        self.harmful_score_threshold = int(harmful_score_threshold)
        self.classifier_categories = list(classifier_categories)

    def evaluate(self, prompt: str, response: str) -> dict[str, Any]:
        raw_evaluation = evaluate_harmfulness(
            prompt,
            response,
            classifier_categories=self.classifier_categories,
        )
        raw_score = raw_evaluation.get("score")
        try:
            score = int(raw_score) if raw_score is not None else None
        except Exception:
            score = None

        jailbroken = score is not None and score >= self.harmful_score_threshold
        return {
            "score": score,
            "jailbroken": jailbroken,
            "evaluation": raw_evaluation,
        }


class JailbreakBenchBenchmark(BenchmarkDefinition):
    """Harmful-only JailbreakBench benchmark using the local CSV dataset."""

    name = "jailbreakbench"
    _SUPPORTED_JUDGE_MODES = ("project",)

    def normalize_config(
        self,
        raw_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = normalize_jailbreakbench_config(raw_config)

        judge_mode = config["judge_mode"]
        if judge_mode not in self._SUPPORTED_JUDGE_MODES:
            valid_values = ", ".join(self._SUPPORTED_JUDGE_MODES)
            raise ValueError(
                f"Unsupported JailbreakBench judge_mode '{judge_mode}'. "
                f"Valid values: {valid_values}"
            )

        max_samples = int(config["max_samples"])
        if max_samples < 0:
            raise ValueError("JAILBREAKBENCH max_samples must be >= 0")

        harmful_score_threshold = int(config["harmful_score_threshold"])
        if harmful_score_threshold < 0 or harmful_score_threshold > 4:
            raise ValueError(
                "JAILBREAKBENCH harmful_score_threshold must be between 0 and 4"
            )

        categories = list(config["categories"])
        invalid_categories = [
            category for category in categories
            if category not in SUPPORTED_JAILBREAKBENCH_CATEGORIES
        ]
        if invalid_categories:
            valid_values = ", ".join(SUPPORTED_JAILBREAKBENCH_CATEGORIES)
            raise ValueError(
                f"Unsupported JailbreakBench categories: {invalid_categories}. "
                f"Valid values: {valid_values}"
            )

        return {
            "judge_mode": judge_mode,
            "max_samples": max_samples,
            "categories": categories,
            "harmful_score_threshold": harmful_score_threshold,
        }

    def load_samples(self, config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        normalized = self.normalize_config(config)
        dataset_path = _resolve_dataset_path()

        allowed_categories = set(normalized["categories"])
        max_samples = normalized["max_samples"]

        samples: list[dict[str, Any]] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for fallback_index, row in enumerate(reader):
                category = (row.get("Category") or "").strip()
                if allowed_categories and category not in allowed_categories:
                    continue

                goal = (row.get("Goal") or "").strip()
                if not goal:
                    continue

                raw_index = (
                    row.get("Index")
                    or row.get("index")
                    or row.get("")
                    or fallback_index
                )
                try:
                    sample_index = int(raw_index)
                except Exception:
                    sample_index = fallback_index

                samples.append(
                    {
                        "id": sample_index,
                        "goal": goal,
                        "target": (row.get("Target") or "").strip(),
                        "behavior": (row.get("Behavior") or "").strip(),
                        "category": category,
                        "source": (row.get("Source") or "").strip(),
                    }
                )

                if max_samples > 0 and len(samples) >= max_samples:
                    break

        return samples

    def evaluate_model(
        self,
        model: Any,
        *,
        config: dict[str, Any],
        classifier_categories: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        normalized = self.normalize_config(config)
        samples = self.load_samples(normalized)
        prompts = [sample["goal"] for sample in samples]

        responses_raw = model.get_responses_batched(prompts) if prompts else []
        responses = [
            extracted if extracted else raw.strip()
            for raw in responses_raw
            for extracted in [extract_response_after_think(raw)]
        ]

        judge = self._build_judge(
            judge_mode=normalized["judge_mode"],
            harmful_score_threshold=normalized["harmful_score_threshold"],
            classifier_categories=classifier_categories,
        )

        records: list[dict[str, Any]] = []
        for sample, response in zip(samples, responses):
            judged = judge.evaluate(sample["goal"], response)
            records.append(
                {
                    **sample,
                    "response": response,
                    "score": judged["score"],
                    "jailbroken": judged["jailbroken"],
                    "evaluation": judged["evaluation"],
                }
            )

        summary = self._summarize_records(records)
        details = {
            "benchmark": self.name,
            "config": normalized,
            "summary": summary,
            "samples": records,
        }
        return {
            "summary": summary,
            "details": details,
        }

    def build_result_block(
        self,
        *,
        original_summary: dict[str, Any],
        modified_summary: dict[str, Any],
        config: dict[str, Any],
        original_details_file: str,
        modified_details_file: str,
    ) -> dict[str, Any]:
        return {
            "original": self._summary_without_buckets(original_summary),
            "modified": self._summary_without_buckets(modified_summary),
            "delta_attack_success_rate": self._delta_attack_success_rate(
                original_summary,
                modified_summary,
            ),
            "config": self.normalize_config(config),
            "details_file": {
                "original": original_details_file,
                "modified": modified_details_file,
            },
            "by_category": self._build_bucket_comparison(
                original_summary.get("by_category", {}),
                modified_summary.get("by_category", {}),
            ),
            "by_source": self._build_bucket_comparison(
                original_summary.get("by_source", {}),
                modified_summary.get("by_source", {}),
            ),
        }

    def _build_judge(
        self,
        *,
        judge_mode: str,
        harmful_score_threshold: int,
        classifier_categories: Sequence[Mapping[str, Any]],
    ) -> BaseJailbreakBenchJudge:
        if judge_mode == "project":
            return ProjectJailbreakJudge(
                harmful_score_threshold=harmful_score_threshold,
                classifier_categories=classifier_categories,
            )

        raise ValueError(f"Unsupported JailbreakBench judge_mode '{judge_mode}'")

    def _summarize_records(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        summary = self._bucket_summary(records)

        by_category: dict[str, list[Mapping[str, Any]]] = {}
        by_source: dict[str, list[Mapping[str, Any]]] = {}

        for record in records:
            category = str(record.get("category", "")).strip()
            source = str(record.get("source", "")).strip()
            by_category.setdefault(category, []).append(record)
            by_source.setdefault(source, []).append(record)

        summary["by_category"] = {
            key: self._bucket_summary(bucket)
            for key, bucket in sorted(by_category.items())
        }
        summary["by_source"] = {
            key: self._bucket_summary(bucket)
            for key, bucket in sorted(by_source.items())
        }
        return summary

    def _bucket_summary(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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

    def _summary_without_buckets(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "attack_success_rate": summary.get("attack_success_rate", 0.0),
            "n_prompts": summary.get("n_prompts", 0),
            "jailbroken_count": summary.get("jailbroken_count", 0),
            "mean_score": summary.get("mean_score"),
            "n_scored": summary.get("n_scored", 0),
        }

    def _build_bucket_comparison(
        self,
        original_buckets: Mapping[str, Mapping[str, Any]],
        modified_buckets: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        keys = sorted(set(original_buckets) | set(modified_buckets))
        comparison: dict[str, Any] = {}
        for key in keys:
            original = self._summary_without_buckets(original_buckets.get(key, {}))
            modified = self._summary_without_buckets(modified_buckets.get(key, {}))
            comparison[key] = {
                "original": original,
                "modified": modified,
                "delta_attack_success_rate": (
                    modified["attack_success_rate"] - original["attack_success_rate"]
                ),
            }
        return comparison

    def _delta_attack_success_rate(
        self,
        original_summary: Mapping[str, Any],
        modified_summary: Mapping[str, Any],
    ) -> float:
        original_rate = float(original_summary.get("attack_success_rate", 0.0))
        modified_rate = float(modified_summary.get("attack_success_rate", 0.0))
        return modified_rate - original_rate
