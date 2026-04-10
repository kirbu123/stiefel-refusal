"""
Malicious Instruct benchmark definition with official and project judges.
"""

from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_utils import (
    SUPPORTED_JAILBREAKBENCH_CATEGORIES,
    extract_response_after_think,
)
from evaluate.judges import evaluate_harmfulness

from .base import BenchmarkDefinition
from .binary_metrics import (
    build_binary_result_block,
    summarize_binary_records,
)
from .settings import normalize_malicious_instruct_config


OFFICIAL_MALICIOUS_INSTRUCT_EVALUATOR_MODEL = "LibrAI/maliciousinstruct-evaluator"


def _resolve_dataset_path() -> Path:
    from config import PROJECT_ROOT

    return Path(PROJECT_ROOT) / "dataset" / "raw" / "malicious_instruct.csv"


def _resolve_positive_label(
    id2label: Mapping[int, str],
) -> tuple[int, str]:
    positive_tokens = (
        "malicious",
        "unsafe",
        "harmful",
        "jailbreak",
        "attack",
        "positive",
        "yes",
    )
    for label_id, label_name in sorted(id2label.items()):
        normalized = str(label_name or "").strip().lower()
        if any(token in normalized for token in positive_tokens):
            return int(label_id), str(label_name)

    if 1 in id2label:
        return 1, str(id2label[1])

    return 1, str(id2label.get(1, "1"))


class BaseMaliciousInstructJudge(ABC):
    """Judge adapter for Malicious Instruct jailbreak verdicts."""

    @abstractmethod
    def evaluate_batch(
        self,
        samples: Sequence[Mapping[str, Any]],
        responses: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Return benchmark verdicts and raw evaluations for each sample."""


class ProjectMaliciousInstructJudge(BaseMaliciousInstructJudge):
    """Bridge from project harmfulness evaluator to benchmark verdicts."""

    def __init__(
        self,
        *,
        harmful_score_threshold: int,
        classifier_categories: Sequence[Mapping[str, Any]],
    ):
        self.harmful_score_threshold = int(harmful_score_threshold)
        self.classifier_categories = list(classifier_categories)

    def evaluate_batch(
        self,
        samples: Sequence[Mapping[str, Any]],
        responses: Sequence[str],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for sample, response in zip(samples, responses):
            raw_evaluation = evaluate_harmfulness(
                str(sample.get("prompt", "")),
                response,
                classifier_categories=self.classifier_categories,
            )
            raw_score = raw_evaluation.get("score")
            try:
                score = int(raw_score) if raw_score is not None else None
            except Exception:
                score = None

            results.append(
                {
                    "score": score,
                    "jailbroken": (
                        score is not None
                        and score >= self.harmful_score_threshold
                    ),
                    "evaluation": raw_evaluation,
                }
            )

        return results


class MaliciousInstructOfficialEvaluator:
    """Lazy-loaded official Malicious Instruct evaluator."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        token = os.getenv("HF_TOKEN")
        self.model_name = model_name
        self.device = device
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            token=token,
        )
        self.model.to(device)
        self.model.eval()

        raw_id2label = getattr(self.model.config, "id2label", {}) or {}
        self.id2label = {
            int(label_id): str(label_name)
            for label_id, label_name in raw_id2label.items()
        }
        if not self.id2label:
            num_labels = int(getattr(self.model.config, "num_labels", 2) or 2)
            self.id2label = {
                label_id: str(label_id) for label_id in range(num_labels)
            }
        self.positive_label_id, self.positive_label = _resolve_positive_label(
            self.id2label
        )

    def evaluate_batch(
        self,
        samples: Sequence[Mapping[str, Any]],
        responses: Sequence[str],
        *,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        instructions = [
            str(sample.get("instruction", "")).strip()
            for sample in samples
        ]
        predictions = self._predict_pairs(
            instructions,
            responses,
            batch_size=batch_size,
        )

        results: list[dict[str, Any]] = []
        for prediction in predictions:
            predicted_label_id = int(prediction["predicted_label_id"])
            label_name = self.id2label.get(
                predicted_label_id,
                str(predicted_label_id),
            )
            probabilities = {
                self.id2label.get(label_id, str(label_id)): float(probability)
                for label_id, probability in enumerate(prediction["probabilities"])
            }
            results.append(
                {
                    "score": None,
                    "jailbroken": predicted_label_id == self.positive_label_id,
                    "evaluation": {
                        "predicted_label_id": predicted_label_id,
                        "predicted_label": label_name,
                        "positive_label_id": self.positive_label_id,
                        "positive_label": self.positive_label,
                        "logits": [
                            float(value) for value in prediction["logits"]
                        ],
                        "probabilities": probabilities,
                        "model_name": self.model_name,
                    },
                }
            )

        return results

    def _predict_pairs(
        self,
        instructions: Sequence[str],
        responses: Sequence[str],
        *,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        effective_batch_size = max(int(batch_size), 1)

        for start in range(0, len(instructions), effective_batch_size):
            instruction_batch = list(instructions[start : start + effective_batch_size])
            response_batch = list(responses[start : start + effective_batch_size])
            tokenized = self.tokenizer(
                instruction_batch,
                response_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            tokenized = {
                key: value.to(self.device)
                for key, value in tokenized.items()
            }

            with self.torch.no_grad():
                output = self.model(**tokenized)
                logits = output.logits
                probabilities = self.torch.softmax(logits, dim=-1)

            for row_logits, row_probabilities in zip(logits, probabilities):
                logits_list = [
                    float(value)
                    for value in row_logits.detach().cpu().tolist()
                ]
                probabilities_list = [
                    float(value)
                    for value in row_probabilities.detach().cpu().tolist()
                ]
                predicted_label_id = max(
                    range(len(probabilities_list)),
                    key=lambda idx: probabilities_list[idx],
                )
                predictions.append(
                    {
                        "predicted_label_id": int(predicted_label_id),
                        "logits": logits_list,
                        "probabilities": probabilities_list,
                    }
                )

        return predictions


@lru_cache(maxsize=4)
def get_cached_official_evaluator(
    model_name: str,
    device: str,
) -> MaliciousInstructOfficialEvaluator:
    return MaliciousInstructOfficialEvaluator(
        model_name=model_name,
        device=device,
    )


class OfficialMaliciousInstructJudge(BaseMaliciousInstructJudge):
    """Official evaluator-backed Malicious Instruct judge."""

    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        device: str,
    ):
        self.batch_size = int(batch_size)
        self.evaluator = get_cached_official_evaluator(
            model_name=model_name,
            device=device,
        )

    def evaluate_batch(
        self,
        samples: Sequence[Mapping[str, Any]],
        responses: Sequence[str],
    ) -> list[dict[str, Any]]:
        return self.evaluator.evaluate_batch(
            samples,
            responses,
            batch_size=self.batch_size,
        )


class MaliciousInstructBenchmark(BenchmarkDefinition):
    """Malicious Instruct benchmark using local CSV prompts."""

    name = "malicious_instruct"
    _SUPPORTED_JUDGE_MODES = ("official", "project")

    def normalize_config(
        self,
        raw_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = normalize_malicious_instruct_config(raw_config)

        judge_mode = config["judge_mode"]
        if judge_mode not in self._SUPPORTED_JUDGE_MODES:
            valid_values = ", ".join(self._SUPPORTED_JUDGE_MODES)
            raise ValueError(
                f"Unsupported Malicious Instruct judge_mode '{judge_mode}'. "
                f"Valid values: {valid_values}"
            )

        max_samples = int(config["max_samples"])
        if max_samples < 0:
            raise ValueError("MALICIOUS_INSTRUCT max_samples must be >= 0")

        harmful_score_threshold = int(config["harmful_score_threshold"])
        if harmful_score_threshold < 0 or harmful_score_threshold > 4:
            raise ValueError(
                "MALICIOUS_INSTRUCT harmful_score_threshold must be between 0 and 4"
            )

        categories = list(config["categories"])
        invalid_categories = [
            category
            for category in categories
            if category not in SUPPORTED_JAILBREAKBENCH_CATEGORIES
        ]
        if invalid_categories:
            valid_values = ", ".join(SUPPORTED_JAILBREAKBENCH_CATEGORIES)
            raise ValueError(
                f"Unsupported Malicious Instruct categories: {invalid_categories}. "
                f"Valid values: {valid_values}"
            )

        official_evaluator = dict(config["official_evaluator"])
        official_evaluator["model_name"] = (
            str(
                official_evaluator.get(
                    "model_name",
                    OFFICIAL_MALICIOUS_INSTRUCT_EVALUATOR_MODEL,
                )
            ).strip()
            or OFFICIAL_MALICIOUS_INSTRUCT_EVALUATOR_MODEL
        )
        official_evaluator["batch_size"] = int(
            official_evaluator.get("batch_size", 32)
        )
        official_evaluator["device"] = (
            str(official_evaluator.get("device", "cuda")).strip() or "cuda"
        )
        if official_evaluator["batch_size"] < 1:
            raise ValueError(
                "MALICIOUS_INSTRUCT official evaluator batch_size must be >= 1"
            )

        return {
            "judge_mode": judge_mode,
            "max_samples": max_samples,
            "categories": categories,
            "harmful_score_threshold": harmful_score_threshold,
            "official_evaluator": official_evaluator,
        }

    def load_samples(
        self,
        config: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        normalized = self.normalize_config(config)
        dataset_path = _resolve_dataset_path()

        allowed_categories = set(normalized["categories"])
        max_samples = normalized["max_samples"]

        samples: list[dict[str, Any]] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for fallback_index, row in enumerate(reader):
                instruction = (row.get("text") or row.get("instruction") or "").strip()
                category = (row.get("Category") or row.get("category") or "").strip()
                if not instruction:
                    continue
                if allowed_categories and category not in allowed_categories:
                    continue

                samples.append(
                    {
                        "id": fallback_index,
                        "instruction": instruction,
                        "prompt": instruction,
                        "category": category,
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
        prompts = [sample["prompt"] for sample in samples]

        responses_raw = model.get_responses_batched(prompts) if prompts else []
        responses = [
            extracted if extracted else raw.strip()
            for raw in responses_raw
            for extracted in [extract_response_after_think(raw)]
        ]

        judge = self._build_judge(
            config=normalized,
            classifier_categories=classifier_categories,
        )
        judged_records = judge.evaluate_batch(samples, responses)

        records: list[dict[str, Any]] = []
        for sample, response, judged in zip(samples, responses, judged_records):
            records.append(
                {
                    **sample,
                    "response": response,
                    "score": judged.get("score"),
                    "jailbroken": bool(judged.get("jailbroken")),
                    "evaluation": judged.get("evaluation", {}),
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
        return build_binary_result_block(
            original_summary=original_summary,
            modified_summary=modified_summary,
            config=self.normalize_config(config),
            original_details_file=original_details_file,
            modified_details_file=modified_details_file,
            bucket_keys=("by_category",),
        )

    def _build_judge(
        self,
        *,
        config: Mapping[str, Any],
        classifier_categories: Sequence[Mapping[str, Any]],
    ) -> BaseMaliciousInstructJudge:
        judge_mode = config["judge_mode"]
        if judge_mode == "project":
            return ProjectMaliciousInstructJudge(
                harmful_score_threshold=int(config["harmful_score_threshold"]),
                classifier_categories=classifier_categories,
            )

        if judge_mode == "official":
            official_evaluator = dict(config["official_evaluator"])
            return OfficialMaliciousInstructJudge(
                model_name=official_evaluator["model_name"],
                batch_size=int(official_evaluator["batch_size"]),
                device=official_evaluator["device"],
            )

        raise ValueError(
            f"Unsupported Malicious Instruct judge_mode '{judge_mode}'"
        )

    def _summarize_records(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return summarize_binary_records(
            records,
            bucket_fields={
                "by_category": "category",
            },
        )
