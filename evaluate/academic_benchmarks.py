"""
Academic benchmark evaluation helpers for graph_grpo.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config import (
    ACADEMIC_BENCHMARKS_CONFIG,
    PROJECT_ROOT,
    RESULTS_DIR,
    get_method_results_dir,
)
from data_utils import extract_response_after_think
from evaluate.model_scoring import (
    generate_responses,
    score_text_choice_variants,
)


PREDICTION_PREVIEW_LIMIT = 10
ARC_VARIANTS = ("ARC-Easy", "ARC-Challenge")
_DATASET_RECORDS_CACHE: dict[tuple[str, str | None, str], list[dict[str, Any]]] = {}


def normalize_academic_benchmarks_config(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = copy.deepcopy(ACADEMIC_BENCHMARKS_CONFIG)
    if config:
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(normalized.get(key), dict):
                normalized[key].update(value)
            else:
                normalized[key] = value

    enabled = normalized.get("enabled", [])
    normalized["enabled"] = [
        str(item).strip().lower()
        for item in enabled
        if str(item).strip()
    ]
    normalized["sample_seed"] = int(normalized.get("sample_seed", 42))
    normalized["store_predictions"] = bool(normalized.get("store_predictions", True))

    normalized["tinyhellaswag"] = {
        "dataset": str(
            normalized.get("tinyhellaswag", {}).get(
                "dataset",
                "tinyBenchmarks/tinyHellaswag",
            )
        ),
        "split": str(
            normalized.get("tinyhellaswag", {}).get("split", "validation")
        ),
        "sample_size": _normalize_optional_int(
            normalized.get("tinyhellaswag", {}).get("sample_size", 100)
        ),
    }
    normalized["arc"] = {
        "dataset": str(normalized.get("arc", {}).get("dataset", "allenai/ai2_arc")),
        "split": str(normalized.get("arc", {}).get("split", "validation")),
        "sample_size": _normalize_optional_int(
            normalized.get("arc", {}).get("sample_size", 100)
        ),
    }
    normalized["winogrande"] = {
        "dataset": str(
            normalized.get("winogrande", {}).get("dataset", "allenai/winogrande")
        ),
        "subset": str(
            normalized.get("winogrande", {}).get("subset", "winogrande_xl")
        ),
        "split": str(
            normalized.get("winogrande", {}).get("split", "validation")
        ),
        "sample_size": _normalize_optional_int(
            normalized.get("winogrande", {}).get("sample_size", 100)
        ),
    }
    normalized["gsm8k"] = {
        "dataset": str(normalized.get("gsm8k", {}).get("dataset", "openai/gsm8k")),
        "subset": str(normalized.get("gsm8k", {}).get("subset", "main")),
        "split": str(normalized.get("gsm8k", {}).get("split", "test")),
        "sample_size": _normalize_optional_int(
            normalized.get("gsm8k", {}).get("sample_size", 100)
        ),
        "max_new_tokens": int(
            normalized.get("gsm8k", {}).get("max_new_tokens", 512)
        ),
    }
    normalized["truthfulqa"] = {
        "dataset": str(
            normalized.get("truthfulqa", {}).get(
                "dataset",
                "truthfulqa/truthful_qa",
            )
        ),
        "subset": str(
            normalized.get("truthfulqa", {}).get("subset", "multiple_choice")
        ),
        "split": str(
            normalized.get("truthfulqa", {}).get("split", "validation")
        ),
        "sample_size": _normalize_optional_int(
            normalized.get("truthfulqa", {}).get("sample_size", 100)
        ),
    }
    return normalized


def get_academic_benchmark_config_snapshot(
    config: dict[str, Any] | None,
    benchmark_name: str,
) -> dict[str, Any]:
    normalized = normalize_academic_benchmarks_config(config)
    benchmark_name = benchmark_name.lower()
    if benchmark_name not in normalized:
        raise KeyError(f"Unknown academic benchmark: {benchmark_name}")

    snapshot = {
        "sample_seed": normalized["sample_seed"],
        "store_predictions": normalized["store_predictions"],
        **copy.deepcopy(normalized[benchmark_name]),
    }
    snapshot["benchmark"] = benchmark_name
    return snapshot


def get_academic_metric_names(config: dict[str, Any] | None = None) -> list[str]:
    normalized = normalize_academic_benchmarks_config(config)
    metric_names: list[str] = []
    for benchmark_name in normalized["enabled"]:
        if benchmark_name == "tinyhellaswag":
            metric_names.extend(
                ["tinyhellaswag_irt_plus_plus", "tinyhellaswag_accuracy"]
            )
        elif benchmark_name == "arc":
            metric_names.extend(
                ["arc_easy_accuracy", "arc_challenge_accuracy", "arc_macro_accuracy"]
            )
        elif benchmark_name == "winogrande":
            metric_names.append("winogrande_accuracy")
        elif benchmark_name == "gsm8k":
            metric_names.append("gsm8k_exact_match")
        elif benchmark_name == "truthfulqa":
            metric_names.extend(["truthfulqa_mc1", "truthfulqa_mc2"])
    return metric_names


def get_academic_metric_values(results: dict[str, Any] | None) -> dict[str, float]:
    values: dict[str, float] = {}
    for benchmark_name, benchmark_result in (results or {}).items():
        summary = None
        if isinstance(benchmark_result, dict):
            if "summary" in benchmark_result:
                summary = benchmark_result["summary"]
            elif "modified" in benchmark_result:
                summary = benchmark_result["modified"]
        if not isinstance(summary, dict):
            continue

        if benchmark_name == "tinyhellaswag":
            if summary.get("irt_plus_plus") is not None:
                values["tinyhellaswag_irt_plus_plus"] = float(summary["irt_plus_plus"])
            if summary.get("accuracy") is not None:
                values["tinyhellaswag_accuracy"] = float(summary["accuracy"])
        elif benchmark_name == "arc":
            by_variant = summary.get("by_variant", {})
            easy = by_variant.get("ARC-Easy", {})
            challenge = by_variant.get("ARC-Challenge", {})
            if easy.get("accuracy") is not None:
                values["arc_easy_accuracy"] = float(easy["accuracy"])
            if challenge.get("accuracy") is not None:
                values["arc_challenge_accuracy"] = float(challenge["accuracy"])
            if summary.get("macro_accuracy") is not None:
                values["arc_macro_accuracy"] = float(summary["macro_accuracy"])
        elif benchmark_name == "winogrande" and summary.get("accuracy") is not None:
            values["winogrande_accuracy"] = float(summary["accuracy"])
        elif benchmark_name == "gsm8k" and summary.get("exact_match") is not None:
            values["gsm8k_exact_match"] = float(summary["exact_match"])
        elif benchmark_name == "truthfulqa":
            if summary.get("mc1") is not None:
                values["truthfulqa_mc1"] = float(summary["mc1"])
            if summary.get("mc2") is not None:
                values["truthfulqa_mc2"] = float(summary["mc2"])
    return values


def evaluate_model_on_academic_benchmarks(
    model: Any,
    config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    normalized = normalize_academic_benchmarks_config(config)
    results: dict[str, dict[str, Any]] = {}
    for benchmark_name in normalized["enabled"]:
        evaluator = _EVALUATORS[benchmark_name]
        results[benchmark_name] = evaluator(model, normalized)
    return results


def get_cached_or_evaluate_original_academic_benchmarks(
    model: Any,
    *,
    model_name: str,
    config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    normalized = normalize_academic_benchmarks_config(config)
    if not normalized["enabled"]:
        return {}

    method_results_dir = get_method_results_dir("graph_grpo")
    cache_dir = method_results_dir / "academic_benchmarks" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for benchmark_name in normalized["enabled"]:
        config_snapshot = get_academic_benchmark_config_snapshot(
            normalized,
            benchmark_name,
        )
        cache_file = _academic_cache_file(cache_dir, model_name, benchmark_name, config_snapshot)
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                results[benchmark_name] = json.load(f)
            continue

        result = _EVALUATORS[benchmark_name](model, normalized)
        payload = copy.deepcopy(result)
        payload["cached_at"] = datetime.now().isoformat()
        payload["model_name"] = model_name
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        results[benchmark_name] = payload

    return results


def build_academic_benchmarks_result(
    *,
    config: dict[str, Any] | None,
    original_results: dict[str, dict[str, Any]] | None,
    modified_results: dict[str, dict[str, Any]] | None,
    method_results_dir: Path | None = None,
    detail_prefix: str | None = None,
) -> dict[str, dict[str, Any]]:
    normalized = normalize_academic_benchmarks_config(config)
    result_blocks: dict[str, dict[str, Any]] = {}

    for benchmark_name in normalized["enabled"]:
        original_result = (original_results or {}).get(benchmark_name)
        modified_result = (modified_results or {}).get(benchmark_name)
        if original_result is None or modified_result is None:
            continue

        original_summary = copy.deepcopy(original_result["summary"])
        modified_summary = copy.deepcopy(modified_result["summary"])
        details_file = None
        if (
            normalized["store_predictions"]
            and method_results_dir is not None
            and detail_prefix
        ):
            details_file = save_academic_benchmark_details(
                benchmark_name=benchmark_name,
                method_results_dir=method_results_dir,
                detail_prefix=detail_prefix,
                original_result=original_result,
                modified_result=modified_result,
            )

        result_blocks[benchmark_name] = {
            "original": original_summary,
            "modified": modified_summary,
            "delta_primary_metric": (
                modified_summary["primary_metric_value"]
                - original_summary["primary_metric_value"]
            ),
            "config": get_academic_benchmark_config_snapshot(normalized, benchmark_name),
            "details_file": details_file,
        }

    return result_blocks


def save_academic_benchmark_details(
    *,
    benchmark_name: str,
    method_results_dir: Path,
    detail_prefix: str,
    original_result: dict[str, Any],
    modified_result: dict[str, Any],
) -> str:
    details_dir = method_results_dir / "academic_benchmarks" / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    filename = details_dir / (
        f"{detail_prefix}_{benchmark_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    payload = {
        "original": original_result,
        "modified": modified_result,
        "saved_at": datetime.now().isoformat(),
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return _to_project_relative(filename)


def extract_gsm8k_final_answer(text: str) -> str | None:
    if not text:
        return None

    cleaned = extract_response_after_think(text)
    cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think>", " ", cleaned, flags=re.IGNORECASE)

    marker_match = re.search(r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?)", cleaned)
    if marker_match:
        return _normalize_numeric_text(marker_match.group(1))

    numeric_matches = re.findall(r"[-+]?\$?[\d,]+(?:\.\d+)?", cleaned)
    if numeric_matches:
        return _normalize_numeric_text(numeric_matches[-1])

    return None


def _normalize_optional_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", "null", "all"):
        return None
    return int(value)


def _config_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _to_project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _academic_cache_file(
    cache_dir: Path,
    model_name: str,
    benchmark_name: str,
    config_snapshot: dict[str, Any],
) -> Path:
    model_hash = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
    config_hash = _config_hash(config_snapshot)
    return cache_dir / f"{benchmark_name}_{model_hash}_{config_hash}.json"


def _load_dataset_records(
    dataset_name: str,
    *,
    subset: str | None = None,
    split: str,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    cache_key = (dataset_name, subset, split)
    if cache_key in _DATASET_RECORDS_CACHE:
        return _DATASET_RECORDS_CACHE[cache_key]

    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    records = [dict(dataset[index]) for index in range(len(dataset))]
    _DATASET_RECORDS_CACHE[cache_key] = records
    return records


def _sample_records(
    records: list[dict[str, Any]],
    sample_size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size is None or sample_size >= len(records):
        return list(records)

    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(len(records)), sample_size))
    return [records[index] for index in sampled_indices]


def _prediction_preview(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(predictions[:PREDICTION_PREVIEW_LIMIT])


def _evaluate_tinybenchmarks_score_vector(
    score_vector: list[int],
    benchmark: str,
) -> dict[str, Any]:
    try:
        tinybenchmarks = importlib.import_module("tinyBenchmarks")
    except ImportError as exc:
        raise ImportError(
            "TinyHellaSwag evaluation requires the optional tinyBenchmarks package. "
            "Install it with `pip install git+https://github.com/felipemaiapolo/tinyBenchmarks`."
        ) from exc

    score_array = np.asarray(score_vector, dtype=np.int64)
    result = tinybenchmarks.evaluate(score_array, benchmark)
    if benchmark in result:
        return result[benchmark]
    return result


def _extract_question(record: dict[str, Any], *field_names: str) -> str:
    for field_name in field_names:
        value = record.get(field_name)
        if value:
            return str(value)
    raise ValueError(f"Record is missing question fields {field_names}")


def _extract_choice_texts(value: Any) -> tuple[list[str], list[str] | None]:
    if isinstance(value, dict):
        texts = value.get("text") or value.get("choices")
        labels = value.get("label") or value.get("labels")
        if texts is not None:
            return [str(item) for item in texts], None if labels is None else [str(item) for item in labels]

    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            texts = [str(item.get("text") or item.get("label") or item.get("choice") or "") for item in value]
            labels = [str(item.get("label", "")) for item in value]
            return texts, labels
        return [str(item) for item in value], None

    raise ValueError(f"Unsupported choice structure: {type(value).__name__}")


def _resolve_choice_index(
    answer_key: Any,
    *,
    choice_labels: list[str] | None,
    choice_count: int,
) -> int:
    if isinstance(answer_key, int):
        return int(answer_key)

    if isinstance(answer_key, (list, tuple)):
        labels = [int(item) for item in answer_key]
        return labels.index(1)

    normalized = str(answer_key).strip()
    upper = normalized.upper()

    if choice_labels:
        for index, label in enumerate(choice_labels):
            if upper == str(label).strip().upper():
                return index

    if normalized.isdigit():
        numeric_index = int(normalized)
        if numeric_index == 0:
            return 0
        if 1 <= numeric_index <= choice_count:
            return numeric_index - 1
        if 0 <= numeric_index < choice_count:
            return numeric_index

    letter_index = ord(upper[:1]) - ord("A")
    if 0 <= letter_index < choice_count:
        return letter_index

    raise ValueError(f"Unsupported answer key: {answer_key}")


def _score_text_choices(
    model: Any,
    prompt: str,
    choices: list[str],
) -> tuple[int, list[float]]:
    scores = [
        float(score_text_choice_variants(model, prompt, choice))
        for choice in choices
    ]
    predicted_index = max(range(len(scores)), key=lambda idx: scores[idx])
    return predicted_index, scores


def _evaluate_tinyhellaswag(
    model: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    benchmark_name = "tinyhellaswag"
    benchmark_config = get_academic_benchmark_config_snapshot(config, benchmark_name)
    records = _load_dataset_records(
        benchmark_config["dataset"],
        split=benchmark_config["split"],
    )
    sampled_records = _sample_records(
        records,
        benchmark_config["sample_size"],
        benchmark_config["sample_seed"],
    )

    predictions: list[dict[str, Any]] = []
    score_vector: list[int] = []
    for index, record in enumerate(sampled_records):
        prompt = (
            record.get("input_formatted")
            or record.get("input")
            or record.get("ctx")
            or _extract_question(record, "ctx_a", "query")
        )
        choices, _choice_labels = _extract_choice_texts(
            record.get("endings") or record.get("choices")
        )
        gold_index = _resolve_choice_index(
            record.get("label", record.get("gold", record.get("answer", record.get("target")))),
            choice_labels=None,
            choice_count=len(choices),
        )
        prompt_text = (
            "Choose the most plausible ending for the passage.\n\n"
            f"Passage: {prompt}\n\nEnding:"
        )
        predicted_index, choice_scores = _score_text_choices(model, prompt_text, choices)
        is_correct = predicted_index == gold_index
        score_vector.append(int(is_correct))
        predictions.append(
            {
                "index": index,
                "question": str(prompt),
                "choices": choices,
                "gold_index": gold_index,
                "predicted_index": predicted_index,
                "is_correct": is_correct,
                "choice_scores": choice_scores,
            }
        )

    tinybenchmarks_result = _evaluate_tinybenchmarks_score_vector(score_vector, "hellaswag")
    irt_plus_plus = tinybenchmarks_result.get("gpirt")
    if irt_plus_plus is None:
        raise ValueError(
            "tinyBenchmarks did not return a `gpirt` score for TinyHellaSwag."
        )

    summary = {
        "primary_metric_name": "irt_plus_plus",
        "primary_metric_value": float(irt_plus_plus),
        "irt_plus_plus": float(irt_plus_plus),
        "accuracy": (sum(score_vector) / len(score_vector)) if score_vector else 0.0,
        "correct": int(sum(score_vector)),
        "total": len(score_vector),
        "irt": tinybenchmarks_result.get("irt"),
        "pirt": tinybenchmarks_result.get("pirt"),
        "config_snapshot": benchmark_config,
    }
    result = {
        "summary": summary,
        "prediction_preview": _prediction_preview(predictions),
        "score_vector": score_vector,
    }
    if benchmark_config["store_predictions"]:
        result["predictions"] = predictions
    return result


def _extract_arc_choices(record: dict[str, Any]) -> tuple[list[str], list[str] | None]:
    choices_value = record.get("choices")
    if choices_value is None:
        raise ValueError("ARC record is missing `choices`")
    return _extract_choice_texts(choices_value)


def _evaluate_arc(
    model: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    benchmark_name = "arc"
    benchmark_config = get_academic_benchmark_config_snapshot(config, benchmark_name)
    predictions: list[dict[str, Any]] = []
    by_variant: dict[str, dict[str, Any]] = {}

    for variant in ARC_VARIANTS:
        records = _load_dataset_records(
            benchmark_config["dataset"],
            subset=variant,
            split=benchmark_config["split"],
        )
        sampled_records = _sample_records(
            records,
            benchmark_config["sample_size"],
            benchmark_config["sample_seed"],
        )

        variant_predictions: list[dict[str, Any]] = []
        for record in sampled_records:
            question = _extract_question(record, "question", "input", "prompt")
            choices, choice_labels = _extract_arc_choices(record)
            gold_index = _resolve_choice_index(
                record.get("answerKey", record.get("answer", record.get("gold"))),
                choice_labels=choice_labels,
                choice_count=len(choices),
            )
            prompt_text = (
                "Answer the multiple-choice question with the best answer text.\n\n"
                f"Question: {question}\n"
                "Choices:\n"
                + "\n".join(
                    f"{label or chr(ord('A') + idx)}. {choice}"
                    for idx, (choice, label) in enumerate(
                        zip(choices, choice_labels or [None] * len(choices))
                    )
                )
                + "\nAnswer:"
            )
            predicted_index, choice_scores = _score_text_choices(model, prompt_text, choices)
            prediction = {
                "variant": variant,
                "question": question,
                "choices": choices,
                "choice_labels": choice_labels,
                "gold_index": gold_index,
                "predicted_index": predicted_index,
                "is_correct": predicted_index == gold_index,
                "choice_scores": choice_scores,
            }
            variant_predictions.append(prediction)
            predictions.append(prediction)

        correct = sum(1 for row in variant_predictions if row["is_correct"])
        total = len(variant_predictions)
        by_variant[variant] = {
            "accuracy": (correct / total) if total else 0.0,
            "correct": correct,
            "total": total,
        }

    macro_accuracy = (
        sum(bucket["accuracy"] for bucket in by_variant.values()) / len(by_variant)
        if by_variant
        else 0.0
    )
    summary = {
        "primary_metric_name": "macro_accuracy",
        "primary_metric_value": float(macro_accuracy),
        "macro_accuracy": float(macro_accuracy),
        "by_variant": by_variant,
        "config_snapshot": benchmark_config,
    }
    result = {
        "summary": summary,
        "prediction_preview": _prediction_preview(predictions),
    }
    if benchmark_config["store_predictions"]:
        result["predictions"] = predictions
    return result


def _complete_winogrande_sentence(sentence: str, option: str) -> str:
    if "_" not in sentence:
        return f"{sentence} {option}".strip()
    return sentence.replace("_", option, 1)


def _evaluate_winogrande(
    model: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    benchmark_name = "winogrande"
    benchmark_config = get_academic_benchmark_config_snapshot(config, benchmark_name)
    records = _load_dataset_records(
        benchmark_config["dataset"],
        subset=benchmark_config["subset"],
        split=benchmark_config["split"],
    )
    sampled_records = _sample_records(
        records,
        benchmark_config["sample_size"],
        benchmark_config["sample_seed"],
    )

    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(sampled_records):
        sentence = _extract_question(record, "sentence", "query", "prompt")
        options = [str(record["option1"]), str(record["option2"])]
        completed_options = [
            _complete_winogrande_sentence(sentence, option)
            for option in options
        ]
        gold_index = _resolve_choice_index(
            record.get("answer", record.get("label", record.get("gold"))),
            choice_labels=["1", "2"],
            choice_count=2,
        )
        prompt_text = (
            "Choose the more plausible sentence completion.\n\n"
            f"Sentence: {sentence}\nAnswer:"
        )
        predicted_index, choice_scores = _score_text_choices(
            model,
            prompt_text,
            completed_options,
        )
        predictions.append(
            {
                "index": index,
                "sentence": sentence,
                "options": options,
                "completed_options": completed_options,
                "gold_index": gold_index,
                "predicted_index": predicted_index,
                "is_correct": predicted_index == gold_index,
                "choice_scores": choice_scores,
            }
        )

    total = len(predictions)
    correct = sum(1 for row in predictions if row["is_correct"])
    summary = {
        "primary_metric_name": "accuracy",
        "primary_metric_value": (correct / total) if total else 0.0,
        "accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
        "config_snapshot": benchmark_config,
    }
    result = {
        "summary": summary,
        "prediction_preview": _prediction_preview(predictions),
    }
    if benchmark_config["store_predictions"]:
        result["predictions"] = predictions
    return result


def _normalize_numeric_text(value: str) -> str:
    normalized = value.strip().replace("$", "").replace(",", "")
    if normalized.endswith(".0"):
        normalized = normalized[:-2]
    return normalized


def _extract_gsm8k_target_answer(record: dict[str, Any]) -> str | None:
    answer = record.get("answer")
    if answer:
        extracted = extract_gsm8k_final_answer(str(answer))
        if extracted is not None:
            return extracted

    final_answer = record.get("final_answer")
    if final_answer:
        return _normalize_numeric_text(str(final_answer))

    return None


def _evaluate_gsm8k(
    model: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    benchmark_name = "gsm8k"
    benchmark_config = get_academic_benchmark_config_snapshot(config, benchmark_name)
    records = _load_dataset_records(
        benchmark_config["dataset"],
        subset=benchmark_config["subset"],
        split=benchmark_config["split"],
    )
    sampled_records = _sample_records(
        records,
        benchmark_config["sample_size"],
        benchmark_config["sample_seed"],
    )

    prompts = [
        (
            "Solve the math word problem. End your response with `#### <answer>`.\n\n"
            f"Question: {_extract_question(record, 'question', 'query', 'prompt')}\nAnswer:"
        )
        for record in sampled_records
    ]
    responses = generate_responses(
        model,
        prompts,
        max_new_tokens=benchmark_config["max_new_tokens"],
    )

    predictions: list[dict[str, Any]] = []
    for index, (record, prompt, raw_response) in enumerate(zip(sampled_records, prompts, responses)):
        target_answer = _extract_gsm8k_target_answer(record)
        predicted_answer = extract_gsm8k_final_answer(raw_response)
        predictions.append(
            {
                "index": index,
                "question": _extract_question(record, "question", "query", "prompt"),
                "target_answer": target_answer,
                "predicted_answer": predicted_answer,
                "is_correct": predicted_answer is not None and predicted_answer == target_answer,
                "raw_response": raw_response,
                "prompt": prompt,
            }
        )

    total = len(predictions)
    correct = sum(1 for row in predictions if row["is_correct"])
    summary = {
        "primary_metric_name": "exact_match",
        "primary_metric_value": (correct / total) if total else 0.0,
        "exact_match": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
        "config_snapshot": benchmark_config,
    }
    result = {
        "summary": summary,
        "prediction_preview": _prediction_preview(predictions),
    }
    if benchmark_config["store_predictions"]:
        result["predictions"] = predictions
    return result


def _parse_truthfulqa_answers(value: Any) -> list[str]:
    if isinstance(value, dict):
        choices = value.get("choices") or value.get("text")
        if choices is not None:
            return [str(item) for item in choices]
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    raise ValueError(f"Unsupported TruthfulQA answer structure: {type(value).__name__}")


def _extract_truthfulqa_targets(record: dict[str, Any]) -> tuple[str, list[str], list[int], list[str], list[int]]:
    question = _extract_question(record, "question", "query", "Question")
    mc1_targets = record.get("mc1_targets")
    mc2_targets = record.get("mc2_targets")

    if isinstance(mc1_targets, dict) and isinstance(mc2_targets, dict):
        mc1_choices = _parse_truthfulqa_answers(mc1_targets.get("choices"))
        mc1_labels = [int(item) for item in mc1_targets.get("labels", [])]
        mc2_choices = _parse_truthfulqa_answers(mc2_targets.get("choices"))
        mc2_labels = [int(item) for item in mc2_targets.get("labels", [])]
        return question, mc1_choices, mc1_labels, mc2_choices, mc2_labels

    if "Correct Answers" in record and "Incorrect Answers" in record:
        correct_answers = _parse_truthfulqa_answers(record["Correct Answers"])
        incorrect_answers = _parse_truthfulqa_answers(record["Incorrect Answers"])
        choices = correct_answers + incorrect_answers
        labels = [1] * len(correct_answers) + [0] * len(incorrect_answers)
        return question, choices, labels, choices, labels

    if "choices" in record and "gold" in record:
        choices = _parse_truthfulqa_answers(record["choices"])
        labels = [0] * len(choices)
        labels[int(record["gold"])] = 1
        return question, choices, labels, choices, labels

    raise ValueError("Unsupported TruthfulQA record structure")


def _probability_mass(scores: list[float], labels: list[int]) -> float:
    if not scores:
        return 0.0
    max_score = max(scores)
    weights = [math.exp(score - max_score) for score in scores]
    denominator = sum(weights)
    if denominator == 0:
        return 0.0
    numerator = sum(weight for weight, label in zip(weights, labels) if label)
    return numerator / denominator


def _evaluate_truthfulqa(
    model: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    benchmark_name = "truthfulqa"
    benchmark_config = get_academic_benchmark_config_snapshot(config, benchmark_name)
    records = _load_dataset_records(
        benchmark_config["dataset"],
        subset=benchmark_config.get("subset"),
        split=benchmark_config["split"],
    )
    sampled_records = _sample_records(
        records,
        benchmark_config["sample_size"],
        benchmark_config["sample_seed"],
    )

    predictions: list[dict[str, Any]] = []
    mc1_total = 0.0
    mc2_total = 0.0
    for index, record in enumerate(sampled_records):
        question, mc1_choices, mc1_labels, mc2_choices, mc2_labels = _extract_truthfulqa_targets(record)
        prompt_text = f"Question: {question}\nAnswer:"
        _mc1_predicted_index, mc1_scores = _score_text_choices(model, prompt_text, mc1_choices)
        _mc2_predicted_index, mc2_scores = _score_text_choices(model, prompt_text, mc2_choices)

        best_true_score = max(
            score for score, label in zip(mc1_scores, mc1_labels) if label
        )
        best_false_score = max(
            score for score, label in zip(mc1_scores, mc1_labels) if not label
        )
        mc1_correct = best_true_score > best_false_score
        mc2_mass = _probability_mass(mc2_scores, mc2_labels)
        mc1_total += float(mc1_correct)
        mc2_total += float(mc2_mass)
        predictions.append(
            {
                "index": index,
                "question": question,
                "mc1_choices": mc1_choices,
                "mc1_labels": mc1_labels,
                "mc1_scores": mc1_scores,
                "mc1_correct": mc1_correct,
                "mc2_choices": mc2_choices,
                "mc2_labels": mc2_labels,
                "mc2_scores": mc2_scores,
                "mc2_probability_mass": mc2_mass,
            }
        )

    total = len(predictions)
    summary = {
        "primary_metric_name": "mc1",
        "primary_metric_value": (mc1_total / total) if total else 0.0,
        "mc1": (mc1_total / total) if total else 0.0,
        "mc2": (mc2_total / total) if total else 0.0,
        "total": total,
        "config_snapshot": benchmark_config,
    }
    result = {
        "summary": summary,
        "prediction_preview": _prediction_preview(predictions),
    }
    if benchmark_config["store_predictions"]:
        result["predictions"] = predictions
    return result


_EVALUATORS: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {
    "tinyhellaswag": _evaluate_tinyhellaswag,
    "arc": _evaluate_arc,
    "winogrande": _evaluate_winogrande,
    "gsm8k": _evaluate_gsm8k,
    "truthfulqa": _evaluate_truthfulqa,
}
