"""
Shared MMLU evaluation helpers.

Supports zero-shot and few-shot prompting, deterministic sampling,
summary aggregation, and caching of the original-model baseline.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from config import MMLU_CONFIG, PROJECT_ROOT, RESULTS_DIR
from data_utils import extract_response_after_think
from evaluate.model_scoring import (
    batchify,
    build_chat_prompts,
    generate_responses,
    score_completion_logprob,
    score_text_choice_variants,
)


CHOICE_LETTERS = ("A", "B", "C", "D")
PREDICTION_PREVIEW_LIMIT = 10
_DATASET_RECORDS_CACHE: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
_PREPARED_DATA_CACHE: dict[str, dict[str, Any]] = {}


def normalize_mmlu_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = dict(MMLU_CONFIG)
    if config:
        normalized.update(config)

    normalized["enabled"] = bool(normalized.get("enabled", True))
    normalized["dataset"] = str(normalized.get("dataset", "cais/mmlu"))
    normalized["subset"] = str(normalized.get("subset", "all"))
    normalized["split"] = str(normalized.get("split", "test"))
    normalized["mode"] = str(normalized.get("mode", "zero_shot")).lower()
    normalized["answer_mode"] = str(normalized.get("answer_mode", "generate")).lower()
    normalized["n_shots"] = int(normalized.get("n_shots", 5))
    normalized["sample_seed"] = int(normalized.get("sample_seed", 42))
    normalized["max_new_tokens"] = int(normalized.get("max_new_tokens", 32))
    normalized["store_predictions"] = bool(normalized.get("store_predictions", False))

    sample_size = normalized.get("sample_size")
    if sample_size in ("", "none", "null", "all"):
        sample_size = None
    normalized["sample_size"] = None if sample_size is None else int(sample_size)

    if normalized["mode"] == "zero_shot":
        normalized["n_shots"] = 0

    return normalized


def get_mmlu_config_snapshot(config: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_mmlu_config(config)
    return {
        "dataset": normalized["dataset"],
        "subset": normalized["subset"],
        "split": normalized["split"],
        "mode": normalized["mode"],
        "answer_mode": normalized["answer_mode"],
        "n_shots": normalized["n_shots"],
        "sample_size": normalized["sample_size"],
        "sample_seed": normalized["sample_seed"],
        "max_new_tokens": normalized["max_new_tokens"],
        "store_predictions": normalized["store_predictions"],
    }


def _dataset_cache_key(dataset_name: str, subset: str, split: str) -> tuple[str, str, str]:
    return dataset_name, subset, split


def _config_hash(config_snapshot: dict[str, Any]) -> str:
    payload = json.dumps(config_snapshot, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _to_project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_choice_letter(text: str) -> str | None:
    if not text:
        return None

    cleaned = extract_response_after_think(text)
    cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think>", " ", cleaned, flags=re.IGNORECASE)
    stripped = cleaned.strip().upper()
    if len(stripped) == 1 and stripped in CHOICE_LETTERS:
        return stripped[:1]

    match = re.search(r"\b(?:ANSWER\s*:?\s*)?([ABCD])\b", stripped)
    if match:
        return match.group(1)

    return None


def _extract_subject(record: dict[str, Any], default_subject: str) -> str:
    subject = record.get("subject") or record.get("category") or default_subject
    return str(subject)


def _extract_choices(record: dict[str, Any]) -> list[str]:
    choices = record.get("choices")
    if isinstance(choices, dict):
        return [str(choices[letter]) for letter in CHOICE_LETTERS]
    if isinstance(choices, (list, tuple)) and len(choices) >= 4:
        return [str(choice) for choice in choices[:4]]
    raise ValueError(f"Unsupported MMLU choices format: {type(choices).__name__}")


def _extract_question(record: dict[str, Any]) -> str:
    question = record.get("question") or record.get("input") or record.get("prompt")
    if not question:
        raise ValueError("MMLU record does not contain a question field")
    return str(question)


def _extract_correct_letter(record: dict[str, Any]) -> str:
    answer = record.get("answer")
    if answer is None:
        answer = record.get("target")
    if answer is None:
        answer = record.get("answerKey")

    if isinstance(answer, int):
        return CHOICE_LETTERS[answer]

    answer_str = str(answer).strip().upper()
    if answer_str in CHOICE_LETTERS:
        return answer_str
    if answer_str.isdigit():
        return CHOICE_LETTERS[int(answer_str)]

    raise ValueError(f"Unsupported MMLU answer format: {answer}")


def _load_dataset_records(dataset_name: str, subset: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    cache_key = _dataset_cache_key(dataset_name, subset, split)
    if cache_key in _DATASET_RECORDS_CACHE:
        return _DATASET_RECORDS_CACHE[cache_key]

    dataset = None
    tried_without_subset = False

    if subset:
        try:
            dataset = load_dataset(dataset_name, subset, split=split)
        except Exception:
            if subset != "all":
                raise
    if dataset is None:
        tried_without_subset = True
        dataset = load_dataset(dataset_name, split=split)

    records = [dict(dataset[i]) for i in range(len(dataset))]
    fallback_subject = subset if subset and not tried_without_subset else "all"
    for record in records:
        record.setdefault("subject", _extract_subject(record, fallback_subject))

    _DATASET_RECORDS_CACHE[cache_key] = records
    return records


def _sample_records(records: list[dict[str, Any]], sample_size: int | None, seed: int) -> list[dict[str, Any]]:
    if sample_size is None or sample_size >= len(records):
        return list(records)

    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(len(records)), sample_size))
    return [records[index] for index in sampled_indices]


def _format_question_block(record: dict[str, Any], include_answer: bool) -> str:
    question = _extract_question(record)
    choices = _extract_choices(record)
    lines = [f"Question: {question}"]
    for letter, choice in zip(CHOICE_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:" if include_answer else "Answer:")
    if include_answer:
        lines[-1] += f" {_extract_correct_letter(record)}"
    return "\n".join(lines)


def build_mmlu_prompt(
    record: dict[str, Any],
    *,
    mode: str,
    few_shot_examples: Iterable[dict[str, Any]] | None = None,
) -> str:
    subject = _extract_subject(record, "all")
    intro = (
        f"The following are multiple choice questions about {subject}. "
        "Respond with only the letter A, B, C, or D. Do not include reasoning or <think> tags."
    )
    if mode == "few_shot":
        intro = (
            f"The following are multiple choice questions (with answers) about {subject}. "
            "Respond to the final question with only the letter A, B, C, or D. Do not include reasoning or <think> tags."
        )

    blocks = [intro]
    if mode == "few_shot":
        for example in few_shot_examples or []:
            blocks.append(_format_question_block(example, include_answer=True))
    blocks.append(_format_question_block(record, include_answer=False))
    return "\n\n".join(blocks)


def _prepare_few_shot_examples(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if config["mode"] != "few_shot" or config["n_shots"] <= 0:
        return {}

    subset = config["subset"]
    dataset_name = config["dataset"]
    n_shots = config["n_shots"]

    if subset != "all":
        dev_records = _load_dataset_records(dataset_name, subset, "dev")
        return {subset: dev_records[:n_shots]}

    dev_records = _load_dataset_records(dataset_name, "all", "dev")
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for record in dev_records:
        subject = _extract_subject(record, "all")
        by_subject.setdefault(subject, [])
        if len(by_subject[subject]) < n_shots:
            by_subject[subject].append(record)
    return by_subject


def prepare_mmlu_data(config: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_mmlu_config(config)
    config_snapshot = get_mmlu_config_snapshot(normalized)
    cache_key = _config_hash(config_snapshot)
    if cache_key in _PREPARED_DATA_CACHE:
        return _PREPARED_DATA_CACHE[cache_key]

    records = _load_dataset_records(
        normalized["dataset"],
        normalized["subset"],
        normalized["split"],
    )
    sampled_records = _sample_records(records, normalized["sample_size"], normalized["sample_seed"])
    few_shot_by_subject = _prepare_few_shot_examples(normalized)

    entries = []
    for index, record in enumerate(sampled_records):
        subject = _extract_subject(record, normalized["subset"])
        prompt = build_mmlu_prompt(
            record,
            mode=normalized["mode"],
            few_shot_examples=few_shot_by_subject.get(subject, []),
        )
        entries.append(
            {
                "index": index,
                "subject": subject,
                "question": _extract_question(record),
                "choices": _extract_choices(record),
                "correct_letter": _extract_correct_letter(record),
                "prompt": prompt,
            }
        )

    prepared = {
        "entries": entries,
        "config_snapshot": config_snapshot,
    }
    _PREPARED_DATA_CACHE[cache_key] = prepared
    return prepared


def _generate_choice_responses(model: Any, prompts: list[str], max_new_tokens: int) -> list[str]:
    return generate_responses(model, prompts, max_new_tokens=max_new_tokens)


def _build_prediction_rows(entries: list[dict[str, Any]], responses: list[str]) -> list[dict[str, Any]]:
    rows = []
    for entry, raw_response in zip(entries, responses):
        predicted = parse_choice_letter(raw_response)
        rows.append(
            {
                "index": entry["index"],
                "subject": entry["subject"],
                "question": entry["question"],
                "choices": entry["choices"],
                "correct_letter": entry["correct_letter"],
                "predicted_letter": predicted,
                "is_correct": predicted == entry["correct_letter"],
                "raw_response": raw_response,
                "choice_scores": None,
            }
        )
    return rows


def _build_chat_prompts(model: Any, prompts: list[str]) -> list[str]:
    return build_chat_prompts(model, prompts)


def _score_completion_logprob(model: Any, prompt_text: str, completion_text: str) -> float:
    return score_completion_logprob(model, prompt_text, completion_text)


def _predict_with_logits(model: Any, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    chat_prompts = _build_chat_prompts(model, [entry["prompt"] for entry in entries])

    for entry, prompt_text in zip(entries, chat_prompts):
        choice_scores = {}
        for letter in CHOICE_LETTERS:
            choice_scores[letter] = score_text_choice_variants(
                model,
                prompt_text,
                letter,
                variants=(letter, f" {letter}"),
            )

        predicted = max(choice_scores, key=choice_scores.get)
        rows.append(
            {
                "index": entry["index"],
                "subject": entry["subject"],
                "question": entry["question"],
                "choices": entry["choices"],
                "correct_letter": entry["correct_letter"],
                "predicted_letter": predicted,
                "is_correct": predicted == entry["correct_letter"],
                "raw_response": None,
                "choice_scores": choice_scores,
            }
        )

    return rows


def summarize_mmlu_predictions(
    predictions: list[dict[str, Any]],
    config_snapshot: dict[str, Any],
) -> dict[str, Any]:
    total = len(predictions)
    correct = sum(1 for row in predictions if row["is_correct"])
    invalid_predictions = sum(1 for row in predictions if row["predicted_letter"] is None)

    subject_buckets: dict[str, dict[str, int]] = {}
    for row in predictions:
        bucket = subject_buckets.setdefault(
            row["subject"],
            {"correct": 0, "total": 0, "invalid_predictions": 0},
        )
        bucket["total"] += 1
        if row["is_correct"]:
            bucket["correct"] += 1
        if row["predicted_letter"] is None:
            bucket["invalid_predictions"] += 1

    by_subject = {}
    for subject, bucket in sorted(subject_buckets.items()):
        subject_total = bucket["total"]
        by_subject[subject] = {
            "accuracy": (bucket["correct"] / subject_total) if subject_total else 0.0,
            "correct": bucket["correct"],
            "total": subject_total,
            "invalid_predictions": bucket["invalid_predictions"],
        }

    return {
        "accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
        "invalid_predictions": invalid_predictions,
        "by_subject": by_subject,
        "config_snapshot": copy.deepcopy(config_snapshot),
    }


def _build_prediction_preview(predictions: list[dict[str, Any]], limit: int = PREDICTION_PREVIEW_LIMIT) -> list[dict[str, Any]]:
    preview = []
    for row in predictions[:limit]:
        preview.append(
            {
                "index": row["index"],
                "subject": row["subject"],
                "question": row["question"],
                "correct_letter": row["correct_letter"],
                "predicted_letter": row["predicted_letter"],
                "is_correct": row["is_correct"],
                "raw_response": row.get("raw_response"),
                "choice_scores": copy.deepcopy(row.get("choice_scores")),
            }
        )
    return preview


def _build_answer_comparison_preview(
    original_predictions: list[dict[str, Any]],
    modified_predictions: list[dict[str, Any]],
    limit: int = PREDICTION_PREVIEW_LIMIT,
) -> list[dict[str, Any]]:
    preview = []
    for original_row, modified_row in zip(original_predictions[:limit], modified_predictions[:limit]):
        preview.append(
            {
                "index": original_row.get("index"),
                "subject": original_row.get("subject"),
                "question": original_row.get("question"),
                "correct_letter": original_row.get("correct_letter"),
                "original": {
                    "predicted_letter": original_row.get("predicted_letter"),
                    "is_correct": original_row.get("is_correct"),
                    "raw_response": original_row.get("raw_response"),
                    "choice_scores": copy.deepcopy(original_row.get("choice_scores")),
                },
                "modified": {
                    "predicted_letter": modified_row.get("predicted_letter"),
                    "is_correct": modified_row.get("is_correct"),
                    "raw_response": modified_row.get("raw_response"),
                    "choice_scores": copy.deepcopy(modified_row.get("choice_scores")),
                },
            }
        )
    return preview


def evaluate_model_on_mmlu(model: Any, config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    normalized = normalize_mmlu_config(config)
    if not normalized["enabled"]:
        return None

    prepared = prepare_mmlu_data(normalized)
    if normalized["answer_mode"] == "logits":
        predictions = _predict_with_logits(model, prepared["entries"])
    else:
        prompts = [entry["prompt"] for entry in prepared["entries"]]
        responses = _generate_choice_responses(model, prompts, normalized["max_new_tokens"])
        predictions = _build_prediction_rows(prepared["entries"], responses)
    summary = summarize_mmlu_predictions(predictions, prepared["config_snapshot"])

    result = {
        "summary": summary,
        "prediction_preview": _build_prediction_preview(predictions),
    }
    if normalized["store_predictions"]:
        result["predictions"] = predictions
    return result


def _mmlu_cache_file(model_name: str, config_snapshot: dict[str, Any]) -> Path:
    cache_dir = RESULTS_DIR / "mmlu_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_hash = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
    config_hash = _config_hash(config_snapshot)
    return cache_dir / f"mmlu_{model_hash}_{config_hash}.json"


def get_cached_or_evaluate_original_mmlu(
    model: Any,
    *,
    model_name: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    normalized = normalize_mmlu_config(config)
    if not normalized["enabled"]:
        return None

    config_snapshot = get_mmlu_config_snapshot(normalized)
    cache_file = _mmlu_cache_file(model_name, config_snapshot)

    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    result = evaluate_model_on_mmlu(model, normalized)
    if result is None:
        return None

    payload = copy.deepcopy(result)
    payload["cached_at"] = datetime.now().isoformat()
    payload["model_name"] = model_name
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def save_mmlu_details(
    *,
    method_results_dir: Path,
    detail_prefix: str,
    original_result: dict[str, Any],
    modified_result: dict[str, Any],
) -> str:
    details_dir = method_results_dir / "mmlu"
    details_dir.mkdir(parents=True, exist_ok=True)
    filename = details_dir / f"{detail_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "original": original_result,
        "modified": modified_result,
        "saved_at": datetime.now().isoformat(),
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return _to_project_relative(filename)


def build_mmlu_result(
    *,
    config: dict[str, Any] | None,
    original_result: dict[str, Any] | None,
    modified_result: dict[str, Any] | None,
    method_results_dir: Path | None = None,
    detail_prefix: str | None = None,
) -> dict[str, Any] | None:
    normalized = normalize_mmlu_config(config)
    if not normalized["enabled"] or original_result is None or modified_result is None:
        return None

    original_summary = copy.deepcopy(original_result["summary"])
    modified_summary = copy.deepcopy(modified_result["summary"])
    details_file = None

    if normalized["store_predictions"] and method_results_dir is not None and detail_prefix:
        details_file = save_mmlu_details(
            method_results_dir=method_results_dir,
            detail_prefix=detail_prefix,
            original_result=original_result,
            modified_result=modified_result,
        )

    return {
        "original": original_summary,
        "modified": modified_summary,
        "delta_accuracy": modified_summary["accuracy"] - original_summary["accuracy"],
        "config": get_mmlu_config_snapshot(normalized),
        "answer_comparison_preview": _build_answer_comparison_preview(
            original_result.get("predictions", original_result.get("prediction_preview", [])),
            modified_result.get("predictions", modified_result.get("prediction_preview", [])),
        ),
        "details_file": details_file,
    }
