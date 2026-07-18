"""Dataset preparation and aggregation for RDO locality benchmarks."""

from __future__ import annotations

import math
import random
import re
from typing import Any, Callable


PREDICTION_PREVIEW_LIMIT = 10
ARC_DATASET = "allenai/ai2_arc"
ARC_VARIANTS = {
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
}
GSM8K_DATASET = "openai/gsm8k"
WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_SUBSET = "wikitext-2-raw-v1"

_DATASET_CACHE: dict[tuple[str, str | None, str], list[dict[str, Any]]] = {}


def load_dataset_records(
    dataset: str,
    *,
    subset: str | None = None,
    split: str,
) -> list[dict[str, Any]]:
    """Load and cache a Hugging Face dataset split as plain dictionaries."""
    key = (dataset, subset, split)
    if key not in _DATASET_CACHE:
        from datasets import load_dataset

        loaded = load_dataset(dataset, subset, split=split) if subset else load_dataset(dataset, split=split)
        _DATASET_CACHE[key] = [dict(row) for row in loaded]
    return _DATASET_CACHE[key]


def sample_records(
    records: list[dict[str, Any]],
    sample_size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size is None or sample_size <= 0 or sample_size >= len(records):
        return list(records)
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    return [records[index] for index in indices[:sample_size]]


def prepare_wikitext_windows(
    tokenizer: Any,
    *,
    dataset: str = WIKITEXT_DATASET,
    subset: str = WIKITEXT_SUBSET,
    split: str = "test",
    max_length: int = 512,
    stride: int = 512,
    max_windows: int | None = 100,
) -> list[dict[str, Any]]:
    """Create sliding token windows, counting every corpus token at most once."""
    if max_length < 2:
        raise ValueError("PPL max_length must be at least 2")
    if stride < 1 or stride > max_length:
        raise ValueError("PPL stride must be in [1, max_length]")

    records = load_dataset_records(dataset, subset=subset, split=split)
    text = "\n\n".join(str(row.get("text", "")) for row in records if str(row.get("text", "")).strip())
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]

    windows: list[dict[str, Any]] = []
    previous_end = 0
    begin = 0
    while begin < len(token_ids) - 1:
        end = min(begin + max_length, len(token_ids))
        target_start = max(1, previous_end - begin)
        if end - target_start > 0:
            windows.append(
                {
                    "input_ids": list(token_ids[begin:end]),
                    "target_start": target_start,
                    "begin": begin,
                    "end": end,
                }
            )
        if end == len(token_ids) or (max_windows is not None and max_windows > 0 and len(windows) >= max_windows):
            break
        previous_end = end
        begin += stride
    return windows


def summarize_token_losses(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_tokens = sum(int(row.get("n_tokens", 0)) for row in rows)
    nll_sum = sum(float(row.get("nll_sum", 0.0)) for row in rows)
    cross_entropy = (nll_sum / n_tokens) if n_tokens else None
    perplexity = math.exp(cross_entropy) if cross_entropy is not None else None
    return {
        "perplexity": perplexity,
        "cross_entropy": cross_entropy,
        "n_tokens": n_tokens,
        "n_windows": len(rows),
    }


def _extract_arc_choices(record: dict[str, Any]) -> tuple[list[str], list[str]]:
    choices = record.get("choices")
    if not isinstance(choices, dict) or "text" not in choices:
        raise ValueError("ARC record is missing choices.text")
    texts = [str(value) for value in choices["text"]]
    labels = [str(value) for value in choices.get("label", [])]
    if not labels:
        labels = [chr(ord("A") + index) for index in range(len(texts))]
    return texts, labels


def prepare_arc_entries(
    benchmark: str,
    *,
    dataset: str = ARC_DATASET,
    split: str = "validation",
    sample_size: int | None = 100,
    sample_seed: int = 42,
) -> list[dict[str, Any]]:
    variant = ARC_VARIANTS.get(benchmark)
    if variant is None:
        raise ValueError(f"Unsupported ARC benchmark: {benchmark}")
    records = sample_records(
        load_dataset_records(dataset, subset=variant, split=split),
        sample_size,
        sample_seed,
    )
    entries = []
    for index, record in enumerate(records):
        question = str(record.get("question") or record.get("input") or record.get("prompt") or "")
        choices, labels = _extract_arc_choices(record)
        answer = str(record.get("answerKey", record.get("answer", record.get("gold", "")))).strip().upper()
        try:
            gold_index = next(i for i, label in enumerate(labels) if label.strip().upper() == answer)
        except StopIteration:
            if answer.isdigit() and 1 <= int(answer) <= len(choices):
                gold_index = int(answer) - 1
            else:
                raise ValueError(f"Unsupported ARC answer key: {answer}")
        prompt = (
            "Answer the multiple-choice question with the best answer text.\n\n"
            f"Question: {question}\n"
            "Choices:\n"
            + "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
            + "\nAnswer:"
        )
        entries.append(
            {
                "index": index,
                "variant": variant,
                "question": question,
                "prompt": prompt,
                "choices": choices,
                "choice_labels": labels,
                "gold_index": gold_index,
            }
        )
    return entries


def evaluate_arc_entries(
    entries: list[dict[str, Any]],
    score_completion: Callable[[str, str], float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = []
    for entry in entries:
        scores = []
        for choice in entry["choices"]:
            variants = (choice, f" {choice}") if choice and not choice[0].isspace() else (choice,)
            scores.append(max(float(score_completion(entry["prompt"], variant)) for variant in variants))
        predicted_index = max(range(len(scores)), key=scores.__getitem__)
        predictions.append(
            {
                "index": entry["index"],
                "variant": entry["variant"],
                "question": entry["question"],
                "gold_index": entry["gold_index"],
                "predicted_index": predicted_index,
                "is_correct": predicted_index == entry["gold_index"],
                "choice_scores": scores,
            }
        )
    correct = sum(1 for row in predictions if row["is_correct"])
    total = len(predictions)
    return {
        "accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
    }, predictions


_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_GSM_MARKER_RE = re.compile(r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"[-+]?\$?[\d,]+(?:\.\d+)?")


def normalize_numeric_text(value: str) -> str:
    normalized = value.strip().replace("$", "").replace(",", "")
    if normalized.endswith(".0"):
        normalized = normalized[:-2]
    return normalized


def extract_gsm8k_final_answer(text: str) -> str | None:
    cleaned = _THINK_RE.sub("", str(text))
    marker_matches = _GSM_MARKER_RE.findall(cleaned)
    if marker_matches:
        return normalize_numeric_text(marker_matches[-1])
    number_matches = _NUMBER_RE.findall(cleaned)
    return normalize_numeric_text(number_matches[-1]) if number_matches else None


def prepare_gsm8k_entries(
    *,
    dataset: str = GSM8K_DATASET,
    subset: str = "main",
    split: str = "test",
    sample_size: int | None = 100,
    sample_seed: int = 42,
) -> list[dict[str, Any]]:
    records = sample_records(
        load_dataset_records(dataset, subset=subset, split=split),
        sample_size,
        sample_seed,
    )
    entries = []
    for index, record in enumerate(records):
        question = str(record.get("question") or record.get("query") or record.get("prompt") or "")
        target = extract_gsm8k_final_answer(str(record.get("answer") or record.get("final_answer") or ""))
        entries.append(
            {
                "index": index,
                "question": question,
                "target_answer": target,
                "prompt": (
                    "Solve the math word problem. End your response with `#### <answer>`.\n\n"
                    f"Question: {question}\nAnswer:"
                ),
            }
        )
    return entries


def summarize_gsm8k_responses(
    entries: list[dict[str, Any]],
    responses: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = []
    for entry, response in zip(entries, responses):
        predicted = extract_gsm8k_final_answer(response)
        predictions.append(
            {
                "index": entry["index"],
                "question": entry["question"],
                "target_answer": entry["target_answer"],
                "predicted_answer": predicted,
                "is_correct": predicted is not None and predicted == entry["target_answer"],
                "raw_response": response,
            }
        )
    correct = sum(1 for row in predictions if row["is_correct"])
    total = len(predictions)
    return {
        "exact_match": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
    }, predictions


def build_metric_block(
    *,
    config: dict[str, Any],
    metric_name: str,
    initial: dict[str, Any],
    refined: dict[str, Any] | None,
    initial_predictions: list[dict[str, Any]] | None = None,
    refined_predictions: list[dict[str, Any]] | None = None,
    store_predictions: bool = False,
) -> dict[str, Any]:
    initial_value = initial.get(metric_name)
    refined_value = refined.get(metric_name) if refined else None
    block: dict[str, Any] = {
        "config": config,
        "metric_name": metric_name,
        "initial": initial,
        "refined": refined,
        "delta": (refined_value - initial_value) if initial_value is not None and refined_value is not None else None,
    }
    if initial_predictions is not None:
        block["preview"] = {
            "initial": initial_predictions[:PREDICTION_PREVIEW_LIMIT],
            "refined": refined_predictions[:PREDICTION_PREVIEW_LIMIT] if refined_predictions is not None else None,
        }
        if store_predictions:
            block["predictions"] = {
                "initial": initial_predictions,
                "refined": refined_predictions,
            }
    return block


def metrics_to_csv_rows(locality_metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten locality result blocks into the shared eval CSV schema."""
    rows: list[dict[str, Any]] = []
    for benchmark, block in locality_metrics.items():
        metric_name = block.get("metric_name")
        for phase in ("initial", "refined"):
            for stat_name, stat_value in (block.get(phase) or {}).items():
                if not isinstance(stat_value, (int, float)) or isinstance(stat_value, bool):
                    continue
                rows.append(
                    {
                        "benchmark": benchmark,
                        "backend": "",
                        "group": "",
                        "phase": phase,
                        "metric": stat_name,
                        "value": stat_value,
                    }
                )
        if metric_name and block.get("delta") is not None:
            rows.append(
                {
                    "benchmark": benchmark,
                    "backend": "",
                    "group": "",
                    "phase": "delta",
                    "metric": metric_name,
                    "value": block["delta"],
                }
            )
    return rows
