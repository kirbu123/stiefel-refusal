"""
Utilities for loading and processing data.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


SUPPORTED_CATEGORY_DATASET_SOURCES = (
    "combined",
    "jailbreakbench",
)

SUPPORTED_JAILBREAKBENCH_CATEGORIES = (
    "Harassment/Discrimination",
    "Malware/Hacking",
    "Physical harm",
    "Economic harm",
    "Fraud/Deception",
    "Disinformation",
    "Sexual/Adult content",
    "Privacy",
    "Expert advice",
    "Government decision-making",
)

SUPPORTED_HARMBENCH_FUNCTIONAL_CATEGORIES = (
    "contextual",
    "copyright",
    "standard",
)

SUPPORTED_HARMBENCH_SEMANTIC_CATEGORIES = (
    "chemical_biological",
    "copyright",
    "cybercrime_intrusion",
    "harassment_bullying",
    "harmful",
    "illegal",
    "misinformation_disinformation",
)


def _load_combined_datasets_with_categories() -> Tuple[List[Dict], List[str]]:
    """Load all processed datasets and combine them."""
    from config import CATEGORIES

    dataset_dir = Path(__file__).parent / "dataset" / "processed"
    all_data = []

    for json_file in sorted(dataset_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            if "category" in item:
                combined_item = dict(item)
                combined_item["source"] = json_file.stem
                all_data.append(combined_item)

    return all_data, list(CATEGORIES)


def _validate_category_filter(category_filter: str | None) -> str | None:
    if category_filter is None:
        return None

    normalized = category_filter.strip()
    if not normalized:
        return None

    if normalized not in SUPPORTED_JAILBREAKBENCH_CATEGORIES:
        valid_values = ", ".join(SUPPORTED_JAILBREAKBENCH_CATEGORIES)
        raise ValueError(
            f"Unsupported JailbreakBench category '{normalized}'. "
            f"Valid values: {valid_values}"
        )

    return normalized


def _load_jailbreakbench_datasets(
    category_filter: str | None = None,
) -> Tuple[List[Dict], List[str]]:
    dataset_dir = Path(__file__).parent / "dataset" / "raw"
    filter_value = _validate_category_filter(category_filter)
    all_data = []

    csv_file = dataset_dir / "jailbreakbench.csv"
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = (row.get("Category") or "").strip()
            if not category:
                continue
            if filter_value is not None and category != filter_value:
                continue

            instruction = (row.get("Goal") or "").strip()
            if not instruction:
                continue

            all_data.append(
                {
                    "instruction": instruction,
                    "category": category,
                    "source": "jailbreakbench",
                }
            )

    categories = [
        category for category in SUPPORTED_JAILBREAKBENCH_CATEGORIES
        if filter_value is None or category == filter_value
    ]
    return all_data, categories


def load_datasets_with_categories(
    source: str = "combined",
    category_filter: str | None = None,
) -> Tuple[List[Dict], List[str]]:
    normalized_source = (source or "combined").strip()
    if normalized_source not in SUPPORTED_CATEGORY_DATASET_SOURCES:
        valid_sources = ", ".join(SUPPORTED_CATEGORY_DATASET_SOURCES)
        raise ValueError(
            f"Unsupported category dataset source '{normalized_source}'. "
            f"Valid values: {valid_sources}"
        )

    if normalized_source != "jailbreakbench" and category_filter:
        raise ValueError(
            "Category filtering is only supported with "
            "source='jailbreakbench'."
        )

    if normalized_source == "jailbreakbench":
        return _load_jailbreakbench_datasets(category_filter)

    return _load_combined_datasets_with_categories()


def load_all_datasets_with_categories():
    """Backward-compatible wrapper for the default combined dataset source."""
    all_data, _ = load_datasets_with_categories()
    return all_data


def extract_response_after_think(response: str) -> str:
    """
    Extract the part of the response after the </think> tag.

    Args:
        response: Full model response

    Returns:
        Part of response after </think> with leading whitespace stripped
    """
    parts = response.split('</think>')
    if len(parts) > 1:
        return parts[1].strip()
    return response.strip()
