"""
Utilities for loading and processing data.
"""

import json
from pathlib import Path
from typing import List, Dict


def load_all_datasets_with_categories():
    """Load all processed datasets and combine them."""
    dataset_dir = Path(__file__).parent / 'dataset' / 'processed'

    all_data = []

    for json_file in dataset_dir.glob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            if 'category' in item:
                item['source'] = json_file.stem
                all_data.append(item)

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
