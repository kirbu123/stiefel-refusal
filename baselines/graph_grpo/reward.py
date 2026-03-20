"""
Reward computation: wrapper around evaluate_harmfulness.
"""

import time
from typing import Dict, List, Optional

from evaluate.judges import evaluate_harmfulness


def compute_reward(
    questions: List[str],
    responses: List[str],
    classifier_categories: Optional[List[Dict]] = None,
    backend: Optional[str] = None,
) -> List[float]:
    """
    Batch evaluation using evaluate_harmfulness.

    Args:
        questions: List of question strings
        responses: List of response strings (same length as questions)
        classifier_categories: Optional categories for local_llm_judge
        backend: Evaluation backend (local_llm_judge, llamaguard, etc.)

    Returns:
        List of float scores (0-4 harmfulness scale)
    """
    t0 = time.time()
    total = len(questions)
    scores = []
    empty_count = 0
    error_count = 0
    for idx, (question, response) in enumerate(zip(questions, responses)):
        if not response:
            scores.append(0.0)
            empty_count += 1
            continue
        try:
            result = evaluate_harmfulness(
                question,
                response,
                classifier_categories=classifier_categories,
                backend=backend,
            )
            score = result.get("score")
            scores.append(float(score) if score is not None else 0.0)
        except Exception as e:
            scores.append(0.0)
            error_count += 1
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            elapsed = time.time() - t0
            print(f"      [reward] {idx+1}/{total} evaluated ({elapsed:.1f}s)", flush=True)

    if empty_count or error_count:
        print(f"      [reward] warnings: {empty_count} empty responses, {error_count} errors")
    return scores
