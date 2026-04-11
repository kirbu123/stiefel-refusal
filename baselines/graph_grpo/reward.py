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
    reward_metric: str = "harmfulness",
) -> List[float]:
    """
    Batch evaluation using evaluate_harmfulness.

    Args:
        questions: List of question strings
        responses: List of response strings (same length as questions)
        classifier_categories: Optional categories for local_llm_judge
        backend: Evaluation backend (local_llm_judge, llamaguard, etc.)
        reward_metric: Metric to extract from evaluator output:
            harmfulness uses the numeric score, llamaguard_unsafe uses
            1.0 for LlamaGuard unsafe labels and 0.0 otherwise

    Returns:
        List of float scores on the active reward metric scale
    """
    normalized_metric = reward_metric.strip().lower()
    if normalized_metric not in {"harmfulness", "llamaguard_unsafe"}:
        raise ValueError(
            f"Unsupported reward metric '{reward_metric}'. "
            "Expected one of: harmfulness, llamaguard_unsafe."
        )

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
            if normalized_metric == "llamaguard_unsafe":
                label = str(result.get("label", "")).strip().lower()
                scores.append(1.0 if label == "unsafe" else 0.0)
            else:
                score = result.get("score")
                scores.append(float(score) if score is not None else 0.0)
        except Exception:
            scores.append(0.0)
            error_count += 1
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            elapsed = time.time() - t0
            print(f"      [reward] {idx+1}/{total} evaluated ({elapsed:.1f}s)", flush=True)

    if empty_count or error_count:
        print(f"      [reward] warnings: {empty_count} empty responses, {error_count} errors")
    return scores
