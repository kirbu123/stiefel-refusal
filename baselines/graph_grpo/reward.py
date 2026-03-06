"""
Reward computation: wrapper around evaluate_harmfulness.
"""

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
    scores = []
    for question, response in zip(questions, responses):
        if not response:
            scores.append(0.0)
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
        except Exception:
            scores.append(0.0)
    return scores
