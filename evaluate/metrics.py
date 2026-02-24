"""
Batch evaluation helpers for scoring responses and measuring locality.

These functions encapsulate the repeated pattern of:
  classify question -> call judge -> collect scores
so that baselines and runners don't need to duplicate it.
"""

import numpy as np
from typing import Dict, List, Tuple

from evaluate.judges import (
    evaluate_harmfulness_with_local_judge,
    classify_question_category_with_local_llm,
)
from config import JUDGE_API_URL, CLASSIFIER_API_URL, JUDGE_MODEL, FEW_SHOTS_PATH


def evaluate_responses(
    questions: List[str],
    responses: List[str],
    classifier_categories: List[Dict],
    description: str = "",
) -> Tuple[List[int], List[Dict]]:
    """
    Evaluate harmfulness for a list of question-response pairs.

    Returns:
        Tuple of (list of integer scores, list of full score dicts)
    """
    scores: List[int] = []
    score_data_list: List[Dict] = []

    print(f"\nEvaluating {description}...")
    print(f"Total responses to evaluate: {len(responses)}")

    for i, (question, response) in enumerate(zip(questions, responses)):
        if not response:
            continue

        category_id = "unknown"
        if classifier_categories:
            try:
                pred = classify_question_category_with_local_llm(
                    question,
                    classifier_categories,
                    api_url=CLASSIFIER_API_URL,
                    model=JUDGE_MODEL,
                    temperature=0.0,
                    max_tokens=120,
                )
                category_id = pred.get("category_id", "unknown")
            except Exception as e:
                if i < 5:
                    print(f"      Classification error: {e}")

        try:
            score_data = evaluate_harmfulness_with_local_judge(
                question,
                response,
                api_key="dummy",
                api_url=JUDGE_API_URL,
                model=JUDGE_MODEL,
                category_id=category_id,
                few_shots_path=str(FEW_SHOTS_PATH) if FEW_SHOTS_PATH.exists() else None,
            )
            score = score_data.get("score")
            if score is not None:
                score_int = int(score)
                scores.append(score_int)
                score_data_list.append({
                    "question": question,
                    "response": response,
                    "score": score_int,
                    "category_id": category_id,
                    "explanation": score_data.get("explanation", ""),
                    "full_evaluation": score_data,
                })
        except Exception as e:
            if i < 5:
                print(f"      Evaluation error: {e}")

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{len(responses)} responses...")

    print(f"Got {len(scores)} scores for {description}")
    return scores, score_data_list


def evaluate_locality(
    harmless_questions: List[str],
    original_harmless_responses: List[str],
    modified_harmless_responses: List[str],
    classifier_categories: List[Dict],
) -> Tuple[List[Dict], float]:
    """
    Evaluate locality by comparing harmfulness scores on harmless questions
    before and after model modification.

    Returns:
        Tuple of (locality score dicts, average change or None)
    """
    locality_scores: List[Dict] = []

    print(f"\nEvaluating locality...")
    print(f"Total response pairs to evaluate: {len(harmless_questions)}")

    original_scores_dict: Dict[str, int] = {}
    original_evaluations_dict: Dict[str, Dict] = {}

    for i, (question, response) in enumerate(zip(harmless_questions, original_harmless_responses)):
        if not response:
            continue

        category_id = "unknown"
        if classifier_categories:
            try:
                pred = classify_question_category_with_local_llm(
                    question,
                    classifier_categories,
                    api_url=CLASSIFIER_API_URL,
                    model=JUDGE_MODEL,
                    temperature=0.0,
                    max_tokens=120,
                )
                category_id = pred.get("category_id", "unknown")
            except Exception as e:
                if i < 5:
                    print(f"      Classification error (original): {e}")

        try:
            score_data = evaluate_harmfulness_with_local_judge(
                question,
                response,
                api_key="dummy",
                api_url=JUDGE_API_URL,
                model=JUDGE_MODEL,
                category_id=category_id,
                few_shots_path=str(FEW_SHOTS_PATH) if FEW_SHOTS_PATH.exists() else None,
            )
            score = score_data.get("score")
            if score is not None:
                original_scores_dict[question] = int(score)
                original_evaluations_dict[question] = score_data
        except Exception as e:
            if i < 5:
                print(f"      Evaluation error (original): {e}")

        if (i + 1) % 10 == 0:
            print(f"  Evaluated original responses: {i + 1}/{len(harmless_questions)}...")

    for i, (question, mod_response) in enumerate(zip(harmless_questions, modified_harmless_responses)):
        if not mod_response:
            continue

        orig_score = original_scores_dict.get(question)
        if orig_score is None:
            continue

        category_id = "unknown"
        if classifier_categories:
            try:
                pred = classify_question_category_with_local_llm(
                    question,
                    classifier_categories,
                    api_url=CLASSIFIER_API_URL,
                    model=JUDGE_MODEL,
                    temperature=0.0,
                    max_tokens=120,
                )
                category_id = pred.get("category_id", "unknown")
            except Exception as e:
                if i < 5:
                    print(f"      Classification error (modified): {e}")

        try:
            mod_score_data = evaluate_harmfulness_with_local_judge(
                question,
                mod_response,
                api_key="dummy",
                api_url=JUDGE_API_URL,
                model=JUDGE_MODEL,
                category_id=category_id,
                few_shots_path=str(FEW_SHOTS_PATH) if FEW_SHOTS_PATH.exists() else None,
            )
            mod_score = mod_score_data.get("score")
            if mod_score is not None:
                mod_score_int = int(mod_score)
                difference = mod_score_int - orig_score
                question_idx = harmless_questions.index(question)
                locality_scores.append({
                    "question": question,
                    "original_response": original_harmless_responses[question_idx],
                    "modified_response": mod_response,
                    "original_score": orig_score,
                    "modified_score": mod_score_int,
                    "difference": difference,
                    "original_evaluation": original_evaluations_dict.get(question, {}),
                    "modified_evaluation": mod_score_data,
                })
        except Exception as e:
            if i < 5:
                print(f"      Evaluation error (modified): {e}")

        if (i + 1) % 10 == 0:
            print(f"  Evaluated modified responses: {i + 1}/{len(harmless_questions)}...")

    valid_differences = [s["difference"] for s in locality_scores if s["difference"] is not None]
    average_locality_change = (sum(valid_differences) / len(valid_differences)) if valid_differences else None

    print(f"Got {len(locality_scores)} score pairs for locality")
    if average_locality_change is not None:
        print(f"Average harmfulness change on harmless questions (locality): {average_locality_change:+.2f}")

    return locality_scores, average_locality_change
