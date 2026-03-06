#!/usr/bin/env python3
"""
Скрипт для оценки ответов модели из JSON файлов с помощью LlamaGuard.

Поддерживает два формата JSON файлов:
1. "Final" файлы (category_*_final_*.json):
   results -> param_key -> {questions, original_responses, modified_responses, ...}
2. "Answers" файлы (answers_*.json):
   harmful_questions -> {questions, original_responses, modified_responses, ...}

Использование:
    1. Впишите пути к JSON файлам в переменную FILES_TO_EVALUATE ниже
    2. Запустите: python evaluate_answers.py
"""

import os
import sys
import json
import shutil
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# ============================================================================
# НАСТРОЙКИ - ИЗМЕНИТЕ ПОД СВОИ НУЖДЫ
# ============================================================================

# GPU для LlamaGuard
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Список путей к JSON файлам для оценки
# Можно указывать как абсолютные, так и относительные пути
FILES_TO_EVALUATE = [
    "results/graph_average/answers/answers_Physical_harm_max_weight=2.5_&max_weight_position=0.7_&min_weight=0.0_&min_weight_distance=0.3_20260302_091612.json",
    "results/graph_average/answers/answers_Physical_harm_max_weight=2.5_&max_weight_position=0.7_&min_weight=1.0_&min_weight_distance=0.3_20260302_094125.json",
    "results/graph_average/answers/answers_Physical_harm_max_weight=3.0_&max_weight_position=0.7_&min_weight=0.0_&min_weight_distance=0.3_20260302_100637.json",
    "results/graph_average/answers/answers_Physical_harm_max_weight=3.0_&max_weight_position=0.7_&min_weight=1.0_&min_weight_distance=0.3_20260302_103254.json",
]

# Оценивать ли оригинальные ответы (до модификации)
EVALUATE_ORIGINAL = True

# Оценивать ли модифицированные ответы (после модификации)
EVALUATE_MODIFIED = True

# Пропускать ли уже оцененные файлы (если scores не пустые)
SKIP_ALREADY_EVALUATED = True

# ============================================================================

# Добавляем путь к проекту
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from evaluation_llamaguard import LlamaGuardEvaluator


def compute_score_statistics(scores: List[int]) -> Dict:
    """Вычисляет статистику по оценкам."""
    if not scores:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "count": 0,
        }
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "min": int(np.min(scores)),
        "max": int(np.max(scores)),
        "count": len(scores),
    }


def evaluate_responses(
    evaluator: LlamaGuardEvaluator,
    questions: List[str],
    responses: List[str],
    label: str = "",
) -> tuple:
    """Оценивает список пар (вопрос, ответ).

    Returns:
        (scores, score_data): списки оценок и полных данных оценки
    """
    scores = []
    score_data = []
    total = len(questions)

    for i, (question, response) in enumerate(zip(questions, responses)):
        if not response:
            continue

        result = evaluator.evaluate(question, response)
        score = result.get("score")

        if score is not None:
            scores.append(int(score))
            score_data.append({
                "question": question,
                "response": response,
                "score": int(score),
                "unsafe_probability": result.get("unsafe_probability"),
                "label": result.get("label"),
                "violated_categories": result.get("violated_categories", []),
                "explanation": result.get("explanation", ""),
            })

        if (i + 1) % 10 == 0 or (i + 1) == total:
            avg = np.mean(scores) if scores else 0
            print(f"    {label}: {i + 1}/{total} оценено (средний score: {avg:.2f})")

    return scores, score_data


def process_final_format(data: Dict, evaluator: LlamaGuardEvaluator) -> Dict:
    """Обработка файла в формате 'final' (category_*_final_*.json).

    Структура: results -> param_key -> {questions, original_responses, modified_responses, ...}
    """
    results = data.get("results", {})
    total_params = len(results)

    for param_idx, (param_key, result_data) in enumerate(results.items(), 1):
        print(f"\n  Параметры {param_idx}/{total_params}: {param_key}")

        questions = result_data.get("questions", [])
        original_responses = result_data.get("original_responses", [])
        modified_responses = result_data.get("modified_responses", [])

        if not questions:
            print(f"    Пропуск: нет вопросов")
            continue

        # Проверяем, не оценены ли уже
        existing_orig_scores = result_data.get("original_scores", [])
        existing_mod_scores = result_data.get("modified_scores", [])

        # Оцениваем оригинальные ответы
        if EVALUATE_ORIGINAL and original_responses:
            if SKIP_ALREADY_EVALUATED and existing_orig_scores:
                print(f"    Оригинальные ответы уже оценены ({len(existing_orig_scores)} оценок), пропуск")
            else:
                print(f"    Оценка оригинальных ответов ({len(original_responses)} шт.)...")
                orig_scores, orig_score_data = evaluate_responses(
                    evaluator, questions, original_responses, label="original"
                )
                result_data["original_scores"] = orig_scores
                result_data["original_score_data"] = orig_score_data

        # Оцениваем модифицированные ответы
        if EVALUATE_MODIFIED and modified_responses:
            if SKIP_ALREADY_EVALUATED and existing_mod_scores:
                print(f"    Модифицированные ответы уже оценены ({len(existing_mod_scores)} оценок), пропуск")
            else:
                print(f"    Оценка модифицированных ответов ({len(modified_responses)} шт.)...")
                mod_scores, mod_score_data = evaluate_responses(
                    evaluator, questions, modified_responses, label="modified"
                )
                result_data["modified_scores"] = mod_scores
                result_data["modified_score_data"] = mod_score_data

        # Обновляем статистику
        orig_scores_final = result_data.get("original_scores", [])
        mod_scores_final = result_data.get("modified_scores", [])

        result_data["score_statistics"] = {
            "original_mean": float(np.mean(orig_scores_final)) if orig_scores_final else None,
            "original_median": float(np.median(orig_scores_final)) if orig_scores_final else None,
            "modified_mean": float(np.mean(mod_scores_final)) if mod_scores_final else None,
            "modified_median": float(np.median(mod_scores_final)) if mod_scores_final else None,
            "original_count": len(orig_scores_final),
            "modified_count": len(mod_scores_final),
        }

        if orig_scores_final:
            print(f"    Оригинальные: mean={np.mean(orig_scores_final):.2f}, "
                  f"median={np.median(orig_scores_final):.1f}, count={len(orig_scores_final)}")
        if mod_scores_final:
            print(f"    Модифицированные: mean={np.mean(mod_scores_final):.2f}, "
                  f"median={np.median(mod_scores_final):.1f}, count={len(mod_scores_final)}")

    return data


def process_answers_format(data: Dict, evaluator: LlamaGuardEvaluator) -> Dict:
    """Обработка файла в формате 'answers' (answers_*.json).

    Структура: harmful_questions -> {questions, original_responses, modified_responses, ...}
    """
    harmful = data.get("harmful_questions", {})

    questions = harmful.get("questions", [])
    original_responses = harmful.get("original_responses", [])
    modified_responses = harmful.get("modified_responses", [])

    if not questions:
        print("  Пропуск: нет вопросов в harmful_questions")
        return data

    existing_orig_evals = harmful.get("original_evaluations", [])
    existing_mod_evals = harmful.get("modified_evaluations", [])

    # Оцениваем оригинальные ответы
    if EVALUATE_ORIGINAL and original_responses:
        if SKIP_ALREADY_EVALUATED and existing_orig_evals:
            print(f"  Оригинальные ответы уже оценены ({len(existing_orig_evals)} оценок), пропуск")
        else:
            print(f"  Оценка оригинальных ответов ({len(original_responses)} шт.)...")
            orig_scores, orig_score_data = evaluate_responses(
                evaluator, questions, original_responses, label="original"
            )
            harmful["original_evaluations"] = orig_score_data

    # Оцениваем модифицированные ответы
    if EVALUATE_MODIFIED and modified_responses:
        if SKIP_ALREADY_EVALUATED and existing_mod_evals:
            print(f"  Модифицированные ответы уже оценены ({len(existing_mod_evals)} оценок), пропуск")
        else:
            print(f"  Оценка модифицированных ответов ({len(modified_responses)} шт.)...")
            mod_scores, mod_score_data = evaluate_responses(
                evaluator, questions, modified_responses, label="modified"
            )
            harmful["modified_evaluations"] = mod_score_data

    # Выводим статистику
    orig_evals = harmful.get("original_evaluations", [])
    mod_evals = harmful.get("modified_evaluations", [])

    if orig_evals:
        orig_s = [e["score"] for e in orig_evals if e.get("score") is not None]
        if orig_s:
            print(f"  Оригинальные: mean={np.mean(orig_s):.2f}, "
                  f"median={np.median(orig_s):.1f}, count={len(orig_s)}")
    if mod_evals:
        mod_s = [e["score"] for e in mod_evals if e.get("score") is not None]
        if mod_s:
            print(f"  Модифицированные: mean={np.mean(mod_s):.2f}, "
                  f"median={np.median(mod_s):.1f}, count={len(mod_s)}")

    return data


def detect_format(data: Dict) -> str:
    """Определяет формат JSON файла.

    Returns:
        "final" или "answers"
    """
    if "results" in data:
        return "final"
    elif "harmful_questions" in data:
        return "answers"
    else:
        return "unknown"


def process_file(filepath: str, evaluator: LlamaGuardEvaluator) -> bool:
    """Обрабатывает один JSON файл.

    Returns:
        True если файл успешно обработан, False иначе
    """
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"ОШИБКА: Файл не найден: {filepath}")
        return False

    print(f"\nЗагрузка файла: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"ОШИБКА при загрузке JSON: {e}")
        return False

    fmt = detect_format(data)
    print(f"  Формат: {fmt}")

    if fmt == "final":
        n_params = len(data.get("results", {}))
        print(f"  Количество комбинаций параметров: {n_params}")
        data = process_final_format(data, evaluator)
    elif fmt == "answers":
        n_questions = len(data.get("harmful_questions", {}).get("questions", []))
        print(f"  Количество вопросов: {n_questions}")
        data = process_answers_format(data, evaluator)
    else:
        print(f"ОШИБКА: Неизвестный формат файла")
        return False

    # Создаем backup
    backup_path = filepath.with_suffix(".bak")
    try:
        shutil.copy2(filepath, backup_path)
        print(f"  Backup создан: {backup_path}")
    except Exception as e:
        print(f"  Предупреждение: не удалось создать backup: {e}")

    # Сохраняем результаты
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Результаты сохранены в: {filepath}")
    except Exception as e:
        print(f"ОШИБКА при сохранении: {e}")
        return False

    return True


def main():
    print("=" * 80)
    print("ОЦЕНКА ОТВЕТОВ С ПОМОЩЬЮ LLAMAGUARD")
    print("=" * 80)
    print(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'не задано')}")
    print(f"Оценивать оригинальные: {EVALUATE_ORIGINAL}")
    print(f"Оценивать модифицированные: {EVALUATE_MODIFIED}")
    print(f"Пропускать уже оцененные: {SKIP_ALREADY_EVALUATED}")
    print()

    if not FILES_TO_EVALUATE:
        print("ОШИБКА: Список FILES_TO_EVALUATE пуст!")
        print("Впишите пути к JSON файлам в переменную FILES_TO_EVALUATE в начале скрипта.")
        return

    # Проверяем существование файлов
    valid_files = []
    for filepath in FILES_TO_EVALUATE:
        p = Path(filepath)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            valid_files.append(str(p))
            print(f"  [OK] {filepath}")
        else:
            print(f"  [НЕ НАЙДЕН] {filepath}")

    if not valid_files:
        print("\nОШИБКА: Ни один файл не найден!")
        return

    print(f"\nФайлов для обработки: {len(valid_files)}")

    # Загружаем LlamaGuard
    print()
    start_time = time.time()
    evaluator = LlamaGuardEvaluator()
    load_time = time.time() - start_time
    print(f"Модель загружена за {load_time:.1f}с")

    # Обрабатываем файлы
    success_count = 0
    for file_idx, filepath in enumerate(valid_files, 1):
        print(f"\n{'=' * 80}")
        print(f"ФАЙЛ {file_idx}/{len(valid_files)}")
        print(f"{'=' * 80}")

        file_start = time.time()
        success = process_file(filepath, evaluator)
        file_time = time.time() - file_start

        if success:
            success_count += 1
            print(f"  Время обработки: {file_time:.1f}с")
        else:
            print(f"  Файл не обработан")

    # Итоги
    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print("ОЦЕНКА ЗАВЕРШЕНА")
    print(f"{'=' * 80}")
    print(f"Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Общее время: {total_time:.1f}с")
    print(f"Успешно обработано: {success_count}/{len(valid_files)} файлов")


if __name__ == "__main__":
    main()
