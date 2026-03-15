#!/usr/bin/env python3
"""
Скрипт для оценки ответов модели из JSON файлов.

Поддерживает два evaluator-а:
  llamaguard  — локальная модель Meta-Llama-Guard-2-8B (по умолчанию)
  deepinfra   — API DeepInfra (OpenAI-compatible)

Поддерживает два формата JSON файлов:
1. "Final" файлы (category_*_final_*.json):
   results -> param_key -> {questions, original_responses, modified_responses, ...}
2. "Answers" файлы (answers_*.json):
   harmful_questions -> {questions, original_responses, modified_responses, ...}

Использование:
    1. Впишите пути к JSON файлам в переменную FILES_TO_EVALUATE ниже
    2. Запустите: python evaluate_answers.py [--evaluator llamaguard|deepinfra] [опции]

Примеры:
    python evaluate_answers.py
    python evaluate_answers.py --evaluator deepinfra --model meta-llama/Meta-Llama-Guard-3-8B
    python evaluate_answers.py --evaluator llamaguard --cuda-device 0
"""

import argparse
import os
import sys
import json
import shutil
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# ============================================================================
# НАСТРОЙКИ - ИЗМЕНИТЕ ПОД СВОИ НУЖДЫ
# ============================================================================

# Список путей к JSON файлам для оценки
# Можно указывать как абсолютные, так и относительные пути
FILES_TO_EVALUATE = [
    "results/graph_average/answers/answers_Physical_harm_max_weight=2.5_&max_weight_position=0.7_&min_weight=0.0_&min_weight_distance=0.3_20260311_235916.json",
    "results/graph_average/answers/answers_Physical_harm_max_weight=2.5_&max_weight_position=0.7_&min_weight=1.0_&min_weight_distance=0.3_20260312_003330.json",
    "results/graph_average/answers/answers_Physical_harm_max_weight=3.0_&max_weight_position=0.7_&min_weight=0.0_&min_weight_distance=0.3_20260312_010834.json",
    "results/graph_average/answers/answers_Physical_harm_max_weight=3.0_&max_weight_position=0.7_&min_weight=1.0_&min_weight_distance=0.3_20260312_014338.json",
    # "results/graph_average/answers/graph_average_physical_harm_deepinfra_4.json"
    # "results/graph_average/answers/answers_Physical_harm_max_weight=2.5_&max_weight_position=0.7_&min_weight=1.0_&min_weight_distance=0.3_20260302_094125.json",
    # "results/graph_average/answers/answers_Physical_harm_max_weight=3.0_&max_weight_position=0.7_&min_weight=0.0_&min_weight_distance=0.3_20260302_100637.json",
    # "results/graph_average/answers/answers_Physical_harm_max_weight=3.0_&max_weight_position=0.7_&min_weight=1.0_&min_weight_distance=0.3_20260302_103254.json",
]

# Оценивать ли оригинальные ответы (до модификации)
EVALUATE_ORIGINAL = False

# Оценивать ли модифицированные ответы (после модификации)
EVALUATE_MODIFIED = True

# Пропускать ли уже оцененные файлы (если scores не пустые)
SKIP_ALREADY_EVALUATED = False

# ============================================================================

# Добавляем путь к проекту
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))


def _is_llamaguard_evaluation(eval_item: Dict) -> bool:
    """Возвращает True, если оценка от LlamaGuard (её нужно пропускать)."""
    exp = eval_item.get("explanation") or ""
    return str(exp).strip().startswith("LlamaGuard:")


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
    evaluator: Any,
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
            explanation = result.get("explanation", "")
            if _is_llamaguard_evaluation({"explanation": explanation}):
                continue  # Пропускаем оценки от LlamaGuard
            scores.append(int(score))
            score_data.append({
                "question": question,
                "response": response,
                "score": int(score),
                "unsafe_probability": result.get("unsafe_probability"),
                "label": result.get("label"),
                "violated_categories": result.get("violated_categories", []),
                "confidence": result.get("confidence"),
                "category_id": result.get("category_id"),
                "explanation": explanation,
            })

        if (i + 1) % 10 == 0 or (i + 1) == total:
            avg = np.mean(scores) if scores else 0
            print(f"    {label}: {i + 1}/{total} оценено (средний score: {avg:.2f})")

    return scores, score_data


def process_final_format(data: Dict, evaluator: Any) -> Dict:
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

        # Обновляем статистику (фильтруем оценки LlamaGuard при наличии score_data)
        orig_score_data = result_data.get("original_score_data", [])
        mod_score_data = result_data.get("modified_score_data", [])
        if orig_score_data:
            orig_scores_final = [e["score"] for e in orig_score_data if e.get("score") is not None and not _is_llamaguard_evaluation(e)]
        else:
            orig_scores_final = result_data.get("original_scores", [])
        if mod_score_data:
            mod_scores_final = [e["score"] for e in mod_score_data if e.get("score") is not None and not _is_llamaguard_evaluation(e)]
        else:
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


def process_answers_format(data: Dict, evaluator: Any) -> Dict:
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
        orig_s = [e["score"] for e in orig_evals if e.get("score") is not None and not _is_llamaguard_evaluation(e)]
        if orig_s:
            print(f"  Оригинальные: mean={np.mean(orig_s):.2f}, "
                  f"median={np.median(orig_s):.1f}, count={len(orig_s)}")
    if mod_evals:
        mod_s = [e["score"] for e in mod_evals if e.get("score") is not None and not _is_llamaguard_evaluation(e)]
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


def process_file(filepath: str, evaluator: Any) -> bool:
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Оценка ответов модели из JSON файлов с помощью LlamaGuard или DeepInfra.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--evaluator",
        choices=["llamaguard", "deepinfra"],
        default="llamaguard",
        help="Используемый evaluator.",
    )
    # LlamaGuard options
    parser.add_argument(
        "--cuda-device",
        type=str,
        default="1",
        help="CUDA_VISIBLE_DEVICES для LlamaGuard (игнорируется при --evaluator deepinfra).",
    )
    # DeepInfra options
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-Guard-3-8B",
        help="Модель DeepInfra (игнорируется при --evaluator llamaguard).",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="DEEPINFRA_API_KEY",
        help="Имя переменной окружения с API-ключом DeepInfra.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="https://api.deepinfra.com/v1/openai/chat/completions",
        help="DeepInfra OpenAI-compatible endpoint.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.5)
    parser.add_argument(
        "--few-shots-path",
        type=str,
        default="prompts/few-shots.json",
        help="Путь к JSON с few-shot примерами (для --evaluator deepinfra).",
    )
    parser.add_argument(
        "--category-id",
        type=str,
        default=None,
        help="Фиксированный category_id для оценки (для --evaluator deepinfra). "
             "По умолчанию используется первая категория из few-shots JSON.",
    )
    return parser


def main():
    args = _build_arg_parser().parse_args()

    evaluator_name = args.evaluator.upper()

    print("=" * 80)
    print(f"ОЦЕНКА ОТВЕТОВ С ПОМОЩЬЮ {evaluator_name}")
    print("=" * 80)
    print(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Evaluator: {args.evaluator}")
    if args.evaluator == "llamaguard":
        print(f"CUDA_VISIBLE_DEVICES: {args.cuda_device}")
    else:
        print(f"Модель: {args.model}")
        print(f"API URL: {args.api_url}")
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
    print()

    start_time = time.time()

    if args.evaluator == "llamaguard":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        from evaluation_llamaguard import LlamaGuardEvaluator
        evaluator: Any = LlamaGuardEvaluator()
    else:
        from evaluation_deepinfra import make_deepinfra_evaluator
        evaluator = make_deepinfra_evaluator(
            model=args.model,
            api_key_env=args.api_key_env,
            api_url=args.api_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            retry_delay=args.retry_delay,
            few_shots_path=args.few_shots_path,
            category_id=args.category_id,
        )

    load_time = time.time() - start_time
    print(f"Evaluator инициализирован за {load_time:.1f}с")

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
