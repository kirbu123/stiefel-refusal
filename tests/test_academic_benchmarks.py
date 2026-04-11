import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import evaluate.academic_benchmarks as academic


class TestAcademicBenchmarks(unittest.TestCase):
    def setUp(self):
        academic._DATASET_RECORDS_CACHE.clear()

    def test_normalize_academic_benchmarks_defaults(self):
        normalized = academic.normalize_academic_benchmarks_config()

        self.assertEqual(
            normalized["enabled"],
            ["tinyhellaswag", "arc", "winogrande", "gsm8k", "truthfulqa"],
        )
        self.assertEqual(normalized["sample_seed"], 42)
        self.assertTrue(normalized["store_predictions"])
        self.assertEqual(normalized["tinyhellaswag"]["sample_size"], 100)
        self.assertEqual(normalized["arc"]["split"], "validation")
        self.assertEqual(normalized["winogrande"]["subset"], "winogrande_xl")
        self.assertEqual(normalized["gsm8k"]["max_new_tokens"], 512)
        self.assertEqual(normalized["truthfulqa"]["subset"], "multiple_choice")

    def test_extract_gsm8k_final_answer_uses_marker_and_fallback(self):
        self.assertEqual(
            academic.extract_gsm8k_final_answer("<think>calc</think>\n#### 1,234"),
            "1234",
        )
        self.assertEqual(
            academic.extract_gsm8k_final_answer("Reasoning...\nThe answer is 56."),
            "56",
        )
        self.assertIsNone(academic.extract_gsm8k_final_answer("No numeric answer"))

    def test_build_academic_benchmarks_result_saves_details(self):
        original_results = {
            "gsm8k": {
                "summary": {
                    "primary_metric_name": "exact_match",
                    "primary_metric_value": 0.2,
                    "exact_match": 0.2,
                },
                "predictions": [{"index": 0}],
            }
        }
        modified_results = {
            "gsm8k": {
                "summary": {
                    "primary_metric_name": "exact_match",
                    "primary_metric_value": 0.4,
                    "exact_match": 0.4,
                },
                "predictions": [{"index": 1}],
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            block = academic.build_academic_benchmarks_result(
                config={
                    "enabled": ["gsm8k"],
                    "store_predictions": True,
                    "gsm8k": {"sample_size": 5},
                },
                original_results=original_results,
                modified_results=modified_results,
                method_results_dir=Path(tmpdir),
                detail_prefix="graph_grpo_test",
            )

        self.assertIn("gsm8k", block)
        self.assertAlmostEqual(block["gsm8k"]["delta_primary_metric"], 0.2)
        self.assertIsNotNone(block["gsm8k"]["details_file"])
        self.assertEqual(block["gsm8k"]["config"]["sample_size"], 5)

    def test_original_academic_results_use_cache(self):
        fake_result = {
            "summary": {
                "primary_metric_name": "accuracy",
                "primary_metric_value": 0.5,
                "accuracy": 0.5,
            },
            "prediction_preview": [],
        }

        evaluator_calls = []

        def fake_evaluator(model, config):
            evaluator_calls.append("called")
            return fake_result

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(academic, "get_method_results_dir", return_value=Path(tmpdir)):
                with patch.dict(academic._EVALUATORS, {"winogrande": fake_evaluator}, clear=False):
                    config = {
                        "enabled": ["winogrande"],
                        "store_predictions": False,
                        "winogrande": {"sample_size": 3},
                    }
                    first = academic.get_cached_or_evaluate_original_academic_benchmarks(
                        object(),
                        model_name="model-a",
                        config=config,
                    )
                    second = academic.get_cached_or_evaluate_original_academic_benchmarks(
                        object(),
                        model_name="model-a",
                        config=config,
                    )

        self.assertEqual(first["winogrande"]["summary"]["accuracy"], 0.5)
        self.assertEqual(second["winogrande"]["summary"]["accuracy"], 0.5)
        self.assertEqual(len(evaluator_calls), 1)

    def test_evaluate_tinyhellaswag_uses_gpirt_as_primary_metric(self):
        records = [
            {
                "input_formatted": "A person starts cooking.",
                "endings": [
                    "They plate the food.",
                    "They fly to the moon.",
                    "They paint the wall.",
                    "They water a cactus.",
                ],
                "label": 0,
            }
        ]

        with patch.object(academic, "_load_dataset_records", return_value=records):
            with patch.object(academic, "_score_text_choices", return_value=(0, [0.0, -1.0, -2.0, -3.0])):
                with patch.object(
                    academic,
                    "_evaluate_tinybenchmarks_score_vector",
                    return_value={"gpirt": 0.63, "irt": 0.58, "pirt": 0.6},
                ):
                    result = academic._evaluate_tinyhellaswag(
                        object(),
                        {
                            "enabled": ["tinyhellaswag"],
                            "tinyhellaswag": {"sample_size": 1},
                        },
                    )

        self.assertAlmostEqual(result["summary"]["irt_plus_plus"], 0.63)
        self.assertEqual(result["summary"]["primary_metric_name"], "irt_plus_plus")
        self.assertAlmostEqual(result["summary"]["accuracy"], 1.0)

    def test_tinyhellaswag_requires_optional_tinybenchmarks_dependency(self):
        with patch.object(academic.importlib, "import_module", side_effect=ImportError):
            with self.assertRaisesRegex(ImportError, "tinyBenchmarks"):
                academic._evaluate_tinybenchmarks_score_vector([1, 0, 1], "hellaswag")

    def test_tinybenchmarks_score_vector_is_numpy_array(self):
        captured = {}
        fake_tinybenchmarks = types.SimpleNamespace()

        def fake_evaluate(score_vector, benchmark):
            captured["score_vector"] = score_vector
            captured["benchmark"] = benchmark
            return {"hellaswag": {"gpirt": 0.5}}

        fake_tinybenchmarks.evaluate = fake_evaluate

        with patch.object(academic.importlib, "import_module", return_value=fake_tinybenchmarks):
            result = academic._evaluate_tinybenchmarks_score_vector([1, 0, 1], "hellaswag")

        self.assertEqual(result["gpirt"], 0.5)
        self.assertEqual(captured["benchmark"], "hellaswag")
        self.assertIsInstance(captured["score_vector"], np.ndarray)
        self.assertEqual(captured["score_vector"].shape, (3,))

    def test_evaluate_arc_aggregates_variants(self):
        arc_easy_records = [
            {
                "question": "Easy question",
                "choices": {"text": ["A1", "A2", "A3", "A4"], "label": ["A", "B", "C", "D"]},
                "answerKey": "A",
            }
        ]
        arc_challenge_records = [
            {
                "question": "Challenge question",
                "choices": {"text": ["C1", "C2", "C3", "C4"], "label": ["A", "B", "C", "D"]},
                "answerKey": "B",
            }
        ]

        def fake_load_dataset_records(dataset_name, *, subset=None, split):
            return arc_easy_records if subset == "ARC-Easy" else arc_challenge_records

        def fake_score_text_choices(model, prompt, choices):
            if "Easy question" in prompt:
                return 0, [0.0, -1.0, -2.0, -3.0]
            return 2, [-1.0, -0.1, 0.0, -2.0]

        with patch.object(academic, "_load_dataset_records", side_effect=fake_load_dataset_records):
            with patch.object(academic, "_score_text_choices", side_effect=fake_score_text_choices):
                result = academic._evaluate_arc(
                    object(),
                    {
                        "enabled": ["arc"],
                        "arc": {"sample_size": 1},
                    },
                )

        self.assertAlmostEqual(result["summary"]["by_variant"]["ARC-Easy"]["accuracy"], 1.0)
        self.assertAlmostEqual(result["summary"]["by_variant"]["ARC-Challenge"]["accuracy"], 0.0)
        self.assertAlmostEqual(result["summary"]["macro_accuracy"], 0.5)
        self.assertEqual(result["summary"]["primary_metric_name"], "macro_accuracy")

    def test_evaluate_winogrande_scores_completed_sentences(self):
        records = [
            {
                "sentence": "The trophy doesn't fit in the suitcase because _ is too large.",
                "option1": "the trophy",
                "option2": "the suitcase",
                "answer": "1",
            }
        ]

        with patch.object(academic, "_load_dataset_records", return_value=records):
            with patch.object(
                academic,
                "_score_text_choices",
                return_value=(0, [0.0, -1.0]),
            ):
                result = academic._evaluate_winogrande(
                    object(),
                    {
                        "enabled": ["winogrande"],
                        "winogrande": {"sample_size": 1},
                    },
                )

        self.assertAlmostEqual(result["summary"]["accuracy"], 1.0)
        preview = result["prediction_preview"][0]
        self.assertIn("the trophy", preview["completed_options"][0])

    def test_evaluate_gsm8k_uses_final_answer_parsing(self):
        records = [
            {
                "question": "If Alice has 2 apples and gets 3 more, how many apples does she have?",
                "answer": "#### 5",
            }
        ]

        with patch.object(academic, "_load_dataset_records", return_value=records):
            with patch.object(
                academic,
                "generate_responses",
                return_value=["Let's think.\n#### 5"],
            ):
                result = academic._evaluate_gsm8k(
                    object(),
                    {
                        "enabled": ["gsm8k"],
                        "gsm8k": {"sample_size": 1, "max_new_tokens": 32},
                    },
                )

        self.assertAlmostEqual(result["summary"]["exact_match"], 1.0)
        self.assertEqual(result["prediction_preview"][0]["predicted_answer"], "5")

    def test_evaluate_truthfulqa_reports_mc1_and_mc2(self):
        records = [
            {
                "question": "What color is the sun from space?",
                "mc1_targets": {
                    "choices": ["The sun is white.", "The sun is yellow."],
                    "labels": [1, 0],
                },
                "mc2_targets": {
                    "choices": ["The sun is white.", "The sun is yellow.", "The sun is blue."],
                    "labels": [1, 0, 0],
                },
            }
        ]

        score_calls = iter(
            [
                (0, [0.0, -2.0]),
                (0, [0.0, -1.0, -2.0]),
            ]
        )

        with patch.object(academic, "_load_dataset_records", return_value=records):
            with patch.object(
                academic,
                "_score_text_choices",
                side_effect=lambda model, prompt, choices: next(score_calls),
            ):
                result = academic._evaluate_truthfulqa(
                    object(),
                    {
                        "enabled": ["truthfulqa"],
                        "truthfulqa": {"sample_size": 1},
                    },
                )

        self.assertAlmostEqual(result["summary"]["mc1"], 1.0)
        self.assertGreater(result["summary"]["mc2"], 0.0)
        self.assertEqual(result["summary"]["primary_metric_name"], "mc1")


if __name__ == "__main__":
    unittest.main()
