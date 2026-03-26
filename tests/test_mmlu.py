import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluate.mmlu as mmlu


class FakeDataset:
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def make_record(index, subject="math", answer=0):
    return {
        "subject": subject,
        "question": f"Question {index}?",
        "choices": [f"choice {index}A", f"choice {index}B", f"choice {index}C", f"choice {index}D"],
        "answer": answer,
    }


class TestMMLU(unittest.TestCase):
    def setUp(self):
        mmlu._DATASET_RECORDS_CACHE.clear()
        mmlu._PREPARED_DATA_CACHE.clear()

    def test_parse_choice_letter(self):
        self.assertEqual(mmlu.parse_choice_letter("A"), "A")
        self.assertEqual(mmlu.parse_choice_letter("B."), "B")
        self.assertEqual(mmlu.parse_choice_letter("Answer: c"), "C")
        self.assertEqual(mmlu.parse_choice_letter("<think>reasoning</think>\nD"), "D")
        self.assertIsNone(mmlu.parse_choice_letter("I am not sure"))

    def test_build_prompt_variants(self):
        record = make_record(1, subject="history", answer=2)
        shot = make_record(0, subject="history", answer=1)

        zero_shot_prompt = mmlu.build_mmlu_prompt(record, mode="zero_shot")
        few_shot_prompt = mmlu.build_mmlu_prompt(record, mode="few_shot", few_shot_examples=[shot])

        self.assertIn("Respond with only the letter A, B, C, or D.", zero_shot_prompt)
        self.assertNotIn("(with answers)", zero_shot_prompt)
        self.assertIn("(with answers)", few_shot_prompt)
        self.assertIn("Answer: B", few_shot_prompt)
        self.assertTrue(few_shot_prompt.strip().endswith("Answer:"))

    def test_prepare_mmlu_data_sampling_is_deterministic(self):
        eval_records = [make_record(i, subject="math", answer=i % 4) for i in range(10)]

        with patch.object(mmlu, "_load_dataset_records", return_value=eval_records):
            config = {
                "enabled": True,
                "dataset": "cais/mmlu",
                "subset": "all",
                "split": "test",
                "mode": "zero_shot",
                "sample_size": 4,
                "sample_seed": 7,
            }
            prepared_one = mmlu.prepare_mmlu_data(config)
            mmlu._PREPARED_DATA_CACHE.clear()
            prepared_two = mmlu.prepare_mmlu_data(config)

        questions_one = [entry["question"] for entry in prepared_one["entries"]]
        questions_two = [entry["question"] for entry in prepared_two["entries"]]
        self.assertEqual(questions_one, questions_two)
        self.assertEqual(len(questions_one), 4)

    def test_summarize_predictions_by_subject(self):
        predictions = [
            {"subject": "math", "predicted_letter": "A", "correct_letter": "A", "is_correct": True},
            {"subject": "math", "predicted_letter": None, "correct_letter": "B", "is_correct": False},
            {"subject": "history", "predicted_letter": "C", "correct_letter": "D", "is_correct": False},
        ]

        summary = mmlu.summarize_mmlu_predictions(predictions, {"subset": "all"})

        self.assertAlmostEqual(summary["accuracy"], 1 / 3)
        self.assertEqual(summary["invalid_predictions"], 1)
        self.assertEqual(summary["by_subject"]["math"]["correct"], 1)
        self.assertEqual(summary["by_subject"]["math"]["invalid_predictions"], 1)
        self.assertAlmostEqual(summary["by_subject"]["history"]["accuracy"], 0.0)

    def test_original_mmlu_uses_cache(self):
        fake_result = {"summary": {"accuracy": 0.5, "correct": 1, "total": 2, "invalid_predictions": 0, "by_subject": {}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mmlu, "RESULTS_DIR", Path(tmpdir)):
                with patch.object(mmlu, "evaluate_model_on_mmlu", return_value=fake_result) as evaluate_mock:
                    config = {"enabled": True, "store_predictions": False}
                    first = mmlu.get_cached_or_evaluate_original_mmlu(object(), model_name="model-a", config=config)
                    second = mmlu.get_cached_or_evaluate_original_mmlu(object(), model_name="model-a", config=config)

        self.assertEqual(first["summary"]["accuracy"], 0.5)
        self.assertEqual(second["summary"]["accuracy"], 0.5)
        self.assertEqual(evaluate_mock.call_count, 1)

    def test_build_mmlu_result_saves_details(self):
        original_result = {
            "summary": {"accuracy": 0.4, "correct": 2, "total": 5, "invalid_predictions": 0, "by_subject": {}},
            "predictions": [{"index": 0}],
        }
        modified_result = {
            "summary": {"accuracy": 0.6, "correct": 3, "total": 5, "invalid_predictions": 0, "by_subject": {}},
            "predictions": [{"index": 1}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = mmlu.build_mmlu_result(
                config={"enabled": True, "store_predictions": True},
                original_result=original_result,
                modified_result=modified_result,
                method_results_dir=Path(tmpdir),
                detail_prefix="mmlu_test",
            )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["delta_accuracy"], 0.2)
        self.assertIsNotNone(result["details_file"])
        self.assertTrue(result["answer_comparison_preview"])

    def test_evaluate_model_on_mmlu_smoke_with_mocks(self):
        eval_records = [
            make_record(0, subject="math", answer=0),
            make_record(1, subject="history", answer=1),
        ]
        dev_records = [
            make_record(10, subject="math", answer=2),
            make_record(11, subject="history", answer=3),
        ]

        def fake_load_records(dataset_name, subset, split):
            if split == "dev":
                return dev_records
            return eval_records

        with patch.object(mmlu, "_load_dataset_records", side_effect=fake_load_records):
            with patch.object(mmlu, "_generate_choice_responses", return_value=["A", "B"]):
                result = mmlu.evaluate_model_on_mmlu(
                    object(),
                    {
                        "enabled": True,
                        "dataset": "cais/mmlu",
                        "subset": "all",
                        "split": "test",
                        "mode": "few_shot",
                        "n_shots": 1,
                        "sample_size": 2,
                        "sample_seed": 42,
                        "store_predictions": True,
                    },
                )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["summary"]["accuracy"], 1.0)
        self.assertEqual(len(result["predictions"]), 2)
        self.assertEqual(len(result["prediction_preview"]), 2)

    def test_evaluate_model_on_mmlu_logits_mode(self):
        entries = [
            {
                "index": 0,
                "subject": "math",
                "question": "Question 0?",
                "choices": ["A0", "B0", "C0", "D0"],
                "correct_letter": "B",
                "prompt": "prompt-0",
            }
        ]

        fake_predictions = [
            {
                "index": 0,
                "subject": "math",
                "question": "Question 0?",
                "choices": ["A0", "B0", "C0", "D0"],
                "correct_letter": "B",
                "predicted_letter": "B",
                "is_correct": True,
                "raw_response": None,
                "choice_scores": {"A": -2.0, "B": -0.1, "C": -1.0, "D": -3.0},
            }
        ]

        with patch.object(mmlu, "prepare_mmlu_data", return_value={"entries": entries, "config_snapshot": {"answer_mode": "logits"}}):
            with patch.object(mmlu, "_predict_with_logits", return_value=fake_predictions):
                result = mmlu.evaluate_model_on_mmlu(object(), {"enabled": True, "answer_mode": "logits"})

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["summary"]["accuracy"], 1.0)
        self.assertEqual(result["prediction_preview"][0]["predicted_letter"], "B")
        self.assertIn("choice_scores", result["prediction_preview"][0])


if __name__ == "__main__":
    unittest.main()
