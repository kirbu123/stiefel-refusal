import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from baselines.graph_grpo import optuna_optimizer as optuna_module
    from model_utils import LearnableDirectionWeights


class _DummyModel:
    def __init__(self):
        self.reload_calls = 0

    def reload_model(self):
        self.reload_calls += 1

    def get_responses_batched(self, questions):
        return [f"<think>hidden</think>answer-{idx}" for idx, _ in enumerate(questions)]


@unittest.skipUnless(torch is not None, "torch is required for graph_grpo optuna tests")
class TestGraphGrpoOptuna(unittest.TestCase):
    def test_evaluate_scalar_weights_returns_metrics(self):
        direction_weights = LearnableDirectionWeights(
            n_directions=2,
            n_layers=1,
            hidden_size=2,
            init_type="average",
            mode="scalar",
        )
        model = _DummyModel()
        extracted_directions = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        ]
        apply_calls = []

        def fake_apply(*args):
            apply_calls.append(args)

        with patch.object(optuna_module, "apply_abliteration_with_hyperparams", side_effect=fake_apply):
            with patch.object(optuna_module, "compute_reward", return_value=[0.0, 1.0, 1.0]):
                result = optuna_module.evaluate_scalar_weights(
                    direction_weights=direction_weights,
                    extracted_directions=extracted_directions,
                    scalar_weights=torch.tensor([1.5, -0.5], dtype=torch.float32),
                    model=model,
                    questions=["q1", "q2", "q3"],
                    abliteration_params={
                        "max_weight": 2.0,
                        "max_weight_position": 0.5,
                        "min_weight": 0.25,
                        "min_weight_distance": 0.4,
                    },
                    classifier_categories=[],
                    n_layers=1,
                    ref_alpha=1.5,
                    backend="llamaguard",
                )

        self.assertEqual(model.reload_calls, 1)
        self.assertEqual(len(apply_calls), 1)
        self.assertEqual(result["weights"], [1.5, -0.5])
        self.assertAlmostEqual(result["mean_reward"], 2.0 / 3.0)
        self.assertEqual(result["best_reward"], 1.0)
        self.assertEqual(result["n_questions"], 3)
        self.assertEqual(result["scores"], [0.0, 1.0, 1.0])

    def test_optimize_scalar_weights_updates_module_and_records_history(self):
        direction_weights = LearnableDirectionWeights(
            n_directions=2,
            n_layers=1,
            hidden_size=2,
            init_type="average",
            mode="scalar",
        )
        model = _DummyModel()
        extracted_directions = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        ]

        def fake_evaluate(
            direction_weights,
            extracted_directions,
            scalar_weights,
            model,
            questions,
            abliteration_params,
            classifier_categories,
            n_layers,
            ref_alpha,
            backend=None,
        ):
            score = float(scalar_weights.sum().item())
            return {
                "weights": scalar_weights.detach().cpu().tolist(),
                "responses": [],
                "scores": [score],
                "mean_reward": score,
                "best_reward": score,
                "n_questions": len(questions),
            }

        with patch.object(optuna_module, "evaluate_scalar_weights", side_effect=fake_evaluate):
            result = optuna_module.optimize_scalar_weights_with_optuna(
                direction_weights=direction_weights,
                extracted_directions=extracted_directions,
                model=model,
                questions=["q1", "q2"],
                abliteration_params={
                    "max_weight": 2.0,
                    "max_weight_position": 0.5,
                    "min_weight": 0.25,
                    "min_weight_distance": 0.4,
                },
                classifier_categories=[],
                n_layers=1,
                ref_alpha=1.0,
                n_trials=3,
                sampler_seed=42,
                weight_min=-2.0,
                weight_max=2.0,
                backend="llamaguard",
            )

        self.assertEqual(len(result["optimization_history"]), 3)
        self.assertIsInstance(result["best_trial_number"], int)
        self.assertEqual(len(result["best_weights"]), 2)
        best_from_history = max(item["mean_reward"] for item in result["optimization_history"])
        self.assertAlmostEqual(result["best_value"], best_from_history)
        self.assertTrue(torch.allclose(
            direction_weights.weights.detach().cpu(),
            torch.tensor(result["best_weights"], dtype=direction_weights.weights.dtype),
        ))
        for item in result["optimization_history"]:
            self.assertIn("weights", item)
            self.assertIn("mean_reward", item)
            self.assertIn("best_reward", item)


if __name__ == "__main__":
    unittest.main()
