import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import optuna
except ModuleNotFoundError:
    optuna = None

if torch is not None and optuna is not None:
    from baselines.graph_grpo import optuna_optimizer as optuna_module
    from model_utils import LearnableDirectionWeights


class _DummyModel:
    def __init__(self):
        self.reload_calls = 0

    def reload_model(self):
        self.reload_calls += 1

    def get_responses_batched(self, questions):
        return [f"<think>hidden</think>answer-{idx}" for idx, _ in enumerate(questions)]


@unittest.skipUnless(torch is not None and optuna is not None, "torch and optuna are required for graph_grpo optuna tests")
class TestGraphGrpoOptuna(unittest.TestCase):
    def test_create_optuna_sampler_supports_configured_names(self):
        tpe = optuna_module.create_optuna_sampler("tpe", sampler_seed=42)
        random = optuna_module.create_optuna_sampler("random", sampler_seed=42)
        gp = optuna_module.create_optuna_sampler("gp", sampler_seed=42)
        cmaes = optuna_module.create_optuna_sampler("cmaes", sampler_seed=42)
        qmc = optuna_module.create_optuna_sampler("qmc", sampler_seed=42)

        self.assertEqual(type(tpe).__name__, "TPESampler")
        self.assertEqual(type(random).__name__, "RandomSampler")
        self.assertEqual(type(gp).__name__, "GPSampler")
        self.assertEqual(type(cmaes).__name__, "CmaEsSampler")
        self.assertEqual(type(qmc).__name__, "QMCSampler")

        with self.assertRaisesRegex(ValueError, "Unsupported Optuna sampler"):
            optuna_module.create_optuna_sampler("nsga2", sampler_seed=42)

    def test_suggest_trial_weights_supports_dense_shape(self):
        direction_weights = LearnableDirectionWeights(
            n_directions=2,
            n_layers=1,
            hidden_size=2,
            init_type="average",
            mode="dense",
        )
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        dense_weights = optuna_module.suggest_trial_weights(
            trial=trial,
            direction_weights=direction_weights,
            weight_min=-2.0,
            weight_max=2.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(tuple(dense_weights.shape), (2, 2, 2))
        self.assertEqual(len(trial.params), 8)
        reconstructed = optuna_module.reconstruct_weights_from_params(
            params=trial.params,
            direction_weights=direction_weights,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(dense_weights, reconstructed))

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
            with patch.object(
                optuna_module,
                "compute_sequence_log_probs",
                side_effect=[
                    (
                        torch.tensor(
                            [[0.7, 0.6], [0.7, 0.6], [0.7, 0.6]],
                            dtype=torch.float32,
                        ),
                        torch.ones((3, 2), dtype=torch.float32),
                    ),
                    (
                        torch.tensor(
                            [[0.4, 0.3], [0.4, 0.3], [0.4, 0.3]],
                            dtype=torch.float32,
                        ),
                        torch.ones((3, 2), dtype=torch.float32),
                    ),
                ],
            ):
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
                        reward_sign=1.0,
                        kl_loss_coef=0.01,
                        backend="llamaguard",
                    )

        self.assertEqual(model.reload_calls, 2)
        self.assertEqual(len(apply_calls), 1)
        self.assertEqual(result["weights"], [1.5, -0.5])
        self.assertEqual(result["weights_mode"], "scalar")
        self.assertAlmostEqual(result["mean_reward"], 2.0 / 3.0)
        self.assertEqual(result["best_reward"], 1.0)
        self.assertEqual(result["n_questions"], 3)
        self.assertEqual(result["scores"], [0.0, 1.0, 1.0])
        self.assertIn("mean_kl", result)
        self.assertIn("mean_objective", result)
        self.assertAlmostEqual(
            result["mean_objective"],
            result["mean_reward"] - 0.01 * result["mean_kl"],
        )

    def test_evaluate_dense_weights_returns_metrics(self):
        direction_weights = LearnableDirectionWeights(
            n_directions=2,
            n_layers=1,
            hidden_size=2,
            init_type="average",
            mode="dense",
        )
        model = _DummyModel()
        extracted_directions = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        ]
        dense_weights = torch.tensor(
            [
                [[1.0, 0.5], [0.2, -0.1]],
                [[-0.5, 0.3], [0.9, 0.7]],
            ],
            dtype=torch.float32,
        )
        apply_calls = []

        def fake_apply(*args):
            apply_calls.append(args)

        with patch.object(optuna_module, "apply_abliteration_with_hyperparams", side_effect=fake_apply):
            with patch.object(
                optuna_module,
                "compute_sequence_log_probs",
                side_effect=[
                    (
                        torch.tensor(
                            [[0.6, 0.5], [0.6, 0.5]],
                            dtype=torch.float32,
                        ),
                        torch.ones((2, 2), dtype=torch.float32),
                    ),
                    (
                        torch.tensor(
                            [[0.2, 0.1], [0.2, 0.1]],
                            dtype=torch.float32,
                        ),
                        torch.ones((2, 2), dtype=torch.float32),
                    ),
                ],
            ):
                with patch.object(optuna_module, "compute_reward", return_value=[1.0, 0.0]):
                    result = optuna_module.evaluate_dense_weights(
                        direction_weights=direction_weights,
                        extracted_directions=extracted_directions,
                        dense_weights=dense_weights,
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
                        reward_sign=1.0,
                        kl_loss_coef=0.01,
                        backend="llamaguard",
                    )

        self.assertEqual(model.reload_calls, 2)
        self.assertEqual(len(apply_calls), 1)
        self.assertEqual(result["weights_mode"], "dense")
        self.assertEqual(tuple(torch.tensor(result["weights"]).shape), (2, 2, 2))
        self.assertAlmostEqual(result["mean_reward"], 0.5)
        self.assertEqual(result["best_reward"], 1.0)
        self.assertEqual(result["n_questions"], 2)
        self.assertIn("mean_kl", result)
        self.assertIn("mean_objective", result)
        self.assertAlmostEqual(
            result["mean_objective"],
            result["mean_reward"] - 0.01 * result["mean_kl"],
        )

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
            weights,
            model,
            questions,
            abliteration_params,
            classifier_categories,
            n_layers,
            ref_alpha,
            reward_sign,
            kl_loss_coef,
            loss_agg_mode="token-mean",
            backend=None,
            harmless_questions=None,
        ):
            score = float(weights.sum().item())
            mean_kl = 0.5
            return {
                "weights": weights.detach().cpu().tolist(),
                "responses": [],
                "scores": [score],
                "mean_reward": score,
                "best_reward": score,
                "mean_harmfulness": score,
                "best_harmfulness": score,
                "mean_kl": mean_kl,
                "kl_loss": mean_kl,
                "mean_objective": score - kl_loss_coef * mean_kl,
                "n_questions": len(questions),
                "weights_mode": "scalar",
            }

        with patch.object(optuna_module, "evaluate_weights", side_effect=fake_evaluate):
            result = optuna_module.optimize_weights_with_optuna(
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
                reward_sign=1.0,
                kl_loss_coef=0.01,
                n_trials=3,
                sampler_name="random",
                sampler_seed=42,
                weight_min=-2.0,
                weight_max=2.0,
                loss_agg_mode="token-mean",
                backend="llamaguard",
            )

        self.assertEqual(len(result["optimization_history"]), 3)
        self.assertIsInstance(result["best_trial_number"], int)
        self.assertEqual(len(result["best_weights"]), 2)
        self.assertEqual(result["best_trial_questions"], ["q1", "q2"])
        self.assertEqual(result["sampler_name"], "random")
        best_from_history = max(item["mean_objective"] for item in result["optimization_history"])
        self.assertAlmostEqual(result["best_value"], best_from_history)
        self.assertTrue(torch.allclose(
            direction_weights.weights.detach().cpu(),
            torch.tensor(result["best_weights"], dtype=direction_weights.weights.dtype),
        ))
        for item in result["optimization_history"]:
            self.assertIn("weights", item)
            self.assertIn("mean_reward", item)
            self.assertIn("best_reward", item)
            self.assertIn("mean_kl", item)
            self.assertIn("mean_objective", item)
            self.assertEqual(item["weights_mode"], "scalar")

    def test_optimize_uses_question_sampler_per_trial_and_returns_best_trial_questions(self):
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
        sampled_batches = [
            ["trial-0"],
            ["trial-1a", "trial-1b"],
            ["trial-2"],
        ]
        score_by_batch = {
            "trial-0": 0.1,
            "trial-1a": 0.9,
            "trial-2": 0.4,
        }
        sampler_calls = []

        def question_sampler():
            batch = sampled_batches[len(sampler_calls)]
            sampler_calls.append(list(batch))
            return list(batch)

        def fake_evaluate(
            direction_weights,
            extracted_directions,
            weights,
            model,
            questions,
            abliteration_params,
            classifier_categories,
            n_layers,
            ref_alpha,
            reward_sign,
            kl_loss_coef,
            loss_agg_mode="token-mean",
            backend=None,
            harmless_questions=None,
        ):
            score = score_by_batch[questions[0]]
            mean_kl = 0.25
            return {
                "weights": weights.detach().cpu().tolist(),
                "responses": [],
                "scores": [score for _ in questions],
                "mean_reward": score,
                "best_reward": score,
                "mean_harmfulness": score,
                "best_harmfulness": score,
                "mean_kl": mean_kl,
                "kl_loss": mean_kl,
                "mean_objective": score - kl_loss_coef * mean_kl,
                "n_questions": len(questions),
                "weights_mode": "scalar",
            }

        with patch.object(optuna_module, "evaluate_weights", side_effect=fake_evaluate):
            result = optuna_module.optimize_weights_with_optuna(
                direction_weights=direction_weights,
                extracted_directions=extracted_directions,
                model=model,
                questions=None,
                abliteration_params={
                    "max_weight": 2.0,
                    "max_weight_position": 0.5,
                    "min_weight": 0.25,
                    "min_weight_distance": 0.4,
                },
                classifier_categories=[],
                n_layers=1,
                ref_alpha=1.0,
                reward_sign=1.0,
                kl_loss_coef=0.01,
                n_trials=3,
                sampler_name="random",
                sampler_seed=42,
                weight_min=-2.0,
                weight_max=2.0,
                question_sampler=question_sampler,
                loss_agg_mode="token-mean",
                backend="llamaguard",
            )

        self.assertEqual(sampler_calls, sampled_batches)
        self.assertEqual(result["best_trial_number"], 1)
        self.assertEqual(result["best_trial_questions"], sampled_batches[1])
        self.assertEqual(
            [item["n_questions"] for item in result["optimization_history"]],
            [1, 2, 1],
        )

    def test_optimize_dense_weights_updates_module_and_records_summary_history(self):
        direction_weights = LearnableDirectionWeights(
            n_directions=2,
            n_layers=1,
            hidden_size=2,
            init_type="average",
            mode="dense",
        )
        model = _DummyModel()
        extracted_directions = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        ]

        def fake_evaluate(
            direction_weights,
            extracted_directions,
            weights,
            model,
            questions,
            abliteration_params,
            classifier_categories,
            n_layers,
            ref_alpha,
            reward_sign,
            kl_loss_coef,
            loss_agg_mode="token-mean",
            backend=None,
            harmless_questions=None,
        ):
            score = float(weights.sum().item())
            mean_kl = 0.5
            return {
                "weights": weights.detach().cpu().tolist(),
                "responses": [],
                "scores": [score],
                "mean_reward": score,
                "best_reward": score,
                "mean_harmfulness": score,
                "best_harmfulness": score,
                "mean_kl": mean_kl,
                "kl_loss": mean_kl,
                "mean_objective": score - kl_loss_coef * mean_kl,
                "n_questions": len(questions),
                "weights_mode": "dense",
            }

        with patch.object(optuna_module, "evaluate_weights", side_effect=fake_evaluate):
            result = optuna_module.optimize_weights_with_optuna(
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
                reward_sign=1.0,
                kl_loss_coef=0.01,
                n_trials=3,
                sampler_name="random",
                sampler_seed=42,
                weight_min=-2.0,
                weight_max=2.0,
                loss_agg_mode="token-mean",
                backend="llamaguard",
            )

        self.assertEqual(len(result["optimization_history"]), 3)
        self.assertIsInstance(result["best_trial_number"], int)
        best_weights_tensor = torch.tensor(result["best_weights"], dtype=direction_weights.weights.dtype)
        self.assertEqual(tuple(best_weights_tensor.shape), tuple(direction_weights.weights.shape))
        self.assertEqual(result["best_trial_questions"], ["q1", "q2"])
        best_from_history = max(item["mean_objective"] for item in result["optimization_history"])
        self.assertAlmostEqual(result["best_value"], best_from_history)
        self.assertTrue(torch.allclose(
            direction_weights.weights.detach().cpu(),
            best_weights_tensor,
        ))
        for item in result["optimization_history"]:
            self.assertEqual(item["weights_mode"], "dense")
            self.assertNotIn("weights", item)
            self.assertIn("weights_shape", item)
            self.assertIn("weights_mean", item)
            self.assertIn("weights_std", item)
            self.assertIn("weights_min", item)
            self.assertIn("weights_max", item)
            self.assertIn("weights_norm", item)
            self.assertIn("mean_kl", item)
            self.assertIn("mean_objective", item)


if __name__ == "__main__":
    unittest.main()
