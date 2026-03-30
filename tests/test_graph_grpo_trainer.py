import types
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from baselines.graph_grpo import trainer as trainer_module
    from model_utils import LearnableDirectionWeights


class _DummyHandle:
    def __init__(self, state):
        self._state = state

    def remove(self):
        self._state["current_direction"] = None


class _DummyModel:
    def __init__(self):
        self.reload_calls = 0
        if torch is not None:
            self.model = torch.nn.Linear(1, 1)

    def reload_model(self):
        self.reload_calls += 1

    def get_responses_batched(self, questions):
        return [f"response-{self.reload_calls}-{idx}" for idx, _ in enumerate(questions)]


@unittest.skipUnless(torch is not None, "torch is required for graph_grpo trainer tests")
class TestGraphGrpoTrainer(unittest.TestCase):
    def test_build_rollout_weight_variants_returns_detached_samples(self):
        torch.manual_seed(0)
        base_weights = torch.ones(2, 2, 3, requires_grad=True)

        variants = trainer_module._build_rollout_weight_variants(
            base_weights=base_weights,
            n_groups=4,
            noise_scale=0.25,
        )

        self.assertEqual(len(variants), 4)
        self.assertTrue(torch.equal(variants[0], base_weights.detach()))
        self.assertFalse(variants[0].requires_grad)
        self.assertTrue(any(not torch.equal(variant, variants[0]) for variant in variants[1:]))
        self.assertTrue(all(not variant.requires_grad for variant in variants))

    def test_build_rollout_weight_variants_supports_scalar_weights(self):
        torch.manual_seed(0)
        base_weights = torch.tensor([0.2, -0.4, 0.8], dtype=torch.float32, requires_grad=True)

        variants = trainer_module._build_rollout_weight_variants(
            base_weights=base_weights,
            n_groups=4,
            noise_scale=0.25,
        )

        self.assertEqual(len(variants), 4)
        self.assertEqual(tuple(variants[0].shape), (3,))
        self.assertTrue(torch.equal(variants[0], base_weights.detach()))
        self.assertTrue(any(not torch.equal(variant, variants[0]) for variant in variants[1:]))
        self.assertTrue(all(not variant.requires_grad for variant in variants))

    def test_scalar_combine_with_weights_broadcasts_scalars(self):
        direction_weights = LearnableDirectionWeights(
            n_directions=2,
            n_layers=1,
            hidden_size=2,
            init_type="average",
            mode="scalar",
        )
        refusal_directions = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        ]

        combined = direction_weights.combine_with_weights(
            refusal_directions,
            torch.tensor([2.0, 1.0], dtype=torch.float32),
        )

        expected = torch.stack(
            [
                2.0 * refusal_directions[0][0] + 1.0 * refusal_directions[1][0],
                2.0 * refusal_directions[0][1] + 1.0 * refusal_directions[1][1],
            ],
            dim=0,
        )
        expected = torch.nn.functional.normalize(expected, p=2, dim=1)

        self.assertTrue(torch.allclose(combined, expected))

    def test_combine_with_weights_rejects_invalid_shape(self):
        direction_weights = LearnableDirectionWeights(
            n_directions=2,
            n_layers=1,
            hidden_size=2,
            init_type="average",
            mode="scalar",
        )
        refusal_directions = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        ]

        with self.assertRaisesRegex(ValueError, "Direction weights must be either"):
            direction_weights.combine_with_weights(
                refusal_directions,
                torch.ones(2, 2, dtype=torch.float32),
            )

    def test_train_step_uses_sampled_w_rollouts_and_shared_ref_alpha(self):
        torch.manual_seed(0)
        direction_weights = LearnableDirectionWeights(
            n_directions=2,
            n_layers=1,
            hidden_size=2,
            init_type="average",
        )
        optimizer = torch.optim.SGD(direction_weights.parameters(), lr=0.01)
        model = _DummyModel()
        extracted_directions = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.5, 0.5], [1.0, -1.0]], dtype=torch.float32),
        ]
        questions = ["q1", "q2", "q3", "q4"]
        hook_state = {"current_direction": None}
        rollout_calls = []

        def fake_apply_abliteration_with_hyperparams(
            _model,
            combined_direction,
            max_weight,
            max_weight_position,
            min_weight,
            min_weight_distance,
            n_layers,
        ):
            rollout_calls.append(
                {
                    "direction": combined_direction.detach().clone(),
                    "max_weight": max_weight,
                    "max_weight_position": max_weight_position,
                    "min_weight": min_weight,
                    "min_weight_distance": min_weight_distance,
                    "n_layers": n_layers,
                }
            )

        def fake_register_abliteration_hooks(
            _model,
            combined_direction,
            alpha,
            n_layers,
            max_weight,
            max_weight_position,
            min_weight,
            min_weight_distance,
        ):
            hook_state["current_direction"] = combined_direction
            rollout_calls.append(
                {
                    "hook_alpha": alpha,
                    "hook_max_weight": max_weight,
                    "hook_min_weight": min_weight,
                    "hook_n_layers": n_layers,
                    "hook_direction": combined_direction.detach().clone(),
                }
            )
            return [_DummyHandle(hook_state)]

        def fake_remove_hooks(handles):
            for handle in handles:
                handle.remove()

        def fake_compute_sequence_log_probs(_model, prompts, responses, batch_size=8):
            seq_len = 2
            base = torch.ones((len(prompts), seq_len), dtype=torch.float32)
            if hook_state["current_direction"] is not None:
                signal = hook_state["current_direction"].sum()
                log_probs = base * signal
            else:
                log_probs = base * 0.5
            mask = torch.ones((len(prompts), seq_len), dtype=torch.float32)
            return log_probs, mask

        def fake_compute_reward(questions, responses, classifier_categories=None, backend=None):
            return [float(idx + 1) for idx, _ in enumerate(responses)]

        def fake_compute_advantage(
            token_level_rewards,
            response_mask,
            index,
            is_weights=None,
            epsilon=1e-6,
            norm_adv_by_std_in_grpo=True,
            config=None,
        ):
            advantages = torch.ones_like(token_level_rewards)
            return advantages, advantages.clone()

        def fake_compute_rollout_correction_weights(
            log_ratio,
            response_mask,
            rollout_is,
            rollout_is_threshold,
        ):
            return torch.ones_like(log_ratio), {"mean": 1.0}

        def fake_compute_policy_loss_vanilla(
            old_log_prob,
            log_prob,
            advantages,
            response_mask,
            rollout_is_weights,
            config,
        ):
            loss = (log_prob * response_mask).sum() / response_mask.sum()
            return loss, {"loss": float(loss.detach().item())}

        def fake_compute_offpolicy_metrics(old_log_prob, rollout_log_prob, response_mask):
            return {"kl": 0.0}

        with patch.object(trainer_module, "apply_abliteration_with_hyperparams", side_effect=fake_apply_abliteration_with_hyperparams):
            with patch.object(trainer_module, "register_abliteration_hooks", side_effect=fake_register_abliteration_hooks):
                with patch.object(trainer_module, "remove_hooks", side_effect=fake_remove_hooks):
                    with patch.object(trainer_module, "compute_sequence_log_probs", side_effect=fake_compute_sequence_log_probs):
                        with patch.object(trainer_module, "compute_reward", side_effect=fake_compute_reward):
                            with patch.object(trainer_module, "compute_grpo_is_advantage", side_effect=fake_compute_advantage):
                                with patch.object(trainer_module, "empty_cache", return_value=None):
                                    with patch("verl.trainer.ppo.rollout_corr_helper.compute_rollout_correction_weights", side_effect=fake_compute_rollout_correction_weights):
                                        with patch("verl.trainer.ppo.rollout_corr_helper.compute_offpolicy_metrics", side_effect=fake_compute_offpolicy_metrics):
                                            with patch("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla", side_effect=fake_compute_policy_loss_vanilla):
                                                metrics = trainer_module.train_grpo_is_step(
                                                    direction_weights=direction_weights,
                                                    extracted_directions=extracted_directions,
                                                    model=model,
                                                    questions=questions,
                                                    n_groups=4,
                                                    noise_scale=0.2,
                                                    abliteration_params={
                                                        "max_weight": 2.0,
                                                        "max_weight_position": 0.5,
                                                        "min_weight": 0.5,
                                                        "min_weight_distance": 0.5,
                                                    },
                                                    optimizer=optimizer,
                                                    classifier_categories=[],
                                                    n_layers=1,
                                                    ref_alpha=1.5,
                                                    is_clip_ratio=5.0,
                                                    clip_ratio=0.2,
                                                    loss_agg_mode="token-mean",
                                                    backend="llamaguard",
                                                )

        rollout_abliterations = [call for call in rollout_calls if "direction" in call]
        hook_calls = [call for call in rollout_calls if "hook_alpha" in call]

        self.assertEqual(len(rollout_abliterations), 4)
        self.assertEqual(model.reload_calls, 5)
        self.assertTrue(all(call["max_weight"] == 3.0 for call in rollout_abliterations))
        self.assertTrue(all(call["min_weight"] == 0.75 for call in rollout_abliterations))
        self.assertTrue(any(not torch.equal(call["direction"], rollout_abliterations[0]["direction"]) for call in rollout_abliterations[1:]))
        self.assertTrue(all(call["hook_alpha"] == 1.5 for call in hook_calls))
        self.assertEqual(len(hook_calls), 4)
        self.assertTrue(all(not param.requires_grad for param in model.model.parameters()))
        self.assertIsNotNone(direction_weights.weights.grad)
        self.assertGreater(direction_weights.weights.grad.abs().sum().item(), 0.0)
        self.assertIn("mean_reward", metrics)
        self.assertIn("weights_norm", metrics)

    def test_train_step_supports_scalar_direction_weights(self):
        torch.manual_seed(0)
        direction_weights = LearnableDirectionWeights(
            n_directions=2,
            n_layers=1,
            hidden_size=2,
            init_type="average",
            mode="scalar",
        )
        optimizer = torch.optim.SGD(direction_weights.parameters(), lr=0.01)
        model = _DummyModel()
        extracted_directions = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.5, 0.5], [1.0, -1.0]], dtype=torch.float32),
        ]
        questions = ["q1", "q2", "q3", "q4"]
        hook_state = {"current_direction": None}
        rollout_calls = []

        def fake_apply_abliteration_with_hyperparams(
            _model,
            combined_direction,
            max_weight,
            max_weight_position,
            min_weight,
            min_weight_distance,
            n_layers,
        ):
            rollout_calls.append(
                {
                    "direction": combined_direction.detach().clone(),
                    "max_weight": max_weight,
                    "max_weight_position": max_weight_position,
                    "min_weight": min_weight,
                    "min_weight_distance": min_weight_distance,
                    "n_layers": n_layers,
                }
            )

        def fake_register_abliteration_hooks(
            _model,
            combined_direction,
            alpha,
            n_layers,
            max_weight,
            max_weight_position,
            min_weight,
            min_weight_distance,
        ):
            hook_state["current_direction"] = combined_direction
            rollout_calls.append(
                {
                    "hook_alpha": alpha,
                    "hook_max_weight": max_weight,
                    "hook_min_weight": min_weight,
                    "hook_n_layers": n_layers,
                    "hook_direction": combined_direction.detach().clone(),
                }
            )
            return [_DummyHandle(hook_state)]

        def fake_remove_hooks(handles):
            for handle in handles:
                handle.remove()

        def fake_compute_sequence_log_probs(_model, prompts, responses, batch_size=8):
            seq_len = 2
            base = torch.ones((len(prompts), seq_len), dtype=torch.float32)
            if hook_state["current_direction"] is not None:
                signal = hook_state["current_direction"].sum()
                log_probs = base * signal
            else:
                log_probs = base * 0.5
            mask = torch.ones((len(prompts), seq_len), dtype=torch.float32)
            return log_probs, mask

        def fake_compute_reward(questions, responses, classifier_categories=None, backend=None):
            return [float(idx + 1) for idx, _ in enumerate(responses)]

        def fake_compute_advantage(
            token_level_rewards,
            response_mask,
            index,
            is_weights=None,
            epsilon=1e-6,
            norm_adv_by_std_in_grpo=True,
            config=None,
        ):
            advantages = torch.ones_like(token_level_rewards)
            return advantages, advantages.clone()

        def fake_compute_rollout_correction_weights(
            log_ratio,
            response_mask,
            rollout_is,
            rollout_is_threshold,
        ):
            return torch.ones_like(log_ratio), {"mean": 1.0}

        def fake_compute_policy_loss_vanilla(
            old_log_prob,
            log_prob,
            advantages,
            response_mask,
            rollout_is_weights,
            config,
        ):
            loss = (log_prob * response_mask).sum() / response_mask.sum()
            return loss, {"loss": float(loss.detach().item())}

        def fake_compute_offpolicy_metrics(old_log_prob, rollout_log_prob, response_mask):
            return {"kl": 0.0}

        with patch.object(trainer_module, "apply_abliteration_with_hyperparams", side_effect=fake_apply_abliteration_with_hyperparams):
            with patch.object(trainer_module, "register_abliteration_hooks", side_effect=fake_register_abliteration_hooks):
                with patch.object(trainer_module, "remove_hooks", side_effect=fake_remove_hooks):
                    with patch.object(trainer_module, "compute_sequence_log_probs", side_effect=fake_compute_sequence_log_probs):
                        with patch.object(trainer_module, "compute_reward", side_effect=fake_compute_reward):
                            with patch.object(trainer_module, "compute_grpo_is_advantage", side_effect=fake_compute_advantage):
                                with patch.object(trainer_module, "empty_cache", return_value=None):
                                    with patch("verl.trainer.ppo.rollout_corr_helper.compute_rollout_correction_weights", side_effect=fake_compute_rollout_correction_weights):
                                        with patch("verl.trainer.ppo.rollout_corr_helper.compute_offpolicy_metrics", side_effect=fake_compute_offpolicy_metrics):
                                            with patch("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla", side_effect=fake_compute_policy_loss_vanilla):
                                                metrics = trainer_module.train_grpo_is_step(
                                                    direction_weights=direction_weights,
                                                    extracted_directions=extracted_directions,
                                                    model=model,
                                                    questions=questions,
                                                    n_groups=4,
                                                    noise_scale=0.2,
                                                    abliteration_params={
                                                        "max_weight": 2.0,
                                                        "max_weight_position": 0.5,
                                                        "min_weight": 0.5,
                                                        "min_weight_distance": 0.5,
                                                    },
                                                    optimizer=optimizer,
                                                    classifier_categories=[],
                                                    n_layers=1,
                                                    ref_alpha=1.5,
                                                    is_clip_ratio=5.0,
                                                    clip_ratio=0.2,
                                                    loss_agg_mode="token-mean",
                                                    backend="llamaguard",
                                                )

        rollout_abliterations = [call for call in rollout_calls if "direction" in call]
        hook_calls = [call for call in rollout_calls if "hook_alpha" in call]

        self.assertEqual(tuple(direction_weights.weights.shape), (2,))
        self.assertEqual(len(rollout_abliterations), 4)
        self.assertEqual(model.reload_calls, 5)
        self.assertTrue(all(call["max_weight"] == 3.0 for call in rollout_abliterations))
        self.assertTrue(all(call["min_weight"] == 0.75 for call in rollout_abliterations))
        self.assertTrue(any(not torch.equal(call["direction"], rollout_abliterations[0]["direction"]) for call in rollout_abliterations[1:]))
        self.assertTrue(all(call["hook_alpha"] == 1.5 for call in hook_calls))
        self.assertEqual(len(hook_calls), 4)
        self.assertIsNotNone(direction_weights.weights.grad)
        self.assertGreater(direction_weights.weights.grad.abs().sum().item(), 0.0)
        self.assertIn("mean_reward", metrics)
        self.assertIn("weights_norm", metrics)


if __name__ == "__main__":
    unittest.main()
