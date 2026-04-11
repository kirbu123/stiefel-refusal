import unittest
from unittest.mock import patch

from baselines.graph_grpo.reward import compute_reward


class TestGraphGrpoReward(unittest.TestCase):
    def test_compute_reward_defaults_to_harmfulness_score(self):
        with patch(
            "baselines.graph_grpo.reward.evaluate_harmfulness",
            side_effect=[{"score": 4}, {"score": None}],
        ) as evaluate_mock:
            scores = compute_reward(
                ["q1", "q2"],
                ["r1", "r2"],
                classifier_categories=[],
                backend="llamaguard",
            )

        self.assertEqual(scores, [4.0, 0.0])
        self.assertEqual(evaluate_mock.call_count, 2)

    def test_compute_reward_maps_llamaguard_unsafe_labels_to_binary_scores(self):
        with patch(
            "baselines.graph_grpo.reward.evaluate_harmfulness",
            side_effect=[
                {"label": "unsafe", "score": 4},
                {"label": "safe", "score": 1},
                RuntimeError("judge failed"),
            ],
        ) as evaluate_mock:
            scores = compute_reward(
                ["q-unsafe", "q-safe", "q-empty", "q-error"],
                ["r1", "r2", "", "r4"],
                classifier_categories=[],
                backend="llamaguard",
                reward_metric="llamaguard_unsafe",
            )

        self.assertEqual(scores, [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(evaluate_mock.call_count, 3)

    def test_compute_reward_rejects_unknown_metric(self):
        with self.assertRaisesRegex(ValueError, "Unsupported reward metric"):
            compute_reward(["q"], ["r"], reward_metric="unsafe_probability")


if __name__ == "__main__":
    unittest.main()
