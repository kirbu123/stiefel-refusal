import unittest

import torch

from baselines.additive_rotation_ops import (
    additive_subspace_rotation,
    dense_additive_rotation_matrix,
    infer_non_identity_layers,
    select_activation_additive_layers,
    select_baseline_ablation_layers,
)


class TestAdditiveRotationOps(unittest.TestCase):
    def setUp(self):
        self.basis = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.rotation = torch.tensor(
            [
                [0.0, -1.0],
                [1.0, 0.0],
            ]
        )

    def test_leaves_orthogonal_component_unchanged(self):
        x = torch.tensor([[2.0, 3.0, 5.0, 7.0]])

        actual = additive_subspace_rotation(x, self.basis, self.rotation)

        torch.testing.assert_close(actual[..., 2:], x[..., 2:])

    def test_forward_and_inverse_round_trip(self):
        x = torch.randn(3, 5, 4)

        rotated = additive_subspace_rotation(x, self.basis, self.rotation)
        restored = additive_subspace_rotation(
            rotated,
            self.basis,
            self.rotation,
            use_transpose=False,
        )

        torch.testing.assert_close(restored, x)

    def test_factorized_and_dense_forms_match(self):
        x = torch.randn(2, 4)
        dense = dense_additive_rotation_matrix(self.basis, self.rotation)

        factorized = additive_subspace_rotation(x, self.basis, self.rotation)

        torch.testing.assert_close(factorized, x @ dense.T)

    def test_protect_direction_reverses_attack_direction(self):
        x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        attack = additive_subspace_rotation(x, self.basis, self.rotation)
        protect = additive_subspace_rotation(
            x,
            self.basis,
            self.rotation,
            use_transpose=False,
        )

        torch.testing.assert_close(attack[..., :2], torch.tensor([[0.0, 1.0]]))
        torch.testing.assert_close(protect[..., :2], torch.tensor([[0.0, -1.0]]))

    def test_gradients_reach_basis_and_rotation(self):
        basis = self.basis.clone().requires_grad_()
        rotation = self.rotation.clone().requires_grad_()
        x = torch.randn(3, 4)

        additive_subspace_rotation(x, basis, rotation).square().sum().backward()

        self.assertIsNotNone(basis.grad)
        self.assertIsNotNone(rotation.grad)

    def test_nol_zero_selects_all_layers(self):
        selected = select_activation_additive_layers(
            n_layers=6,
            num_opt_layers=0,
            best_layer=3,
            layer_scores=[0.0] * 6,
        )

        self.assertEqual(selected, set(range(6)))

    def test_nol_one_selects_only_best_layer(self):
        selected = select_activation_additive_layers(
            n_layers=6,
            num_opt_layers=1,
            best_layer=3,
            layer_scores=[0.0] * 6,
        )

        self.assertEqual(selected, {3})

    def test_effective_all_layer_count_selects_all_layers(self):
        selected = select_activation_additive_layers(
            n_layers=6,
            num_opt_layers=6,
            best_layer=3,
            layer_scores=[0.0] * 6,
        )

        self.assertEqual(selected, set(range(6)))

    def test_nol_selects_exact_ranked_middle_layers(self):
        selected = select_activation_additive_layers(
            n_layers=7,
            num_opt_layers=3,
            best_layer=3,
            layer_scores=[0.0, 1.0, 8.0, 0.1, 7.0, 2.0, 0.0],
        )

        self.assertEqual(selected, {2, 3, 4})

    def test_nol_caps_selection_to_middle_layers(self):
        selected = select_activation_additive_layers(
            n_layers=5,
            num_opt_layers=99,
            best_layer=2,
            layer_scores=[0.0] * 5,
        )

        self.assertEqual(selected, {1, 2, 3})

    def test_nol_rejects_negative_value(self):
        with self.assertRaises(ValueError):
            select_activation_additive_layers(
                n_layers=5,
                num_opt_layers=-1,
                best_layer=2,
                layer_scores=[0.0] * 5,
            )

    def test_checkpoint_active_layer_inference(self):
        matrices = torch.eye(3).repeat(4, 1, 1)
        matrices[1, 0, 0] = 2.0
        matrices[3, 1, 2] = 0.5

        self.assertEqual(infer_non_identity_layers(matrices), {1, 3})

    def test_baseline_nol_zero_preserves_all_layer_ablation(self):
        selected = select_baseline_ablation_layers(
            n_layers=6,
            num_opt_layers=0,
            best_layer=3,
            layer_scores=[0.0] * 6,
        )

        self.assertEqual(selected, set(range(6)))

    def test_baseline_nol_selects_exact_ranked_layers_and_best(self):
        selected = select_baseline_ablation_layers(
            n_layers=7,
            num_opt_layers=3,
            best_layer=3,
            layer_scores=[0.0, 1.0, 8.0, 0.1, 7.0, 2.0, 0.0],
        )

        self.assertEqual(selected, {2, 3, 4})

    def test_baseline_nol_caps_to_middle_layers(self):
        selected = select_baseline_ablation_layers(
            n_layers=5,
            num_opt_layers=99,
            best_layer=2,
            layer_scores=[0.0] * 5,
        )

        self.assertEqual(selected, {1, 2, 3})

    def test_baseline_nol_rejects_negative_value(self):
        with self.assertRaises(ValueError):
            select_baseline_ablation_layers(
                n_layers=5,
                num_opt_layers=-1,
                best_layer=2,
                layer_scores=[0.0] * 5,
            )


if __name__ == "__main__":
    unittest.main()
