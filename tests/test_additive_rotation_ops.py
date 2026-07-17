import unittest

import torch

from baselines.additive_rotation_ops import (
    additive_subspace_rotation,
    dense_additive_rotation_matrix,
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


if __name__ == "__main__":
    unittest.main()
