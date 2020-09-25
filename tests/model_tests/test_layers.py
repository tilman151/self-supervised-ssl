import unittest

import torch

from models import layers


class TestDeflatten(unittest.TestCase):
    def test_deflattening(self):
        inputs = torch.randn(16, 512)
        expected_shape = (16, 8, 64)
        module = layers.DeFlatten(64, 8)

        deflattened = module(inputs)
        reflattened = deflattened.view(16, -1)

        self.assertEqual(expected_shape, deflattened.shape)
        self.assertEqual(0.0, torch.sum(inputs - reflattened).item())


class TestGradientReversal(unittest.TestCase):
    def test_against_identity(self):
        inputs = torch.randn(16, 64, 1)
        weight = torch.randn(1, 32, 64)
        weight.requires_grad = True
        grad_reversal = layers.GradientReversalLayer()

        normal_loss = torch.mean(weight @ inputs)
        normal_loss.backward()
        normal_gradient = weight.grad.clone()
        weight.grad = None

        reverse_loss = torch.mean(grad_reversal(weight @ inputs))
        reverse_loss.backward()
        reverse_gradient = weight.grad.clone()

        self.assertEqual(0.0, torch.sum(normal_gradient + reverse_gradient))
