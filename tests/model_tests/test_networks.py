import unittest

import torch

from models import networks


class ModelTestsMixin:
    @torch.no_grad()
    def test_shape(self):
        outputs = self.net(self.test_inputs)
        self.assertEqual(self.expected_shape, outputs.shape)

    @torch.no_grad()
    @unittest.skipUnless(torch.cuda.is_available(), "No GPU was detected")
    def test_device_moving(self):
        self.net.eval()

        torch.manual_seed(42)
        outputs_cpu = self.net(self.test_inputs)

        net_on_gpu = self.net.to("cuda:0")
        torch.manual_seed(42)
        outputs_gpu = net_on_gpu(self.test_inputs.to("cuda:0"))

        torch.manual_seed(42)
        net_back_on_cpu = net_on_gpu.cpu()
        outputs_back_on_cpu = net_back_on_cpu(self.test_inputs)

        self.assertAlmostEqual(
            0.0, torch.sum(outputs_cpu - outputs_gpu.cpu()).item(), delta=10e-6
        )
        self.assertAlmostEqual(
            0.0, torch.sum(outputs_cpu - outputs_back_on_cpu).item(), delta=10e-6
        )

    def test_batch_independence(self):
        inputs = self.test_inputs.clone()
        inputs.requires_grad = True

        # Compute forward pass in eval mode to deactivate batch norm
        self.net.eval()
        outputs = self.net(inputs)
        self.net.train()

        # Mask loss for certain samples in batch
        batch_size = inputs.shape[0]
        mask_idx = torch.randint(0, batch_size, ())
        mask = torch.ones_like(outputs)
        mask[mask_idx] = 0
        outputs = outputs * mask

        # Compute backward pass
        loss = outputs.mean()
        loss.backward()

        # Check if gradient exists and is zero for masked samples
        for i, grad in enumerate(inputs.grad):
            if i == mask_idx:
                self.assertTrue(torch.all(grad == 0).item())
            else:
                self.assertTrue(not torch.all(grad == 0))

    def test_all_parameters_updated(self):
        optim = torch.optim.SGD(self.net.parameters(), lr=0.1)

        outputs = self.net(self.test_inputs)
        loss = outputs.mean()
        loss.backward()
        optim.step()

        for param_name, param in self.net.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    self.assertNotEqual(0.0, torch.sum(param.grad ** 2))


class TestEncoder(unittest.TestCase, ModelTestsMixin):
    def setUp(self):
        self.net = networks.Encoder(14, 16, 3, 6, 64, 30, 0.1, True)
        self.test_inputs = torch.randn(16, 14, 30)
        self.expected_shape = (16, 64)


class TestDecoder(unittest.TestCase, ModelTestsMixin):
    def setUp(self):
        self.net = networks.Decoder(14, 16, 3, 6, 64, 30, 0.1)
        self.test_inputs = torch.randn(16, 64)
        self.expected_shape = (16, 14, 30)


class TestRegressor(unittest.TestCase, ModelTestsMixin):
    def setUp(self):
        self.net = networks.Regressor(64)
        self.test_inputs = torch.randn(16, 64)
        self.expected_shape = (16,)
