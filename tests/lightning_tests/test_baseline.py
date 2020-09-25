import unittest
from unittest import mock

import torch

import building
from lightning import baseline


class TestBaselineTemplate:
    @torch.no_grad()
    def test_encoder(self):
        inputs = torch.randn(16, 14, 30)
        outputs = self.net.encoder(inputs)
        self.assertEqual(self.encoder_shape, outputs.shape)

    @torch.no_grad()
    def test_regressor(self):
        inputs = torch.randn(*self.encoder_shape)
        outputs = self.net.regressor(inputs)
        self.assertEqual(torch.Size((16,)), outputs.shape)

    def test_batch_independence(self):
        inputs = torch.randn(16, 14, 30)
        inputs.requires_grad = True

        # Compute forward pass in eval mode to deactivate batch norm
        self.net.eval()
        outputs = self.net(inputs)
        self.net.train()

        # Mask loss for certain samples in batch
        batch_size = outputs.shape[0]
        mask_idx = torch.randint(0, batch_size, ())
        mask = torch.ones_like(outputs)
        mask[mask_idx] = 0
        output = outputs * mask

        # Compute backward pass
        loss = output.mean()
        loss.backward(retain_graph=True)

        # Check if gradient exists and is zero for masked samples
        for i, grad in enumerate(inputs.grad[:batch_size]):
            if i == mask_idx:
                self.assertTrue(torch.all(grad == 0).item())
            else:
                self.assertTrue(not torch.all(grad == 0))
        inputs.grad = None

    def test_all_parameters_updated(self):
        optim = torch.optim.SGD(self.net.parameters(), lr=0.1)

        loss = self.net.training_step(
            (torch.randn(16, 14, 30), torch.ones(16)), batch_idx=0
        )
        loss.backward()
        optim.step()

        for param_name, param in self.net.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    self.assertNotEqual(0.0, torch.sum(param.grad ** 2))

    @mock.patch("pytorch_lightning.LightningModule.log")
    @torch.no_grad()
    def test_metric_val_reduction(self, mock_log):
        source_batches = 600
        source_prediction = torch.randn(source_batches, 32) + 30
        self._mock_predictions(source_prediction)

        self._feed_dummy_val(source_batches)

        source_regression_loss = torch.sqrt(torch.mean(source_prediction ** 2))
        expected_logs = {
            "val/regression_loss": (source_regression_loss, 0.001),
        }
        self.net.validation_epoch_end([])

        self._assert_logs(mock_log, expected_logs)

    def test_metric_val_updates(self):
        source_batches = 600
        source_prediction = torch.randn(source_batches, 32) + 30
        self._mock_predictions(source_prediction)

        self._feed_dummy_val(source_batches)
        self.assertEqual(source_batches, self.net.regression_metrics[1].sample_counter)

    @mock.patch("pytorch_lightning.LightningModule.log")
    @torch.no_grad()
    def test_metric_test_reduction(self, mock_log):
        source_batches = 600
        source_prediction = [
            torch.randn(source_batches, 32) + i * 10 for i in range(1, 5)
        ]
        self._mock_predictions(torch.cat(source_prediction))

        self._feed_dummy_test(source_batches)

        expected_logs = {}
        for i in range(4):
            source_regression_loss = torch.sqrt(torch.mean(source_prediction[i] ** 2))
            metric_name = f"test/regression_loss_fd{i+1}"
            expected_logs[metric_name] = (source_regression_loss, 0.001)
        self.net.test_epoch_end([])

        self._assert_logs(mock_log, expected_logs)

    def test_metric_test_updates(self):
        source_batches = 600
        source_prediction = [
            torch.randn(source_batches, 32) + i * 10 for i in range(1, 5)
        ]
        self._mock_predictions(torch.cat(source_prediction))

        self._feed_dummy_test(source_batches)
        for fd in range(1, 5):
            self.assertEqual(
                source_batches, self.net.regression_metrics[fd].sample_counter
            )

    def _mock_predictions(self, prediction):
        self.net.regressor.forward = mock.MagicMock(side_effect=prediction)

    def _feed_dummy_val(self, num_batches):
        for i in range(num_batches):
            self.net.validation_step(
                (torch.zeros(32, 14, 30), torch.zeros(32)),
                batch_idx=i,
            )

    def _feed_dummy_test(self, num_batches):
        for fd in range(4):
            for i in range(num_batches):
                self.net.test_step(
                    (torch.zeros(32, 14, 30), torch.zeros(32)),
                    batch_idx=i,
                    dataloader_idx=fd,
                )

    def _assert_logs(self, mock_log, expected_logs):
        mock_log.assert_called()
        for call in mock_log.mock_calls:
            metric = call[1][0]
            self.assertIn(metric, expected_logs, "Unexpected logged metric found.")
            expected_value, delta = expected_logs[metric]
            expected_value = expected_value.item()
            actual_value = call[1][1].item()
            with self.subTest(metric):
                self.assertAlmostEqual(expected_value, actual_value, delta=delta)


class TestCnnBaseline(unittest.TestCase, TestBaselineTemplate):
    def setUp(self):
        self.encoder_shape = torch.Size((16, 64))
        self.trade_off = 0.5
        self.net = baseline.Baseline(
            in_channels=14,
            seq_len=30,
            num_layers=4,
            kernel_size=3,
            base_filters=16,
            latent_dim=64,
            dropout=0.1,
            optim_type="adam",
            lr=0.01,
        )

    def test_load_from_rbm(self):
        self.net.encoder.layers[1].bias.data = torch.randn_like(
            self.net.encoder.layers[1].bias.data
        )
        old_weight_shape = self.net.encoder.layers[0].weight.shape
        old_bias_shape = self.net.encoder.layers[1].bias.shape
        rbm = building.build_rbm(14, 16)
        self.assertNotEqual(
            0,
            torch.dist(rbm.interaction.module.weight, self.net.encoder.layers[0].weight),
        )
        self.assertNotEqual(
            0, torch.dist(rbm.hidden.bias, self.net.encoder.layers[1].bias)
        )
        self.assertTrue(self.net.encoder.layers[0].weight.requires_grad)
        self.assertTrue(self.net.encoder.layers[1].bias.requires_grad)

        self.net.load_from_rbm(rbm)
        self.assertEqual(
            0,
            torch.dist(rbm.interaction.module.weight, self.net.encoder.layers[0].weight),
        )
        self.assertEqual(0, torch.dist(rbm.hidden.bias, self.net.encoder.layers[1].bias))
        self.assertTrue(self.net.encoder.layers[0].weight.requires_grad)
        self.assertTrue(self.net.encoder.layers[1].bias.requires_grad)
        self.assertEqual(old_weight_shape, self.net.encoder.layers[0].weight.shape)
        self.assertEqual(old_bias_shape, self.net.encoder.layers[1].bias.shape)
