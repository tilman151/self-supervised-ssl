import unittest
from unittest import mock

import torch

from lightning import autoencoder


class TestUnsupervisedPretraining(unittest.TestCase):
    def setUp(self):
        self.net = autoencoder.AutoencoderPretraining(
            in_channels=14,
            seq_len=30,
            num_layers=4,
            kernel_size=3,
            base_filters=16,
            latent_dim=64,
            dropout=0.1,
            domain_tradeoff=0.0,
            domain_disc_dim=32,
            num_disc_layers=2,
            weight_decay=0,
            lr=0.01,
        )

    def test_mode_hparam(self):
        self.assertLess(1, len(self.net.hparams))
        self.assertIn("mode", self.net.hparams)
        self.assertEqual("autoencoder", self.net.hparams["mode"])

    @torch.no_grad()
    def test_encoder(self):
        inputs = torch.randn(16, 14, 30)
        outputs = self.net.encoder(inputs)
        self.assertEqual(torch.Size((16, 64)), outputs.shape)

    @torch.no_grad()
    def test_decoder(self):
        inputs = torch.randn(16, 64)
        outputs = self.net.decoder(inputs)
        self.assertEqual(torch.Size((16, 14, 30)), outputs.shape)

    @torch.no_grad()
    def test_forward(self):
        inputs = torch.randn(16, 14, 30)
        outputs = self.net(inputs)
        self.assertEqual(torch.Size((16, 14, 30)), outputs.shape)

    def test_batch_independence(self):
        torch.autograd.set_detect_anomaly(True)

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

        torch.autograd.set_detect_anomaly(False)

    def test_all_parameters_updated(self):
        optim = torch.optim.SGD(self.net.parameters(), lr=0.1)

        inputs = (
            torch.randn(16, 14, 30),
            torch.randn(16, 14, 30),
            torch.randn(16),
            torch.randn(16),
        )
        loss = self.net.training_step(inputs, batch_idx=0)
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
        num_batches = 600
        embeddings = torch.randn(num_batches, 64, 64)
        reconstructions = torch.randn(num_batches, 64, 14, 30)
        domain_predictions = torch.randn(num_batches, 32)
        self._mock_predictions(embeddings, reconstructions, domain_predictions)

        self._feed_dummy_val(num_batches)

        regression_loss = torch.mean(reconstructions ** 2)
        domain_loss = torch.tensor(0.0)
        checkpoint_score = regression_loss - 0.1 * domain_loss
        expected_logs = {
            "val/regression_loss": (regression_loss, 0.0001),
            "val/domain_loss": (domain_loss, 0.0001),
            "val/checkpoint_score": (checkpoint_score, 0.0001),
        }
        self.net.validation_epoch_end([])

        self._assert_logs(mock_log, expected_logs)

    def test_metric_val_updates(self):
        num_batches = 600
        embeddings = torch.randn(num_batches, 64, 64)
        reconstructions = torch.randn(num_batches, 64, 14, 30)
        domain_predictions = torch.randn(num_batches, 32)
        self._mock_predictions(embeddings, reconstructions, domain_predictions)

        self._feed_dummy_val(num_batches)
        self.assertEqual(num_batches, self.net.regression_metric.sample_counter)
        self.assertEqual(num_batches, self.net.domain_metric.sample_counter)

    def _mock_predictions(self, embeddings, reconstructions, domain_predictions):
        self.net.encoder.forward = mock.MagicMock(side_effect=embeddings)
        self.net.decoder.forward = mock.MagicMock(side_effect=reconstructions)

    def _feed_dummy_val(self, num_batches):
        for i in range(num_batches):
            self.net.validation_step(
                (
                    torch.zeros(32, 14, 30),
                    torch.zeros(32, 14, 30),
                    torch.zeros(32),
                    torch.zeros(32),
                ),
                batch_idx=i,
                dataloader_idx=0,
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
