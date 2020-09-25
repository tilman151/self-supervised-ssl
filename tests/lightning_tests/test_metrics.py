import unittest

import matplotlib.pyplot as plt
import torch

from lightning import metrics


class TestEmbeddingViz(unittest.TestCase):
    def setUp(self):
        self.metric = metrics.EmbeddingViz(32, 128)

    def test_updating(self):
        embeddings = torch.randn(16, 128)
        labels = torch.ones(16)
        ruls = torch.arange(0, 16)

        self.metric.update(embeddings, labels, ruls)
        self.metric.update(embeddings, labels, ruls)

        self.assertEqual(0, torch.sum(self.metric.embeddings[:16] - embeddings))
        self.assertEqual(0, torch.sum(self.metric.labels[:16] - labels))
        self.assertEqual(0, torch.sum(self.metric.embeddings[16:32] - embeddings))
        self.assertEqual(0, torch.sum(self.metric.labels[16:32] - labels))

    def test_compute(self):
        embeddings = torch.randn(32, 128)
        labels = torch.cat([torch.ones(16), torch.zeros(16)])
        ruls = torch.arange(0, 32)

        self.metric.update(embeddings, labels, ruls)
        fig = self.metric.compute()

        self.assertIsInstance(fig, plt.Figure)
        self.assertListEqual([20.0, 10.0], list(fig.get_size_inches()))


class TestRULScore(unittest.TestCase):
    def setUp(self):
        self.metric = metrics.RULScore()

    def test_negative(self):
        inputs = torch.ones(2)
        targets = torch.ones(2) * 2
        actual_score = self.metric(inputs, targets)
        expected_score = torch.sum(
            torch.exp((inputs - targets) / self.metric.neg_factor) - 1
        )
        self.assertEqual(expected_score, actual_score)

    def test_positive(self):
        inputs = torch.ones(2) * 2
        targets = torch.ones(2)
        actual_score = self.metric(inputs, targets)
        expected_score = torch.sum(
            torch.exp((inputs - targets) / self.metric.pos_factor) - 1
        )
        self.assertEqual(expected_score, actual_score)


class TestRMSE(unittest.TestCase):
    def setUp(self):
        self.metric = metrics.RMSELoss()

    def test_update(self):
        expected_sse, batch_size = self._add_one_batch()
        self.assertEqual(1, self.metric.sample_counter)
        self.assertEqual(expected_sse, self.metric.losses[0])
        self.assertEqual(batch_size, self.metric.sizes[0])

    def test_reset(self):
        self._add_one_batch()
        self.metric.reset()
        self.assertEqual(0, self.metric.sample_counter)
        self.assertEqual(0, self.metric.losses.sum())
        self.assertEqual(0, self.metric.sizes.sum())

    def _add_one_batch(self):
        batch_size = 100
        inputs = torch.ones(batch_size) * 2
        targets = torch.zeros_like(inputs)
        summed_squares = torch.sum((inputs - targets) ** 2)

        self.metric.update(inputs, targets)

        return summed_squares, batch_size

    def test_compute(self):
        batch_sizes = [3000, 3000, 3000, 1000]
        inputs = torch.randn(sum(batch_sizes)) + 100
        targets = torch.randn_like(inputs)
        expected_rmse = torch.sqrt(torch.mean((inputs - targets) ** 2))

        batched_inputs = torch.split(inputs, batch_sizes)
        batched_targets = torch.split(targets, batch_sizes)
        for inp, tgt in zip(batched_inputs, batched_targets):
            self.metric.update(inp, tgt)
        actual_rmse = self.metric.compute()

        self.assertAlmostEqual(expected_rmse.item(), actual_rmse.item(), delta=0.001)

    def test_compute_fails_on_empty_metric(self):
        with self.assertRaises(RuntimeError):
            self.metric.compute()


class TestMeanMetric(unittest.TestCase):
    def setUp(self):
        self.mean_metric = metrics.SimpleMetric()
        self.sum_metric = metrics.SimpleMetric(reduction="sum")

    def test_update(self):
        expected_loss, batch_size = self._add_one_batch()
        self.assertEqual(1, self.mean_metric.sample_counter)
        self.assertEqual(expected_loss, self.mean_metric.losses[0])
        self.assertEqual(batch_size, self.mean_metric.sizes[0])

    def test_reset(self):
        self._add_one_batch()
        self.mean_metric.reset()
        self.assertEqual(0, self.mean_metric.sample_counter)
        self.assertEqual(0, self.mean_metric.losses.sum())
        self.assertEqual(0, self.mean_metric.sizes.sum())

    def _add_one_batch(self):
        batch_size = 100
        loss = torch.tensor(500)

        self.mean_metric.update(loss, batch_size)

        return loss, batch_size

    def test_compute_mean(self):
        batch_sizes = [512] * 50 + [100]
        losses = torch.randn(sum(batch_sizes)) + 2
        expected_loss = losses.mean()

        batched_inputs = torch.split(losses, batch_sizes)
        for inp, sizes in zip(batched_inputs, batch_sizes):
            self.mean_metric.update(inp.mean(), sizes)
        actual_loss = self.mean_metric.compute()

        self.assertAlmostEqual(expected_loss.item(), actual_loss.item(), places=5)

    def test_compute_sum(self):
        batch_sizes = [512] * 50 + [100]
        losses = torch.randn(sum(batch_sizes)) + 2
        expected_loss = losses.sum()

        batched_inputs = torch.split(losses, batch_sizes)
        for inp, sizes in zip(batched_inputs, batch_sizes):
            self.sum_metric.update(inp.sum(), sizes)
        actual_loss = self.sum_metric.compute()

        self.assertAlmostEqual(expected_loss.item(), actual_loss.item(), delta=0.1)

    def test_compute_fails_on_empty_metric(self):
        with self.assertRaises(RuntimeError):
            self.mean_metric.compute()
        with self.assertRaises(RuntimeError):
            self.sum_metric.compute()
