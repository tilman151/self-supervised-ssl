import pytorch_lightning as pl
import torch

from lightning import metrics
from lightning.mixins import LoadEncoderMixin
from models import networks


class Baseline(pl.LightningModule, LoadEncoderMixin):
    def __init__(
        self,
        in_channels,
        seq_len,
        num_layers,
        kernel_size,
        base_filters,
        latent_dim,
        dropout=0.0,
        optim_type="adam",
        lr=0.001,
        record_embeddings=False,
        encoder="cnn",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.optim_type = optim_type
        self.lr = lr
        self.record_embeddings = record_embeddings
        self.encoder_type = encoder

        self.encoder = self._get_encoder()
        self.regressor = networks.Regressor(latent_dim)

        self.criterion_regression = metrics.RMSELoss(num_elements=0)

        self.embedding_metric = metrics.EmbeddingViz(
            20000, self.latent_dim, combined=False
        )
        self.regression_metrics = {i: metrics.RMSELoss() for i in range(1, 5)}

        self.save_hyperparameters()

    def _get_encoder(self):
        if "cnn" in self.encoder_type:
            encoder = networks.Encoder(
                self.in_channels,
                self.base_filters,
                self.kernel_size,
                self.num_layers,
                self.latent_dim,
                self.seq_len,
                dropout=self.dropout,
                norm_outputs=False,
            )
        else:
            raise ValueError(
                f"Unknown encoder type {self.encoder_type}. Must contain 'cnn'."
            )

        return encoder

    def add_data_hparams(self, data):
        self.hparams.update(data.hparams)

    @property
    def example_input_array(self):
        common = torch.randn(16, self.in_channels, self.seq_len)

        return common

    def configure_optimizers(self):
        param_groups = [
            {"params": self.encoder.parameters()},
            {"params": self.regressor.parameters()},
        ]
        if self.optim_type == "adam":
            return torch.optim.Adam(param_groups, lr=self.lr)
        else:
            return torch.optim.SGD(
                param_groups, lr=self.lr, momentum=0.9, weight_decay=0.01
            )

    def forward(self, inputs):
        latent_code = self.encoder(inputs)
        prediction = self.regressor(latent_code)

        return prediction

    def training_step(self, batch, batch_idx):
        source, source_labels = batch
        predictions = self(source)
        loss = self.criterion_regression(predictions, source_labels)

        self.log("train/regression_loss", loss)

        return loss

    def on_validation_epoch_start(self):
        self._reset_all_metrics()

    def on_test_epoch_start(self):
        self._reset_all_metrics()

    def _reset_all_metrics(self):
        self.embedding_metric.reset()
        for metric in self.regression_metrics.values():
            metric.reset()

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, metric_id=1)

    def test_step(self, batch, batch_idx, dataloader_idx):
        self._evaluate(batch, metric_id=dataloader_idx + 1)

    def _evaluate(self, batch, metric_id):
        features, labels = batch
        predictions = self(features)
        self.regression_metrics[metric_id].update(predictions, labels)
        if self.record_embeddings:
            latent_code = self.encoder(features)
            self.embedding_metric.update(latent_code, torch.zeros_like(labels), labels)

    def validation_epoch_end(self, outputs):
        self.log("val/regression_loss", self.regression_metrics[1].compute())
        if self.record_embeddings:
            fig_class, fig_rul = self.embedding_metric.compute()
            self.logger.log_figure("val/embeddings_class", fig_class, self.global_step)
            self.logger.log_figure("val/embeddings_rul", fig_rul, self.global_step)
            self.embedding_metric.reset()

    def test_epoch_end(self, outputs):
        for fd, metric in self.regression_metrics.items():
            self.log(f"test/regression_loss_fd{fd}", metric.compute())
