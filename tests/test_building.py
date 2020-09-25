import unittest
from unittest import mock

from building import build, get_logdir
from lightning import loggers


class TestBuildingFunctions(unittest.TestCase):
    def setUp(self):
        self.config = {
            "num_layers": 8,
            "base_filters": 16,
            "domain_tradeoff": 1.0,
            "latent_dim": 32,
            "dropout": 0.1,
            "num_disc_layers": 1,
            "lr": 0.01,
            "batch_size": 512,
        }
        self.pretraining_config = {
            "domain_tradeoff": 2.0,
            "dropout": 0.3,
            "lr": 0.02,
            "batch_size": 256,
            "distance_mode": "piecewise",
        }

    @mock.patch("lightning.loggers.MinEpochModelCheckpoint")
    @mock.patch("lightning.loggers.MLTBLogger")
    @mock.patch("building.build.build_pretraining_from_config")
    @mock.patch("building.build.build_datamodule")
    @mock.patch("building.build.build_trainer")
    def test_build_pretraining_metric(
        self,
        mock_build_trainer,
        mock_datamodule,
        mock_pretraining_from_config,
        mock_logger,
        mock_checkpoint,
    ):
        self._check_build_pretraining(
            mock_build_trainer,
            mock_checkpoint,
            mock_datamodule,
            mock_logger,
            mock_pretraining_from_config,
            "metric",
        )

    @mock.patch("lightning.loggers.MinEpochModelCheckpoint")
    @mock.patch("lightning.loggers.MLTBLogger")
    @mock.patch("building.build.build_autoencoder_from_config")
    @mock.patch("building.build.build_datamodule")
    @mock.patch("building.build.build_trainer")
    def test_build_pretraining_autoencoder(
        self,
        mock_build_trainer,
        mock_datamodule,
        mock_pretraining_from_config,
        mock_logger,
        mock_checkpoint,
    ):
        self._check_build_pretraining(
            mock_build_trainer,
            mock_checkpoint,
            mock_datamodule,
            mock_logger,
            mock_pretraining_from_config,
            "autoencoder",
        )

    def _check_build_pretraining(
        self,
        mock_build_trainer,
        mock_checkpoint,
        mock_datamodule,
        mock_logger,
        mock_pretraining_from_config,
        mode,
    ):
        mock_logger.return_value = "mock_logger"
        mock_checkpoint.return_value = "mock_checkpoint"
        build.build_pretraining(
            2,
            1,
            0.8,
            0.5,
            self.config,
            self.pretraining_config,
            "cnn",
            mode,
            False,
            1,
            42,
            "version",
        )
        mock_logger.assert_called_with(
            get_logdir(),
            loggers.pretraining_experiment_name(2, 1),
            tag="version",
            tensorboard_struct={
                "pb": 0.8,
                "dt": self.pretraining_config["domain_tradeoff"],
            },
        )
        mock_build_trainer.assert_called_with(
            "mock_logger",
            "mock_checkpoint",
            max_epochs=100,
            val_interval=1.0,
            gpu=1,
            seed=42,
        )
        mock_datamodule.assert_called_with(
            2,
            1,
            0.8,
            0.5,
            self.pretraining_config["batch_size"],
            True,
            "piecewise",
            1,
        )
        mock_pretraining_from_config.assert_called_with(
            self.config,
            self.pretraining_config,
            mock_datamodule().window_size,
            "cnn",
            False,
            True,
        )

    @mock.patch("lightning.pretraining.UnsupervisedPretraining")
    def test_build_pretraining_from_config(self, mock_pretraining):
        self._check_pretraining_from_config(
            build.build_pretraining_from_config, mock_pretraining
        )

    @mock.patch("lightning.autoencoder.AutoencoderPretraining")
    def test_build_autoencoder_from_config(self, mock_autoencoder):
        self._check_pretraining_from_config(
            build.build_autoencoder_from_config, mock_autoencoder
        )

    def _check_pretraining_from_config(self, building_func, mock_pretraining):
        building_func(
            self.config, self.pretraining_config, 30, "cnn", False, use_adaption=True
        )
        mock_pretraining.assert_called_with(
            in_channels=14,
            seq_len=30,
            num_layers=self.config["num_layers"],
            kernel_size=3,
            base_filters=self.config["base_filters"],
            latent_dim=self.config["latent_dim"],
            encoder="cnn",
            dropout=self.pretraining_config["dropout"],
            domain_tradeoff=self.pretraining_config["domain_tradeoff"],
            domain_disc_dim=self.config["latent_dim"],
            num_disc_layers=self.config["num_disc_layers"],
            lr=self.pretraining_config["lr"],
            weight_decay=0.0,
            record_embeddings=False,
        )
        building_func(
            self.config, self.pretraining_config, 30, "cnn", False, use_adaption=False
        )
        mock_pretraining.assert_called_with(
            in_channels=14,
            seq_len=30,
            num_layers=self.config["num_layers"],
            kernel_size=3,
            base_filters=self.config["base_filters"],
            latent_dim=self.config["latent_dim"],
            encoder="cnn",
            dropout=self.pretraining_config["dropout"],
            domain_tradeoff=0.0,
            domain_disc_dim=self.config["latent_dim"],
            num_disc_layers=self.config["num_disc_layers"],
            lr=self.pretraining_config["lr"],
            weight_decay=0.0,
            record_embeddings=False,
        )

    @mock.patch("pytorch_lightning.callbacks.ModelCheckpoint")
    @mock.patch("lightning.loggers.MLTBLogger")
    @mock.patch("building.build.build_baseline_from_config")
    @mock.patch("datasets.BaselineDataModule")
    @mock.patch("building.build.build_trainer")
    def test_build_baseline(
        self,
        mock_build_trainer,
        mock_datamodule,
        mock_baseline_from_config,
        mock_logger,
        mock_checkpoint,
    ):
        mock_logger.return_value = "mock_logger"
        mock_checkpoint.return_value = "mock_checkpoint"
        with self.subTest("no_encoder"):
            build.build_baseline(2, 1.0, self.config, "cnn", None, 1, 42, "version")
            mock_logger.assert_called_with(
                get_logdir(),
                loggers.baseline_experiment_name(2),
                tag="version",
            )
            mock_build_trainer.assert_called_with(
                "mock_logger",
                "mock_checkpoint",
                max_epochs=100,
                val_interval=1.0,
                gpu=1,
                seed=42,
            )
            mock_datamodule.assert_called_with(
                fd_source=2, percent_fail_runs=1.0, batch_size=self.config["batch_size"]
            )
            mock_baseline_from_config.assert_called_with(
                self.config,
                mock_datamodule().window_size,
                "cnn",
                None,
                False,
            )
        with self.subTest("with_encoder"):
            build.build_baseline(
                2, 1.0, self.config, "cnn", "encoder_path", 1, 42, "version"
            )
            mock_build_trainer.assert_called_with(
                "mock_logger",
                "mock_checkpoint",
                max_epochs=100,
                val_interval=1.0,
                gpu=1,
                seed=42,
            )
            mock_datamodule.assert_called_with(
                fd_source=2, percent_fail_runs=1.0, batch_size=self.config["batch_size"]
            )
            mock_baseline_from_config.assert_called_with(
                self.config,
                mock_datamodule().window_size,
                "cnn",
                "encoder_path",
                False,
            )

    @mock.patch("lightning.baseline.Baseline")
    def test_build_baseline_from_config(self, mock_baseline):
        with self.subTest("no_encoder"):
            build.build_baseline_from_config(self.config, 30, "cnn", None)
            self._assert_baseline_build_correctly(mock_baseline)
            mock_baseline().load_encoder.assert_not_called()
        with self.subTest("with_encoder"):
            build.build_baseline_from_config(self.config, 30, "cnn", "encoder_path")
            self._assert_baseline_build_correctly(mock_baseline)
            mock_baseline().load_encoder.assert_called_with("encoder_path")

    def _assert_baseline_build_correctly(self, mock_baseline):
        mock_baseline.assert_called_with(
            in_channels=14,
            seq_len=30,
            num_layers=self.config["num_layers"],
            kernel_size=3,
            base_filters=self.config["base_filters"],
            latent_dim=self.config["latent_dim"],
            dropout=self.config["dropout"],
            optim_type="adam",
            lr=self.config["lr"],
            record_embeddings=False,
            encoder="cnn",
        )
