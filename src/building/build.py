import pytorch_lightning as pl
import torch
from pytorch_probgraph import (
    InteractionModule,
    RestrictedBoltzmannMachineCD,
)

import datasets
from building.build_common import add_hparams, build_trainer, get_logdir
from lightning import autoencoder, baseline, loggers, pretraining
from models.layers import GaussianSequenceLayer, RectifiedLinearLayer


def build_baseline(
    source,
    fails,
    config,
    encoder,
    pretrained_encoder_path,
    gpu,
    seed,
    version,
    record_embeddings=False,
):
    logger = loggers.MLTBLogger(
        get_logdir(), loggers.baseline_experiment_name(source), tag=version
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val/regression_loss")
    trainer = build_trainer(
        logger,
        checkpoint_callback,
        max_epochs=100,
        val_interval=1.0,
        gpu=gpu,
        seed=seed,
    )
    data = datasets.BaselineDataModule(
        fd_source=source, batch_size=config["batch_size"], percent_fail_runs=fails
    )
    model = build_baseline_from_config(
        config, data.window_size, encoder, pretrained_encoder_path, record_embeddings
    )
    add_hparams(model, data, seed)

    return trainer, data, model


def build_baseline_from_config(
    config, seq_len, encoder, pretrained_encoder_path, record_embeddings=False
):
    model = baseline.Baseline(
        in_channels=14,
        seq_len=seq_len,
        num_layers=config["num_layers"],
        kernel_size=3,
        base_filters=config["base_filters"],
        latent_dim=config["latent_dim"],
        dropout=config["dropout"],
        optim_type="adam",
        lr=config["lr"],
        record_embeddings=record_embeddings,
        encoder=encoder,
    )
    if pretrained_encoder_path is not None:
        model.load_encoder(pretrained_encoder_path)

    return model


def build_pretraining(
    source,
    target,
    percent_broken,
    percent_fail_runs,
    arch_config,
    config,
    encoder,
    mode,
    record_embeddings,
    gpu,
    seed,
    version,
):
    logger = loggers.MLTBLogger(
        get_logdir(),
        loggers.pretraining_experiment_name(source, target),
        tag=version,
        tensorboard_struct={"pb": percent_broken, "dt": config["domain_tradeoff"]},
    )
    checkpoint_callback = loggers.MinEpochModelCheckpoint(
        monitor="val/checkpoint_score", min_epochs_before_saving=1
    )
    trainer = build_trainer(
        logger,
        checkpoint_callback,
        max_epochs=100,
        val_interval=1.0,
        gpu=gpu,
        seed=seed,
    )
    truncate_val = not record_embeddings
    distance_mode = config["distance_mode"]
    min_distance = config["min_distance"] if "min_distance" in config else 1
    data = build_datamodule(
        source,
        target,
        percent_broken,
        percent_fail_runs,
        config["batch_size"],
        truncate_val,
        distance_mode,
        min_distance,
    )
    use_adaption = target is not None
    if mode == "metric":
        model = build_pretraining_from_config(
            arch_config,
            config,
            data.window_size,
            encoder,
            record_embeddings,
            use_adaption,
        )
    elif mode == "autoencoder":
        model = build_autoencoder_from_config(
            arch_config,
            config,
            data.window_size,
            encoder,
            record_embeddings,
            use_adaption,
        )
    else:
        raise ValueError(f"Unrecognized pre-training mode {mode}.")

    add_hparams(model, data, seed)

    return trainer, data, model


def build_datamodule(
    source,
    target,
    percent_broken,
    percent_fail_runs,
    batch_size,
    truncate_val,
    distance_mode,
    min_distance,
    window_size=None,
):
    if target is None:
        return datasets.PretrainingBaselineDataModule(
            fd_source=source,
            num_samples=25000,
            batch_size=batch_size,
            window_size=window_size,
            min_distance=min_distance,
            percent_broken=percent_broken,
            percent_fail_runs=percent_fail_runs,
            truncate_val=truncate_val,
            distance_mode=distance_mode,
        )
    else:
        return datasets.PretrainingAdaptionDataModule(
            fd_source=source,
            fd_target=target,
            num_samples=50000,
            batch_size=batch_size,
            window_size=window_size,
            min_distance=min_distance,
            percent_broken=percent_broken,
            percent_fail_runs=percent_fail_runs,
            truncate_target_val=truncate_val,
            distance_mode=distance_mode,
        )


def build_pretraining_from_config(
    arch_config, config, seq_len, encoder, record_embeddings, use_adaption
):
    model = pretraining.UnsupervisedPretraining(
        in_channels=14,
        seq_len=seq_len,
        num_layers=arch_config["num_layers"],
        kernel_size=3,
        base_filters=arch_config["base_filters"],
        latent_dim=arch_config["latent_dim"],
        dropout=config["dropout"],
        domain_tradeoff=config["domain_tradeoff"] if use_adaption else 0.0,
        domain_disc_dim=arch_config["latent_dim"],
        num_disc_layers=arch_config["num_disc_layers"],
        lr=config["lr"],
        weight_decay=0.0,
        record_embeddings=record_embeddings,
        encoder=encoder,
    )

    return model


def build_autoencoder_from_config(
    arch_config, config, seq_len, encoder, record_embeddings, use_adaption
):
    model = autoencoder.AutoencoderPretraining(
        in_channels=14,
        seq_len=seq_len,
        num_layers=arch_config["num_layers"],
        kernel_size=3,
        base_filters=arch_config["base_filters"],
        latent_dim=arch_config["latent_dim"],
        dropout=config["dropout"],
        domain_tradeoff=config["domain_tradeoff"] if use_adaption else 0.0,
        domain_disc_dim=arch_config["latent_dim"],
        num_disc_layers=arch_config["num_disc_layers"],
        lr=config["lr"],
        weight_decay=0.0,
        record_embeddings=record_embeddings,
        encoder=encoder,
    )

    return model


def build_rbm(in_units, out_units, he_init=False):
    l0bias = torch.zeros([1, in_units, 1])
    l0bias.requires_grad = True
    l1bias = torch.zeros([1, out_units, 1])
    l1bias.requires_grad = True

    l0 = GaussianSequenceLayer(l0bias, torch.ones_like(l0bias))
    l0.logsigma.requires_grad = False
    l1 = RectifiedLinearLayer(l1bias)

    module = torch.nn.Conv1d(14, out_units, kernel_size=3, bias=False)
    i0 = InteractionModule(module, l0.bias.shape[1:])
    if he_init:
        torch.nn.init.kaiming_uniform_(i0.weight, nonlinearity="relu")

    # build the RBM
    rbm = RestrictedBoltzmannMachineCD(l0, l1, i0, ksteps=1)

    return rbm
