import os
from functools import partial

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

import building
import datasets
from lightning import loggers


def tune_supervised(config, source, arch_config, encoder):
    arch_config.update(config)

    logger = pl_loggers.TensorBoardLogger(
        _get_hyperopt_logdir(),
        loggers.baseline_experiment_name(source),
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val/regression_loss")
    tune_callback = TuneReportCallback(
        {
            "reg_loss": "val/regression_loss",
        },
        on="validation_end",
    )
    trainer = building.build_trainer(
        logger,
        checkpoint_callback,
        max_epochs=100,
        val_interval=1.0,
        gpu=1,
        seed=None,
        callbacks=[tune_callback],
        check_sanity=False,
    )

    data = datasets.BaselineDataModule(
        fd_source=source, batch_size=arch_config["batch_size"]
    )
    model = building.build_baseline_from_config(
        arch_config, data.window_size, encoder, None
    )
    building.add_hparams(model, data, None)

    trainer.fit(model, datamodule=data)


def _get_hyperopt_logdir():
    script_path = os.path.dirname(__file__)
    log_dir = os.path.normpath(os.path.join(script_path, "..", "..", "hyperopt"))

    return log_dir


def optimize_supervised(source, arch_config, encoder, num_trials):
    config = {
        "dropout": tune.quniform(0.0, 0.5, 0.1),
        "lr": tune.qloguniform(1e-4, 1e-1, 5e-5),
        "batch_size": tune.choice([64, 128, 256, 512]),
    }

    scheduler = tune.schedulers.ASHAScheduler(
        max_t=100, grace_period=10, reduction_factor=2
    )
    reporter = tune.CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["reg_loss"],
    )

    tune_func = partial(
        tune_supervised,
        source=source,
        arch_config=arch_config,
        encoder=encoder,
    )
    analysis = tune.run(
        tune_func,
        resources_per_trial={"cpu": 3, "gpu": 0.5},
        metric="reg_loss",
        mode="min",
        config=config,
        num_samples=num_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_supervised_asha",
        fail_fast=True,
    )

    print("Best hyperparameters found were: ", analysis.best_config)

    return analysis.best_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for supervised training"
    )
    parser.add_argument("--source", type=int, required=True, help="source FD number")
    parser.add_argument(
        "--arch_config", required=True, help="path to architecture base config"
    )
    parser.add_argument(
        "--encoder",
        default="cnn",
        choices=["cnn", "lstm"],
        help="encoder type",
    )
    parser.add_argument(
        "--num_trials", type=int, required=True, help="number of hyperopt trials"
    )
    opt = parser.parse_args()

    _arch_config = building.load_config(opt.arch_config)
    optimize_supervised(opt.source, _arch_config, opt.encoder, opt.num_trials)
