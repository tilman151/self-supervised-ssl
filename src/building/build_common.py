import json
import os

import pytorch_lightning as pl


def build_trainer(
    logger,
    checkpoint_callback,
    max_epochs,
    val_interval,
    gpu,
    seed=None,
    callbacks=None,
    check_sanity=True,
):
    if seed is not None:
        pl.trainer.seed_everything(seed)
    trainer = pl.Trainer(
        num_sanity_val_steps=2 if check_sanity else 0,
        gpus=[gpu],
        max_epochs=max_epochs,
        logger=logger,
        deterministic=True,
        log_every_n_steps=10,
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=1.0,
        val_check_interval=val_interval,
        callbacks=callbacks,
    )

    return trainer


def add_hparams(model, data, seed):
    model.add_data_hparams(data)
    model.hparams.update({"seed": seed})


def get_logdir():
    script_path = os.path.dirname(__file__)
    log_dir = os.path.normpath(os.path.join(script_path, "..", ".."))

    return log_dir


def load_config(config_path):
    with open(config_path, mode="rt") as f:
        config = json.load(f)

    return config
