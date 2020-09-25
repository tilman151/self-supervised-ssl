import random
from datetime import datetime

import building


def run(
    source,
    fails,
    config,
    encoder,
    seed,
    gpu,
    pretrained_encoder_path,
    version,
    record_embeddings=False,
):
    trainer, data, model = building.build_baseline(
        source,
        fails,
        config,
        encoder,
        pretrained_encoder_path,
        gpu,
        seed,
        version,
        record_embeddings,
    )
    trainer.fit(model, datamodule=data)
    trainer.test(datamodule=data)


def run_multiple(
    source,
    fails,
    config,
    encoder,
    replications,
    gpu,
    pretrained_encoder_path,
    version=None,
):
    random.seed(999)
    seeds = [random.randint(0, 9999999) for _ in range(replications)]
    if version is None:
        version = datetime.now().timestamp()

    for f in fails:
        for s in seeds:
            run(source, f, config, encoder, s, gpu, pretrained_encoder_path, version)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline experiment")
    parser.add_argument("--source", type=int, help="FD number of the source data")
    parser.add_argument(
        "-f", "--fails", nargs="+", type=float, help="percent fail runs to use"
    )
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--encoder",
        default="cnn",
        choices=["cnn", "lstm"],
        help="encoder type",
    )
    parser.add_argument(
        "--pretrained_encoder",
        default=None,
        help="Path to checkpoint file form pretraining",
    )
    parser.add_argument(
        "-r", "--replications", type=int, default=3, help="replications for each run"
    )
    parser.add_argument("--gpu", type=int, default=0, help="id of GPU to use")
    opt = parser.parse_args()

    _config = building.load_config(opt.config)
    run_multiple(
        opt.source,
        opt.fails,
        _config,
        opt.encoder,
        opt.replications,
        opt.gpu,
        opt.pretrained_encoder,
    )
