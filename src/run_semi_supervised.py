import random
from datetime import datetime

import ray
from sklearn.model_selection import ShuffleSplit

import building
from datasets.loader import CMAPSSLoader
from run_baseline import run as run_baseline
from run_pretraining import run as run_pretraining


def run(
    source,
    percent_broken,
    percent_fails,
    arch_config,
    pre_config,
    pretrain,
    encoder,
    mode,
    record_embeddings,
    replications,
    gpu,
    master_seed,
    version=None,
):
    if version is None:
        version = datetime.now().timestamp()
    if master_seed:
        random.seed(master_seed)
    seeds = [random.randint(0, 9999999) for _ in range(replications)]

    if percent_fails < 1:
        splitter = ShuffleSplit(
            n_splits=replications, train_size=percent_fails, random_state=42
        )
    else:
        splitter = AllDataSplitter(replications)
    run_idx = range(CMAPSSLoader.NUM_TRAIN_RUNS[source])
    process_ids = []
    for (failed_idx, _), s in zip(splitter.split(run_idx), seeds):
        process_ids.append(
            ray_train.remote(
                source,
                percent_broken,
                failed_idx,
                arch_config,
                pre_config,
                encoder,
                mode,
                record_embeddings,
                s,
                gpu,
                version,
                pretrain,
            )
        )

    ray.get(process_ids)


class AllDataSplitter:
    """Splitter that returns the whole dataset as training and nothing as test split."""

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def split(self, x):
        for _ in range(self.n_splits):
            yield list(x), []


@ray.remote(num_cpus=3, num_gpus=0.5)
def ray_train(
    source,
    percent_broken,
    failed_idx,
    arch_config,
    pre_config,
    encoder,
    mode,
    record_embeddings,
    s,
    gpu,
    version,
    pretrain,
):
    if pretrain:
        checkpoint, _ = run_pretraining(
            source,
            None,
            percent_broken,
            failed_idx,
            arch_config,
            pre_config,
            encoder,
            mode,
            record_embeddings,
            s,
            gpu,
            version,
        )
    else:
        checkpoint = None
    run_baseline(
        source,
        failed_idx,
        arch_config,
        encoder,
        s,
        gpu,
        checkpoint,
        version,
        record_embeddings,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run semi-supervised experiment")
    parser.add_argument(
        "--source", required=True, type=int, help="FD number of the source data"
    )
    parser.add_argument(
        "-b", "--broken", required=True, type=float, help="percent broken to use"
    )
    parser.add_argument(
        "-f", "--fails", required=True, type=float, help="percent fail runs to use"
    )
    parser.add_argument(
        "--arch_config", required=True, help="path to architecture config file"
    )
    parser.add_argument(
        "--pre_config", required=True, help="path to pretraining config file"
    )
    parser.add_argument("--pretrain", action="store_true", help="use pre-training")
    parser.add_argument(
        "-r",
        "--replications",
        type=int,
        default=1,
        help="runs of the cross-validation",
    )
    parser.add_argument(
        "--encoder",
        default="cnn",
        choices=["cnn", "lstm"],
        help="encoder type",
    )
    parser.add_argument(
        "--mode",
        default="metric",
        choices=["metric", "autoencoder"],
        help="metric or autoencoder pre-training mode",
    )
    parser.add_argument(
        "--record_embeddings",
        action="store_true",
        help="whether to record embeddings of val data",
    )
    parser.add_argument("--gpu", type=int, default=0, help="id of GPU to use")
    parser.add_argument(
        "--seed", default=999, help="master seed used to produce all seeds"
    )
    parser.add_argument("--version", help="version tag to group runs together")
    opt = parser.parse_args()

    _arch_config = building.load_config(opt.arch_config)
    _pre_config = building.load_config(opt.pre_config)
    ray.init()
    run(
        opt.source,
        opt.broken,
        opt.fails,
        _arch_config,
        _pre_config,
        opt.pretrain,
        opt.encoder,
        opt.mode,
        opt.record_embeddings,
        opt.replications,
        opt.gpu,
        opt.seed,
        opt.version,
    )
    ray.shutdown()
