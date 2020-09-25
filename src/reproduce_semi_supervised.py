import os
import random
from datetime import datetime

import ray

import building
from run_rbm import run as run_rbm
from run_semi_supervised import run as run_ae_or_metric

script_path = os.path.dirname(__file__)
config_root = os.path.join(script_path, "..", "configs")


def reproduce(base_version, percent_fails, percent_broken, encoder, mode, master_seed):
    ray.init()

    if base_version is None:
        base_version = datetime.now().timestamp()
    error_log = []
    fds = [1, 2, 3, 4]
    pre_mode = "pre" if mode == "metric" else "ae"
    random.seed(master_seed)
    seeds = [
        [
            [random.randint(0, 99999999) for _ in range(len(percent_fails))]
            for _ in range(len(percent_broken))
        ]
        for _ in range(len(fds))
    ]

    for num_fd, fd in enumerate(fds):
        arch_config_path = os.path.join(config_root, f"{encoder}_fd{fd}.json")
        arch_config = building.load_config(arch_config_path)
        pre_config_path = os.path.join(config_root, f"{encoder}_{pre_mode}_fd{fd}.json")
        pre_config = building.load_config(pre_config_path)
        for num_broken, broken in enumerate(percent_broken):
            for num_fails, fails in enumerate(percent_fails):
                seed = seeds[num_fd][num_broken][num_fails]
                try:
                    run(
                        fd,
                        broken,
                        fails,
                        arch_config,
                        pre_config,
                        encoder,
                        mode,
                        seed,
                        base_version,
                    )
                except Exception as e:
                    error_log.append(e)

    ray.shutdown()

    if error_log:
        for e in error_log:
            print(type(e), e)
    else:
        print("Everything went well.")


def run(fd, broken, fails, arch_config, pre_config, encoder, mode, seed, base_version):
    if mode == "rbm":
        run_rbm(
            fd,
            broken,
            fails,
            arch_config,
            encoder,
            replications=10,
            gpu=0,
            master_seed=seed,
            version=f"{base_version}_pretrained@{broken:.2f}@{fails:.2f}",
        )
    else:
        run_ae_or_metric(
            fd,
            broken,
            fails,
            arch_config,
            pre_config,
            pretrain=True,
            encoder=encoder,
            mode=mode,
            record_embeddings=False,
            replications=10,
            gpu=[0],
            master_seed=seed,
            version=f"{base_version}_pretrained@{broken:.2f}@{fails:.2f}",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reproduce all semi-supervised experiments."
    )
    parser.add_argument(
        "base_version", default=None, help="common prefix for the version tag"
    )
    parser.add_argument(
        "--percent_fails",
        nargs="*",
        default=[1.0, 0.4, 0.2, 0.1, 0.02],
        help="percentage of failed runs",
    )
    parser.add_argument(
        "--percent_broken",
        nargs="*",
        type=float,
        default=[0.9, 0.8, 0.7, 0.6, 0.4],
        help="percentage of degradation of unfailed runs",
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
        choices=["metric", "autoencoder", "rbm"],
        help="metric or autoencoder pre-training mode",
    )
    parser.add_argument(
        "--seed", default=21, help="master seed used to produce all other seeds"
    )
    opt = parser.parse_args()

    reproduce(
        opt.base_version,
        opt.percent_fails,
        opt.percent_broken,
        opt.encoder,
        opt.mode,
        opt.seed,
    )
