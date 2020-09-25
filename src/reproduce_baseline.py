import os
from datetime import datetime
import random

import ray

import building
from run_semi_supervised import run


script_path = os.path.dirname(__file__)
config_root = os.path.join(script_path, "..", "configs")


def reproduce(base_version, percent_fails, encoder, master_seed):
    ray.init()

    if base_version is None:
        base_version = datetime.now().timestamp()
    error_log = []
    fds = [1, 2, 3, 4]
    random.seed(master_seed)
    seeds = [
        [random.randint(0, 99999999) for _ in range(len(percent_fails))]
        for _ in range(len(fds))
    ]

    for num_fd, fd in enumerate(fds):
        arch_config_path = os.path.join(config_root, f"baseline_fd{fd}.json")
        arch_config = building.load_config(arch_config_path)
        pre_config_path = os.path.join(config_root, f"baseline_pre_fd{fd}.json")
        pre_config = building.load_config(pre_config_path)
        for num_fails, fails in enumerate(percent_fails):
            seed = seeds[num_fd][num_fails]
            try:
                run(
                    fd,
                    None,
                    fails,
                    arch_config,
                    pre_config,
                    pretrain=False,
                    encoder=encoder,
                    mode="metric",
                    record_embeddings=False,
                    replications=10,
                    gpu=[0],
                    master_seed=seed,
                    version=f"{base_version}_baseline@{fails:.2f}",
                )
            except Exception as e:
                error_log.append(e)

    ray.shutdown()

    if error_log:
        for e in error_log:
            print(type(e), e)
    else:
        print("Everything went well.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reproduce all supervised baseline experiments."
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
        "--encoder",
        default="cnn",
        choices=["cnn", "lstm"],
        help="encoder type",
    )
    parser.add_argument(
        "--seed", default=42, help="master seed used to produce all other seeds"
    )
    opt = parser.parse_args()

    reproduce(opt.base_version, opt.percent_fails, opt.encoder, opt.seed)
