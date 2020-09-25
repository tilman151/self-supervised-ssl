import random
from datetime import datetime

import ray
import torch.optim
from pytorch_probgraph import RestrictedBoltzmannMachineCD
from sklearn.model_selection import ShuffleSplit

import building
from datasets.loader import CMAPSSLoader


def run(
    source,
    percent_broken,
    percent_fails,
    arch_config,
    encoder,
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
                encoder,
                s,
                gpu,
                version,
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
def ray_train(source, percent_broken, failed_idx, arch_config, encoder, s, gpu, version):
    pretrained_rbm = run_pretraining(
        source,
        percent_broken,
        failed_idx,
        arch_config,
        gpu,
    )
    run_baseline(
        source,
        failed_idx,
        arch_config,
        encoder,
        s,
        gpu,
        pretrained_rbm,
        version,
    )


def run_pretraining(source, percent_broken, failed_idx, arch_config, gpu):
    print("Pre-train RBM layer...")
    device = torch.device(f"cuda:{gpu}")
    dm = building.build_datamodule(
        source,
        None,
        percent_broken,
        failed_idx,
        arch_config["batch_size"],
        truncate_val=True,
        distance_mode="linear",
        min_distance=1,
    )
    dm.prepare_data()
    dm.setup()
    data = (torch.cat([anchor, query]) for anchor, query, *_ in dm.train_dataloader())
    rbm: RestrictedBoltzmannMachineCD = building.build_rbm(
        14, arch_config["base_filters"]
    ).to(device)
    adam = torch.optim.Adam(rbm.parameters(), lr=1e-4)
    rbm.train(data, epochs=5, optimizer=adam, device=device)

    data = (torch.cat([anchor, query]) for anchor, query, *_ in dm.val_dataloader()[0])
    recon_error = 0
    num_elem = 0
    with torch.no_grad():
        for batch in data:
            batch = batch.to(device)
            recon = rbm.reconstruct(visible_input=batch)
            recon_error += torch.sum((batch - recon) ** 2).cpu().item()
            num_elem += len(batch)
        recon_error /= num_elem * dm.window_size

    print(f"Recon error: {recon_error:.2f}")

    return rbm


def run_baseline(source, fails, config, encoder, seed, gpu, pretrained_rbm, version):
    trainer, data, model = building.build_baseline(
        source, fails, config, encoder, None, gpu, seed, version
    )
    model.hparams["encoder"] = f"rbm_{encoder}"
    model.load_from_rbm(pretrained_rbm)
    trainer.fit(model, datamodule=data)
    trainer.test(datamodule=data)


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
    parser.add_argument("--gpu", type=int, default=0, help="id of GPU to use")
    parser.add_argument(
        "--seed", default=999, help="master seed used to produce all seeds"
    )
    parser.add_argument("--version", help="version tag to group runs together")
    opt = parser.parse_args()

    _arch_config = building.load_config(opt.arch_config)
    ray.init()
    run(
        opt.source,
        opt.broken,
        opt.fails,
        _arch_config,
        opt.encoder,
        opt.replications,
        opt.gpu,
        opt.seed,
        opt.version,
    )
    ray.shutdown()
