import os
import re

import mlflow
import torch
import tqdm

import datasets
from lightning import metrics
from lightning.baseline import Baseline

LIST_PATTERN = re.compile(r"\d{1,3}")


def add_rul_score_all_fd(mlflow_uri):
    client = mlflow.tracking.MlflowClient(mlflow_uri)

    experiments = client.list_experiments()
    exp_pattern = re.compile("cmapss_.+_baseline$")
    experiments = [e for e in experiments if exp_pattern.match(e.name) is not None]
    print("Found experiments %d" % len(experiments))

    for e in experiments:
        runs = client.search_runs(e.experiment_id)

        errors = []
        for r in tqdm.tqdm(runs, desc=e.name):
            try:
                _process_run(client, r)
            except Exception as e:
                errors.append(e)

        print("Errors:")
        for e in errors:
            print(e)


def _process_run(client, r):
    # if run complete and not already processed
    if (
        "epoch" in r.data.metrics
        and r.data.metrics["epoch"] == 99.0
        and "val/rul_score" not in r.data.metrics
    ):
        dm = _get_datamodule(r)
        model_artifact = client.list_artifacts(r.info.run_id, "checkpoints")[0]
        model = _load_model(model_artifact, r)
        score = _get_score(model, dm.val_dataloader())
        _log_metric(client, r, score, "val/rul_score")
        for fd, test_loader in enumerate(dm.test_dataloader(), start=1):
            score = _get_score(model, test_loader)
            _log_metric(client, r, score, f"test/rul_score_fd{fd}")


def _get_datamodule(r):
    data = datasets.BaselineDataModule(
        fd_source=int(r.data.params["fd_source"]),
        batch_size=512,
    )
    data.prepare_data()
    data.setup()

    return data


def _load_model(model_artifact, r):
    artifact_path = os.path.join(
        r.info.artifact_uri.replace("file://", ""), model_artifact.path
    )
    model = Baseline.load_from_checkpoint(artifact_path, map_location="cpu")
    if "pretrained_checkpoint" in r.data.params:
        model.encoder.norm_outputs = True
    model.eval()

    return model


@torch.no_grad()
def _get_score(model, dataloader):
    model = model.to("cuda:0")
    rul_score = metrics.RULScore()
    rul_score_metric = metrics.SimpleMetric(reduction="sum")
    for batch in dataloader:
        features, labels = [b.to("cuda:0") for b in batch]
        batch_size = len(features)
        score = rul_score(model(features), labels)
        rul_score_metric.update(score, batch_size)

    return rul_score_metric.compute().item()


def _log_metric(client, r, score, metric):
    if metric not in r.data.metrics:
        client.log_metric(r.info.run_id, metric, score)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add RUL score val and test metric for all FDs of CMAPSS"
    )
    parser.add_argument("mlflow_uri", help="tracking URI for mlflow")
    opt = parser.parse_args()

    add_rul_score_all_fd(opt.mlflow_uri)
