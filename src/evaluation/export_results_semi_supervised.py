"""Exports all RUL experiments to a data frame."""

import argparse
import re

import mlflow
import numpy as np
import pandas as pd

LIST_PATTERN = re.compile(r".?\d{1,3}")
VERSION_PATTERN = re.compile(r".*?(@(?P<percent_broken>.*))?@(?P<percent_fail_runs>.*)")


def export_all(mlflow_uri, tags):
    """Export all transfer and baseline experiments to CSV."""
    client = mlflow.tracking.MlflowClient(mlflow_uri)

    print("Export baseline experiments...")
    df = export_matching_experiments(
        client, "cmapss_.{3,5}_baseline$", tags, _runs_of_semi_supervised
    )
    if df is not None:
        df.to_csv("semi_supervised.csv")


def export_matching_experiments(client, exp_name_regex, tags, func):
    regex = re.compile(exp_name_regex)
    experiments = client.list_experiments()
    filtered_experiments = [e for e in experiments if regex.match(e.name) is not None]
    print("Found %d experiments..." % len(filtered_experiments))
    df = []
    for e in filtered_experiments:
        print('Evaluate experiment "%s" with id %s' % (e.name, e.experiment_id))
        df.append(func(client, e, tags))
    if df:
        df = pd.concat(df)
    else:
        df = None

    return df


def _runs_of_semi_supervised(client, e, tags):
    replications = _get_replications(client, e)
    replications = _filter_complete_with_tag(replications, tags)
    df = pd.DataFrame(
        np.zeros((len(replications), 7)),
        columns=[
            "source",
            "num_labeled",
            "percent_broken",
            "pretrained",
            "test",
            "val",
            "score",
        ],
        index=[e.name] * len(replications),
    )
    for i, run in enumerate(replications):
        statics = [
            run.data.params["fd_source"],
            _get_num_labeled(run.data.params["percent_fail_runs"]),
            _get_percent_broken(run.data.tags["version"]),
            _get_pretraining_mode(run),
        ]
        test_rmse = _get_test_value(f"regression_loss_fd{statics[0]}", run)
        test_score = _get_test_value(f"rul_score_fd{statics[0]}", run)
        val_rmse = _get_val_value(client, "regression_loss", "regression_loss", run)
        df.iloc[i] = statics + [test_rmse, val_rmse, test_score]
    print("Return %d runs..." % len(df))

    return df


def _get_replications(client, experiment):
    runs = client.search_runs(experiment_ids=experiment.experiment_id, max_results=9999)
    print("Found %d top-level runs..." % len(runs))

    return runs


def _filter_complete_with_tag(replications, tags):
    versions = [
        r.data.tags["version"]
        for r in replications
        if any(r.data.tags["version"].startswith(t) for t in tags)
    ]
    versions, version_counts = np.unique(versions, return_counts=True)
    versions = versions[version_counts == 10]
    replications = [r for r in replications if r.data.tags["version"] in versions]

    return replications


def _get_num_labeled(percent_labeled):
    return len(LIST_PATTERN.findall(percent_labeled))


def _get_pretraining_mode(run):
    version = run.data.tags["version"]
    if "baseline" in version:
        return "no pretraining"
    elif "autoencoder" in version:
        return "autoencoder"
    elif "rbm" in version:
        return "rbm"
    else:
        return "metric"


def _get_percent_broken(version):
    match = VERSION_PATTERN.match(version)
    if match is not None:
        percent_broken = match.group("percent_broken") or "0.0"
        percent_broken = float(percent_broken)
    else:
        raise ValueError(f"Could not parse version tag '{version}'")

    return percent_broken


def _get_test_value(metric, run):
    """Return test value of selected metric."""
    test_metric = f"test/{metric}"
    if test_metric in run.data.metrics:
        best_value = run.data.metrics[test_metric]
    else:
        best_value = np.nan

    return best_value


def _get_val_value(client, metric, indicator_metric, run):
    """Return test value of selected metric."""
    val_metric = f"val/{metric}"
    val_indicator = f"val/{indicator_metric}"
    if val_metric in run.data.metrics and val_indicator in run.data.metrics:
        indicator_values = client.get_metric_history(run.info.run_id, val_indicator)
        best_indicator_idx = np.argmin([m.value for m in indicator_values])
        metric_value = client.get_metric_history(run.info.run_id, val_metric)
        best_value = metric_value[best_indicator_idx].value
    else:
        best_value = np.nan

    return best_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exports semi_supervised experiments to CSV"
    )
    parser.add_argument("mlflow_uri", help="URI for MLFlow Server.")
    parser.add_argument(
        "--tags", "-t", nargs="+", help="MLFlow tag prefix of runs to export"
    )
    opt = parser.parse_args()

    export_all(opt.mlflow_uri, opt.tags)
