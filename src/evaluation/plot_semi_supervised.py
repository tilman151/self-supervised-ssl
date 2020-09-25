import re

import numpy as np
import pandas as pd
import plotnine
from matplotlib import cm, colors
from plotnine import aes, ggplot
from tabulate import tabulate


def plot(file_path):
    df = pd.read_csv(file_path, index_col=0)
    df = _preprocess_data(df)

    percent_broken = [40, 60, 70, 80, 90]
    percent_labeled = [2, 10, 20, 40, 100]

    plotnine.options.figure_size = (15, 8)
    for broken in percent_broken:
        plot_df = df[(df["percent_broken"] == broken) | (df["percent_broken"] == 0.0)]
        _plot_val_test(plot_df, broken, file_path)
    for labeled in percent_labeled:
        plot_df = df[(df["num_labeled"] == labeled) & (df["percent_broken"] > 0.0)]
        gg = _vs_broken(plot_df, "test")
        gg.save(file_path.replace(".csv", f"_test_labeled@{labeled}.pdf"))
        gg = _vs_broken(plot_df, "score", log_scale=True)
        gg.save(file_path.replace(".csv", f"_score_labeled@{labeled}.pdf"))
    _generate_table(df, "test")
    _generate_table(df, "score")


def _preprocess_data(df):
    method2displayname = {
        "no pretraining": "None",
        "metric": "Self-Supervised",
        "autoencoder": "Autoencoder",
        "rbm": "RBM",
    }
    df["Pre-Training"] = df["pretrained"].map(method2displayname)
    num2percent_labeled = {
        1.0: 2,
        4.0: 2,
        3.0: 2,
        8.0: 10,
        20.0: 10,
        19.0: 10,
        16.0: 20,
        41.0: 20,
        39.0: 20,
        32.0: 40,
        83.0: 40,
        79.0: 40,
        80.0: 100,
        208.0: 100,
        199.0: 100,
    }
    df["num_labeled"] = df["num_labeled"].map(num2percent_labeled)
    num2subset = {1: "FD001", 2: "FD002", 3: "FD003", 4: "FD004"}
    df["source"] = df["source"].map(num2subset)
    df["percent_broken"] = (df["percent_broken"] * 100).astype(int)

    return df


def _plot_val_test(plot_df, broken, file_path):
    plot_df.loc[:, "Pre-Training"] = pd.Categorical(
        plot_df["Pre-Training"],
        ["None", "Self-Supervised", "Autoencoder", "RBM"],
        ordered=True,
    )
    gg = _vs_labeled(plot_df, "val")
    gg.save(file_path.replace(".csv", f"_val@{broken:.2f}.pdf"))
    gg = _vs_labeled(plot_df, "test")
    gg.save(file_path.replace(".csv", f"_test@{broken:.2f}.pdf"))
    gg = _vs_labeled(plot_df, "score", log_scale=True)
    gg.save(file_path.replace(".csv", f"_score@{broken:.2f}.pdf"))


def _vs_labeled(df, column, log_scale=False):
    df = df.sort_values("Pre-Training")
    gg = (
        ggplot(df, aes(x="num_labeled", y=column))
        + plotnine.stat_boxplot(
            aes(
                y=column,
                x="factor(num_labeled)",
                color="Pre-Training",
            )
        )
        + plotnine.facet_wrap("source", nrow=2, ncol=2, scales="free_x")
        + plotnine.scale_color_manual(
            [colors.rgb2hex(c) for c in cm.get_cmap("tab10").colors]
        )
        + plotnine.xlab("Percent Labeled")
        + plotnine.theme_classic(base_size=20)
        + plotnine.theme(subplots_adjust={"hspace": 0.25})
    )
    if column == "score":
        gg += plotnine.ylab("RUL Score")
    else:
        gg += plotnine.ylab("RMSE")
    if log_scale:
        gg += plotnine.scale_y_log10()

    return gg


def _vs_broken(df, column, log_scale=False):
    df.loc[:, "Pre-Training"] = pd.Categorical(
        df["Pre-Training"],
        ["Self-Supervised", "Autoencoder", "RBM"],
        ordered=True,
    )
    df = df.sort_values("Pre-Training")
    gg = (
        ggplot(df, aes(x="percent_broken", y=column))
        + plotnine.stat_boxplot(
            aes(
                y=column,
                x="factor(percent_broken)",
                color="Pre-Training",
            )
        )
        + plotnine.facet_wrap("source", nrow=2, ncol=2)
        + plotnine.scale_color_manual(
            [colors.rgb2hex(c) for c in cm.get_cmap("tab10").colors[1:]]
        )
        + plotnine.xlab("Grade of Degradation")
        + plotnine.theme_classic(base_size=20)
        + plotnine.theme(subplots_adjust={"hspace": 0.25})
    )
    if column == "score":
        gg += plotnine.ylab("RUL Score")
    else:
        gg += plotnine.ylab("RMSE")
    if log_scale:
        gg += plotnine.scale_y_log10()

    return gg


POWER_PATTERN = re.compile(r"(?P<base>[\d.]+)e(?P<exponent>[+-]\d+)")


def _replace_power_fmt(string):
    for match in POWER_PATTERN.finditer(string):
        base = match.group("base")
        exponent = int(match.group("exponent"))
        string = string.replace(match.group(), fr"{base}e{exponent}")

    return string


def _generate_table(df: pd.DataFrame, column):
    df = df.filter(
        items=["source", "Pre-Training", "percent_broken", "num_labeled", column],
        axis=1,
    )
    df = df.groupby(["source", "Pre-Training", "percent_broken", "num_labeled"])
    df = df.agg(
        mean=pd.NamedAgg(column, "mean"), std=pd.NamedAgg(column, "std")
    ).reset_index()
    if column == "score":
        df[column] = df["mean"].combine(
            df["std"], func=lambda mean, std: f"${mean:.2e} \pm {std:.2e}$"
        )
        df[column] = df[column].apply(_replace_power_fmt)
    else:
        df[column] = df["mean"].combine(
            df["std"], func=lambda mean, std: f"${mean:.2f} \pm {std:.2f}$"
        )
    argmins = _get_argmins(df)
    df = df.pivot(
        index=["Pre-Training", "percent_broken"],
        columns=["source", "num_labeled"],
        values=[column],
    )
    df = (
        df.reset_index(level="Pre-Training")
        .sort_index(level="percent_broken")
        .droplevel(0, axis=1)
    )
    df = _mark_best(df, argmins)
    df = _rename_methods(df)
    df = _make_multirows(df)

    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        print(tabulate(df.loc[:, ["", fd]], headers="keys", tablefmt="latex_raw"))


def _get_argmins(df):
    min_df = df.pivot(
        index=["Pre-Training", "percent_broken"],
        columns=["source", "num_labeled"],
        values=["mean"],
    )
    worse_than_baseline = (min_df - min_df.loc[("None", 0), :]) > 0
    worse_than_baseline = worse_than_baseline.groupby(["percent_broken"]).agg(all)
    argmins = min_df.reset_index().groupby(["percent_broken"]).agg(np.argmin)
    argmins = argmins.drop(("Pre-Training", "", ""), axis=1)
    argmins[worse_than_baseline] = -1
    argmins = argmins.droplevel(0, axis=1)

    return argmins


def _mark_best(df, argmins):
    row_iter = argmins.iterrows()
    next(row_iter)  # skip baseline row
    for percent_broken, row in row_iter:
        for col_idx in row.index:
            min_idx = int(argmins.loc[percent_broken, col_idx])  #
            if min_idx > -1:
                stripped_value = df.loc[percent_broken, col_idx].iloc[min_idx].strip("$")
                df.loc[percent_broken, col_idx].iloc[
                    min_idx
                ] = f"$\mathbf{{{stripped_value}}}$"

    return df


def _rename_methods(df):
    method_dict = {
        "Autoencoder": "AE",
        "Self-Supervised": "Ours",
        "RBM": "RBM",
        "None": "None",
    }
    df.loc[:, ("", "")] = df.loc[:, ("", "")].map(method_dict)

    return df


def _make_multirows(df):
    new_index = []
    for i, value in enumerate(df.index):
        if value == 0:
            new_index.append("")
        elif i % 3 == 1:
            new_index.append(f"\multirow{{3}}{{*}}{{{value}\%}}")
        else:
            new_index.append("")
    df.index = new_index

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot results of semi-supervised experiments"
    )
    parser.add_argument("file_path", help="path to result CSV file")
    opt = parser.parse_args()

    plot(opt.file_path)
