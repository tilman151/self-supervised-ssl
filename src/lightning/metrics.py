import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import umap
from torch import nn as nn


class EmbeddingViz(pl.metrics.Metric):
    def __init__(self, num_elements, embedding_size, combined=True):
        super().__init__()

        self.num_elements = num_elements
        self.embedding_size = embedding_size
        self.combined = combined

        self.add_state(
            "embeddings",
            default=torch.zeros(num_elements, embedding_size),
            dist_reduce_fx=None,
        )
        self.add_state("labels", default=torch.zeros(num_elements), dist_reduce_fx=None)
        self.add_state("ruls", default=torch.zeros(num_elements), dist_reduce_fx=None)
        self.add_state("sample_counter", default=torch.tensor(0), dist_reduce_fx=None)

        self.class_cm = plt.get_cmap("tab10")
        self.rul_cm = plt.get_cmap("viridis")

    def update(self, embeddings, labels, ruls):
        """
        Add embedding, labels and RUL to the metric.

        :param embeddings: hidden layer embedding of the data points
        :param labels: domain labels with 0 being source and 1 being target
        :param ruls: Remaining Useful Lifetime values of the data points
        """
        start = self.sample_counter
        end = self.sample_counter + embeddings.shape[0]
        self.sample_counter = end

        self.embeddings[start:end] = embeddings
        self.labels[start:end] = labels
        self.ruls[start:end] = ruls

    def compute(self):
        """Compute UMAP and plot points to 2d scatter plot."""
        logged_embeddings = self.embeddings[: self.sample_counter].detach().cpu().numpy()
        logged_labels = self.labels[: self.sample_counter].detach().cpu().int()
        logged_ruls = self.ruls[: self.sample_counter].detach().cpu()
        viz_embeddings = umap.UMAP(random_state=42).fit_transform(logged_embeddings)

        if self.combined:
            fig, (ax_class, ax_rul) = plt.subplots(
                1, 2, sharex="all", sharey="all", figsize=(20, 10), dpi=300
            )
        else:
            fig_class = plt.figure(figsize=(10, 10), dpi=300)
            ax_class = fig_class.gca()
            fig_rul = plt.figure(figsize=(10, 10), dpi=300)
            ax_rul = fig_rul.gca()
            fig = [fig_class, fig_rul]

        self._scatter_class(ax_class, viz_embeddings, logged_labels)
        self._scatter_rul(ax_rul, viz_embeddings, logged_ruls)

        return fig

    def _scatter_class(self, ax, embeddings, labels):
        class_colors = [self.class_cm.colors[c] for c in labels]
        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=class_colors,
            alpha=0.4,
            edgecolors="none",
            s=[4],
            rasterized=True,
        )
        ax.legend(["Source", "Target"], loc="lower left")

    def _scatter_rul(self, ax, embeddings, ruls):
        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=ruls,
            alpha=0.4,
            edgecolors="none",
            s=[4],
            rasterized=True,
        )
        color_bar = plt.cm.ScalarMappable(
            mplcolors.Normalize(ruls.min(), ruls.max()), self.rul_cm
        )
        color_bar.set_array(ruls)
        plt.colorbar(color_bar)


class RMSELoss(pl.metrics.Metric):
    def __init__(self, num_elements: int = 1000):
        super().__init__()

        self.mse = nn.MSELoss()

        self.add_state("losses", default=torch.zeros(num_elements), dist_reduce_fx=None)
        self.add_state("sizes", default=torch.zeros(num_elements), dist_reduce_fx=None)
        self.add_state("sample_counter", default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        summed_square = nn.functional.mse_loss(inputs, targets, reduction="sum")
        batch_size = inputs.shape[0]

        self.losses[self.sample_counter] = summed_square
        self.sizes[self.sample_counter] = batch_size
        self.sample_counter += 1

    def compute(self) -> torch.Tensor:
        if self.sample_counter == 0:
            raise RuntimeError("RMSE metric was not used. Computation impossible.")
        summed_squares = self.losses[: self.sample_counter]
        batch_sizes = self.sizes[: self.sample_counter]
        rmse = torch.sqrt(summed_squares.sum() / batch_sizes.sum())

        return rmse

    def forward(self, inputs, targets):
        return torch.sqrt(self.mse(inputs, targets))


class SimpleMetric(pl.metrics.Metric):
    def __init__(self, num_elements: int = 1000, reduction="mean"):
        super().__init__()

        if reduction not in ["sum", "mean"]:
            raise ValueError(f"Unsupported reduction {reduction}")
        self.reduction = reduction

        self.add_state("losses", default=torch.zeros(num_elements), dist_reduce_fx=None)
        self.add_state("sizes", default=torch.zeros(num_elements), dist_reduce_fx=None)
        self.add_state("sample_counter", default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, loss: torch.Tensor, batch_size: int):
        self.losses[self.sample_counter] = loss
        self.sizes[self.sample_counter] = batch_size
        self.sample_counter += 1

    def compute(self) -> torch.Tensor:
        if self.sample_counter == 0:
            raise RuntimeError("RMSE metric was not used. Computation impossible.")
        if self.reduction == "mean":
            loss = self._weighted_mean()
        else:
            loss = self._sum()

        return loss

    def _weighted_mean(self):
        weights = self.sizes[: self.sample_counter]
        weights = weights / weights.sum()
        loss = self.losses[: self.sample_counter]
        loss = torch.sum(loss * weights)

        return loss

    def _sum(self):
        loss = self.losses[: self.sample_counter]
        loss = torch.sum(loss)

        return loss


class RULScore:
    def __init__(self, pos_factor=10, neg_factor=-13):
        self.pos_factor = pos_factor
        self.neg_factor = neg_factor

    def __call__(self, inputs, targets):
        dist = inputs - targets
        for i, d in enumerate(dist):
            dist[i] = (d / self.neg_factor) if d < 0 else (d / self.pos_factor)
        dist = torch.exp(dist) - 1
        score = dist.sum()

        return score
