from typing import Union

import torch
from pytorch_probgraph import GaussianLayer, UnitLayer
from torch import nn as nn


class DeFlatten(nn.Module):
    def __init__(self, seq_len, num_channels):
        super().__init__()

        self.seq_len = seq_len
        self.num_channels = num_channels

    def forward(self, inputs):
        return inputs.view(-1, self.num_channels, self.seq_len)


class _GradientReverse(torch.autograd.Function):
    """Gradient reversal forward and backward definitions."""

    @staticmethod
    def forward(ctx, inputs, **kwargs):
        """Forward pass as identity mapping."""
        return inputs

    @staticmethod
    def backward(ctx, grad):
        """Backward pass as negative of gradient."""
        return -grad


def gradient_reversal(x):
    """Perform gradient reversal on input."""
    return _GradientReverse.apply(x)


class GradientReversalLayer(nn.Module):
    """Module for gradient reversal."""

    def forward(self, inputs):
        """Perform forward pass of gradient reversal."""
        return gradient_reversal(inputs)


class RectifiedLinearLayer(UnitLayer):
    """
    A UnitLayer of rectified linear units.
    """

    def __init__(self, bias: torch.Tensor):
        """
        :param bias: Bias for the ReLU unit.
        """
        super().__init__()
        self.register_parameter("bias", torch.nn.Parameter(bias))

    def mean_cond(
        self, interaction: Union[torch.Tensor, None] = None, N: int = 1
    ) -> torch.Tensor:
        return nn.functional.relu(interaction + self.bias)

    def sample_cond(
        self, interaction: Union[torch.Tensor, None] = None, N: int = 1
    ) -> torch.Tensor:
        interaction = interaction + self.bias
        return nn.functional.relu(interaction + torch.sigmoid(interaction))

    def transform(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def transform_invert(self, transformed_input: torch.Tensor) -> torch.Tensor:
        return transformed_input

    def logprob_cond(
        self, input: torch.Tensor, interaction: Union[torch.Tensor, float] = 0.0
    ) -> torch.Tensor:
        raise NotImplementedError

    def logprob_joint(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sum(input * self.bias, dim=list(range(1, len(input.shape))))

    def free_energy(self, interaction: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def backward(
        self, input: torch.Tensor, factor: Union[torch.Tensor, float] = 1.0
    ) -> None:
        if self.bias.requires_grad:
            grad_bias = (factor * input).mean(dim=0, keepdim=True)
            grad_bias = grad_bias.mean(dim=-1, keepdim=True)
            self.bias.backward(grad_bias.detach())


class GaussianSequenceLayer(GaussianLayer):
    def backward(
        self, input: torch.Tensor, factor: Union[torch.Tensor, float] = 1.0
    ) -> None:
        if self.bias.requires_grad:
            grad_bias = (factor * input).sum(dim=0, keepdim=True) / input.shape[0]
            grad_bias = grad_bias.sum(dim=-1, keepdim=True) / input.shape[-1]
            self.bias.backward(grad_bias.detach())
        if self.logsigma.requires_grad:
            var = (input ** 2).sum(dim=0, keepdim=True) / input.shape[0]
            grad_logsigma = factor * var * torch.exp(-2 * self.logsigma) - 1.0
            self.logsigma.backward(grad_logsigma.detach())
