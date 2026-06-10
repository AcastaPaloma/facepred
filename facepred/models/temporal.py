"""Causal temporal encoders for local and long-range conversational context."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class CausalConv1d(nn.Conv1d):
    """One-dimensional convolution that never reads future timesteps."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        self.left_padding = (kernel_size - 1) * dilation
        super().__init__(
            channels,
            channels,
            kernel_size,
            padding=self.left_padding,
            dilation=dilation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = super().forward(inputs)
        return output[..., : inputs.shape[-1]]


class CausalResidualBlock(nn.Module):
    """Residual causal convolution block operating on ``[B, T, D]`` tensors."""

    def __init__(
        self,
        dim: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = CausalConv1d(dim, kernel_size, dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(inputs).transpose(1, 2)
        hidden = self.conv(hidden).transpose(1, 2)
        return inputs + self.dropout(self.activation(hidden))


class MultiRateCausalEncoder(nn.Module):
    """Combine short-range causal convolutions with a full-context causal GRU."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int | None = None,
        dilations: Sequence[int] = (1, 2, 4, 8),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or input_dim)
        self.input_dim = int(input_dim)
        self.hidden_dim = hidden_dim
        self.local = nn.Sequential(
            *[
                CausalResidualBlock(
                    self.input_dim,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.long_context = nn.GRU(self.input_dim, hidden_dim, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(self.input_dim + hidden_dim, self.input_dim),
            nn.LayerNorm(self.input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"inputs must be [batch, steps, dim], got {tuple(inputs.shape)}")
        local = self.local(inputs)
        long_context, _ = self.long_context(inputs)
        return self.output(torch.cat([local, long_context], dim=-1))
