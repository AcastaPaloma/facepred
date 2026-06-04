"""Modality encoders for FacePred.

Each encoder accepts a tensor shaped ``[batch, steps, features]`` and projects it
into a compact per-timestep representation. The modules are deliberately small
for iteration 0 so they can run in CPU smoke tests and with synthetic data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class EncoderSpec:
    """Configuration for a single modality encoder."""

    input_dim: int
    output_dim: int
    hidden_dim: int | None = None
    num_layers: int = 2
    dropout: float = 0.0


class MLPEncoder(nn.Module):
    """Small MLP applied independently to each timestep."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")

        hidden_dim = hidden_dim or max(input_dim, output_dim)
        num_layers = max(1, num_layers)

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    activation(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` while preserving leading batch/sequence dimensions."""
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dim {self.input_dim}, got {x.shape[-1]}")
        return self.net(x.float())


class IdentityOrLinearEncoder(nn.Module):
    """Projection that skips work when dimensions already match."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if input_dim == output_dim and dropout <= 0:
            self.net: nn.Module = nn.Identity()
        else:
            layers: list[nn.Module] = [nn.Linear(input_dim, output_dim)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dim {self.input_dim}, got {x.shape[-1]}")
        return self.net(x.float())


def _get(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def spec_from_config(config: Any) -> EncoderSpec:
    """Build an :class:`EncoderSpec` from a dict/OmegaConf-like object."""
    input_dim = int(_get(config, "input_dim"))
    output_dim = int(_get(config, "output_dim"))
    hidden_dim = _get(config, "hidden_dim", None)
    return EncoderSpec(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=int(hidden_dim) if hidden_dim is not None else None,
        num_layers=int(_get(config, "num_layers", 2)),
        dropout=float(_get(config, "dropout", 0.0)),
    )


class ModalityEncoders(nn.Module):
    """Container that owns all configured modality encoders."""

    def __init__(self, specs: Mapping[str, EncoderSpec]) -> None:
        super().__init__()
        self.specs = dict(specs)
        self.encoders = nn.ModuleDict(
            {
                name: MLPEncoder(
                    spec.input_dim,
                    spec.output_dim,
                    hidden_dim=spec.hidden_dim,
                    num_layers=spec.num_layers,
                    dropout=spec.dropout,
                )
                for name, spec in self.specs.items()
            }
        )
        self.output_dims = {name: spec.output_dim for name, spec in self.specs.items()}
        self.input_dims = {name: spec.input_dim for name, spec in self.specs.items()}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ModalityEncoders:
        specs = {name: spec_from_config(value) for name, value in config.items()}
        return cls(specs)

    def forward(self, features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode all present modalities."""
        encoded: dict[str, torch.Tensor] = {}
        for name, tensor in features.items():
            if name not in self.encoders:
                continue
            encoded[name] = self.encoders[name](tensor)
        if not encoded:
            known = ", ".join(self.encoders.keys())
            received = ", ".join(features.keys())
            raise ValueError(f"No known modalities received. Known={known}; received={received}")
        return encoded

    def zeros_like_input(
        self,
        modality: str,
        batch_size: int,
        steps: int,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return encoded zeros for a missing modality."""
        if modality not in self.specs:
            raise KeyError(modality)
        return torch.zeros(batch_size, steps, self.specs[modality].output_dim, device=device)
