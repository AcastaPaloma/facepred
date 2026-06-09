"""Compatibility helpers for loading trusted FacePred PyTorch artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_trusted_torch_artifact(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> Any:
    """Load an artifact created by FacePred, including non-tensor training state.

    FacePred checkpoints and cache progress files contain optimizer, scaler, RNG,
    and metadata state. PyTorch 2.6 defaults ``torch.load`` to
    ``weights_only=True``, which rejects those resumability fields. Explicitly
    disable that restriction only at this trusted-artifact boundary.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch releases without weights_only.
        return torch.load(path, map_location=map_location)
