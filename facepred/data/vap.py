"""Voice Activity Projection target utilities for continuous dyadic data."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def build_vap_state_targets(
    voice_activity: torch.Tensor,
    *,
    bin_steps: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode future two-speaker activity bins as categorical VAP states.

    ``voice_activity`` is shaped ``[steps, 2]``. ``bin_steps`` contains the
    widths of successive future bins. A bin is active when a majority of its
    frames contain voice activity. The resulting state uses one bit per
    speaker and bin, matching the 256-state objective for four bins.
    """

    if voice_activity.ndim != 2 or voice_activity.shape[-1] != 2:
        raise ValueError("voice_activity must have shape [steps, 2]")
    widths = [int(value) for value in bin_steps]
    if not widths or any(value <= 0 for value in widths):
        raise ValueError("bin_steps must contain positive widths")

    activity = voice_activity.bool()
    steps = activity.shape[0]
    future_steps = sum(widths)
    targets = torch.full((steps,), -100, dtype=torch.long, device=activity.device)
    valid = torch.zeros(steps, dtype=torch.bool, device=activity.device)
    for step in range(max(0, steps - future_steps)):
        offset = step + 1
        state = 0
        bit = 0
        for width in widths:
            window = activity[offset : offset + width]
            voiced = window.float().mean(dim=0) > 0.5
            for speaker in range(2):
                state |= int(voiced[speaker]) << bit
                bit += 1
            offset += width
        targets[step] = state
        valid[step] = True
    return targets, valid
