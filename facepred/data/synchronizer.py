"""Fixed-rate synchronization utilities for multimodal feature streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - only for torch-free metadata environments.
    torch = None


@dataclass(frozen=True)
class ModalitySpec:
    """A feature stream and its timestamps before synchronization."""

    values: Any
    timestamps_s: Sequence[float] | None = None
    method: str = "nearest"
    tolerance_ms: float | None = None
    fill_value: float = 0.0
    timestamp_col: str = "time_s"
    feature_cols: Sequence[str] | None = None


@dataclass(frozen=True)
class SynchronizedBatch:
    """Features aligned to one fixed-rate time grid."""

    grid_s: np.ndarray
    features: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    step_duration_ms: int = 100

    @property
    def length(self) -> int:
        return int(self.grid_s.shape[0])

    def as_torch(self) -> dict[str, Any]:
        """Return torch tensors while preserving the public dict structure."""

        if torch is None:
            raise ImportError("torch is required for SynchronizedBatch.as_torch()")
        return {
            "grid_s": torch.as_tensor(self.grid_s, dtype=torch.float32),
            "features": {
                key: torch.as_tensor(value, dtype=torch.float32)
                for key, value in self.features.items()
            },
            "masks": {
                key: torch.as_tensor(value, dtype=torch.bool) for key, value in self.masks.items()
            },
        }


class MultimodalSynchronizer:
    """Align timestamped modality arrays onto the world-model step grid."""

    def __init__(
        self,
        step_duration_ms: int = 100,
        tolerance_ms: float | None = None,
        include_endpoint: bool = False,
    ) -> None:
        self.step_duration_ms = int(step_duration_ms)
        self.step_s = self.step_duration_ms / 1000.0
        self.tolerance_ms = tolerance_ms if tolerance_ms is not None else self.step_duration_ms / 2
        self.include_endpoint = include_endpoint

    def grid(self, start_s: float, end_s: float) -> np.ndarray:
        """Create a fixed-rate time grid in seconds."""

        return make_time_grid(
            start_s=start_s,
            end_s=end_s,
            step_duration_ms=self.step_duration_ms,
            include_endpoint=self.include_endpoint,
        )

    def align_values(
        self,
        values: Any,
        timestamps_s: Sequence[float],
        grid_s: Sequence[float] | None = None,
        start_s: float | None = None,
        end_s: float | None = None,
        method: str = "nearest",
        tolerance_ms: float | None = None,
        fill_value: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align one modality stream and return ``(features, valid_mask)``."""

        value_array = _as_2d_array(values)
        timestamps = np.asarray(timestamps_s, dtype=float)
        if value_array.shape[0] != timestamps.shape[0]:
            raise ValueError("values and timestamps_s must have the same first dimension")
        if grid_s is None:
            if start_s is None:
                start_s = float(np.nanmin(timestamps))
            if end_s is None:
                end_s = float(np.nanmax(timestamps)) + self.step_s
            grid = self.grid(float(start_s), float(end_s))
        else:
            grid = np.asarray(grid_s, dtype=float)

        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        value_array = value_array[order]
        method_key = method.lower()
        tolerance_s = (self.tolerance_ms if tolerance_ms is None else tolerance_ms) / 1000.0

        if method_key == "nearest":
            aligned, mask = _align_nearest(value_array, timestamps, grid, tolerance_s, fill_value)
        elif method_key == "previous":
            aligned, mask = _align_previous(value_array, timestamps, grid, tolerance_s, fill_value)
        elif method_key == "next":
            aligned, mask = _align_next(value_array, timestamps, grid, tolerance_s, fill_value)
        elif method_key == "linear":
            aligned, mask = _align_linear(value_array, timestamps, grid, fill_value)
        else:
            raise ValueError("method must be one of: nearest, previous, next, linear")
        return aligned, mask

    def align_modalities(
        self,
        modalities: Mapping[str, Any | ModalitySpec],
        timestamps: Mapping[str, Sequence[float]] | None = None,
        start_s: float | None = None,
        end_s: float | None = None,
        methods: Mapping[str, str] | None = None,
        feature_cols: Mapping[str, Sequence[str]] | None = None,
    ) -> SynchronizedBatch:
        """Align multiple feature streams to a shared grid."""

        specs = {
            name: value if isinstance(value, ModalitySpec) else ModalitySpec(value)
            for name, value in modalities.items()
        }
        extracted = {
            name: _extract_values_and_timestamps(
                spec,
                timestamps.get(name) if timestamps else None,
                feature_cols.get(name) if feature_cols else None,
            )
            for name, spec in specs.items()
        }

        all_times = np.concatenate([times for _, times in extracted.values() if len(times) > 0])
        if all_times.size == 0:
            raise ValueError("At least one modality must contain timestamps")
        grid_start = float(np.nanmin(all_times)) if start_s is None else float(start_s)
        grid_end = float(np.nanmax(all_times)) + self.step_s if end_s is None else float(end_s)
        grid = self.grid(grid_start, grid_end)

        features: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        for name, spec in specs.items():
            values, times = extracted[name]
            method = methods.get(name, spec.method) if methods else spec.method
            aligned, mask = self.align_values(
                values,
                times,
                grid_s=grid,
                method=method,
                tolerance_ms=spec.tolerance_ms,
                fill_value=spec.fill_value,
            )
            features[name] = aligned
            masks[name] = mask

        return SynchronizedBatch(
            grid_s=grid,
            features=features,
            masks=masks,
            step_duration_ms=self.step_duration_ms,
        )


def make_time_grid(
    start_s: float,
    end_s: float,
    step_duration_ms: int = 100,
    include_endpoint: bool = False,
) -> np.ndarray:
    """Create a monotonic time grid in seconds."""

    step_s = step_duration_ms / 1000.0
    if end_s <= start_s:
        return np.asarray([float(start_s)], dtype=np.float32)
    stop = end_s + (step_s * 0.5 if include_endpoint else 0.0)
    return np.arange(float(start_s), float(stop), step_s, dtype=np.float32)


def make_synthetic_synchronized_batch(
    duration_s: float = 5.0,
    step_duration_ms: int = 100,
    feature_dims: Mapping[str, int] | None = None,
    seed: int = 42,
) -> SynchronizedBatch:
    """Generate a deterministic aligned multimodal batch for smoke tests."""

    dims = feature_dims or {"audio_prosody": 88, "visual": 1493, "text": 384, "quality": 4}
    grid = make_time_grid(0.0, duration_s, step_duration_ms=step_duration_ms)
    rng = np.random.default_rng(seed)
    features = {
        name: rng.normal(0.0, 0.1, size=(len(grid), dim)).astype(np.float32)
        for name, dim in dims.items()
    }
    if "quality" in features and features["quality"].shape[1] >= 4:
        features["quality"][:, :4] = np.asarray([0.8, 0.9, 0.75, 1.0], dtype=np.float32)
    masks = {name: np.ones(len(grid), dtype=bool) for name in features}
    return SynchronizedBatch(grid, features, masks, step_duration_ms)


def _extract_values_and_timestamps(
    spec: ModalitySpec,
    explicit_timestamps: Sequence[float] | None,
    explicit_feature_cols: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(spec.values, pd.DataFrame):
        timestamp_col = spec.timestamp_col
        if explicit_timestamps is None and timestamp_col not in spec.values:
            raise ValueError(f"DataFrame modality is missing timestamp column {timestamp_col!r}")
        timestamps = (
            np.asarray(explicit_timestamps, dtype=float)
            if explicit_timestamps is not None
            else spec.values[timestamp_col].to_numpy(dtype=float)
        )
        cols = explicit_feature_cols or spec.feature_cols
        if cols is None:
            cols = [column for column in spec.values.columns if column != timestamp_col]
        values = spec.values[list(cols)].to_numpy(dtype=np.float32)
        return values, timestamps

    if explicit_timestamps is None and spec.timestamps_s is None:
        raise ValueError("Non-DataFrame modalities require timestamps_s")
    timestamps = explicit_timestamps if explicit_timestamps is not None else spec.timestamps_s
    return _as_2d_array(spec.values), np.asarray(timestamps, dtype=float)


def _align_nearest(
    values: np.ndarray,
    timestamps: np.ndarray,
    grid: np.ndarray,
    tolerance_s: float,
    fill_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(timestamps, grid)
    left = np.clip(positions - 1, 0, len(timestamps) - 1)
    right = np.clip(positions, 0, len(timestamps) - 1)
    left_dist = np.abs(grid - timestamps[left])
    right_dist = np.abs(grid - timestamps[right])
    chosen = np.where(right_dist < left_dist, right, left)
    distance = np.minimum(left_dist, right_dist)
    mask = distance <= tolerance_s
    aligned = np.full((len(grid), values.shape[1]), fill_value, dtype=np.float32)
    aligned[mask] = values[chosen[mask]]
    return aligned, mask


def _align_previous(
    values: np.ndarray,
    timestamps: np.ndarray,
    grid: np.ndarray,
    tolerance_s: float,
    fill_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(timestamps, grid, side="right") - 1
    mask = positions >= 0
    distance = np.full(len(grid), np.inf, dtype=np.float32)
    distance[mask] = grid[mask] - timestamps[positions[mask]]
    mask &= distance <= tolerance_s
    aligned = np.full((len(grid), values.shape[1]), fill_value, dtype=np.float32)
    aligned[mask] = values[positions[mask]]
    return aligned, mask


def _align_next(
    values: np.ndarray,
    timestamps: np.ndarray,
    grid: np.ndarray,
    tolerance_s: float,
    fill_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(timestamps, grid, side="left")
    mask = positions < len(timestamps)
    distance = np.full(len(grid), np.inf, dtype=np.float32)
    distance[mask] = timestamps[positions[mask]] - grid[mask]
    mask &= distance <= tolerance_s
    aligned = np.full((len(grid), values.shape[1]), fill_value, dtype=np.float32)
    aligned[mask] = values[positions[mask]]
    return aligned, mask


def _align_linear(
    values: np.ndarray,
    timestamps: np.ndarray,
    grid: np.ndarray,
    fill_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (grid >= timestamps[0]) & (grid <= timestamps[-1])
    aligned = np.full((len(grid), values.shape[1]), fill_value, dtype=np.float32)
    for dim in range(values.shape[1]):
        aligned[:, dim] = np.interp(grid, timestamps, values[:, dim], left=fill_value, right=fill_value)
    aligned[~mask] = fill_value
    return aligned.astype(np.float32), mask


def _as_2d_array(values: Any) -> np.ndarray:
    if torch is not None and isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("Expected modality values with shape [time, dim]")
    return array.astype(np.float32, copy=False)
