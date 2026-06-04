"""Numpy/torch-friendly data augmentations for multimodal training batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - only for torch-free metadata environments.
    torch = None


FeatureDict = Mapping[str, Any]


@dataclass(frozen=True)
class AugmentationResult:
    """Augmented features plus lightweight provenance metadata."""

    features: dict[str, Any]
    modality_mask: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdditiveGaussianNoise:
    """Add Gaussian noise at a random target SNR."""

    snr_range_db: tuple[float, float] = (5.0, 30.0)
    p: float = 1.0
    modalities: Sequence[str] | None = None
    seed: int | None = None

    def __call__(self, features: FeatureDict, rng: np.random.Generator | None = None) -> AugmentationResult:
        generator = _rng(rng, self.seed)
        output = dict(features)
        snrs: dict[str, float] = {}
        for name in _selected_modalities(features, self.modalities):
            if generator.random() > self.p:
                continue
            snr_db = float(generator.uniform(*self.snr_range_db))
            output[name] = add_gaussian_noise(features[name], snr_db=snr_db, rng=generator)
            snrs[name] = snr_db
        return AugmentationResult(output, metadata={"snr_db": snrs})


@dataclass
class ModalityDropout:
    """Zero whole modalities with independent Bernoulli masks."""

    dropout_prob: float = 0.2
    modalities: Sequence[str] | None = None
    min_keep: int = 1
    fill_value: float = 0.0
    seed: int | None = None

    def __call__(self, features: FeatureDict, rng: np.random.Generator | None = None) -> AugmentationResult:
        generator = _rng(rng, self.seed)
        names = list(_selected_modalities(features, self.modalities))
        keep = {name: bool(generator.random() >= self.dropout_prob) for name in names}
        if names and sum(keep.values()) < self.min_keep:
            restore = list(generator.choice(names, size=min(self.min_keep, len(names)), replace=False))
            for name in restore:
                keep[name] = True

        output = dict(features)
        for name in names:
            if not keep[name]:
                output[name] = full_like(features[name], self.fill_value)
        return AugmentationResult(output, modality_mask=keep)


@dataclass
class TemporalJitter:
    """Shift time-major feature arrays by a small integer number of frames."""

    max_shift_frames: int = 1
    p: float = 1.0
    modalities: Sequence[str] | None = None
    same_shift: bool = False
    fill_value: float = 0.0
    seed: int | None = None

    def __call__(self, features: FeatureDict, rng: np.random.Generator | None = None) -> AugmentationResult:
        generator = _rng(rng, self.seed)
        names = list(_selected_modalities(features, self.modalities))
        output = dict(features)
        shifts: dict[str, int] = {}
        shared_shift = self._sample_shift(generator) if self.same_shift else None

        for name in names:
            if generator.random() > self.p:
                shifts[name] = 0
                continue
            shift = shared_shift if shared_shift is not None else self._sample_shift(generator)
            output[name] = temporal_shift(features[name], shift=shift, fill_value=self.fill_value)
            shifts[name] = shift
        return AugmentationResult(output, metadata={"shift_frames": shifts})

    def _sample_shift(self, rng: np.random.Generator) -> int:
        if self.max_shift_frames <= 0:
            return 0
        return int(rng.integers(-self.max_shift_frames, self.max_shift_frames + 1))


@dataclass
class MultimodalAugmentor:
    """Compose the default iteration-0 augmentation stack."""

    noise: AdditiveGaussianNoise | None = None
    modality_dropout: ModalityDropout | None = None
    temporal_jitter: TemporalJitter | None = None
    seed: int | None = None

    def __call__(self, features: FeatureDict) -> AugmentationResult:
        generator = _rng(None, self.seed)
        output = dict(features)
        modality_mask: dict[str, bool] = {}
        metadata: dict[str, Any] = {}

        for transform in (self.noise, self.temporal_jitter, self.modality_dropout):
            if transform is None:
                continue
            result = transform(output, rng=generator)
            output = result.features
            modality_mask.update(result.modality_mask)
            metadata.update(result.metadata)
        return AugmentationResult(output, modality_mask=modality_mask, metadata=metadata)

    @classmethod
    def from_training_config(
        cls,
        modality_dropout: float = 0.2,
        noise_enabled: bool = True,
        snr_range: Sequence[float] = (5.0, 30.0),
        temporal_jitter_enabled: bool = True,
        max_shift_frames: int = 1,
        seed: int | None = None,
    ) -> MultimodalAugmentor:
        """Build the default stack from the current Hydra training keys."""

        return cls(
            noise=AdditiveGaussianNoise(tuple(snr_range), seed=seed) if noise_enabled else None,
            modality_dropout=ModalityDropout(modality_dropout, seed=seed),
            temporal_jitter=TemporalJitter(max_shift_frames, seed=seed)
            if temporal_jitter_enabled
            else None,
            seed=seed,
        )


def add_gaussian_noise(
    values: Any,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> Any:
    """Add Gaussian noise to ``values`` while preserving numpy/torch type."""

    array, restore = _to_numpy(values)
    generator = _rng(rng, None)
    signal_power = float(np.mean(np.square(array.astype(np.float32))))
    if signal_power <= 1e-12:
        return restore(array.copy())
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = generator.normal(0.0, np.sqrt(noise_power), size=array.shape).astype(np.float32)
    return restore(array.astype(np.float32) + noise)


def temporal_shift(values: Any, shift: int, fill_value: float = 0.0) -> Any:
    """Shift a time-major array by ``shift`` frames without wraparound."""

    array, restore = _to_numpy(values)
    shifted = np.full_like(array, fill_value)
    if shift == 0 or array.shape[0] == 0:
        return restore(array.copy())
    if abs(shift) >= array.shape[0]:
        return restore(shifted)
    if shift > 0:
        shifted[shift:] = array[:-shift]
    else:
        shifted[:shift] = array[-shift:]
    return restore(shifted)


def full_like(values: Any, fill_value: float = 0.0) -> Any:
    """Create a filled array/tensor preserving the input shape and type."""

    array, restore = _to_numpy(values)
    return restore(np.full_like(array, fill_value))


def _selected_modalities(features: FeatureDict, modalities: Sequence[str] | None) -> tuple[str, ...]:
    if modalities is None:
        return tuple(features.keys())
    return tuple(name for name in modalities if name in features)


def _rng(rng: np.random.Generator | None, seed: int | None) -> np.random.Generator:
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _to_numpy(values: Any) -> tuple[np.ndarray, Any]:
    if torch is not None and isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()

        def restore(array_value: np.ndarray) -> Any:
            return torch.as_tensor(array_value, dtype=values.dtype, device=values.device)

        return array, restore

    array = np.asarray(values)

    def restore(array_value: np.ndarray) -> np.ndarray:
        return array_value.astype(array.dtype, copy=False)

    return array, restore
