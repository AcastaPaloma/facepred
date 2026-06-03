"""Calibration and uncertainty metrics for classification heads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Reduction = Literal["mean", "sum", "none"]


@dataclass(frozen=True)
class CalibrationBin:
    """Top-label calibration statistics for one confidence bin."""

    lower: float
    upper: float
    count: int
    accuracy: float
    confidence: float
    gap: float


@dataclass(frozen=True)
class CalibrationSummary:
    """Compact set of calibration metrics for a classification head."""

    ece: float
    brier: float
    n_samples: int
    bins: tuple[CalibrationBin, ...]

    @property
    def mce(self) -> float:
        """Maximum calibration error over non-empty bins."""
        gaps = [bin_stats.gap for bin_stats in self.bins if bin_stats.count > 0]
        return max(gaps, default=0.0)


def _to_numpy(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def softmax(logits: Any, *, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    logits_array = _to_numpy(logits).astype(np.float64, copy=False)
    shifted = logits_array - np.max(logits_array, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def normalize_probabilities(
    probabilities: Any,
    *,
    axis: int = -1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Clip and renormalize probabilities along the class axis."""
    probs = _to_numpy(probabilities).astype(np.float64, copy=False)
    if np.any(probs < -eps):
        raise ValueError("Probabilities cannot contain negative values.")
    probs = np.clip(probs, 0.0, None)
    totals = np.sum(probs, axis=axis, keepdims=True)
    if np.any(totals <= eps):
        raise ValueError("Probabilities must have positive mass along the class axis.")
    return probs / totals


def as_probabilities(
    values: Any,
    *,
    from_logits: bool = False,
    axis: int = -1,
    normalize: bool = True,
) -> np.ndarray:
    """Convert logits or probability-like values to a normalized probability array."""
    if from_logits:
        return softmax(values, axis=axis)
    probs = _to_numpy(values).astype(np.float64, copy=False)
    if normalize:
        return normalize_probabilities(probs, axis=axis)
    return probs


def predictive_entropy(
    values: Any,
    *,
    from_logits: bool = False,
    axis: int = -1,
    normalized: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute categorical entropy from logits or probabilities."""
    probs = as_probabilities(values, from_logits=from_logits, axis=axis)
    entropy = -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=axis)
    if not normalized:
        return entropy

    num_classes = probs.shape[axis]
    if num_classes <= 1:
        return np.zeros_like(entropy)
    return entropy / np.log(num_classes)


def top_confidence(
    values: Any,
    *,
    from_logits: bool = False,
    axis: int = -1,
) -> np.ndarray:
    """Return max class probability for each sample."""
    probs = as_probabilities(values, from_logits=from_logits, axis=axis)
    return np.max(probs, axis=axis)


def confidence_margin(
    values: Any,
    *,
    from_logits: bool = False,
    axis: int = -1,
) -> np.ndarray:
    """Return top-1 minus top-2 probability margin for each sample."""
    probs = as_probabilities(values, from_logits=from_logits, axis=axis)
    if probs.shape[axis] < 2:
        raise ValueError("confidence_margin requires at least two classes.")
    sorted_probs = np.sort(probs, axis=axis)
    return np.take(sorted_probs, -1, axis=axis) - np.take(sorted_probs, -2, axis=axis)


def _prepare_classification_arrays(
    values: Any,
    targets: Any,
    *,
    from_logits: bool,
    axis: int,
    mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    probs = as_probabilities(values, from_logits=from_logits, axis=axis)
    probability_shape = probs.shape
    if axis != -1:
        probs = np.moveaxis(probs, axis, -1)
    if probs.ndim < 2:
        raise ValueError("Expected probabilities/logits with a class dimension.")

    num_classes = probs.shape[-1]
    flat_probs = probs.reshape(-1, num_classes)

    target_array = _to_numpy(targets)
    if target_array.shape == probability_shape:
        if axis != -1:
            target_array = np.moveaxis(target_array, axis, -1)
        labels = target_array.reshape(-1, num_classes).argmax(axis=-1)
    elif target_array.shape == probs.shape:
        labels = target_array.reshape(-1, num_classes).argmax(axis=-1)
    else:
        labels = target_array.reshape(-1).astype(np.int64)

    if labels.shape[0] != flat_probs.shape[0]:
        raise ValueError(
            "Targets must match all non-class prediction dimensions. "
            f"Got {labels.shape[0]} labels for {flat_probs.shape[0]} predictions."
        )
    if np.any((labels < 0) | (labels >= num_classes)):
        raise ValueError(f"Targets must be class indices in [0, {num_classes - 1}].")

    if mask is not None:
        flat_mask = _to_numpy(mask).reshape(-1).astype(bool)
        if flat_mask.shape[0] != flat_probs.shape[0]:
            raise ValueError(
                f"Mask has {flat_mask.shape[0]} values for {flat_probs.shape[0]} predictions."
            )
        flat_probs = flat_probs[flat_mask]
        labels = labels[flat_mask]

    if flat_probs.shape[0] == 0:
        raise ValueError("No samples remain after applying mask.")

    return flat_probs, labels


def _reduce(values: np.ndarray, reduction: Reduction) -> float | np.ndarray:
    if reduction == "none":
        return values
    if reduction == "sum":
        return float(np.sum(values))
    if reduction == "mean":
        return float(np.mean(values))
    raise ValueError(f"Unknown reduction: {reduction!r}.")


def brier_score(
    values: Any,
    targets: Any,
    *,
    from_logits: bool = False,
    axis: int = -1,
    mask: Any | None = None,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """Compute the multiclass Brier score."""
    probs, labels = _prepare_classification_arrays(
        values,
        targets,
        from_logits=from_logits,
        axis=axis,
        mask=mask,
    )
    one_hot = np.eye(probs.shape[-1], dtype=np.float64)[labels]
    scores = np.sum((probs - one_hot) ** 2, axis=-1)
    return _reduce(scores, reduction)


def calibration_bins(
    values: Any,
    targets: Any,
    *,
    n_bins: int = 15,
    from_logits: bool = False,
    axis: int = -1,
    mask: Any | None = None,
) -> tuple[CalibrationBin, ...]:
    """Compute top-label reliability bins."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    probs, labels = _prepare_classification_arrays(
        values,
        targets,
        from_logits=from_logits,
        axis=axis,
        mask=mask,
    )
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.searchsorted(edges, confidences, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bins: list[CalibrationBin] = []
    for bin_index in range(n_bins):
        in_bin = bin_ids == bin_index
        count = int(np.sum(in_bin))
        if count == 0:
            bins.append(
                CalibrationBin(
                    lower=float(edges[bin_index]),
                    upper=float(edges[bin_index + 1]),
                    count=0,
                    accuracy=0.0,
                    confidence=0.0,
                    gap=0.0,
                )
            )
            continue

        accuracy = float(np.mean(correct[in_bin]))
        confidence = float(np.mean(confidences[in_bin]))
        bins.append(
            CalibrationBin(
                lower=float(edges[bin_index]),
                upper=float(edges[bin_index + 1]),
                count=count,
                accuracy=accuracy,
                confidence=confidence,
                gap=abs(accuracy - confidence),
            )
        )
    return tuple(bins)


def expected_calibration_error(
    values: Any,
    targets: Any,
    *,
    n_bins: int = 15,
    from_logits: bool = False,
    axis: int = -1,
    mask: Any | None = None,
) -> float:
    """Compute top-label expected calibration error."""
    bins = calibration_bins(
        values,
        targets,
        n_bins=n_bins,
        from_logits=from_logits,
        axis=axis,
        mask=mask,
    )
    total = sum(bin_stats.count for bin_stats in bins)
    if total == 0:
        return 0.0
    return float(sum((bin_stats.count / total) * bin_stats.gap for bin_stats in bins))


def calibration_summary(
    values: Any,
    targets: Any,
    *,
    n_bins: int = 15,
    from_logits: bool = False,
    axis: int = -1,
    mask: Any | None = None,
) -> CalibrationSummary:
    """Return ECE, Brier score, and reliability bins together."""
    probs, labels = _prepare_classification_arrays(
        values,
        targets,
        from_logits=from_logits,
        axis=axis,
        mask=mask,
    )
    bins = calibration_bins(probs, labels, n_bins=n_bins)
    total = probs.shape[0]
    ece = float(sum((bin_stats.count / total) * bin_stats.gap for bin_stats in bins))
    return CalibrationSummary(
        ece=ece,
        brier=float(brier_score(probs, labels)),
        n_samples=total,
        bins=bins,
    )


def one_hot(labels: Sequence[int] | np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to a one-hot matrix."""
    label_array = _to_numpy(labels).reshape(-1).astype(np.int64)
    if np.any((label_array < 0) | (label_array >= num_classes)):
        raise ValueError(f"Labels must be in [0, {num_classes - 1}].")
    return np.eye(num_classes, dtype=np.float64)[label_array]
