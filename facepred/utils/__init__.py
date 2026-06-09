"""Shared lightweight utilities for FacePred."""

from facepred.utils.calibration import (
    CalibrationBin,
    CalibrationSummary,
    as_probabilities,
    brier_score,
    calibration_bins,
    calibration_summary,
    confidence_margin,
    expected_calibration_error,
    normalize_probabilities,
    one_hot,
    predictive_entropy,
    softmax,
    top_confidence,
)
from facepred.utils.logging_utils import coerce_log_level, get_logger, log_once, setup_logging
from facepred.utils.seeding import (
    DEFAULT_SEED,
    MAX_NUMPY_SEED,
    make_torch_generator,
    resolve_seed,
    seed_everything,
    seed_worker,
)
from facepred.utils.timing import Timer, format_seconds, time_block
from facepred.utils.torch_io import load_trusted_torch_artifact

__all__ = [
    "CalibrationBin",
    "CalibrationSummary",
    "DEFAULT_SEED",
    "MAX_NUMPY_SEED",
    "Timer",
    "as_probabilities",
    "brier_score",
    "calibration_bins",
    "calibration_summary",
    "coerce_log_level",
    "confidence_margin",
    "expected_calibration_error",
    "format_seconds",
    "get_logger",
    "log_once",
    "load_trusted_torch_artifact",
    "make_torch_generator",
    "normalize_probabilities",
    "one_hot",
    "predictive_entropy",
    "resolve_seed",
    "seed_everything",
    "seed_worker",
    "setup_logging",
    "softmax",
    "time_block",
    "top_confidence",
]
