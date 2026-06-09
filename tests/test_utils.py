from __future__ import annotations

import io
import logging
import random

import numpy as np
import torch

from facepred.utils import (
    Timer,
    brier_score,
    calibration_summary,
    confidence_margin,
    expected_calibration_error,
    format_seconds,
    load_trusted_torch_artifact,
    log_once,
    predictive_entropy,
    seed_everything,
    seed_worker,
    setup_logging,
    softmax,
)


def test_seed_everything_reproducible_for_python_and_numpy() -> None:
    seed_everything(123)
    python_first = random.random()
    numpy_first = np.random.rand(3)

    seed_everything(123)

    assert random.random() == python_first
    np.testing.assert_allclose(np.random.rand(3), numpy_first)


def test_seed_worker_modulos_large_base_seed() -> None:
    assert seed_worker(1, base_seed=2**40) == 1


def test_timer_with_custom_clock() -> None:
    ticks = iter([10.0, 10.25, 10.5, 10.75])
    timer = Timer(clock=lambda: next(ticks))

    assert timer.stop() == 0.25
    assert timer.reset(start=True).stop() == 0.25
    assert format_seconds(0.25) == "250.000ms"


def test_logging_setup_and_log_once() -> None:
    stream = io.StringIO()
    setup_logging(level="INFO", stream=stream, force=True, fmt="%(levelname)s:%(message)s")
    logger = logging.getLogger("facepred.tests")

    assert log_once(logger, "INFO", "hello", key="greeting") is True
    assert log_once(logger, "INFO", "hello again", key="greeting") is False

    assert stream.getvalue().strip() == "INFO:hello"


def test_calibration_metrics_for_known_probabilities() -> None:
    probs = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.4, 0.6],
            [0.7, 0.3],
        ]
    )
    labels = np.array([0, 0, 1, 1])

    np.testing.assert_allclose(predictive_entropy([[0.5, 0.5]], normalized=True), [1.0])
    assert brier_score([[1.0, 0.0], [0.0, 1.0]], [0, 1]) == 0.0
    np.testing.assert_allclose(confidence_margin([[0.65, 0.35]]), [0.3])

    ece = expected_calibration_error(probs, labels, n_bins=2)
    summary = calibration_summary(probs, labels, n_bins=2)

    assert ece == summary.ece
    assert summary.n_samples == 4
    assert summary.brier > 0.0


def test_brier_score_supports_non_last_class_axis_one_hot_targets() -> None:
    probs = np.array([[[1.0], [0.0]], [[0.0], [1.0]]])
    targets = np.array([[[1.0], [0.0]], [[0.0], [1.0]]])

    assert brier_score(probs, targets, axis=0) == 0.0


def test_softmax_accepts_logits() -> None:
    logits = np.array([[1.0, 2.0, 3.0]])
    probs = softmax(logits)

    np.testing.assert_allclose(probs.sum(axis=-1), [1.0])
    assert probs.argmax(axis=-1).item() == 2


def test_load_trusted_torch_artifact_restores_structured_training_state(tmp_path) -> None:
    artifact_path = tmp_path / "checkpoint.pt"
    rng_state = np.random.RandomState(42).get_state()
    torch.save({"model_state_dict": {"weight": torch.ones(2)}, "numpy_rng": rng_state}, artifact_path)

    artifact = load_trusted_torch_artifact(artifact_path, map_location="cpu")

    assert artifact["numpy_rng"][0] == "MT19937"
    np.testing.assert_array_equal(artifact["numpy_rng"][1], rng_state[1])
    torch.testing.assert_close(artifact["model_state_dict"]["weight"], torch.ones(2))
