from __future__ import annotations

import pandas as pd
import pytest
import torch

from facepred.data import TurnLabelConfig, derive_utterance_labels
from facepred.engine.precompute import GateThresholds, PrecomputeEngine
from scripts.calibrate_yield import (
    select_threshold_for_minimum_precision,
    threshold_for_minimum_precision,
)
from scripts.prepare_meld_cache import corrected_transition_event, project_dialogue_to_steps
from scripts.train_world_model import binary_probability_metrics, compute_yield_pos_weight


def test_projected_events_are_sparse_and_countdown_decreases() -> None:
    dialogue = pd.DataFrame(
        {
            "dialogue_id": [0, 0],
            "utterance_id": [0, 1],
            "speaker": ["A", "B"],
            "utterance": ["hello", "hi"],
            "emotion": ["neutral", "neutral"],
            "sentiment": ["neutral", "neutral"],
            "start_s": [0.0, 0.8],
            "end_s": [0.5, 1.2],
        }
    )
    labeled = derive_utterance_labels(dialogue, TurnLabelConfig())

    projected = project_dialogue_to_steps(
        labeled,
        step_ms=100,
        label_config=TurnLabelConfig(),
    )

    assert (projected["turn_taking"] == 1).sum() == 1
    assert projected.loc[0, "time_to_yield_s"] > projected.loc[1, "time_to_yield_s"]
    assert projected.loc[0, "end_of_turn"] >= projected.loc[1, "end_of_turn"]


def test_corrected_transition_marks_any_cross_speaker_interruption_as_overlap() -> None:
    current = pd.Series({"speaker": "A", "start_s": 0.0, "end_s": 10.0})
    following = pd.Series({"speaker": "B", "start_s": 9.9, "end_s": 10.1})

    label_id, gap_s = corrected_transition_event(current, following, TurnLabelConfig())

    assert label_id == 3
    assert gap_s == pytest.approx(-0.1)


def test_yield_metrics_and_precision_threshold() -> None:
    probabilities = torch.tensor([0.95, 0.8, 0.7, 0.1])
    labels = torch.tensor([1, 1, 0, 0])

    threshold = threshold_for_minimum_precision(probabilities, labels, 0.9)
    metrics = binary_probability_metrics(probabilities, labels, threshold=threshold)

    assert threshold == pytest.approx(0.8)
    assert metrics["precision"] == 1.0
    assert metrics["average_precision"] > metrics["prevalence"]


def test_infeasible_precision_policy_abstains() -> None:
    policy = select_threshold_for_minimum_precision(
        torch.tensor([0.9, 0.8, 0.7, 0.6]),
        torch.tensor([0, 1, 0, 1]),
        0.9,
        minimum_commits=2,
    )

    assert policy["status"] == "infeasible_abstain"
    assert policy["policy_satisfied"] is False
    assert policy["threshold"] > 1.0
    assert policy["commits"] == 0


def test_mild_yield_weighting_modes() -> None:
    loader = [
        {
            "targets": {
                "yield": torch.tensor([[[1, 1], [0, 0], [0, 0], [0, 0]]]),
            }
        }
    ]

    torch.testing.assert_close(
        compute_yield_pos_weight(loader, "sqrt_balanced"),
        torch.tensor([3.0**0.5, 3.0**0.5]),
    )
    torch.testing.assert_close(
        compute_yield_pos_weight(loader, "capped_balanced", cap=2.0),
        torch.tensor([2.0, 2.0]),
    )


def test_gate_uses_calibrated_commit_and_planning_horizons() -> None:
    engine = PrecomputeEngine(
        GateThresholds(
            yield_threshold=0.99,
            entropy_max=1.0,
            margin_min=0.0,
            commit_horizon_index=0,
            planning_horizon_index=1,
            yield_temperatures=(2.0, 1.0),
            yield_thresholds=(0.6, 0.9),
        )
    )
    predictions = {
        "yield_logits": torch.tensor([[[2.0, 3.0]]]),
        "turn_probs": torch.tensor([[[0.7, 0.1, 0.2, 0.0]]]),
    }

    decision = engine.decide(predictions)

    assert decision.should_commit is True
    assert decision.yield_probability == torch.sigmoid(torch.tensor(1.0)).item()
    assert decision.candidates[0].name == "answer"


def test_gate_prefers_commit_safety_verifier_over_direct_yield() -> None:
    engine = PrecomputeEngine(
        GateThresholds(
            yield_threshold=0.8,
            entropy_max=1.0,
            margin_min=0.0,
            commit_horizon_index=0,
        )
    )
    predictions = {
        "yield_logits": torch.tensor([[[10.0, 10.0]]]),
        "commit_safety_logits": torch.tensor([[[-10.0, -10.0]]]),
        "turn_probs": torch.tensor([[[0.1, 0.8, 0.1, 0.0]]]),
    }

    decision = engine.decide(predictions)

    assert decision.should_commit is False
    assert decision.reason == "blocked_by_yield_probability"
