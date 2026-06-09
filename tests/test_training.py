from __future__ import annotations

import json

import torch

from scripts.train_world_model import (
    capture_rng_state,
    compute_class_weights,
    make_scheduler,
    restore_rng_state,
)
from scripts.tune_and_train_colab import read_score


def test_inverse_class_weights_equalize_present_class_mass() -> None:
    loader = [
        {
            "targets": {
                "turn_taking": torch.tensor([[0, 0, 0, 0, 1]]),
            }
        },
        {
            "targets": {
                "turn_taking": torch.tensor([[0, 0, 0, 1, -100]]),
            }
        },
    ]

    weights = compute_class_weights(loader, specs={"turn_taking": (3, "inverse")})["turn_taking"]
    counts = torch.tensor([7.0, 2.0, 0.0])

    assert torch.allclose(counts[:2] * weights[:2], torch.full((2,), counts[0] * weights[0]))
    assert weights[2] == 0


def test_long_schedule_does_not_reach_min_lr_at_stage1_end() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-4)
    scheduler = make_scheduler(
        optimizer,
        {"name": "cosine_warmup", "warmup_fraction": 0.05, "min_lr": 1.0e-6},
        total_steps=75,
    )

    for _ in range(5):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] > 9.0e-5


def test_restore_rng_state_normalizes_remapped_torch_state() -> None:
    state = capture_rng_state()
    state["torch"] = state["torch"].to(dtype=torch.int16)

    restore_rng_state(state)

    assert torch.get_rng_state().dtype == torch.uint8
    assert torch.get_rng_state().device.type == "cpu"


def test_tuning_score_marks_majority_baseline_as_collapsed(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "epoch": 0,
                "best_score": 0.216,
                "val": {
                    "turn_macro_f1_mean": 0.216,
                    "turn_majority_macro_f1_mean": 0.216,
                    "turn_active_classes_mean": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = read_score(metrics_path)

    assert result["collapsed"] is True
    assert result["improvement_over_majority"] == 0.0


def test_tuning_score_is_bounded_to_requested_stage(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    records = [
        {
            "epoch": 4,
            "best_score": 0.25,
            "val": {
                "turn_macro_f1_mean": 0.25,
                "turn_majority_macro_f1_mean": 0.21,
                "turn_active_classes_mean": 3.0,
            },
        },
        {
            "epoch": 12,
            "best_score": 0.40,
            "val": {
                "turn_macro_f1_mean": 0.40,
                "turn_majority_macro_f1_mean": 0.21,
                "turn_active_classes_mean": 4.0,
            },
        },
    ]
    metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    result = read_score(metrics_path, max_epochs=5)

    assert result["score"] == 0.25
    assert result["best_epoch"] == 4
