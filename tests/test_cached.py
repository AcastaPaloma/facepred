from __future__ import annotations

import pandas as pd
import pytest
import torch

from facepred.data import (
    CachedSequenceDataset,
    CacheManifest,
    CacheShard,
    collate_cached_sequences,
    event_balanced_sample_weights,
    save_cache_shard,
    validate_event_hazard_cache,
    validate_safe_yield_cache,
    write_cache_manifest,
)
from scripts.prepare_meld_audio_cache import extract_causal_audio_features
from scripts.prepare_meld_cache import build_event_hazard_targets, build_step_targets
from scripts.train_world_model import WorldMetricAccumulator
from scripts.validate_campaign_v5_cache import summarize_cache_build


def test_cached_sequence_dataset_round_trip(tmp_path) -> None:
    shard_path = tmp_path / "train" / "train_0000.pt"
    features = {
        "vad": torch.randn(2, 4, 3),
        "text": torch.randn(2, 4, 8),
        "quality": torch.ones(2, 4, 4),
    }
    targets = {
        "turn_taking": torch.randint(0, 4, (2, 4)),
        "end_of_turn": torch.randint(0, 10, (2, 4)),
        "dialog_act": torch.randint(0, 13, (2, 4)),
        "emotion": torch.randint(0, 7, (2, 4)),
        "valence_arousal": torch.randn(2, 4, 2),
    }
    mask = torch.ones(2, 4, dtype=torch.bool)
    save_cache_shard(
        shard_path,
        features=features,
        targets=targets,
        mask=mask,
        metadata={"items": [{"id": "a"}, {"id": "b"}]},
    )
    write_cache_manifest(
        tmp_path,
        modalities=["vad", "text", "quality"],
        sequence_length=4,
        step_duration_ms=100,
        splits={"train": [CacheShard(path="train/train_0000.pt", num_sequences=2)]},
    )

    dataset = CachedSequenceDataset(tmp_path, split="train")
    batch = collate_cached_sequences([dataset[0], dataset[1]])

    assert len(dataset) == 2
    assert batch["features"]["vad"].shape == (2, 4, 3)
    assert batch["targets"]["valence_arousal"].shape == (2, 4, 2)
    assert batch["mask"].dtype == torch.bool
    assert batch["metadata"][0]["id"] == "a"


def test_future_targets_are_shifted_and_masked() -> None:
    projected = pd.DataFrame(
        {
            "turn_taking": [0, 1, 2, 3],
            "end_of_turn": [0, 1, 2, 3],
            "dialog_act": [0, 1, 2, 3],
            "emotion": [0, 1, 2, 3],
            "valence": [0.0, 0.1, 0.2, 0.3],
            "arousal": [0.0, 0.1, 0.2, 0.3],
        }
    )

    targets = build_step_targets(projected, horizon_steps=[1, 3])

    assert targets["turn_taking"].tolist() == [[1, 1], [2, -100], [3, -100], [-100, -100]]
    assert targets["yield"].tolist() == [[1, 1], [0, -100], [0, -100], [-100, -100]]
    assert targets["horizon_mask"].tolist() == [
        [True, True],
        [True, False],
        [True, False],
        [False, False],
    ]


def test_causal_audio_features_have_model_contract() -> None:
    waveform = torch.sin(torch.linspace(0.0, 100.0, 6400))
    audio, vad, quality = extract_causal_audio_features(waveform, num_frames=4)

    assert audio.shape == (4, 32)
    assert vad.shape == (4, 3)
    assert quality.shape == (4, 4)
    assert torch.isfinite(audio).all()


def test_causal_audio_features_are_prefix_invariant() -> None:
    waveform = torch.sin(torch.linspace(0.0, 100.0, 6400))
    changed_future = waveform.clone()
    changed_future[3200:] = torch.randn_like(changed_future[3200:]) * 10.0

    first = extract_causal_audio_features(waveform, num_frames=4)
    second = extract_causal_audio_features(changed_future, num_frames=4)

    for first_tensor, second_tensor in zip(first, second, strict=True):
        torch.testing.assert_close(first_tensor[:2], second_tensor[:2])


def test_cache_v1_manifest_remains_loadable(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"version": 1, "modalities": [], "sequence_length": 4, '
        '"step_duration_ms": 100, "splits": {}, "metadata": {}}',
        encoding="utf-8",
    )

    assert CacheManifest.load(tmp_path).version == 1


def test_safe_yield_cache_validation_explains_missing_target(tmp_path) -> None:
    save_cache_shard(
        tmp_path / "dev" / "dev_0000.pt",
        features={"vad": torch.ones(1, 4, 3)},
        targets={"turn_taking": torch.zeros(1, 4, 2, dtype=torch.long)},
        mask=torch.ones(1, 4, dtype=torch.bool),
    )
    write_cache_manifest(
        tmp_path,
        modalities=["vad"],
        sequence_length=4,
        step_duration_ms=100,
        splits={"dev": [CacheShard(path="dev/dev_0000.pt", num_sequences=1)]},
        metadata={"target_schema": "legacy"},
    )

    with pytest.raises(ValueError, match="missing targets.*yield"):
        validate_safe_yield_cache(tmp_path, splits=("dev",))


def test_campaign_v5_cache_validation_accepts_rich_event_hazard_cache(tmp_path) -> None:
    targets = {
        "turn_taking": torch.zeros(1, 4, 2, dtype=torch.long),
        "yield": torch.zeros(1, 4, 2, dtype=torch.long),
        "end_of_turn": torch.zeros(1, 4, 2, dtype=torch.long),
        "horizon_mask": torch.ones(1, 4, 2, dtype=torch.bool),
        "event_hazard": torch.zeros(1, 4, dtype=torch.long),
    }
    save_cache_shard(
        tmp_path / "train" / "train_0000.pt",
        features={"audio_prosody": torch.ones(1, 4, 32)},
        targets=targets,
        mask=torch.ones(1, 4, dtype=torch.bool),
    )
    write_cache_manifest(
        tmp_path,
        modalities=["audio_prosody"],
        sequence_length=4,
        step_duration_ms=100,
        splits={"train": [CacheShard(path="train/train_0000.pt", num_sequences=1)]},
        metadata={
            "target_schema": "earliest_event_safe_yield_v3",
            "feature_mode": "causal_audio_stats_v3_pitch_voicing",
            "event_hazard_bins_ms": [200, 500, 1000, 2000],
        },
    )

    contract = validate_event_hazard_cache(tmp_path, splits=("train",))

    assert contract["event_hazard_bins_ms"] == [200, 500, 1000, 2000]


def test_campaign_v5_cache_build_summary_reports_interrupted_progress(tmp_path) -> None:
    (tmp_path / "extraction_config.json").write_text("{}", encoding="utf-8")
    for split, count in {"train": 3, "dev": 2, "test": 1}.items():
        progress_dir = tmp_path / ".progress" / split
        progress_dir.mkdir(parents=True)
        for index in range(count):
            (progress_dir / f"{index}.pt").touch()
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "train_0000.pt").touch()

    summary = summarize_cache_build(tmp_path)

    assert summary["manifest_exists"] is False
    assert summary["extraction_config_exists"] is True
    assert summary["progress_dialogues"] == {"train": 3, "dev": 2, "test": 1}
    assert summary["shards"] == {"train": 1, "dev": 0, "test": 0}


def test_event_balanced_weights_allocate_mass_by_window_group() -> None:
    def sample(yield_value: int, turn_value: int) -> dict[str, object]:
        return {
            "targets": {
                "yield": torch.tensor([[yield_value]]),
                "turn_taking": torch.tensor([[turn_value]]),
                "horizon_mask": torch.ones(1, 1, dtype=torch.bool),
            },
            "mask": torch.ones(1, dtype=torch.bool),
        }

    dataset = [
        sample(0, 0),
        sample(0, 0),
        sample(0, 2),
        sample(1, 1),
    ]
    weights = event_balanced_sample_weights(dataset, (0.5, 0.25, 0.25))

    assert weights[:2].sum() == pytest.approx(0.5)
    assert weights[2] == pytest.approx(0.25)
    assert weights[3] == pytest.approx(0.25)


def test_world_metrics_expose_majority_collapse() -> None:
    accumulator = WorldMetricAccumulator()
    logits = torch.zeros(1, 4, 1, 4)
    logits[..., 1] = 10.0
    accumulator.update(
        {
            "turn_taking_logits": logits,
            "turn_taking_entropy": torch.zeros(1, 4, 1),
        },
        {
            "turn_taking": torch.tensor([[[0], [1], [1], [2]]]),
            "mask": torch.ones(1, 4, dtype=torch.bool),
            "horizon_mask": torch.ones(1, 4, 1, dtype=torch.bool),
        },
    )

    metrics = accumulator.metrics()

    assert metrics["turn_active_classes_mean"] == 1.0
    assert metrics["turn_macro_f1_mean"] == metrics["turn_majority_macro_f1_mean"]
    assert metrics["turn_pred_support/h0/class1"] == 4.0


def test_event_hazard_target_encodes_earliest_marked_event_bin() -> None:
    events = torch.tensor([0, 0, 1, 0, 0, 2, 3, 0])

    targets = build_event_hazard_targets(events, bin_upper_steps=[2, 4])

    assert targets[0] == 1
    assert targets[1] == 1
    assert targets[2] == 5
    assert targets[-4:].eq(-100).all()


def test_calibrated_threshold_applies_only_to_selected_commit_score() -> None:
    accumulator = WorldMetricAccumulator(
        yield_thresholds=(0.9,),
        operating_score_source="commit_safety",
    )
    accumulator.update(
        {
            "yield_probs": torch.tensor([[[0.6]]]),
            "commit_safety_probs": torch.tensor([[[0.8]]]),
        },
        {
            "yield": torch.tensor([[[1]]]),
            "mask": torch.ones(1, 1, dtype=torch.bool),
            "horizon_mask": torch.ones(1, 1, 1, dtype=torch.bool),
        },
    )

    metrics = accumulator.metrics()

    assert metrics["yield_commits/h0"] == 1
    assert metrics["commit_safety_commits/h0"] == 0
