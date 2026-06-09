from __future__ import annotations

import pandas as pd
import torch

from facepred.data import (
    CacheManifest,
    CachedSequenceDataset,
    CacheShard,
    collate_cached_sequences,
    save_cache_shard,
    write_cache_manifest,
)
from scripts.prepare_meld_audio_cache import extract_causal_audio_features
from scripts.prepare_meld_cache import build_step_targets
from scripts.train_world_model import WorldMetricAccumulator


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
    waveform = torch.sin(torch.linspace(0.0, 100.0, 1600))
    audio, vad, quality = extract_causal_audio_features(waveform, num_frames=4)

    assert audio.shape == (4, 25)
    assert vad.shape == (4, 3)
    assert quality.shape == (4, 4)
    assert torch.isfinite(audio).all()


def test_causal_audio_features_are_prefix_invariant() -> None:
    waveform = torch.sin(torch.linspace(0.0, 100.0, 1600))
    changed_future = waveform.clone()
    changed_future[800:] = torch.randn_like(changed_future[800:]) * 10.0

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
