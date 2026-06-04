from __future__ import annotations

import torch

from facepred.data import (
    CachedSequenceDataset,
    CacheShard,
    collate_cached_sequences,
    save_cache_shard,
    write_cache_manifest,
)


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
