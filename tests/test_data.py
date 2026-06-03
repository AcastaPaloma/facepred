from __future__ import annotations

import torch

from facepred.data import (
    MELDDataset,
    MultimodalAugmentor,
    MultimodalSynchronizer,
    collate_meld_dialogues,
    make_synthetic_synchronized_batch,
)


def test_synthetic_meld_dataset_collates_dialogues() -> None:
    dataset = MELDDataset.from_synthetic(num_dialogues=2, utterances_per_dialogue=3, seed=7)

    assert len(dataset) == 2
    sample = dataset[0]
    assert "features" in sample
    assert sample["labels"]["turn_taking"].shape == (3,)

    batch = collate_meld_dialogues([dataset[0], dataset[1]])
    assert batch["mask"].shape == (2, 3)
    assert batch["features"]["audio_prosody"].shape[:2] == (2, 3)
    assert batch["labels"]["valence_arousal"].shape == (2, 3, 2)


def test_synchronizer_aligns_modalities_and_exports_torch() -> None:
    synchronizer = MultimodalSynchronizer(step_duration_ms=100)
    batch = synchronizer.align_modalities(
        {"audio": [[1.0], [2.0], [3.0]], "visual": [[0.5], [0.7]]},
        timestamps={"audio": [0.0, 0.1, 0.2], "visual": [0.0, 0.2]},
        start_s=0.0,
        end_s=0.3,
    )
    torch_batch = batch.as_torch()

    assert batch.length == 3
    assert torch_batch["features"]["audio"].shape == (3, 1)
    assert torch_batch["masks"]["visual"].dtype == torch.bool


def test_augmentor_preserves_feature_shapes() -> None:
    synced = make_synthetic_synchronized_batch(duration_s=0.3, feature_dims={"audio": 4, "quality": 4})
    augmentor = MultimodalAugmentor.from_training_config(seed=3)
    result = augmentor(synced.features)

    assert result.features["audio"].shape == synced.features["audio"].shape
    assert result.features["quality"].shape == synced.features["quality"].shape
