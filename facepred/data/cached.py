"""Cached sequence datasets for cloud training.

The cache format is intentionally simple and Colab-friendly:

``manifest.json`` points to one or more ``.pt`` shards per split. Each shard
contains stacked tensors for features, targets, masks, and optional metadata.
This lets expensive feature extraction happen once and keeps training loops
focused on reading compact tensor shards.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

CACHE_VERSION = 1


@dataclass(frozen=True)
class CacheShard:
    """One cache shard entry from ``manifest.json``."""

    path: str
    num_sequences: int


@dataclass(frozen=True)
class CacheManifest:
    """Parsed cache manifest."""

    cache_dir: Path
    version: int
    modalities: tuple[str, ...]
    sequence_length: int
    step_duration_ms: int
    splits: dict[str, tuple[CacheShard, ...]]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, cache_dir: str | Path) -> CacheManifest:
        root = Path(cache_dir)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        splits = {
            split: tuple(CacheShard(path=item["path"], num_sequences=int(item["num_sequences"]))
                         for item in shards)
            for split, shards in raw.get("splits", {}).items()
        }
        return cls(
            cache_dir=root,
            version=int(raw.get("version", CACHE_VERSION)),
            modalities=tuple(raw.get("modalities", [])),
            sequence_length=int(raw.get("sequence_length", 0)),
            step_duration_ms=int(raw.get("step_duration_ms", 100)),
            splits=splits,
            metadata=dict(raw.get("metadata", {})),
        )

    def shard_paths(self, split: str) -> list[Path]:
        if split not in self.splits:
            raise KeyError(f"Split {split!r} not found in cache. Available: {sorted(self.splits)}")
        return [self.cache_dir / shard.path for shard in self.splits[split]]


class CachedSequenceDataset(Dataset):
    """Load cached multimodal sequence tensors for one split."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: str = "train",
        *,
        load_to_memory: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> None:
        self.manifest = CacheManifest.load(cache_dir)
        self.split = split
        self.map_location = map_location
        self.shard_paths = self.manifest.shard_paths(split)
        if not self.shard_paths:
            raise ValueError(f"No shards found for split {split!r}")

        self._shards: list[dict[str, Any]] | None = None
        self._index: list[tuple[int, int]] = []

        if load_to_memory:
            self._shards = [load_cache_shard(path, map_location=map_location) for path in self.shard_paths]
            for shard_idx, shard in enumerate(self._shards):
                count = _sequence_count(shard)
                self._index.extend((shard_idx, item_idx) for item_idx in range(count))
        else:
            for shard_idx, path in enumerate(self.shard_paths):
                header = torch.load(path, map_location=map_location)
                count = _sequence_count(header)
                self._index.extend((shard_idx, item_idx) for item_idx in range(count))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_idx, item_idx = self._index[index]
        shard = self._load_shard(shard_idx)
        return {
            "features": {
                name: tensor[item_idx]
                for name, tensor in shard["features"].items()
            },
            "targets": {
                name: tensor[item_idx]
                for name, tensor in shard["targets"].items()
            },
            "mask": shard["mask"][item_idx],
            "metadata": _metadata_at(shard.get("metadata", {}), item_idx),
        }

    def _load_shard(self, shard_idx: int) -> dict[str, Any]:
        if self._shards is not None:
            return self._shards[shard_idx]
        return load_cache_shard(self.shard_paths[shard_idx], map_location=self.map_location)


def load_cache_shard(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a cache shard and validate the expected top-level keys."""
    shard = torch.load(Path(path), map_location=map_location)
    required = {"features", "targets", "mask"}
    missing = sorted(required - set(shard))
    if missing:
        raise ValueError(f"Cache shard {path} is missing required keys: {missing}")
    return shard


def save_cache_shard(
    path: str | Path,
    *,
    features: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save one shard with a stable tensor-cache structure."""
    shard_path = Path(path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": dict(features),
            "targets": dict(targets),
            "mask": mask.bool(),
            "metadata": dict(metadata or {}),
        },
        shard_path,
    )


def write_cache_manifest(
    cache_dir: str | Path,
    *,
    modalities: Sequence[str],
    sequence_length: int,
    step_duration_ms: int,
    splits: Mapping[str, Sequence[CacheShard | Mapping[str, Any]]],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``manifest.json`` for a cache directory."""
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    serializable_splits: dict[str, list[dict[str, Any]]] = {}
    for split, shards in splits.items():
        serializable_splits[split] = []
        for shard in shards:
            if isinstance(shard, CacheShard):
                serializable_splits[split].append(
                    {"path": shard.path, "num_sequences": shard.num_sequences}
                )
            else:
                serializable_splits[split].append(
                    {
                        "path": str(shard["path"]),
                        "num_sequences": int(shard["num_sequences"]),
                    }
                )

    manifest = {
        "version": CACHE_VERSION,
        "modalities": list(modalities),
        "sequence_length": int(sequence_length),
        "step_duration_ms": int(step_duration_ms),
        "splits": serializable_splits,
        "metadata": dict(metadata or {}),
    }
    manifest_path = root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest_path


def collate_cached_sequences(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate cached sequence samples into a world-model training batch."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    feature_names = samples[0]["features"].keys()
    target_names = samples[0]["targets"].keys()
    return {
        "features": {
            name: torch.stack([sample["features"][name] for sample in samples], dim=0)
            for name in feature_names
        },
        "targets": {
            name: torch.stack([sample["targets"][name] for sample in samples], dim=0)
            for name in target_names
        },
        "mask": torch.stack([sample["mask"] for sample in samples], dim=0).bool(),
        "metadata": [sample.get("metadata", {}) for sample in samples],
    }


def make_cached_dataloader(
    cache_dir: str | Path,
    split: str,
    *,
    batch_size: int = 32,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    load_to_memory: bool = True,
) -> DataLoader:
    """Build a DataLoader for one cached split."""
    dataset = CachedSequenceDataset(cache_dir, split=split, load_to_memory=load_to_memory)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_cached_sequences,
    )


def _sequence_count(shard: Mapping[str, Any]) -> int:
    features = shard.get("features", {})
    if not features:
        raise ValueError("Cache shard contains no features")
    first = next(iter(features.values()))
    return int(first.shape[0])


def _metadata_at(metadata: Mapping[str, Any], index: int) -> dict[str, Any]:
    items = metadata.get("items")
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes)) and index < len(items):
        item = items[index]
        return dict(item) if isinstance(item, Mapping) else {"item": item}
    return {}
