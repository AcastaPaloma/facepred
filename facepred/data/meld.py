"""MELD dataset helpers and lightweight synthetic fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from facepred.data.label_derivation import (
    EMOTIONS,
    SENTIMENTS,
    TurnLabelConfig,
    derive_utterance_labels,
    ensure_temporal_columns,
)

try:  # Torch is a core training dependency, but keep imports graceful for metadata tools.
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover - exercised only in torch-free environments.
    torch = None

    class TorchDataset:  # type: ignore[no-redef]
        pass


MELD_SPLIT_ALIASES: Mapping[str, str] = {
    "train": "train",
    "training": "train",
    "dev": "dev",
    "val": "dev",
    "validation": "dev",
    "test": "test",
}

MELD_SPLIT_FILENAMES: Mapping[str, str] = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

DEFAULT_FEATURE_DIMS: Mapping[str, int] = {
    "audio_prosody": 88,
    "audio_ssl": 768,
    "visual": 1493,
    "vad": 3,
    "text": 384,
    "quality": 4,
}


@dataclass(frozen=True)
class MELDConfig:
    """Small runtime config for MELD dataframe and dataset scaffolds."""

    data_root: str | Path | None = None
    split: str = "train"
    step_duration_ms: int = 100
    sequence_length_s: float = 5.0
    feature_dims: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_FEATURE_DIMS))
    synthetic: bool = False
    synthetic_dialogues: int = 4
    synthetic_utterances_per_dialogue: int = 8
    seed: int = 42
    return_tensors: bool = True
    include_synthetic_features: bool = True


@dataclass(frozen=True)
class MELDRecord:
    """Normalized utterance metadata for one MELD row."""

    record_id: str
    split: str
    dialogue_id: int | str
    utterance_id: int | str
    speaker: str
    utterance: str
    start_s: float
    end_s: float
    emotion: str = "neutral"
    sentiment: str = "neutral"
    media_path: str | None = None


class MELDDataset(TorchDataset):
    """Utterance-sequence dataset grouped by dialogue.

    The dataset intentionally works at the normalized dataframe level. It does
    not decode media or run feature extractors, so iteration-0 smoke tests can
    run without OpenCV, MediaPipe, openSMILE, or Hugging Face datasets.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        split: str | None = None,
        dataframe: pd.DataFrame | None = None,
        config: MELDConfig | None = None,
        label_config: TurnLabelConfig | None = None,
        return_tensors: bool | None = None,
    ) -> None:
        self.config = config or MELDConfig(data_root=root, split=split or "train")
        self.label_config = label_config or TurnLabelConfig(
            step_duration_ms=self.config.step_duration_ms
        )
        self.split = canonical_meld_split(split or self.config.split)
        self.return_tensors = (
            self.config.return_tensors if return_tensors is None else return_tensors
        )

        if dataframe is None:
            if self.config.synthetic:
                dataframe = make_synthetic_meld_dataframe(
                    num_dialogues=self.config.synthetic_dialogues,
                    utterances_per_dialogue=self.config.synthetic_utterances_per_dialogue,
                    split=self.split,
                    seed=self.config.seed,
                )
            else:
                root_path = Path(root or self.config.data_root or "data")
                dataframe = load_meld_split(root_path, self.split)

        normalized = normalize_meld_dataframe(dataframe, split=self.split)
        self.frame = derive_utterance_labels(normalized, config=self.label_config)
        self.dialogue_ids = list(self.frame["dialogue_id"].drop_duplicates())

    @classmethod
    def from_synthetic(
        cls,
        num_dialogues: int = 4,
        utterances_per_dialogue: int = 8,
        split: str = "train",
        seed: int = 42,
        **kwargs: Any,
    ) -> MELDDataset:
        """Construct a deterministic synthetic MELD-like dataset."""

        config = MELDConfig(
            split=split,
            synthetic=True,
            synthetic_dialogues=num_dialogues,
            synthetic_utterances_per_dialogue=utterances_per_dialogue,
            seed=seed,
        )
        return cls(config=config, **kwargs)

    def __len__(self) -> int:
        return len(self.dialogue_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        dialogue_id = self.dialogue_ids[index]
        dialogue = self.frame[self.frame["dialogue_id"] == dialogue_id].sort_values(
            ["start_s", "end_s"]
        )
        sample = {
            "dialogue_id": dialogue_id,
            "record_id": dialogue["record_id"].tolist(),
            "utterance": dialogue["utterance"].tolist(),
            "speaker": dialogue["speaker"].tolist(),
            "start_s": dialogue["start_s"].to_numpy(dtype=np.float32),
            "end_s": dialogue["end_s"].to_numpy(dtype=np.float32),
            "labels": _labels_from_frame(dialogue),
        }

        if self.config.synthetic and self.config.include_synthetic_features:
            sample["features"] = make_synthetic_feature_dict(
                len(dialogue),
                feature_dims=self.config.feature_dims,
                seed=self.config.seed + int(index),
            )

        if self.return_tensors:
            return _tensorize_sample(sample)
        return sample

    def records(self) -> list[MELDRecord]:
        """Return normalized rows as dataclass records for metadata consumers."""

        rows: list[MELDRecord] = []
        for _, row in self.frame.iterrows():
            rows.append(
                MELDRecord(
                    record_id=str(row["record_id"]),
                    split=str(row["split"]),
                    dialogue_id=row["dialogue_id"],
                    utterance_id=row["utterance_id"],
                    speaker=str(row["speaker"]),
                    utterance=str(row["utterance"]),
                    start_s=float(row["start_s"]),
                    end_s=float(row["end_s"]),
                    emotion=str(row.get("emotion", "neutral")),
                    sentiment=str(row.get("sentiment", "neutral")),
                    media_path=_nullable_string(row.get("media_path")),
                )
            )
        return rows


def canonical_meld_split(split: str) -> str:
    """Normalize split aliases to MELD's train/dev/test names."""

    key = str(split).strip().lower()
    if key not in MELD_SPLIT_ALIASES:
        raise ValueError(f"Unknown MELD split {split!r}; expected train, dev/val, or test")
    return MELD_SPLIT_ALIASES[key]


def find_meld_csv(root: str | Path, split: str) -> Path:
    """Find a MELD split CSV in a raw or lightly unpacked dataset directory."""

    root_path = Path(root)
    canonical = canonical_meld_split(split)
    filename = MELD_SPLIT_FILENAMES[canonical]
    candidates = [
        root_path / canonical / filename,
        root_path / filename,
        root_path / "MELD.Raw" / canonical / filename,
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = list(root_path.rglob(filename)) if root_path.exists() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {filename} under {root_path}")


def load_meld_split(root: str | Path, split: str = "train") -> pd.DataFrame:
    """Load and normalize one MELD split CSV from local disk."""

    csv_path = find_meld_csv(root, split)
    frame = pd.read_csv(csv_path)
    normalized = normalize_meld_dataframe(frame, split=canonical_meld_split(split))
    return add_meld_media_paths(normalized, root=root, split=split)


def normalize_meld_dataframe(
    frame: pd.DataFrame,
    split: str = "train",
    add_record_id: bool = True,
) -> pd.DataFrame:
    """Normalize common MELD CSV columns into snake_case, typed fields."""

    result = frame.copy()
    result = result.rename(columns={column: _snake_case(column) for column in result.columns})
    result = result.rename(
        columns={
            "dialogue_id": "dialogue_id",
            "dialogueid": "dialogue_id",
            "utterance_id": "utterance_id",
            "utteranceid": "utterance_id",
            "starttime": "start_time",
            "endtime": "end_time",
        }
    )

    if "dialogue_id" not in result:
        result["dialogue_id"] = 0
    if "utterance_id" not in result:
        result["utterance_id"] = result.groupby("dialogue_id").cumcount()
    if "speaker" not in result:
        result["speaker"] = "speaker_0"
    if "utterance" not in result:
        result["utterance"] = ""
    if "emotion" not in result:
        result["emotion"] = "neutral"
    if "sentiment" not in result:
        result["sentiment"] = "neutral"

    result["split"] = canonical_meld_split(split)
    result["speaker"] = result["speaker"].fillna("unknown").astype(str)
    result["utterance"] = result["utterance"].fillna("").astype(str)
    result["emotion"] = _clean_category(result["emotion"], EMOTIONS, default="neutral")
    result["sentiment"] = _clean_category(result["sentiment"], SENTIMENTS, default="neutral")
    result = ensure_temporal_columns(result)

    if add_record_id:
        result["record_id"] = [
            f"{row.split}:dia{row.dialogue_id}_utt{row.utterance_id}"
            for row in result.itertuples(index=False)
        ]

    return result.sort_values(["dialogue_id", "start_s", "utterance_id"]).reset_index(drop=True)


def add_meld_media_paths(
    frame: pd.DataFrame,
    root: str | Path,
    split: str = "train",
) -> pd.DataFrame:
    """Add best-effort local ``media_path`` values when raw videos are present."""

    result = frame.copy()
    root_path = Path(root)
    canonical = canonical_meld_split(split)
    split_dirs = {
        "train": ("train_splits",),
        "dev": ("dev_splits_complete", "dev_splits"),
        "test": ("output_repeated_splits_test", "test_splits"),
    }
    media_dirs = [root_path / canonical / dirname for dirname in split_dirs[canonical]]
    media_dirs.extend(root_path.rglob(split_dirs[canonical][0]) if root_path.exists() else [])

    paths: list[str | None] = []
    for row in result.itertuples(index=False):
        filename = f"dia{row.dialogue_id}_utt{row.utterance_id}.mp4"
        found = None
        for media_dir in media_dirs:
            candidate = media_dir / filename
            if candidate.exists():
                found = str(candidate)
                break
        paths.append(found)
    result["media_path"] = paths
    return result


def make_synthetic_meld_dataframe(
    num_dialogues: int = 4,
    utterances_per_dialogue: int = 8,
    split: str = "train",
    seed: int = 42,
) -> pd.DataFrame:
    """Build deterministic MELD-like metadata for tests and demos."""

    rng = np.random.default_rng(seed)
    speakers = np.asarray(["Monica", "Chandler", "Rachel", "Ross"])
    rows: list[dict[str, Any]] = []

    for dialogue_id in range(num_dialogues):
        time_s = float(rng.uniform(0.0, 0.3))
        speaker_a, speaker_b = rng.choice(speakers, size=2, replace=False)
        for utterance_id in range(utterances_per_dialogue):
            duration_s = float(rng.uniform(0.45, 2.2))
            if utterance_id > 0:
                time_s += float(rng.uniform(-0.15, 0.55))
                time_s = max(0.0, time_s)
            speaker = speaker_a if utterance_id % 2 == 0 else speaker_b
            emotion = str(rng.choice(EMOTIONS, p=[0.42, 0.12, 0.06, 0.1, 0.18, 0.04, 0.08]))
            sentiment = "neutral"
            if emotion in {"joy", "surprise"}:
                sentiment = "positive"
            elif emotion in {"anger", "disgust", "fear", "sadness"}:
                sentiment = "negative"
            rows.append(
                {
                    "split": canonical_meld_split(split),
                    "dialogue_id": dialogue_id,
                    "utterance_id": utterance_id,
                    "speaker": speaker,
                    "utterance": _synthetic_utterance(utterance_id, emotion),
                    "emotion": emotion,
                    "sentiment": sentiment,
                    "start_s": round(time_s, 3),
                    "end_s": round(time_s + duration_s, 3),
                }
            )
            time_s += duration_s

    return normalize_meld_dataframe(pd.DataFrame(rows), split=split)


def make_synthetic_feature_dict(
    length: int,
    feature_dims: Mapping[str, int] | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create deterministic feature arrays with model-config-compatible shapes."""

    dims = dict(DEFAULT_FEATURE_DIMS)
    if feature_dims:
        dims.update(feature_dims)
    rng = np.random.default_rng(seed)
    features: dict[str, np.ndarray] = {}
    for name, dim in dims.items():
        if name == "vad":
            vad = np.zeros((length, dim), dtype=np.float32)
            if dim > 0:
                vad[:, 0] = 1.0
            if dim > 2:
                vad[:, 2] = np.arange(length, dtype=np.float32) % 2
            features[name] = vad
        elif name == "quality":
            quality = np.ones((length, dim), dtype=np.float32)
            if dim >= 4:
                quality[:, :4] = np.asarray([0.8, 0.9, 0.75, 1.0], dtype=np.float32)
            features[name] = quality
        else:
            features[name] = rng.normal(0.0, 0.1, size=(length, dim)).astype(np.float32)
    return features


def collate_meld_dialogues(
    samples: Sequence[Mapping[str, Any]],
    pad_label_value: int = -100,
) -> dict[str, Any]:
    """Pad a batch of dialogue samples emitted by ``MELDDataset``."""

    if torch is None:
        raise ImportError("torch is required for collate_meld_dialogues")
    if not samples:
        raise ValueError("Cannot collate an empty sample list")

    lengths = torch.tensor([len(sample["start_s"]) for sample in samples], dtype=torch.long)
    max_len = int(lengths.max().item())
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    batch: dict[str, Any] = {
        "dialogue_id": [sample["dialogue_id"] for sample in samples],
        "record_id": [sample["record_id"] for sample in samples],
        "utterance": [sample["utterance"] for sample in samples],
        "speaker": [sample["speaker"] for sample in samples],
        "lengths": lengths,
        "mask": mask,
        "start_s": _pad_1d([sample["start_s"] for sample in samples], max_len, 0.0),
        "end_s": _pad_1d([sample["end_s"] for sample in samples], max_len, 0.0),
        "labels": {},
    }

    label_keys = samples[0]["labels"].keys()
    for key in label_keys:
        values = [sample["labels"][key] for sample in samples]
        pad_value = float("nan") if _is_float_tensor(values[0]) else pad_label_value
        batch["labels"][key] = _pad_tensor(values, max_len, pad_value)

    if "features" in samples[0]:
        batch["features"] = {}
        for key in samples[0]["features"]:
            batch["features"][key] = _pad_tensor(
                [sample["features"][key] for sample in samples],
                max_len,
                0.0,
            )
    return batch


def _labels_from_frame(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    labels = {
        "turn_taking": frame["turn_label_id"].to_numpy(dtype=np.int64),
        "emotion": frame["emotion_id"].to_numpy(dtype=np.int64),
        "sentiment": frame["sentiment_id"].to_numpy(dtype=np.int64),
        "dialog_act": frame["dialog_act_id"].to_numpy(dtype=np.int64),
        "time_to_yield_s": frame["time_to_yield_s"].to_numpy(dtype=np.float32),
        "end_of_turn_bucket": frame["end_of_turn_bucket_id"].to_numpy(dtype=np.int64),
        "valence_arousal": frame[["valence", "arousal"]].to_numpy(dtype=np.float32),
    }
    for column in frame.columns:
        if column.startswith("yield_within_"):
            labels[column] = frame[column].to_numpy(dtype=np.int64)
    return labels


def _tensorize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    if torch is None:
        raise ImportError("torch is required when return_tensors=True")
    result = dict(sample)
    result["start_s"] = torch.as_tensor(sample["start_s"], dtype=torch.float32)
    result["end_s"] = torch.as_tensor(sample["end_s"], dtype=torch.float32)
    result["labels"] = {
        key: torch.as_tensor(np.asarray(value).copy(), dtype=_torch_dtype_for(value))
        for key, value in sample["labels"].items()
    }
    if "features" in sample:
        result["features"] = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in sample["features"].items()
        }
    return result


def _pad_1d(tensors: Sequence[Any], max_len: int, value: float) -> Any:
    return _pad_tensor(tensors, max_len, value)


def _pad_tensor(tensors: Sequence[Any], max_len: int, value: float | int) -> Any:
    converted = [torch.as_tensor(tensor) for tensor in tensors]
    shape = (len(converted), max_len, *converted[0].shape[1:])
    out = torch.full(shape, value, dtype=converted[0].dtype)
    for idx, tensor in enumerate(converted):
        out[idx, : tensor.shape[0]] = tensor
    return out


def _is_float_tensor(value: Any) -> bool:
    tensor = torch.as_tensor(value)
    return tensor.dtype.is_floating_point


def _torch_dtype_for(value: np.ndarray) -> Any:
    if torch is None:
        return None
    return torch.float32 if np.issubdtype(value.dtype, np.floating) else torch.long


def _clean_category(series: pd.Series, labels: Sequence[str], default: str) -> pd.Series:
    allowed = set(labels)
    return series.fillna(default).map(
        lambda value: value if str(value).strip().lower() in allowed else default
    )


def _snake_case(value: object) -> str:
    text = str(value).strip().replace("-", "_").replace(" ", "_").replace(".", "")
    chars = [char.lower() if char.isalnum() else "_" for char in text]
    compact = "_".join(part for part in "".join(chars).split("_") if part)
    return compact


def _synthetic_utterance(utterance_id: int, emotion: str) -> str:
    templates = {
        "neutral": "I see what you mean.",
        "surprise": "Wait, really?",
        "fear": "I am not sure this is safe.",
        "sadness": "That is really hard.",
        "joy": "That sounds great!",
        "disgust": "I do not like that.",
        "anger": "That is not okay.",
    }
    if utterance_id % 5 == 3:
        return "Yeah."
    return templates.get(emotion, "I see.")


def _nullable_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)
