"""Prepare cache shards for Colab-friendly FacePred training.

This first real-training cache path uses cheap features that can be generated
from MELD metadata alone:

- VAD-like activity from utterance timing
- deterministic hashed text features from utterance text
- quality/completeness signals

Expensive media features can be added later as additional cache modalities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facepred.data import (
    CacheShard,
    TurnLabelConfig,
    canonical_meld_split,
    derive_utterance_labels,
    load_meld_split,
    make_synthetic_meld_dataframe,
    make_time_grid,
    normalize_meld_dataframe,
    save_cache_shard,
    write_cache_manifest,
)
from facepred.engine.trainer import load_project_config

DEFAULT_CHEAP_MODALITIES = ("vad", "text", "quality")
TURN_IGNORE_INDEX = -100
EVENT_GAP_BUCKETS = {
    0: "overlap",
    1: "gap_le_250ms",
    2: "gap_250_750ms",
    3: "gap_gt_750ms",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="Project config YAML.")
    parser.add_argument("--data-root", default=None, help="Root containing raw MELD split CSVs.")
    parser.add_argument("--output-dir", required=True, help="Directory where cache shards are written.")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"], help="Splits to cache.")
    parser.add_argument("--modalities", default=",".join(DEFAULT_CHEAP_MODALITIES), help="Comma-separated modalities to write.")
    parser.add_argument("--include-zero-modalities", action="store_true", help="Also write zero tensors for omitted model modalities.")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic MELD metadata instead of reading disk.")
    parser.add_argument("--allow-synthetic", action="store_true", help="Fall back to synthetic metadata if a real split is missing.")
    parser.add_argument("--synthetic-dialogues", type=int, default=8, help="Synthetic dialogue count per split.")
    parser.add_argument("--utterances-per-dialogue", type=int, default=8, help="Synthetic utterances per dialogue.")
    parser.add_argument("--sequence-length-s", type=float, default=None, help="Override sequence window length.")
    parser.add_argument("--stride-s", type=float, default=None, help="Override sequence stride.")
    parser.add_argument("--shard-size", type=int, default=512, help="Sequences per shard.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic synthetic/cache seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_project_config(args.config)
    model_cfg = config["model"]
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    step_ms = int(model_cfg.get("step_duration_ms", 100))
    sequence_length_s = float(
        args.sequence_length_s
        or training_cfg.get("dataloader", {}).get("sequence_length_s", 5.0)
    )
    stride_s = float(args.stride_s or max(sequence_length_s / 2.0, step_ms / 1000.0))
    sequence_length = max(1, int(round(sequence_length_s * 1000 / step_ms)))
    stride_steps = max(1, int(round(stride_s * 1000 / step_ms)))
    input_dims = model_input_dims(model_cfg)
    requested_modalities = parse_modalities(args.modalities)
    modalities = resolve_modalities(requested_modalities, input_dims, args.include_zero_modalities)
    label_config = label_config_from_project(data_cfg, model_cfg)
    horizons_ms = [int(value) for value in model_cfg.get("prediction_horizons_ms", [200, 1000])]
    horizon_steps = [max(1, int(round(value / step_ms))) for value in horizons_ms]
    event_hazard_bins_ms = [int(value) for value in model_cfg.get("event_hazard_bins_ms", [])]
    event_hazard_steps = [
        max(1, int(round(value / step_ms))) for value in event_hazard_bins_ms
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_shards: dict[str, list[CacheShard]] = {}

    for split in args.splits:
        canonical_split = canonical_meld_split(split)
        frame = load_or_synthesize_split(
            canonical_split,
            args,
            data_root=args.data_root,
        )
        sequences = build_split_sequences(
            frame=frame,
            split=canonical_split,
            modalities=modalities,
            input_dims=input_dims,
            label_config=label_config,
            step_ms=step_ms,
            sequence_length=sequence_length,
            stride_steps=stride_steps,
            horizon_steps=horizon_steps,
            event_hazard_steps=event_hazard_steps,
            seed=args.seed,
        )
        split_shards[canonical_split] = write_split_shards(
            output_dir=output_dir,
            split=canonical_split,
            sequences=sequences,
            shard_size=args.shard_size,
        )

    manifest_path = write_cache_manifest(
        output_dir,
        modalities=modalities,
        sequence_length=sequence_length,
        step_duration_ms=step_ms,
        splits=split_shards,
        metadata={
            "source": "synthetic" if args.synthetic else "meld",
            "feature_mode": "cheap_timing_text_quality",
            "sequence_length_s": sequence_length_s,
            "stride_s": stride_s,
            "config": str(Path(args.config)),
            "horizons_ms": horizons_ms,
            "target_schema": (
                "earliest_event_safe_yield_v3"
                if event_hazard_steps
                else "earliest_event_safe_yield_v2"
            ),
            "event_hazard_bins_ms": event_hazard_bins_ms,
            "event_gap_buckets": EVENT_GAP_BUCKETS,
            "causal_features": False,
            "oracle_text": "text" in modalities,
        },
    )

    summary = {
        "cache_dir": str(output_dir),
        "manifest": str(manifest_path),
        "modalities": modalities,
        "sequence_length": sequence_length,
        "step_duration_ms": step_ms,
        "splits": {
            split: {
                "shards": len(shards),
                "sequences": sum(shard.num_sequences for shard in shards),
            }
            for split, shards in split_shards.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_modalities(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def model_input_dims(model_cfg: Mapping[str, Any]) -> dict[str, int]:
    encoders = model_cfg.get("encoders", {})
    return {
        str(name): int(cfg.get("input_dim", 0))
        for name, cfg in encoders.items()
        if isinstance(cfg, Mapping) and int(cfg.get("input_dim", 0)) > 0
    }


def resolve_modalities(
    requested: Sequence[str],
    input_dims: Mapping[str, int],
    include_zero_modalities: bool,
) -> list[str]:
    if include_zero_modalities:
        return list(input_dims)
    missing = [name for name in requested if name not in input_dims]
    if missing:
        raise KeyError(f"Unknown modality/modalities {missing}; available: {sorted(input_dims)}")
    return list(requested)


def label_config_from_project(data_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> TurnLabelConfig:
    dataset_cfg = data_cfg.get("dataset", {}) if isinstance(data_cfg, Mapping) else {}
    turn_cfg = dataset_cfg.get("turn_labels", {}) if isinstance(dataset_cfg, Mapping) else {}
    heads = model_cfg.get("heads", {})
    eot = heads.get("end_of_turn", {}) if isinstance(heads, Mapping) else {}
    return TurnLabelConfig(
        silence_threshold_ms=int(turn_cfg.get("silence_threshold_ms", 300)),
        backchannel_max_duration_ms=int(turn_cfg.get("backchannel_max_duration_ms", 1500)),
        overlap_iou_threshold=float(turn_cfg.get("overlap_iou_threshold", 0.1)),
        step_duration_ms=int(model_cfg.get("step_duration_ms", 100)),
        horizons_ms=tuple(int(value) for value in model_cfg.get("prediction_horizons_ms", [200, 1000])),
        end_of_turn_num_buckets=int(eot.get("num_buckets", 10)),
    )


def load_or_synthesize_split(
    split: str,
    args: argparse.Namespace,
    data_root: str | None,
) -> pd.DataFrame:
    if args.synthetic:
        return make_synthetic_meld_dataframe(
            num_dialogues=args.synthetic_dialogues,
            utterances_per_dialogue=args.utterances_per_dialogue,
            split=split,
            seed=args.seed + split_seed_offset(split),
        )

    if not data_root:
        if args.allow_synthetic:
            return load_or_synthesize_split(split, _synthetic_args(args), data_root=None)
        raise ValueError("--data-root is required unless --synthetic or --allow-synthetic is set")

    try:
        return load_meld_split(data_root, split)
    except FileNotFoundError:
        if args.allow_synthetic:
            return load_or_synthesize_split(split, _synthetic_args(args), data_root=None)
        raise


def build_split_sequences(
    *,
    frame: pd.DataFrame,
    split: str,
    modalities: Sequence[str],
    input_dims: Mapping[str, int],
    label_config: TurnLabelConfig,
    step_ms: int,
    sequence_length: int,
    stride_steps: int,
    horizon_steps: Sequence[int],
    event_hazard_steps: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    normalized = normalize_meld_dataframe(frame, split=split)
    labeled = derive_utterance_labels(normalized, config=label_config)
    sequences: list[dict[str, Any]] = []

    for dialogue_id, dialogue in labeled.groupby("dialogue_id", sort=False):
        dialogue = dialogue.sort_values(["start_s", "end_s", "utterance_id"]).reset_index(drop=True)
        projected = project_dialogue_to_steps(dialogue, step_ms=step_ms, label_config=label_config)
        features = build_step_features(
            projected=projected,
            modalities=modalities,
            input_dims=input_dims,
            seed=seed + stable_int(str(dialogue_id), modulo=100_000),
        )
        targets = build_step_targets(
            projected,
            horizon_steps=horizon_steps,
            event_hazard_steps=event_hazard_steps,
        )
        windows = window_dialogue(
            features=features,
            targets=targets,
            dialogue_id=dialogue_id,
            split=split,
            sequence_length=sequence_length,
            stride_steps=stride_steps,
        )
        sequences.extend(windows)
    return sequences


def project_dialogue_to_steps(
    dialogue: pd.DataFrame,
    *,
    step_ms: int,
    label_config: TurnLabelConfig,
) -> pd.DataFrame:
    start_s = float(dialogue["start_s"].min())
    end_s = float(dialogue["end_s"].max())
    grid = make_time_grid(start_s, end_s + step_ms / 1000.0, step_duration_ms=step_ms)
    rows: list[dict[str, Any]] = []

    for time_s in grid:
        active = dialogue[(dialogue["start_s"] <= time_s) & (time_s < dialogue["end_s"])]
        if active.empty:
            previous = dialogue[dialogue["end_s"] <= time_s].tail(1)
            source = previous.iloc[0] if not previous.empty else None
            is_active = False
        else:
            source = active.iloc[-1]
            is_active = True

        if source is None:
            rows.append(empty_step(time_s))
            continue

        rows.append(
            {
                "time_s": float(time_s),
                "is_active_speech": bool(is_active),
                "utterance": str(source.get("utterance", "")),
                "speaker": str(source.get("speaker", "")),
                "turn_taking": 0,
                "end_of_turn": -1,
                "dialog_act": int(source.get("dialog_act_id", 12)),
                "emotion": int(source.get("emotion_id", 0)),
                "valence": float(source.get("valence", 0.0)),
                "arousal": float(source.get("arousal", 0.0)),
                "time_to_yield_s": np.nan,
                "overlap": int(source.get("turn_label", "") == "overlap"),
                "valid": True,
            }
        )

    result = pd.DataFrame(rows)
    events = []
    ordered = dialogue.sort_values(["start_s", "end_s"]).reset_index(drop=True)
    for position in range(max(0, len(ordered) - 1)):
        current = ordered.iloc[position]
        following = ordered.iloc[position + 1]
        label_id, gap_s = corrected_transition_event(current, following, label_config)
        if label_id != 0:
            events.append((float(following["start_s"]), label_id, gap_bucket(gap_s)))

    times = result["time_s"].to_numpy(dtype=np.float64)
    result["event_gap_bucket"] = -100
    for event_time, label_id, event_gap_bucket in sorted(events):
        event_index = int(np.searchsorted(times, event_time, side="left"))
        if event_index < len(result) and int(result.at[event_index, "turn_taking"]) == 0:
            result.at[event_index, "turn_taking"] = label_id
            result.at[event_index, "event_gap_bucket"] = event_gap_bucket

    shift_times = [event_time for event_time, label_id, _ in events if label_id == 1]
    for index, time_s in enumerate(times):
        future_shifts = [event_time for event_time in shift_times if event_time >= time_s]
        if future_shifts:
            result.at[index, "time_to_yield_s"] = max(0.0, min(future_shifts) - time_s)

    result["end_of_turn"] = result["time_to_yield_s"].map(
        lambda value: -1
        if pd.isna(value)
        else min(
            int(
                float(value)
                / max(label_config.max_end_of_turn_s, 1e-6)
                * label_config.end_of_turn_num_buckets
            ),
            label_config.end_of_turn_num_buckets - 1,
        )
    )
    return result


def gap_bucket(gap_s: float) -> int:
    """Group floor-transfer offsets for event-level diagnostics."""

    if gap_s < 0.0:
        return 0
    if gap_s <= 0.25:
        return 1
    if gap_s <= 0.75:
        return 2
    return 3


def corrected_transition_event(
    current: pd.Series,
    following: pd.Series,
    config: TurnLabelConfig,
) -> tuple[int, float]:
    """Classify a transition using corrected campaign-v3 event semantics."""

    gap_s = float(following["start_s"]) - float(current["end_s"])
    if current["speaker"] == following["speaker"]:
        return 0, gap_s
    if gap_s < 0.0:
        return 3, gap_s
    following_duration_ms = max(
        0.0,
        (float(following["end_s"]) - float(following["start_s"])) * 1000.0,
    )
    if (
        gap_s * 1000.0 <= config.silence_threshold_ms
        and following_duration_ms <= config.backchannel_max_duration_ms
    ):
        return 2, gap_s
    return 1, gap_s


def empty_step(time_s: float) -> dict[str, Any]:
    return {
        "time_s": float(time_s),
        "is_active_speech": False,
        "utterance": "",
        "speaker": "",
        "turn_taking": 0,
        "end_of_turn": 9,
        "dialog_act": 12,
        "emotion": 0,
        "valence": 0.0,
        "arousal": 0.0,
        "time_to_yield_s": np.nan,
        "overlap": 0,
        "valid": True,
    }


def build_step_features(
    *,
    projected: pd.DataFrame,
    modalities: Sequence[str],
    input_dims: Mapping[str, int],
    seed: int,
) -> dict[str, torch.Tensor]:
    length = len(projected)
    total_model_modalities = max(1, len(input_dims))
    present_modalities = {name for name in modalities if name in DEFAULT_CHEAP_MODALITIES}
    features: dict[str, torch.Tensor] = {}

    for modality in modalities:
        dim = input_dims[modality]
        if modality == "vad":
            features[modality] = vad_features(projected, dim)
        elif modality == "text":
            features[modality] = text_features(projected, dim)
        elif modality == "quality":
            features[modality] = quality_features(projected, dim, len(present_modalities), total_model_modalities)
        elif modality == "audio_prosody":
            features[modality] = cheap_prosody_features(projected, dim)
        else:
            features[modality] = torch.zeros(length, dim, dtype=torch.float32)

    if not features:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        features["quality"] = torch.rand(length, input_dims.get("quality", 4), generator=generator)
    return features


def vad_features(projected: pd.DataFrame, dim: int) -> torch.Tensor:
    values = torch.zeros(len(projected), dim, dtype=torch.float32)
    if dim >= 1:
        values[:, 0] = torch.as_tensor(projected["is_active_speech"].to_numpy(dtype=np.float32))
    if dim >= 2:
        values[:, 1] = torch.as_tensor(projected["overlap"].to_numpy(dtype=np.float32))
    if dim >= 3:
        speakers = [stable_int(str(value), modulo=16) / 15.0 if value else 0.0 for value in projected["speaker"]]
        values[:, 2] = torch.as_tensor(speakers, dtype=torch.float32)
    return values


def text_features(projected: pd.DataFrame, dim: int) -> torch.Tensor:
    embeddings = [hashed_text_embedding(str(text), dim) for text in projected["utterance"]]
    return torch.stack(embeddings, dim=0)


def quality_features(
    projected: pd.DataFrame,
    dim: int,
    present_modalities: int,
    total_model_modalities: int,
) -> torch.Tensor:
    values = torch.zeros(len(projected), dim, dtype=torch.float32)
    active = torch.as_tensor(projected["is_active_speech"].to_numpy(dtype=np.float32))
    if dim >= 1:
        values[:, 0] = torch.where(active > 0, torch.full_like(active, 0.8), torch.full_like(active, 0.2))
    if dim >= 2:
        values[:, 1] = 0.0
    if dim >= 3:
        has_text = torch.as_tensor(
            (projected["utterance"].astype(str).str.len().to_numpy() > 0),
            dtype=torch.bool,
        )
        values[:, 2] = torch.where(has_text, torch.full_like(active, 0.6), torch.zeros_like(active))
    if dim >= 4:
        values[:, 3] = float(present_modalities / total_model_modalities)
    return values


def cheap_prosody_features(projected: pd.DataFrame, dim: int) -> torch.Tensor:
    values = torch.zeros(len(projected), dim, dtype=torch.float32)
    active = torch.as_tensor(projected["is_active_speech"].to_numpy(dtype=np.float32))
    if dim >= 1:
        values[:, 0] = active
    if dim >= 2:
        t = torch.as_tensor(projected["time_s"].to_numpy(dtype=np.float32))
        values[:, 1] = (t - t.min()) / (t.max() - t.min()).clamp_min(1e-6)
    if dim >= 3:
        values[:, 2] = torch.as_tensor(projected["valence"].to_numpy(dtype=np.float32))
    if dim >= 4:
        values[:, 3] = torch.as_tensor(projected["arousal"].to_numpy(dtype=np.float32))
    return values


def hashed_text_embedding(text: str, dim: int) -> torch.Tensor:
    vector = torch.zeros(dim, dtype=torch.float32)
    tokens = [token.strip(".,!?;:\"'()[]{}").lower() for token in text.split()]
    tokens = [token for token in tokens if token]
    if not tokens or dim <= 0:
        return vector

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        index = value % dim
        sign = -1.0 if (value >> 8) & 1 else 1.0
        vector[index] += sign
    norm = vector.norm(p=2).clamp_min(1.0)
    return vector / norm


def build_step_targets(
    projected: pd.DataFrame,
    *,
    horizon_steps: Sequence[int],
    event_hazard_steps: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Build earliest-event, safe-yield, and future countdown targets."""

    length = len(projected)
    events = torch.as_tensor(projected["turn_taking"].to_numpy(dtype=np.int64).copy())
    eot = torch.as_tensor(projected["end_of_turn"].to_numpy(dtype=np.int64).copy())
    gap_buckets = torch.as_tensor(
        projected.get("event_gap_bucket", pd.Series([-100] * length)).to_numpy(dtype=np.int64).copy()
    )
    time_to_yield = torch.as_tensor(
        projected.get("time_to_yield_s", pd.Series([-100.0] * length))
        .fillna(-100.0)
        .to_numpy(dtype=np.float32)
        .copy()
    )
    horizon_mask = torch.zeros(length, len(horizon_steps), dtype=torch.bool)
    targets: dict[str, torch.Tensor] = {
        "turn_taking": torch.full(
            (length, len(horizon_steps)), TURN_IGNORE_INDEX, dtype=torch.long
        ),
        "yield": torch.full(
            (length, len(horizon_steps)), TURN_IGNORE_INDEX, dtype=torch.long
        ),
        "end_of_turn": torch.full(
            (length, len(horizon_steps)), TURN_IGNORE_INDEX, dtype=torch.long
        ),
        "event_gap_bucket": torch.full(
            (length, len(horizon_steps)), TURN_IGNORE_INDEX, dtype=torch.long
        ),
        "time_to_yield_s": torch.full(
            (length, len(horizon_steps)), float(TURN_IGNORE_INDEX), dtype=torch.float32
        ),
    }
    if event_hazard_steps:
        targets["event_hazard"] = build_event_hazard_targets(
            events,
            bin_upper_steps=event_hazard_steps,
        )

    for horizon_idx, offset in enumerate(horizon_steps):
        valid = max(0, length - int(offset))
        if valid <= 0:
            continue
        horizon_mask[:valid, horizon_idx] = True
        targets["end_of_turn"][:valid, horizon_idx] = eot[:valid]
        targets["end_of_turn"][:valid, horizon_idx][
            targets["end_of_turn"][:valid, horizon_idx] < 0
        ] = TURN_IGNORE_INDEX
        for step in range(valid):
            future = events[step + 1 : step + offset + 1]
            non_hold = torch.nonzero(future != 0, as_tuple=False)
            event = int(future[non_hold[0, 0]]) if non_hold.numel() else 0
            targets["turn_taking"][step, horizon_idx] = event
            targets["yield"][step, horizon_idx] = int(event == 1)
            if event == 1:
                targets["time_to_yield_s"][step, horizon_idx] = time_to_yield[step]
            if non_hold.numel():
                targets["event_gap_bucket"][step, horizon_idx] = gap_buckets[
                    step + 1 + int(non_hold[0, 0])
                ]

    targets["horizon_mask"] = horizon_mask
    return targets


def build_event_hazard_targets(
    events: torch.Tensor,
    *,
    bin_upper_steps: Sequence[int],
) -> torch.Tensor:
    """Return a discrete competing-risk target for the earliest future event.

    Class zero means no event inside the maximum horizon. Remaining classes
    encode ``time-bin x {shift, backchannel, overlap}``.
    """

    upper = [int(value) for value in bin_upper_steps]
    if not upper or any(value <= 0 for value in upper) or upper != sorted(set(upper)):
        raise ValueError("event hazard bin upper steps must be positive and strictly increasing")
    length = int(events.numel())
    maximum = upper[-1]
    targets = torch.full((length,), TURN_IGNORE_INDEX, dtype=torch.long)
    for step in range(max(0, length - maximum)):
        future = events[step + 1 : step + maximum + 1]
        non_hold = torch.nonzero(future != 0, as_tuple=False)
        if not non_hold.numel():
            targets[step] = 0
            continue
        relative_step = int(non_hold[0, 0]) + 1
        event = int(future[relative_step - 1])
        bin_idx = next(idx for idx, boundary in enumerate(upper) if relative_step <= boundary)
        targets[step] = 1 + bin_idx * 3 + (event - 1)
    return targets


def window_dialogue(
    *,
    features: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    dialogue_id: Any,
    split: str,
    sequence_length: int,
    stride_steps: int,
) -> list[dict[str, Any]]:
    length = next(iter(features.values())).shape[0]
    windows: list[dict[str, Any]] = []
    starts = list(range(0, max(length, 1), stride_steps))
    if starts and starts[-1] + sequence_length < length:
        starts.append(max(0, length - sequence_length))

    for window_idx, start in enumerate(starts):
        end = min(start + sequence_length, length)
        valid_steps = max(0, end - start)
        if valid_steps <= 0:
            continue
        mask = torch.zeros(sequence_length, dtype=torch.bool)
        mask[:valid_steps] = True
        windows.append(
            {
                "features": {
                    name: pad_time(tensor[start:end], sequence_length, value=0.0)
                    for name, tensor in features.items()
                },
                "targets": {
                    name: pad_target(name, tensor[start:end], sequence_length)
                    for name, tensor in targets.items()
                },
                "mask": mask,
                "metadata": {
                    "split": split,
                    "dialogue_id": str(dialogue_id),
                    "window_index": window_idx,
                    "start_step": start,
                    "valid_steps": valid_steps,
                },
            }
        )
    return windows


def pad_time(tensor: torch.Tensor, sequence_length: int, value: float = 0.0) -> torch.Tensor:
    out = torch.full((sequence_length, *tensor.shape[1:]), value, dtype=tensor.dtype)
    out[: tensor.shape[0]] = tensor
    return out


def pad_target(name: str, tensor: torch.Tensor, sequence_length: int) -> torch.Tensor:
    if name == "valence_arousal":
        return pad_time(tensor, sequence_length, value=0.0)
    if name.endswith("_mask"):
        return pad_time(tensor, sequence_length, value=0.0)
    out = torch.full((sequence_length, *tensor.shape[1:]), TURN_IGNORE_INDEX, dtype=tensor.dtype)
    out[: tensor.shape[0]] = tensor
    return out


def write_split_shards(
    *,
    output_dir: Path,
    split: str,
    sequences: Sequence[Mapping[str, Any]],
    shard_size: int,
) -> list[CacheShard]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    shards: list[CacheShard] = []
    shard_size = max(1, int(shard_size))

    for shard_idx, start in enumerate(range(0, len(sequences), shard_size)):
        chunk = list(sequences[start : start + shard_size])
        if not chunk:
            continue
        shard_name = f"{split}_{shard_idx:04d}.pt"
        shard_path = split_dir / shard_name
        save_cache_shard(
            shard_path,
            features=stack_nested([item["features"] for item in chunk]),
            targets=stack_nested([item["targets"] for item in chunk]),
            mask=torch.stack([item["mask"] for item in chunk], dim=0),
            metadata={"items": [item["metadata"] for item in chunk]},
        )
        shards.append(CacheShard(path=str(Path(split) / shard_name), num_sequences=len(chunk)))
    return shards


def stack_nested(items: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = items[0].keys()
    return {key: torch.stack([item[key] for item in items], dim=0) for key in keys}


def stable_int(value: str, modulo: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % modulo


def split_seed_offset(split: str) -> int:
    return {"train": 0, "dev": 10_000, "test": 20_000}.get(split, 30_000)


def _synthetic_args(args: argparse.Namespace) -> argparse.Namespace:
    clone = argparse.Namespace(**vars(args))
    clone.synthetic = True
    return clone


if __name__ == "__main__":
    raise SystemExit(main())
