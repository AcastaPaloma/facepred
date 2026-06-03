"""Lightweight label derivation for MELD-style conversational turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

TURN_LABELS: tuple[str, ...] = ("hold", "shift", "backchannel", "overlap")
TURN_LABEL_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(TURN_LABELS)}

EMOTIONS: tuple[str, ...] = (
    "neutral",
    "surprise",
    "fear",
    "sadness",
    "joy",
    "disgust",
    "anger",
)
EMOTION_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(EMOTIONS)}

SENTIMENTS: tuple[str, ...] = ("negative", "neutral", "positive")
SENTIMENT_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(SENTIMENTS)}

DIALOG_ACT_LABELS: tuple[str, ...] = (
    "statement",
    "question",
    "backchannel",
    "opinion",
    "answer",
    "agreement",
    "appreciation",
    "apology",
    "directive",
    "commissive",
    "greeting",
    "closing",
    "other",
)
DIALOG_ACT_TO_ID: dict[str, int] = {
    label: idx for idx, label in enumerate(DIALOG_ACT_LABELS)
}

_AFFECT_BY_EMOTION: Mapping[str, tuple[float, float]] = {
    "neutral": (0.0, 0.0),
    "surprise": (0.25, 0.75),
    "fear": (-0.75, 0.8),
    "sadness": (-0.7, -0.35),
    "joy": (0.8, 0.55),
    "disgust": (-0.7, 0.45),
    "anger": (-0.75, 0.8),
}

_BACKCHANNEL_TEXT = {
    "ah",
    "aha",
    "hmm",
    "mhm",
    "mm",
    "mmm",
    "oh",
    "okay",
    "ok",
    "right",
    "sure",
    "uh huh",
    "uh-huh",
    "yeah",
    "yep",
    "yes",
}


@dataclass(frozen=True)
class TurnLabelConfig:
    """Config knobs for deriving iteration-0 conversational targets."""

    silence_threshold_ms: int = 300
    backchannel_max_duration_ms: int = 1500
    overlap_iou_threshold: float = 0.1
    step_duration_ms: int = 100
    horizons_ms: tuple[int, ...] = (200, 1000)
    end_of_turn_num_buckets: int = 10
    max_end_of_turn_s: float = 2.0


def parse_time_seconds(value: object) -> float:
    """Parse MELD-style timestamps, timedeltas, or numeric seconds."""

    if value is None or pd.isna(value):
        return float("nan")
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, np.timedelta64):
        return pd.to_timedelta(value).total_seconds()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        pass

    text = text.replace(",", ".")
    try:
        return pd.to_timedelta(text).total_seconds()
    except ValueError as exc:
        raise ValueError(f"Could not parse timestamp value {value!r}") from exc


def ensure_temporal_columns(
    frame: pd.DataFrame,
    start_col: str | None = None,
    end_col: str | None = None,
) -> pd.DataFrame:
    """Return a copy with numeric ``start_s``, ``end_s``, and ``duration_s``."""

    result = frame.copy()
    start_name = start_col or _find_column(result, ("start_s", "start_time", "starttime"))
    end_name = end_col or _find_column(result, ("end_s", "end_time", "endtime"))
    if start_name is None or end_name is None:
        raise ValueError("Expected start/end columns such as start_s/end_s or StartTime/EndTime")

    result["start_s"] = result[start_name].map(parse_time_seconds).astype(float)
    result["end_s"] = result[end_name].map(parse_time_seconds).astype(float)
    result["duration_s"] = (result["end_s"] - result["start_s"]).clip(lower=0.0)
    return result


def derive_utterance_labels(
    utterances: pd.DataFrame,
    config: TurnLabelConfig | None = None,
    dialogue_col: str = "dialogue_id",
    speaker_col: str = "speaker",
    text_col: str = "utterance",
) -> pd.DataFrame:
    """Derive utterance-level turn, end-of-turn, dialog-act, and affect labels."""

    cfg = config or TurnLabelConfig()
    frame = ensure_temporal_columns(utterances)
    _require_columns(frame, (dialogue_col, speaker_col))

    frame["turn_label"] = "shift"
    frame["turn_label_id"] = TURN_LABEL_TO_ID["shift"]
    frame["gap_to_next_s"] = np.nan
    frame["overlap_iou"] = 0.0
    frame["next_speaker"] = None
    frame["next_start_s"] = np.nan
    frame["time_to_yield_s"] = np.nan

    for _, group in frame.groupby(dialogue_col, sort=False):
        sorted_group = group.sort_values(["start_s", "end_s"])
        indices = list(sorted_group.index)
        for pos, idx in enumerate(indices):
            row = frame.loc[idx]
            next_row = frame.loc[indices[pos + 1]] if pos + 1 < len(indices) else None
            future_other = _next_different_speaker(frame, indices, pos, speaker_col)

            if next_row is not None:
                label, gap_s, overlap_iou = _derive_turn_label(row, next_row, cfg, speaker_col)
                frame.at[idx, "turn_label"] = label
                frame.at[idx, "turn_label_id"] = TURN_LABEL_TO_ID[label]
                frame.at[idx, "gap_to_next_s"] = gap_s
                frame.at[idx, "overlap_iou"] = overlap_iou
                frame.at[idx, "next_speaker"] = next_row[speaker_col]
                frame.at[idx, "next_start_s"] = float(next_row["start_s"])

            if future_other is not None:
                time_to_yield_s = max(0.0, float(future_other["start_s"]) - float(row["end_s"]))
                frame.at[idx, "time_to_yield_s"] = time_to_yield_s

    frame["end_of_turn_bucket_id"] = frame["time_to_yield_s"].map(
        lambda value: bucket_time_to_yield(value, cfg)
    )
    for horizon_ms in cfg.horizons_ms:
        col = f"yield_within_{int(horizon_ms)}ms"
        horizon_s = horizon_ms / 1000.0
        frame[col] = (
            frame["time_to_yield_s"].notna() & (frame["time_to_yield_s"] <= horizon_s)
        ).astype(int)

    frame["dialog_act"] = _dialog_act_series(frame, text_col)
    frame["dialog_act_id"] = frame["dialog_act"].map(DIALOG_ACT_TO_ID).astype(int)
    frame["emotion_id"] = _categorical_id_series(frame, "emotion", EMOTION_TO_ID)
    frame["sentiment_id"] = _categorical_id_series(frame, "sentiment", SENTIMENT_TO_ID)

    affect = np.stack(
        [
            emotion_to_valence_arousal(
                row.get("emotion", "neutral"),
                row.get("sentiment", None),
            )
            for _, row in frame.iterrows()
        ],
        axis=0,
    )
    frame["valence"] = affect[:, 0]
    frame["arousal"] = affect[:, 1]
    return frame


def derive_timestep_labels(
    utterances: pd.DataFrame,
    grid_s: Sequence[float],
    config: TurnLabelConfig | None = None,
    dialogue_col: str = "dialogue_id",
    speaker_col: str = "speaker",
) -> pd.DataFrame:
    """Project utterance labels onto a fixed-rate time grid."""

    cfg = config or TurnLabelConfig()
    labeled = derive_utterance_labels(
        utterances,
        config=cfg,
        dialogue_col=dialogue_col,
        speaker_col=speaker_col,
    )
    grid = np.asarray(grid_s, dtype=float)
    rows: list[dict[str, object]] = []

    for time_s in grid:
        active = labeled[(labeled["start_s"] <= time_s) & (time_s < labeled["end_s"])]
        if active.empty:
            previous = labeled[labeled["end_s"] <= time_s].sort_values("end_s").tail(1)
            source = previous.iloc[0] if not previous.empty else None
            is_active = False
        else:
            source = active.sort_values("end_s").iloc[-1]
            is_active = True

        if source is None:
            row: dict[str, object] = {
                "time_s": float(time_s),
                "is_active_speech": False,
                "utterance_index": -1,
                "speaker": None,
                "turn_label_id": TURN_LABEL_TO_ID["hold"],
                "emotion_id": EMOTION_TO_ID["neutral"],
                "sentiment_id": SENTIMENT_TO_ID["neutral"],
                "dialog_act_id": DIALOG_ACT_TO_ID["other"],
                "valence": 0.0,
                "arousal": 0.0,
                "time_to_yield_s": np.nan,
                "end_of_turn_bucket_id": -1,
            }
        else:
            time_to_yield_s = source.get("time_to_yield_s", np.nan)
            if pd.notna(time_to_yield_s):
                time_to_yield_s = max(0.0, float(source["end_s"]) + float(time_to_yield_s) - time_s)
            row = {
                "time_s": float(time_s),
                "is_active_speech": is_active,
                "utterance_index": int(source.name) if isinstance(source.name, int) else source.name,
                "speaker": source.get(speaker_col),
                "turn_label_id": int(source["turn_label_id"]),
                "emotion_id": int(source["emotion_id"]),
                "sentiment_id": int(source["sentiment_id"]),
                "dialog_act_id": int(source["dialog_act_id"]),
                "valence": float(source["valence"]),
                "arousal": float(source["arousal"]),
                "time_to_yield_s": time_to_yield_s,
                "end_of_turn_bucket_id": bucket_time_to_yield(time_to_yield_s, cfg),
            }

        for horizon_ms in cfg.horizons_ms:
            value = row["time_to_yield_s"]
            row[f"yield_within_{int(horizon_ms)}ms"] = int(
                pd.notna(value) and float(value) <= horizon_ms / 1000.0
            )
        rows.append(row)

    return pd.DataFrame(rows)


def bucket_time_to_yield(value: object, config: TurnLabelConfig | None = None) -> int:
    """Discretize seconds-until-yield into ``end_of_turn_num_buckets`` bins."""

    cfg = config or TurnLabelConfig()
    if value is None or pd.isna(value):
        return -1
    seconds = max(0.0, float(value))
    clipped = min(seconds, cfg.max_end_of_turn_s)
    ratio = clipped / max(cfg.max_end_of_turn_s, 1e-6)
    bucket = int(np.floor(ratio * cfg.end_of_turn_num_buckets))
    return min(bucket, cfg.end_of_turn_num_buckets - 1)


def emotion_to_valence_arousal(
    emotion: object,
    sentiment: object | None = None,
) -> np.ndarray:
    """Map categorical MELD affect to a small continuous valence/arousal target."""

    label = str(emotion).strip().lower() if emotion is not None and not pd.isna(emotion) else "neutral"
    valence, arousal = _AFFECT_BY_EMOTION.get(label, _AFFECT_BY_EMOTION["neutral"])

    if sentiment is not None and not pd.isna(sentiment):
        sentiment_label = str(sentiment).strip().lower()
        if sentiment_label == "positive":
            valence = max(valence, 0.35)
        elif sentiment_label == "negative":
            valence = min(valence, -0.35)

    return np.asarray([valence, arousal], dtype=np.float32)


def infer_dialog_act(text: object) -> str:
    """Infer a coarse dialog-act label from text-only cues."""

    if text is None or pd.isna(text):
        return "other"

    raw = str(text).strip()
    lowered = " ".join(raw.lower().replace("!", "").replace(".", "").split())
    if not lowered:
        return "other"
    if lowered in _BACKCHANNEL_TEXT:
        return "backchannel"
    if lowered in {"hi", "hello", "hey", "good morning", "good evening"}:
        return "greeting"
    if lowered in {"bye", "goodbye", "see you", "see ya"}:
        return "closing"
    if "thank" in lowered or lowered in {"thanks", "thank you"}:
        return "appreciation"
    if "sorry" in lowered or "apolog" in lowered:
        return "apology"
    if raw.endswith("?") or lowered.split()[0] in {"who", "what", "when", "where", "why", "how"}:
        return "question"
    if lowered.startswith(("please ", "let's ", "lets ", "could you", "can you")):
        return "directive"
    if lowered.startswith(("i think", "i feel", "i guess", "maybe", "probably")):
        return "opinion"
    if lowered.startswith(("yes", "yeah", "no", "nope", "right", "exactly")):
        return "agreement"
    if lowered.startswith(("i will", "i'll", "we will", "we'll")):
        return "commissive"
    return "statement"


def _derive_turn_label(
    row: pd.Series,
    next_row: pd.Series,
    config: TurnLabelConfig,
    speaker_col: str,
) -> tuple[str, float, float]:
    gap_s = float(next_row["start_s"]) - float(row["end_s"])
    overlap_iou = _interval_iou(
        float(row["start_s"]),
        float(row["end_s"]),
        float(next_row["start_s"]),
        float(next_row["end_s"]),
    )
    same_speaker = row[speaker_col] == next_row[speaker_col]
    if overlap_iou >= config.overlap_iou_threshold:
        return "overlap", gap_s, overlap_iou
    if same_speaker:
        return "hold", gap_s, overlap_iou

    next_duration_ms = max(0.0, float(next_row["duration_s"]) * 1000.0)
    if gap_s * 1000.0 <= config.silence_threshold_ms:
        if next_duration_ms <= config.backchannel_max_duration_ms:
            return "backchannel", gap_s, overlap_iou
        return "shift", gap_s, overlap_iou
    return "shift", gap_s, overlap_iou


def _next_different_speaker(
    frame: pd.DataFrame,
    indices: Sequence[object],
    pos: int,
    speaker_col: str,
) -> pd.Series | None:
    speaker = frame.loc[indices[pos], speaker_col]
    for idx in indices[pos + 1 :]:
        candidate = frame.loc[idx]
        if candidate[speaker_col] != speaker:
            return candidate
    return None


def _interval_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0.0:
        return 0.0
    return intersection / union


def _categorical_id_series(
    frame: pd.DataFrame,
    column: str,
    mapping: Mapping[str, int],
) -> pd.Series:
    if column not in frame:
        default = mapping.get("neutral", 0)
        return pd.Series(default, index=frame.index, dtype=int)
    return (
        frame[column]
        .fillna("neutral")
        .map(lambda value: mapping.get(str(value).strip().lower(), mapping.get("neutral", 0)))
        .astype(int)
    )


def _dialog_act_series(frame: pd.DataFrame, text_col: str) -> pd.Series:
    if text_col not in frame:
        return pd.Series("other", index=frame.index)
    return frame[text_col].map(infer_dialog_act)


def _find_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized = {
        "".join(ch for ch in column.lower() if ch.isalnum() or ch == "_"): column
        for column in frame.columns
    }
    for candidate in candidates:
        key = "".join(ch for ch in candidate.lower() if ch.isalnum() or ch == "_")
        if key in normalized:
            return normalized[key]
    return None


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
