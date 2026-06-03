"""Derive lightweight turn-taking labels from utterance timing metadata."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facepred.engine.trainer import load_project_config

TURN_LABELS = {"hold": 0, "shift": 1, "backchannel": 2, "overlap": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="Project config YAML.")
    parser.add_argument("--input", default=None, help="CSV or JSON utterance metadata.")
    parser.add_argument("--output", default=None, help="Optional JSONL label output.")
    parser.add_argument("--synthetic-count", type=int, default=6, help="Rows to synthesize without input.")
    return parser.parse_args()


def turn_label_config(config: Mapping[str, Any]) -> dict[str, float]:
    data_cfg = config.get("data", config)
    dataset_cfg = data_cfg.get("dataset", {}) if isinstance(data_cfg, Mapping) else {}
    labels_cfg = dataset_cfg.get("turn_labels", {}) if isinstance(dataset_cfg, Mapping) else {}
    return {
        "silence_threshold_ms": float(labels_cfg.get("silence_threshold_ms", 300) or 300),
        "backchannel_max_duration_ms": float(
            labels_cfg.get("backchannel_max_duration_ms", 1500) or 1500
        ),
        "overlap_iou_threshold": float(labels_cfg.get("overlap_iou_threshold", 0.1) or 0.1),
    }


def load_utterances(path: str | Path | None, synthetic_count: int) -> list[dict[str, Any]]:
    if path is None:
        return synthesize_utterances(synthetic_count)

    resolved = Path(path)
    if resolved.suffix.lower() == ".json":
        with resolved.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of utterance objects")
        return [dict(item) for item in data]

    with resolved.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def synthesize_utterances(count: int) -> list[dict[str, Any]]:
    rows = []
    cursor = 0.0
    for index in range(count):
        duration = 800.0 + (index % 3) * 250.0
        gap = 120.0 if index % 2 == 0 else 420.0
        rows.append(
            {
                "dialogue_id": "synthetic",
                "utterance_id": str(index),
                "speaker": f"speaker_{index % 2}",
                "start_ms": cursor,
                "end_ms": cursor + duration,
                "emotion": "neutral",
                "text": f"synthetic utterance {index}",
            }
        )
        cursor += duration + gap
    return rows


def derive_labels(
    utterances: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    label_cfg = turn_label_config(config)
    emotions = _emotion_names(config)
    by_dialogue: dict[str, list[dict[str, Any]]] = {}
    for row in utterances:
        by_dialogue.setdefault(str(row.get("dialogue_id", "default")), []).append(row)

    derived: list[dict[str, Any]] = []
    for dialogue_id, rows in by_dialogue.items():
        rows = sorted(rows, key=lambda row: float(row.get("start_ms", 0.0) or 0.0))
        for index, row in enumerate(rows):
            next_row = rows[index + 1] if index + 1 < len(rows) else None
            label_name, time_to_next = classify_turn(row, next_row, label_cfg)
            emotion = str(row.get("emotion", "neutral"))
            derived.append(
                {
                    "dialogue_id": dialogue_id,
                    "utterance_id": row.get("utterance_id", index),
                    "speaker": row.get("speaker"),
                    "turn_label": label_name,
                    "turn_label_id": TURN_LABELS[label_name],
                    "time_to_next_ms": time_to_next,
                    "emotion": emotion,
                    "emotion_id": emotions.index(emotion) if emotion in emotions else 0,
                    "dialog_act_id": 0,
                }
            )
    return derived


def classify_turn(
    row: Mapping[str, Any],
    next_row: Mapping[str, Any] | None,
    label_cfg: Mapping[str, float],
) -> tuple[str, float | None]:
    if next_row is None:
        return "shift", None

    start = float(row.get("start_ms", 0.0) or 0.0)
    end = float(row.get("end_ms", start) or start)
    next_start = float(next_row.get("start_ms", end) or end)
    next_end = float(next_row.get("end_ms", next_start) or next_start)
    next_duration = max(0.0, next_end - next_start)
    gap = next_start - end

    if gap < 0:
        overlap = min(end, next_end) - max(start, next_start)
        union = max(end, next_end) - min(start, next_start)
        overlap_iou = overlap / union if union > 0 else 0.0
        if overlap_iou >= label_cfg["overlap_iou_threshold"]:
            return "overlap", gap

    if next_duration <= label_cfg["backchannel_max_duration_ms"] and gap <= label_cfg[
        "silence_threshold_ms"
    ]:
        return "backchannel", gap
    if gap <= label_cfg["silence_threshold_ms"]:
        return "hold", gap
    return "shift", gap


def _emotion_names(config: Mapping[str, Any]) -> list[str]:
    data_cfg = config.get("data", config)
    dataset_cfg = data_cfg.get("dataset", {}) if isinstance(data_cfg, Mapping) else {}
    emotions = dataset_cfg.get("emotions", []) if isinstance(dataset_cfg, Mapping) else []
    return list(emotions) if emotions else ["neutral"]


def main() -> int:
    args = parse_args()
    config = load_project_config(args.config)
    utterances = load_utterances(args.input, args.synthetic_count)
    labels = derive_labels(utterances, config)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in labels:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(json.dumps({"rows": len(labels), "output": args.output}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

