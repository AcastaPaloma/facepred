"""Build a resumable, causal real-audio MELD cache for Colab training.

The first serious FacePred baseline deliberately excludes transcript and visual
features. It decodes each MELD utterance MP4 with ffmpeg, derives causal
100-millisecond audio statistics and energy VAD, then writes dialogue progress
files before producing the normal sharded training cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facepred.data import (
    CacheShard,
    canonical_meld_split,
    derive_utterance_labels,
    load_meld_split,
    normalize_meld_dataframe,
    write_cache_manifest,
)
from facepred.engine.trainer import load_project_config
from scripts.prepare_meld_cache import (
    build_step_targets,
    label_config_from_project,
    project_dialogue_to_steps,
    window_dialogue,
    write_split_shards,
)

AUDIO_FEATURE_DIM = 25
SAMPLE_RATE = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-config", default="configs/model/rssm_xs_audio.yaml")
    parser.add_argument("--data-root", required=True, help="Extracted MELD.Raw root.")
    parser.add_argument("--output-dir", required=True, help="Persistent cache directory, normally Drive.")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--sequence-length-s", type=float, default=5.0)
    parser.add_argument("--train-stride-s", type=float, default=2.5)
    parser.add_argument("--eval-stride-s", type=float, default=5.0)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--max-dialogues", type=int, default=None, help="Smoke-test limit per split.")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--allow-config-mismatch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_config = load_project_config(args.config)
    model_config = load_yaml(args.model_config)
    validate_model_config(model_config)
    data_cfg = project_config.get("data", {})
    step_ms = int(model_config.get("step_duration_ms", 100))
    horizons_ms = [int(value) for value in model_config.get("prediction_horizons_ms", [200, 1000])]
    horizon_steps = [max(1, int(round(value / step_ms))) for value in horizons_ms]
    sequence_length = max(1, int(round(args.sequence_length_s * 1000 / step_ms)))
    train_stride_steps = max(1, int(round(args.train_stride_s * 1000 / step_ms)))
    eval_stride_steps = max(1, int(round(args.eval_stride_s * 1000 / step_ms)))
    label_config = label_config_from_project(data_cfg, model_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extraction_config = {
        "feature_mode": "causal_audio_stats_v1",
        "feature_alignment": "previous_frame_right_edge",
        "model_config": model_config,
        "step_ms": step_ms,
        "horizons_ms": horizons_ms,
        "sequence_length": sequence_length,
        "train_stride_steps": train_stride_steps,
        "eval_stride_steps": eval_stride_steps,
    }
    validate_or_write_extraction_config(
        output_dir,
        extraction_config,
        allow_mismatch=args.allow_config_mismatch,
    )

    split_shards: dict[str, list[CacheShard]] = {}
    split_stats: dict[str, dict[str, int]] = {}
    for requested_split in args.splits:
        split = canonical_meld_split(requested_split)
        stride_steps = train_stride_steps if split == "train" else eval_stride_steps
        frame = load_meld_split(args.data_root, split)
        normalized = normalize_meld_dataframe(frame, split=split)
        labeled = derive_utterance_labels(normalized, config=label_config)
        progress_dir = output_dir / ".progress" / split
        progress_dir.mkdir(parents=True, exist_ok=True)

        stats = {"dialogues": 0, "utterances": 0, "decoded": 0, "missing_media": 0, "decode_errors": 0}
        grouped = list(labeled.groupby("dialogue_id", sort=False))
        if args.max_dialogues is not None:
            grouped = grouped[: args.max_dialogues]
        for dialogue_index, (dialogue_id, dialogue) in enumerate(grouped, start=1):
            progress_path = progress_dir / f"{safe_id(dialogue_id)}.pt"
            if progress_path.exists():
                payload = torch.load(progress_path, map_location="cpu")
            else:
                payload = process_dialogue(
                    dialogue=dialogue,
                    dialogue_id=dialogue_id,
                    split=split,
                    step_ms=step_ms,
                    sequence_length=sequence_length,
                    stride_steps=stride_steps,
                    horizon_steps=horizon_steps,
                    label_config=label_config,
                )
                atomic_torch_save(payload, progress_path)
            merge_counts(stats, payload.get("stats", {}))
            if dialogue_index % max(1, args.log_every) == 0 or dialogue_index == len(grouped):
                print(
                    json.dumps(
                        {
                            "split": split,
                            "dialogues_complete": dialogue_index,
                            "dialogues_total": len(grouped),
                            "decoded": stats["decoded"],
                            "missing_media": stats["missing_media"],
                            "decode_errors": stats["decode_errors"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        if stats["decoded"] == 0:
            raise RuntimeError(
                f"No audio was decoded for {split}. Check ffmpeg and the extracted MELD media layout."
            )

        sequences: list[Mapping[str, Any]] = []
        for progress_path in sorted(progress_dir.glob("*.pt")):
            sequences.extend(torch.load(progress_path, map_location="cpu")["sequences"])
        split_shards[split] = write_split_shards(
            output_dir=output_dir,
            split=split,
            sequences=sequences,
            shard_size=args.shard_size,
        )
        split_stats[split] = stats

    manifest_path = write_cache_manifest(
        output_dir,
        modalities=["audio_prosody", "vad", "quality"],
        sequence_length=sequence_length,
        step_duration_ms=step_ms,
        splits=split_shards,
        metadata={
            "source": "meld_raw_media",
            "feature_mode": "causal_audio_stats_v1",
            "feature_alignment": "previous_frame_right_edge",
            "causal_features": True,
            "oracle_text": False,
            "horizons_ms": horizons_ms,
            "model_config": str(args.model_config),
            "stats": split_stats,
        },
    )
    print(
        json.dumps(
            {
                "cache_dir": str(output_dir),
                "manifest": str(manifest_path),
                "splits": split_stats,
                "sequences": {
                    split: sum(shard.num_sequences for shard in shards)
                    for split, shards in split_shards.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def process_dialogue(
    *,
    dialogue: Any,
    dialogue_id: Any,
    split: str,
    step_ms: int,
    sequence_length: int,
    stride_steps: int,
    horizon_steps: Sequence[int],
    label_config: Any,
) -> dict[str, Any]:
    dialogue = dialogue.sort_values(["start_s", "end_s", "utterance_id"]).reset_index(drop=True)
    projected = project_dialogue_to_steps(dialogue, step_ms=step_ms, label_config=label_config)
    length = len(projected)
    audio = torch.zeros(length, AUDIO_FEATURE_DIM, dtype=torch.float32)
    vad = torch.zeros(length, 3, dtype=torch.float32)
    quality = torch.zeros(length, 4, dtype=torch.float32)
    audio_counts = torch.zeros(length, 1, dtype=torch.float32)
    times = projected["time_s"].to_numpy(dtype=np.float32)
    stats = {"dialogues": 1, "utterances": len(dialogue), "decoded": 0, "missing_media": 0, "decode_errors": 0}

    for row in dialogue.itertuples(index=False):
        indices = np.flatnonzero((times >= float(row.start_s)) & (times < float(row.end_s)))
        if len(indices) == 0:
            continue
        media_path = getattr(row, "media_path", None)
        if not media_path or not Path(str(media_path)).exists():
            stats["missing_media"] += 1
            continue
        try:
            waveform = decode_audio(media_path)
            local_audio, local_vad, local_quality = extract_causal_audio_features(
                waveform,
                num_frames=len(indices),
            )
        except Exception:
            stats["decode_errors"] += 1
            continue
        stats["decoded"] += 1
        # Frame k summarizes audio ending at the next 100 ms grid point. Assign
        # it to that right edge so no feature at time t contains audio after t.
        target_indices = indices[1:]
        if len(target_indices) == 0:
            continue
        index_tensor = torch.as_tensor(target_indices, dtype=torch.long)
        audio[index_tensor] += local_audio[: len(target_indices)]
        audio_counts[index_tensor] += 1.0
        vad[index_tensor] = torch.maximum(vad[index_tensor], local_vad[: len(target_indices)])
        quality[index_tensor] = torch.maximum(quality[index_tensor], local_quality[: len(target_indices)])

    present = audio_counts.squeeze(-1) > 0
    audio[present] /= audio_counts[present]
    features = {"audio_prosody": audio, "vad": vad, "quality": quality}
    targets = build_step_targets(projected, horizon_steps=horizon_steps)
    sequences = window_dialogue(
        features=features,
        targets=targets,
        dialogue_id=dialogue_id,
        split=split,
        sequence_length=sequence_length,
        stride_steps=stride_steps,
    )
    return {"sequences": sequences, "stats": stats}


def decode_audio(media_path: str | Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(media_path),
        "-vn",
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
    if waveform.size == 0:
        raise ValueError(f"No audio decoded from {media_path}")
    return torch.from_numpy(waveform)


def extract_causal_audio_features(
    waveform: torch.Tensor,
    *,
    num_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = split_waveform(waveform.float(), num_frames)
    rows = []
    previous = None
    for frame in frames:
        row = frame_statistics(frame, previous)
        rows.append(row)
        previous = row
    audio = torch.stack(rows)
    rms = audio[:, 0].clamp_min(0.0)
    low = torch.quantile(rms, 0.2)
    high = torch.quantile(rms, 0.95)
    speech = ((rms - low) / (high - low).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    vad = torch.stack([speech, torch.zeros_like(speech), speech], dim=-1)
    quality = torch.zeros(num_frames, 4, dtype=torch.float32)
    quality[:, 0] = speech
    quality[:, 3] = 1.0
    return audio, vad, quality


def split_waveform(waveform: torch.Tensor, num_frames: int) -> list[torch.Tensor]:
    waveform = waveform.flatten()
    boundaries = torch.linspace(0, waveform.numel(), steps=num_frames + 1).round().long()
    frames = []
    for idx in range(num_frames):
        frame = waveform[boundaries[idx] : boundaries[idx + 1]]
        frames.append(frame if frame.numel() else torch.zeros(1))
    return frames


def frame_statistics(frame: torch.Tensor, previous: torch.Tensor | None) -> torch.Tensor:
    frame = frame.float()
    eps = 1.0e-8
    abs_frame = frame.abs()
    rms = torch.sqrt(frame.square().mean() + eps)
    mean = frame.mean()
    std = frame.std(unbiased=False)
    peak = abs_frame.max()
    zcr = (frame[1:] * frame[:-1] < 0).float().mean() if frame.numel() > 1 else torch.tensor(0.0)
    quantiles = torch.quantile(frame, torch.tensor([0.25, 0.5, 0.75]))
    crest = (peak / rms.clamp_min(eps)).clamp(max=10.0) / 10.0

    spectrum = torch.fft.rfft(frame * torch.hann_window(frame.numel()))
    magnitude = spectrum.abs().clamp_min(eps)
    power = magnitude.square()
    distribution = power / power.sum().clamp_min(eps)
    frequencies = torch.linspace(0.0, 1.0, distribution.numel())
    centroid = (distribution * frequencies).sum()
    bandwidth = torch.sqrt((distribution * (frequencies - centroid).square()).sum())
    cumulative = distribution.cumsum(0)
    rolloff = frequencies[torch.searchsorted(cumulative, torch.tensor(0.85)).clamp_max(len(frequencies) - 1)]
    flatness = torch.exp(magnitude.log().mean()) / magnitude.mean().clamp_min(eps)
    entropy = -(distribution * distribution.log()).sum() / np.log(max(2, distribution.numel()))
    thirds = torch.tensor_split(distribution, 3)
    bands = [part.sum() for part in thirds]

    base = torch.stack(
        [
            rms,
            mean,
            std,
            abs_frame.mean(),
            frame.max(),
            frame.min(),
            peak,
            zcr,
            quantiles[0],
            quantiles[1],
            quantiles[2],
            crest,
            centroid,
            bandwidth,
            rolloff,
            flatness,
            entropy,
            bands[0],
            bands[1],
            bands[2],
        ]
    )
    if previous is None:
        deltas = torch.zeros(4)
    else:
        deltas = torch.stack(
            [
                (rms - previous[0]).clamp(-1.0, 1.0),
                (zcr - previous[7]).clamp(-1.0, 1.0),
                (centroid - previous[12]).clamp(-1.0, 1.0),
                (std - previous[2]).clamp(-1.0, 1.0),
            ]
        )
    return torch.cat([base, deltas, torch.ones(1)]).float()


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def validate_model_config(model_config: Mapping[str, Any]) -> None:
    dims = {
        name: int(value.get("input_dim", 0))
        for name, value in model_config.get("encoders", {}).items()
    }
    expected = {"audio_prosody": AUDIO_FEATURE_DIM, "vad": 3, "quality": 4}
    if dims != expected:
        raise ValueError(f"Audio cache requires encoder dimensions {expected}, got {dims}")


def validate_or_write_extraction_config(
    output_dir: Path,
    config: Mapping[str, Any],
    *,
    allow_mismatch: bool,
) -> None:
    path = output_dir / "extraction_config.json"
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint and not allow_mismatch:
            raise ValueError("Existing cache progress uses a different extraction config")
    path.write_text(
        json.dumps({"fingerprint": fingerprint, "config": config}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    tmp.replace(path)


def safe_id(value: Any) -> str:
    text = str(value)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=5).hexdigest()
    return f"{''.join(ch if ch.isalnum() else '_' for ch in text)[:40]}_{digest}"


def merge_counts(total: dict[str, int], values: Mapping[str, Any]) -> None:
    for key in total:
        total[key] += int(values.get(key, 0))


if __name__ == "__main__":
    raise SystemExit(main())
