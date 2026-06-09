"""Evaluate a conventional sustained-silence endpointing baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facepred.data import make_cached_dataloader
from scripts.train_world_model import binary_probability_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--silence-frames", type=int, default=3)
    parser.add_argument("--speech-threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loader = make_cached_dataloader(args.cache_dir, args.split, batch_size=64, shuffle=False)
    probabilities: list[list[torch.Tensor]] = []
    labels: list[list[torch.Tensor]] = []
    for batch in loader:
        speech = batch["features"]["vad"][..., 0]
        silence = speech < args.speech_threshold
        sustained = sustained_silence(silence, args.silence_frames).float()
        targets = batch["targets"]["yield"]
        while len(probabilities) < targets.shape[-1]:
            probabilities.append([])
            labels.append([])
        for horizon_idx in range(targets.shape[-1]):
            valid = batch["mask"] & batch["targets"]["horizon_mask"][..., horizon_idx]
            probabilities[horizon_idx].append(sustained[valid])
            labels[horizon_idx].append(targets[..., horizon_idx][valid])
    metrics = {
        f"h{idx}": binary_probability_metrics(torch.cat(probs), torch.cat(labels[idx]))
        for idx, probs in enumerate(probabilities)
    }
    payload = {
        "baseline": "sustained_silence_endpointing",
        "split": args.split,
        "silence_frames": args.silence_frames,
        "speech_threshold": args.speech_threshold,
        "metrics": metrics,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def sustained_silence(silence: torch.Tensor, frames: int) -> torch.Tensor:
    result = torch.zeros_like(silence)
    run = torch.zeros(silence.shape[0], dtype=torch.long)
    for index in range(silence.shape[1]):
        run = torch.where(silence[:, index], run + 1, torch.zeros_like(run))
        result[:, index] = run >= max(1, frames)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
