"""Inspect cached feature signal and target imbalance before expensive training."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facepred.data import CacheManifest, make_cached_dataloader

TARGET_NUM_CLASSES = {
    "turn_taking": 4,
    "yield": 2,
    "end_of_turn": 10,
    "event_gap_bucket": 4,
    "dialog_act": 13,
    "emotion": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = CacheManifest.load(args.cache_dir)
    loader = make_cached_dataloader(
        args.cache_dir,
        args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    payload = inspect_loader(loader, max_batches=args.max_batches)
    payload["cache_dir"] = str(args.cache_dir)
    payload["split"] = args.split
    payload["manifest_metadata"] = manifest.metadata
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def inspect_loader(loader: Any, *, max_batches: int | None = None) -> dict[str, Any]:
    features: dict[str, RunningMoments] = {}
    targets: dict[str, list[torch.Tensor]] = {}
    valid_steps = 0
    batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batches += 1
        mask = batch["mask"].bool()
        valid_steps += int(mask.sum())
        for name, values in batch["features"].items():
            tracker = features.setdefault(name, RunningMoments())
            tracker.update(values[mask].float())
        for name, values in batch["targets"].items():
            if name in {"horizon_mask", "time_to_yield_s"}:
                continue
            flattened = values.reshape(-1, values.shape[-1]) if values.ndim >= 3 else values.reshape(-1, 1)
            per_horizon = targets.setdefault(name, [])
            while len(per_horizon) < flattened.shape[-1]:
                per_horizon.append(
                    torch.zeros(TARGET_NUM_CLASSES.get(name, 0), dtype=torch.long)
                )
            for horizon_idx in range(flattened.shape[-1]):
                labels = flattened[:, horizon_idx].long()
                valid = labels[labels != -100].cpu()
                update = torch.bincount(
                    valid,
                    minlength=max(TARGET_NUM_CLASSES.get(name, 0), per_horizon[horizon_idx].numel()),
                )
                if update.numel() > per_horizon[horizon_idx].numel():
                    per_horizon[horizon_idx] = torch.nn.functional.pad(
                        per_horizon[horizon_idx],
                        (0, update.numel() - per_horizon[horizon_idx].numel()),
                    )
                per_horizon[horizon_idx] += update

    return {
        "batches": batches,
        "valid_steps": valid_steps,
        "features": {name: tracker.report() for name, tracker in features.items()},
        "targets": {
            name: [
                target_report(counts)
                for counts in horizons
            ]
            for name, horizons in targets.items()
        },
    }


class RunningMoments:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_square = 0.0
        self.nonzero = 0
        self.minimum = float("inf")
        self.maximum = -float("inf")

    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().cpu().double().reshape(-1)
        if flat.numel() == 0:
            return
        self.count += flat.numel()
        self.sum += float(flat.sum())
        self.sum_square += float(flat.square().sum())
        self.nonzero += int((flat != 0).sum())
        self.minimum = min(self.minimum, float(flat.min()))
        self.maximum = max(self.maximum, float(flat.max()))

    def report(self) -> dict[str, float | int]:
        mean = self.sum / max(1, self.count)
        variance = max(0.0, self.sum_square / max(1, self.count) - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": variance**0.5,
            "nonzero_fraction": self.nonzero / max(1, self.count),
            "min": self.minimum if self.count else 0.0,
            "max": self.maximum if self.count else 0.0,
        }


def target_report(counts: torch.Tensor) -> Mapping[str, Any]:
    if counts.numel() == 0 or counts.sum() == 0:
        return {"count": 0, "class_counts": [], "majority_accuracy": 0.0, "majority_macro_f1": 0.0}
    total = int(counts.sum())
    majority = int(counts.max())
    majority_f1 = 2 * majority / max(1, total + majority)
    return {
        "count": total,
        "class_counts": counts.tolist(),
        "majority_accuracy": majority / max(1, total),
        "majority_macro_f1": majority_f1 / max(1, counts.numel()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
