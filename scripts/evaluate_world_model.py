"""Evaluate a trained FacePred world-model checkpoint on cached shards."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facepred.data import make_cached_dataloader
from facepred.engine.trainer import load_project_config
from facepred.models import FacePredLoss, FacePredWorldModel
from facepred.utils import load_trusted_torch_artifact
from scripts.train_world_model import MetricAccumulator, WorldMetricAccumulator, move_training_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="Project config YAML fallback.")
    parser.add_argument("--cache-dir", required=True, help="Prepared cache directory.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path, e.g. run/checkpoints/best.pt.")
    parser.add_argument("--split", default="test", help="Cache split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--max-batches", type=int, default=None, help="Debug limit.")
    parser.add_argument(
        "--calibration",
        default=None,
        help="Optional dev-fitted yield calibration artifact.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON metrics output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_trusted_torch_artifact(args.checkpoint, map_location=device)
    config = checkpoint.get("config") or load_project_config(args.config)

    model = FacePredWorldModel.from_config(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn = FacePredLoss(config.get("training", {}).get("loss_weights", {}))
    loader = make_cached_dataloader(
        args.cache_dir,
        args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    calibration = (
        json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        if args.calibration
        else None
    )

    metrics, report = evaluate(
        model=model,
        loader=loader,
        loss_fn=loss_fn,
        device=device,
        max_batches=args.max_batches,
        calibration=calibration,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "num_batches": min(len(loader), args.max_batches) if args.max_batches else len(loader),
        "metrics": metrics,
        "yield_calibration": args.calibration,
        "confusion_matrices": report["confusion_matrices"],
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


@torch.no_grad()
def evaluate(
    *,
    model: FacePredWorldModel,
    loader: Iterable[Mapping[str, Any]],
    loss_fn: FacePredLoss,
    device: torch.device,
    max_batches: int | None,
    calibration: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    model.eval()
    accumulator = MetricAccumulator()
    world_accumulator = WorldMetricAccumulator(
        calibration.get("thresholds", ()) if calibration else ()
    )
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        features, targets = move_training_batch(batch, device)
        outputs = model(features)
        if calibration:
            outputs = apply_yield_calibration(outputs, calibration)
        loss_output = loss_fn(outputs, targets)
        accumulator.update(loss_output.metrics())
        world_accumulator.update(outputs, targets)
    report = world_accumulator.report()
    metrics = accumulator.mean()
    metrics.update(report["metrics"])
    return metrics, report


def apply_yield_calibration(
    outputs: Mapping[str, torch.Tensor],
    calibration: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Apply dev-fitted per-horizon temperatures before operating-point metrics."""

    result = dict(outputs)
    logits = result.get("yield_logits")
    if logits is None:
        return result
    temperatures = tuple(float(value) for value in calibration.get("temperatures", ()))
    if len(temperatures) != logits.shape[-1]:
        raise ValueError(
            f"Calibration has {len(temperatures)} temperatures for {logits.shape[-1]} horizons"
        )
    temperature_tensor = torch.tensor(
        temperatures,
        dtype=logits.dtype,
        device=logits.device,
    ).clamp_min(1.0e-6)
    result["yield_probs"] = torch.sigmoid(logits / temperature_tensor)
    result["yield_entropy"] = -(
        result["yield_probs"] * result["yield_probs"].clamp_min(1.0e-8).log()
        + (1.0 - result["yield_probs"])
        * (1.0 - result["yield_probs"]).clamp_min(1.0e-8).log()
    )
    return result


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


if __name__ == "__main__":
    raise SystemExit(main())
