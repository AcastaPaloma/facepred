"""Fit per-horizon safe-yield temperatures and conservative commit thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facepred.data import make_cached_dataloader
from facepred.models import FacePredWorldModel
from facepred.utils import load_trusted_torch_artifact
from scripts.train_world_model import binary_probability_metrics, move_training_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--minimum-precision", type=float, default=0.9)
    parser.add_argument(
        "--minimum-commits",
        type=int,
        default=25,
        help="Minimum dev commits required for a deployable threshold.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    checkpoint = load_trusted_torch_artifact(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = FacePredWorldModel.from_config(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = make_cached_dataloader(
        args.cache_dir,
        args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    logits, labels = collect_yield_outputs(model, loader, device)
    temperatures = []
    thresholds = []
    metrics = []
    policy_results = []
    for horizon_idx, horizon_logits in enumerate(logits):
        horizon_labels = labels[horizon_idx]
        temperature = fit_temperature(horizon_logits, horizon_labels)
        probabilities = torch.sigmoid(horizon_logits / temperature)
        policy = select_threshold_for_minimum_precision(
            probabilities,
            horizon_labels,
            args.minimum_precision,
            minimum_commits=args.minimum_commits,
        )
        threshold = float(policy["threshold"])
        temperatures.append(temperature)
        thresholds.append(threshold)
        metrics.append(binary_probability_metrics(probabilities, horizon_labels, threshold=threshold))
        policy_results.append(policy)

    artifact = {
        "schema_version": 2,
        "type": "safe_yield_temperature_thresholds",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "horizons_ms": list(config["model"].get("prediction_horizons_ms", [])),
        "temperatures": temperatures,
        "thresholds": thresholds,
        "minimum_precision": args.minimum_precision,
        "minimum_commits": args.minimum_commits,
        "selection_policy": "maximize_recall_at_minimum_precision_or_abstain",
        "policy_results": policy_results,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


@torch.no_grad()
def collect_yield_outputs(
    model: FacePredWorldModel,
    loader: object,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    model.eval()
    logits_by_horizon: list[list[torch.Tensor]] = []
    labels_by_horizon: list[list[torch.Tensor]] = []
    for batch in loader:
        features, targets = move_training_batch(batch, device)
        outputs = model(features)
        logits = outputs["yield_logits"]
        while len(logits_by_horizon) < logits.shape[-1]:
            logits_by_horizon.append([])
            labels_by_horizon.append([])
        for horizon_idx in range(logits.shape[-1]):
            labels = targets["yield"][..., horizon_idx]
            valid = targets["mask"] & targets["horizon_mask"][..., horizon_idx] & (labels != -100)
            logits_by_horizon[horizon_idx].append(logits[..., horizon_idx][valid].cpu())
            labels_by_horizon[horizon_idx].append(labels[valid].cpu())
    return (
        [torch.cat(values) for values in logits_by_horizon],
        [torch.cat(values) for values in labels_by_horizon],
    )


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    candidates = torch.logspace(-1.0, 1.0, steps=81)
    losses = [
        float(F.binary_cross_entropy_with_logits(logits / temperature, labels.float()))
        for temperature in candidates
    ]
    return float(candidates[int(torch.tensor(losses).argmin())])


def threshold_for_minimum_precision(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    minimum_precision: float,
) -> float:
    """Compatibility wrapper returning only the selected threshold."""

    return float(
        select_threshold_for_minimum_precision(
            probabilities,
            labels,
            minimum_precision,
            minimum_commits=1,
        )["threshold"]
    )


def select_threshold_for_minimum_precision(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    minimum_precision: float,
    *,
    minimum_commits: int,
) -> dict[str, float | int | bool | str]:
    """Maximize recall under a precision/coverage constraint, otherwise abstain."""

    probabilities = probabilities.float().flatten()
    labels = labels.long().flatten()
    candidates = torch.unique(probabilities).sort(descending=True).values
    viable: list[tuple[float, float, int, float]] = []
    fallback: list[tuple[float, float, int, float]] = []
    for threshold in candidates:
        predicted = probabilities >= threshold
        tp = int((predicted & (labels == 1)).sum())
        fp = int((predicted & (labels == 0)).sum())
        fn = int((~predicted & (labels == 1)).sum())
        commits = tp + fp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        fallback.append((precision, recall, commits, float(threshold)))
        if precision >= minimum_precision and commits >= minimum_commits:
            viable.append((recall, precision, commits, float(threshold)))
    if viable:
        recall, precision, commits, threshold = max(viable)
        return {
            "status": "satisfied",
            "policy_satisfied": True,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "commits": commits,
        }
    best = max(fallback) if fallback else (0.0, 0.0, 0, 1.0)
    return {
        "status": "infeasible_abstain",
        "policy_satisfied": False,
        "threshold": 1.000001,
        "precision": 0.0,
        "recall": 0.0,
        "commits": 0,
        "best_available_precision": best[0],
        "best_available_recall": best[1],
        "best_available_commits": best[2],
        "best_available_threshold": best[3],
    }


if __name__ == "__main__":
    raise SystemExit(main())
