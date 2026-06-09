"""Evaluate the lightweight FacePred scaffold on synthetic tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facepred.engine.evaluator import FacePredEvaluator
from facepred.engine.trainer import (
    TinyFacePredModel,
    TrainerSettings,
    load_project_config,
    make_synthetic_batches,
)
from facepred.utils import load_trusted_torch_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="Project config YAML.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint from scripts/train.py.")
    parser.add_argument("--batches", type=int, default=2, help="Number of synthetic eval batches.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length in steps.")
    parser.add_argument("--feature-dim", type=int, default=None, help="Override synthetic feature dim.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--seed", type=int, default=100, help="Synthetic data seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_project_config(args.config)
    settings = TrainerSettings.from_config(config)
    if args.batch_size is not None:
        settings.batch_size = args.batch_size
    if args.seq_len is not None:
        settings.sequence_length = args.seq_len
    if args.feature_dim is not None:
        settings.feature_dim = args.feature_dim

    model = TinyFacePredModel.from_settings(settings)
    if args.checkpoint:
        checkpoint = load_trusted_torch_artifact(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])

    batches = make_synthetic_batches(args.batches, settings, seed=args.seed, device=args.device)
    evaluator = FacePredEvaluator(model, settings.loss_weights, device=args.device)
    report = evaluator.evaluate(batches)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
