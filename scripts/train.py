"""Run a lightweight synthetic FacePred training smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facepred.engine.trainer import FacePredTrainer, TrainerSettings, load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="Project config YAML.")
    parser.add_argument("--epochs", type=int, default=1, help="Synthetic training epochs.")
    parser.add_argument("--batches", type=int, default=2, help="Synthetic training batches.")
    parser.add_argument("--val-batches", type=int, default=1, help="Synthetic validation batches.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length in steps.")
    parser.add_argument("--feature-dim", type=int, default=None, help="Override synthetic feature dim.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic data seed.")
    parser.add_argument("--checkpoint-out", default=None, help="Optional checkpoint path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_project_config(args.config)
    settings = TrainerSettings.from_config(config)
    settings.max_epochs = args.epochs
    if args.batch_size is not None:
        settings.batch_size = args.batch_size
    if args.seq_len is not None:
        settings.sequence_length = args.seq_len
    if args.feature_dim is not None:
        settings.feature_dim = args.feature_dim

    trainer = FacePredTrainer(settings=settings, device=args.device)
    result = trainer.fit_synthetic(
        num_batches=args.batches,
        num_val_batches=args.val_batches,
        max_epochs=args.epochs,
        seed=args.seed,
    )

    checkpoint_path = None
    if args.checkpoint_out:
        checkpoint_path = Path(args.checkpoint_out)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": trainer.model.state_dict(),
                "settings": asdict(settings),
                "result": result.as_dict(),
            },
            checkpoint_path,
        )

    payload = result.as_dict()
    payload["checkpoint"] = str(checkpoint_path) if checkpoint_path else None
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
