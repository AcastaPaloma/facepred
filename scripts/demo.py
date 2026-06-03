"""Run a tiny terminal demo of FacePred precompute gating."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facepred.inference import make_synthetic_pipeline, render_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--feature-dim", type=int, default=32, help="Synthetic feature dimension.")
    parser.add_argument("--seq-len", type=int, default=8, help="Synthetic sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic feature seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    features = torch.randn(1, args.seq_len, args.feature_dim, generator=generator)

    pipeline = make_synthetic_pipeline(device=args.device)
    result = pipeline.run_step(features, context={"demo": True})
    print(render_dashboard(result.predictions, result.gate_decision))
    print(f"\nlatency_ms: {result.latency_ms:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
