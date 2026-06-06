"""Evaluate the selected tuning winner without using test data during selection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection_path = Path(args.selection)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    run_dir = Path(selection["final"]["run_dir"])
    checkpoint = run_dir / "checkpoints" / "best.pt"
    output = run_dir / f"{args.split}_metrics.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_world_model.py",
            "--cache-dir",
            args.cache_dir,
            "--checkpoint",
            str(checkpoint),
            "--split",
            args.split,
            "--device",
            args.device,
            "--output",
            str(output),
        ],
        check=True,
    )
    print(json.dumps({"run_dir": str(run_dir), "checkpoint": str(checkpoint), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
