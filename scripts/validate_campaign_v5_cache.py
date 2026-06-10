"""Validate a campaign-v5 cache before copying it or starting GPU training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facepred.data import validate_event_hazard_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = validate_event_hazard_cache(args.cache_dir, splits=args.splits)
    print(json.dumps({"campaign_v5_cache_valid": True, **contract}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
