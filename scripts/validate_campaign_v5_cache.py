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


def summarize_cache_build(cache_dir: str | Path) -> dict[str, object]:
    root = Path(cache_dir)
    progress_root = root / ".progress"
    split_names = ("train", "dev", "test")
    return {
        "cache_dir": str(root),
        "manifest_exists": (root / "manifest.json").exists(),
        "extraction_config_exists": (root / "extraction_config.json").exists(),
        "progress_dialogues": {
            split: len(list((progress_root / split).glob("*.pt")))
            for split in split_names
        },
        "shards": {
            split: len(list((root / split).glob(f"{split}_*.pt")))
            for split in split_names
        },
    }


def main() -> int:
    args = parse_args()
    try:
        contract = validate_event_hazard_cache(args.cache_dir, splits=args.splits)
    except FileNotFoundError as exc:
        progress = summarize_cache_build(args.cache_dir)
        print(
            json.dumps(
                {
                    "campaign_v5_cache_valid": False,
                    "error": str(exc),
                    "recovery": (
                        "Resume scripts/prepare_meld_audio_cache.py with the same arguments. "
                        "Do not delete .progress; completed dialogues will be reused. Validate "
                        "again only after preparation prints the manifest path."
                    ),
                    **progress,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"campaign_v5_cache_valid": True, **contract}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
