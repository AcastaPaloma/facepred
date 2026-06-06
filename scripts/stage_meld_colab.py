"""Download the MELD raw archive and stage it on fast Colab runtime storage."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        default="/content/drive/MyDrive/facepred/data/MELD.Raw.tar.gz",
        help="Persistent archive path, normally on Drive.",
    )
    parser.add_argument(
        "--extract-dir",
        default="/content/facepred_data",
        help="Ephemeral local extraction directory.",
    )
    parser.add_argument(
        "--local-archive",
        default="/content/MELD.Raw.tar.gz",
        help="Fast ephemeral archive copy used for extraction.",
    )
    parser.add_argument("--repo-id", default="declare-lab/MELD")
    parser.add_argument("--filename", default="MELD.Raw.tar.gz")
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive = Path(args.archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("Install huggingface_hub before staging MELD") from exc
        downloaded = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=args.filename,
                repo_type="dataset",
                local_dir=archive.parent,
            )
        )
        if downloaded.resolve() != archive.resolve():
            shutil.move(str(downloaded), archive)

    extract_dir = Path(args.extract_dir)
    expected = list(extract_dir.rglob("train_sent_emo.csv")) if extract_dir.exists() else []
    if args.force_extract or not expected:
        local_archive = Path(args.local_archive)
        if local_archive.resolve() == archive.resolve():
            pass
        elif not local_archive.exists() or local_archive.stat().st_size != archive.stat().st_size:
            shutil.copy2(archive, local_archive)
        extract_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tar", "-xzf", str(local_archive), "-C", str(extract_dir)],
            check=True,
        )

    csvs = sorted(str(path) for path in extract_dir.rglob("*_sent_emo.csv"))
    if not csvs:
        raise FileNotFoundError(f"MELD CSV files were not found after extracting {archive}")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "archive_gb": round(archive.stat().st_size / 1_000_000_000, 2),
                "extract_dir": str(extract_dir),
                "csvs": csvs,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
