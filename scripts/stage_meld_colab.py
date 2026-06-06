"""Download the MELD raw archive and stage it on fast Colab runtime storage."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

ANNOTATION_BASE_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD"
ANNOTATION_FILENAMES = ("train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv")
MIN_ARCHIVE_BYTES = 10_000_000_000
VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


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
    parser.add_argument("--skip-size-check", action="store_true")
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

    archive_size = archive.stat().st_size
    if archive_size < MIN_ARCHIVE_BYTES and not args.skip_size_check:
        preview = archive.read_bytes()[:256].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{archive} is only {archive_size:,} bytes; the real MELD archive is about "
            f"10.9 GB. Delete the bad Drive file and rerun staging. File preview: {preview!r}"
        )

    extract_dir = Path(args.extract_dir)
    media_files = find_media_files(extract_dir)
    if args.force_extract or not media_files:
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
        extract_nested_archives(extract_dir)

    annotation_dir = extract_dir / "MELD.Raw" / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for filename in ANNOTATION_FILENAMES:
        matches = list(extract_dir.rglob(filename))
        if not matches:
            urllib.request.urlretrieve(
                f"{ANNOTATION_BASE_URL}/{filename}",
                annotation_dir / filename,
            )

    csvs = sorted(str(path) for path in extract_dir.rglob("*_sent_emo.csv"))
    if not csvs:
        raise FileNotFoundError(f"MELD CSV files were not found after extracting {archive}")
    media_files = find_media_files(extract_dir)
    if not media_files:
        inventory = extracted_inventory(extract_dir)
        raise FileNotFoundError(
            "MELD video files were not found after extracting "
            f"{archive}. Extracted inventory: {json.dumps(inventory, sort_keys=True)}"
        )
    print(
        json.dumps(
            {
                "archive": str(archive),
                "archive_bytes": archive_size,
                "archive_gb": round(archive_size / 1_000_000_000, 2),
                "extract_dir": str(extract_dir),
                "csvs": csvs,
                "media_files": len(media_files),
                "sample_media": [str(path) for path in media_files[:5]],
                "inventory": extracted_inventory(extract_dir),
            },
            indent=2,
        )
    )
    return 0


def find_media_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]


def extract_nested_archives(root: Path, max_rounds: int = 3) -> None:
    """Extract split archives such as train.tar.gz found inside MELD.Raw."""

    for _ in range(max_rounds):
        archives = [
            path
            for path in root.rglob("*")
            if path.is_file() and archive_kind(path) is not None
        ]
        pending = [path for path in archives if not extraction_marker(path).exists()]
        if not pending:
            return
        for archive in pending:
            kind = archive_kind(archive)
            if kind == "zip":
                subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(archive.parent)], check=True)
            else:
                subprocess.run(["tar", "-xf", str(archive), "-C", str(archive.parent)], check=True)
            extraction_marker(archive).touch()


def archive_kind(path: Path) -> str | None:
    name = path.name.lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith((".tar.gz", ".tgz", ".tar", ".tar.bz2", ".tar.xz")):
        return "tar"
    return None


def extraction_marker(path: Path) -> Path:
    return path.with_name(f".{path.name}.facepred-extracted")


def extracted_inventory(root: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    samples: list[str] = []
    if not root.exists():
        return {"files": 0, "suffixes": {}, "samples": []}
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        total += 1
        suffix = "".join(path.suffixes[-2:]).lower() or "<none>"
        counts[suffix] = counts.get(suffix, 0) + 1
        if len(samples) < 20:
            samples.append(str(path.relative_to(root)))
    return {
        "files": total,
        "suffixes": dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:20]),
        "samples": samples,
    }


if __name__ == "__main__":
    raise SystemExit(main())
