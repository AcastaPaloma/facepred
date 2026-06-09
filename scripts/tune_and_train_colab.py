"""Run a small successive-halving sweep, then continue the winner to a long run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

CANDIDATES = (
    {"name": "xs_lr1e4", "model": "configs/model/rssm_xs_audio.yaml", "lr": "1e-4"},
    {"name": "xs_lr3e4", "model": "configs/model/rssm_xs_audio.yaml", "lr": "3e-4"},
    {"name": "s_lr1e4", "model": "configs/model/rssm_s_audio.yaml", "lr": "1e-4"},
    {"name": "s_lr3e4", "model": "configs/model/rssm_s_audio.yaml", "lr": "3e-4"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-root", required=True, help="Persistent Drive run directory.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=15)
    parser.add_argument("--max-epochs", type=int, default=75)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument(
        "--minimum-stage1-improvement",
        type=float,
        default=0.005,
        help="Required macro-F1 gain over the majority baseline before long training.",
    )
    parser.add_argument("--allow-collapsed-stage1", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage1 = run_stage(CANDIDATES, args.stage1_epochs, args)
    write_json(output_root / "stage1_results.json", stage1)
    viable_stage1 = [item for item in stage1 if not item["collapsed"]]
    if not viable_stage1 and not args.allow_collapsed_stage1:
        raise RuntimeError(
            "All stage-1 candidates collapsed to a majority-class baseline. "
            "The campaign stopped before expensive continuation; inspect the stage-1 diagnostics."
        )
    top2 = sorted(viable_stage1 or stage1, key=lambda item: item["score"], reverse=True)[:2]
    stage2 = run_stage([item["candidate"] for item in top2], args.stage2_epochs, args)
    write_json(output_root / "stage2_results.json", stage2)
    viable_stage2 = [item for item in stage2 if not item["collapsed"]]
    winner = max(viable_stage2 or stage2, key=lambda item: item["score"])["candidate"]
    final = run_candidate(winner, args.max_epochs, args)

    selection = {
        "strategy": "successive_halving",
        "stage1": stage1,
        "stage2": stage2,
        "winner": winner,
        "final": final,
        "test_not_used_for_selection": True,
    }
    selection_path = output_root / "selection.json"
    write_json(selection_path, selection)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


def run_stage(
    candidates: Any,
    epochs: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    return [run_candidate(candidate, epochs, args) for candidate in candidates]


def run_candidate(
    candidate: dict[str, str],
    epochs: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = Path(args.output_root) / candidate["name"]
    command = [
        sys.executable,
        "scripts/train_world_model.py",
        "--config",
        "configs/config.yaml",
        "--model-config",
        candidate["model"],
        "--cache-dir",
        args.cache_dir,
        "--output-dir",
        str(run_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        candidate["lr"],
        "--device",
        args.device,
        "--resume",
        "auto",
        "--save-every-steps",
        str(args.save_every_steps),
        "--keep-step-checkpoints",
        "3",
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--turn-class-weighting",
        "inverse",
        "--aux-class-weighting",
        "sqrt_inverse",
        "--turn-focal-gamma",
        "1.5",
        "--turn-loss-weight",
        "2.0",
        "--modality-dropout",
        "0.05",
        "--schedule-epochs",
        str(args.max_epochs),
    ]
    if not args.no_amp:
        command.append("--amp")
    subprocess.run(command, check=True)
    diagnostics = read_score(
        run_dir / "metrics.jsonl",
        args.minimum_stage1_improvement,
        max_epochs=epochs,
    )
    return {
        "candidate": candidate,
        "epochs_requested": epochs,
        **diagnostics,
        "run_dir": str(run_dir),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_score(
    metrics_path: Path,
    minimum_improvement: float = 0.005,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_epochs is not None:
        records = [record for record in records if int(record["epoch"]) < max_epochs]
    if not records:
        raise ValueError(f"No completed epochs available in {metrics_path}")
    best_record = max(
        records,
        key=lambda record: float(record.get("val", {}).get("turn_macro_f1_mean", -float("inf"))),
    )
    val = best_record.get("val", {})
    score = float(val.get("turn_macro_f1_mean", best_record.get("best_score", -float("inf"))))
    majority_score = float(val.get("turn_majority_macro_f1_mean", 0.0))
    active_classes = float(val.get("turn_active_classes_mean", 0.0))
    completed = max(int(record["epoch"]) for record in records) + 1
    return {
        "epochs_completed": completed,
        "score": score,
        "majority_score": majority_score,
        "improvement_over_majority": score - majority_score,
        "active_classes_mean": active_classes,
        "collapsed": active_classes < 2.0 or score < majority_score + minimum_improvement,
        "best_epoch": int(best_record["epoch"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
