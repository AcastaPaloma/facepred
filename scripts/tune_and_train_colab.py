"""Run the focused v4 safe-yield imbalance/sampling campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facepred.data import validate_safe_yield_cache

CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "name": "gru_s_concat_unweighted",
        "model": "configs/model/gru_s_audio_yield.yaml",
        "lr": "3e-4",
        "yield_weighting": "none",
    },
    {
        "name": "gru_s_concat_sqrt_weight",
        "model": "configs/model/gru_s_audio_yield.yaml",
        "lr": "3e-4",
        "yield_weighting": "sqrt_balanced",
    },
    {
        "name": "gru_s_concat_cap3_weight",
        "model": "configs/model/gru_s_audio_yield.yaml",
        "lr": "3e-4",
        "yield_weighting": "capped_balanced",
        "yield_pos_weight_cap": 3.0,
    },
    {
        "name": "gru_s_concat_event_balanced_unweighted",
        "model": "configs/model/gru_s_audio_yield.yaml",
        "lr": "3e-4",
        "yield_weighting": "none",
        "train_sampling": "event_balanced",
    },
    {
        "name": "gru_s_concat_event_balanced_cap3",
        "model": "configs/model/gru_s_audio_yield.yaml",
        "lr": "3e-4",
        "yield_weighting": "capped_balanced",
        "yield_pos_weight_cap": 3.0,
        "train_sampling": "event_balanced",
    },
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
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--minimum-calibration-commits", type=int, default=25)
    parser.add_argument(
        "--minimum-stage1-improvement",
        type=float,
        default=0.02,
        help="Required average-precision gain over yield prevalence before long training.",
    )
    parser.add_argument("--allow-collapsed-stage1", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_contract = validate_safe_yield_cache(args.cache_dir)
    print(json.dumps({"safe_yield_cache_preflight": cache_contract}, indent=2, sort_keys=True))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    silence_output = output_root / "silence_baseline.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_silence_baseline.py",
            "--cache-dir",
            args.cache_dir,
            "--split",
            "dev",
            "--output",
            str(silence_output),
        ],
        check=True,
    )

    stage1 = run_stage(CANDIDATES, args.stage1_epochs, args)
    write_json(output_root / "stage1_results.json", stage1)
    viable_stage1 = [item for item in stage1 if not item["collapsed"]]
    if not viable_stage1 and not args.allow_collapsed_stage1:
        raise RuntimeError(
            "All stage-1 candidates collapsed to a majority-class baseline. "
            "The campaign stopped before expensive continuation; inspect the stage-1 diagnostics."
        )
    top3 = sorted(viable_stage1 or stage1, key=lambda item: item["score"], reverse=True)[:3]
    stage2 = run_stage([item["candidate"] for item in top3], args.stage2_epochs, args)
    write_json(output_root / "stage2_results.json", stage2)
    viable_stage2 = [item for item in stage2 if not item["collapsed"]]
    winner = max(viable_stage2 or stage2, key=lambda item: item["score"])["candidate"]
    final = run_candidate(winner, args.max_epochs, args)
    calibration_path = Path(final["run_dir"]) / "yield_calibration.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_yield.py",
            "--cache-dir",
            args.cache_dir,
            "--checkpoint",
            str(Path(final["run_dir"]) / "checkpoints" / "best.pt"),
            "--split",
            "dev",
            "--device",
            args.device,
            "--minimum-commits",
            str(args.minimum_calibration_commits),
            "--output",
            str(calibration_path),
        ],
        check=True,
    )

    selection = {
        "campaign_version": 4,
        "strategy": "successive_halving",
        "controlled_variables": [
            "yield_positive_weighting",
            "event_balanced_window_sampling",
        ],
        "stage1": stage1,
        "stage2": stage2,
        "winner": winner,
        "final": final,
        "silence_baseline": str(silence_output),
        "yield_calibration": str(calibration_path),
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
    candidate: dict[str, Any],
    epochs: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = Path(args.output_root) / candidate["name"]
    command = [
        sys.executable,
        "scripts/train_world_model.py",
        "--config",
        "configs/config_yield.yaml",
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
        "0.5",
        "--yield-loss-weight",
        "2.0",
        "--yield-weighting",
        str(candidate.get("yield_weighting", "none")),
        "--yield-pos-weight-cap",
        str(candidate.get("yield_pos_weight_cap", 3.0)),
        "--train-sampling",
        str(candidate.get("train_sampling", "natural")),
        "--modality-dropout",
        "0.05",
        "--schedule-epochs",
        str(args.max_epochs),
    ]
    if candidate.get("fusion"):
        command.extend(["--fusion-type", candidate["fusion"]])
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
        key=lambda record: float(
            record.get("val", {}).get(
                "yield_average_precision_mean",
                record.get("val", {}).get("turn_macro_f1_mean", -float("inf")),
            )
        ),
    )
    val = best_record.get("val", {})
    score = float(
        val.get(
            "yield_average_precision_mean",
            val.get("turn_macro_f1_mean", best_record.get("best_score", -float("inf"))),
        )
    )
    prevalence_values = [
        float(value)
        for key, value in val.items()
        if key.startswith("yield_prevalence/h") and key.count("/") == 1
    ]
    majority_score = (
        sum(prevalence_values) / len(prevalence_values)
        if prevalence_values
        else float(val.get("turn_majority_macro_f1_mean", 0.0))
    )
    active_classes = float(val.get("turn_active_classes_mean", 2.0))
    completed = max(int(record["epoch"]) for record in records) + 1
    return {
        "epochs_completed": completed,
        "score": score,
        "majority_score": majority_score,
        "improvement_over_majority": score - majority_score,
        "active_classes_mean": active_classes,
        "collapsed": score < majority_score + minimum_improvement,
        "best_epoch": int(best_record["epoch"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
