"""Run the campaign-v5 causal-context, event-hazard, and commit-safety study."""

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

from facepred.data import validate_event_hazard_cache

CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "name": "rich_gru_direct",
        "model": "configs/model/gru_s_audio_yield_rich.yaml",
        "event_hazard_weight": 0.0,
        "commit_safety_weight": 0.0,
    },
    {
        "name": "multirate_direct",
        "model": "configs/model/gru_s_audio_yield_multirate_direct.yaml",
        "event_hazard_weight": 0.0,
        "commit_safety_weight": 0.0,
    },
    {
        "name": "multirate_hazard",
        "model": "configs/model/gru_s_audio_yield_multirate.yaml",
        "event_hazard_weight": 0.5,
        "commit_safety_weight": 0.0,
    },
    {
        "name": "multirate_hazard_safety",
        "model": "configs/model/gru_s_audio_yield_multirate_safety.yaml",
        "event_hazard_weight": 0.5,
        "commit_safety_weight": 2.0,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=15)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--minimum-calibration-commits", type=int, default=25)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = validate_event_hazard_cache(args.cache_dir)
    print(json.dumps({"campaign_v5_cache_preflight": contract}, indent=2, sort_keys=True))
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
    top2 = sorted(stage1, key=lambda item: item["score"], reverse=True)[:2]
    stage2 = run_stage([item["candidate"] for item in top2], args.stage2_epochs, args)
    write_json(output_root / "stage2_results.json", stage2)
    winner = max(stage2, key=lambda item: item["score"])["candidate"]
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
        "campaign_version": 5,
        "strategy": "controlled_bridge_architecture",
        "controlled_variables": [
            "multi_rate_causal_context",
            "discrete_competing_risk_event_hazard",
            "learned_commit_safety_verifier",
        ],
        "stage1": stage1,
        "stage2": stage2,
        "winner": winner,
        "final": final,
        "silence_baseline": str(silence_output),
        "yield_calibration": str(calibration_path),
        "test_not_used_for_selection": True,
    }
    write_json(output_root / "selection.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


def run_stage(candidates: Any, epochs: int, args: argparse.Namespace) -> list[dict[str, Any]]:
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
        "configs/config_yield_v5.yaml",
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
        "3e-4",
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
        "sqrt_inverse",
        "--aux-class-weighting",
        "sqrt_inverse",
        "--turn-focal-gamma",
        "1.0",
        "--turn-loss-weight",
        "0.25",
        "--yield-loss-weight",
        "1.0",
        "--event-hazard-loss-weight",
        str(candidate["event_hazard_weight"]),
        "--commit-safety-loss-weight",
        str(candidate["commit_safety_weight"]),
        "--yield-weighting",
        "capped_balanced",
        "--yield-pos-weight-cap",
        "3.0",
        "--train-sampling",
        "natural",
        "--modality-dropout",
        "0.0",
        "--schedule-epochs",
        str(args.max_epochs),
    ]
    if not args.no_amp:
        command.append("--amp")
    subprocess.run(command, check=True)
    return {
        "candidate": candidate,
        "epochs_requested": epochs,
        **read_score(run_dir / "metrics.jsonl", max_epochs=epochs),
        "run_dir": str(run_dir),
    }


def read_score(metrics_path: Path, *, max_epochs: int | None = None) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_epochs is not None:
        records = [record for record in records if int(record["epoch"]) < max_epochs]
    if not records:
        raise ValueError(f"No completed epochs available in {metrics_path}")

    def score(record: dict[str, Any]) -> float:
        val = record.get("val", {})
        return float(
            val.get(
                "commit_safety_average_precision_mean",
                val.get("yield_average_precision_mean", -float("inf")),
            )
        )

    best = max(records, key=score)
    val = best.get("val", {})
    source = (
        "commit_safety"
        if "commit_safety_average_precision_mean" in val
        else "yield"
    )
    prevalence = [
        float(value)
        for key, value in val.items()
        if key.startswith(f"{source}_prevalence/h") and key.count("/") == 1
    ]
    baseline = sum(prevalence) / max(1, len(prevalence))
    return {
        "epochs_completed": max(int(record["epoch"]) for record in records) + 1,
        "score": score(best),
        "score_source": source,
        "prevalence_baseline": baseline,
        "improvement_over_prevalence": score(best) - baseline,
        "best_epoch": int(best["epoch"]),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
