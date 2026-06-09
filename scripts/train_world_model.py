"""Train the real FacePred world model from cached sequence shards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facepred.data import CacheManifest, make_cached_dataloader
from facepred.engine.trainer import load_project_config
from facepred.models import FacePredLoss, FacePredWorldModel
from facepred.utils import load_trusted_torch_artifact, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="Project config YAML.")
    parser.add_argument("--model-config", default=None, help="Optional model YAML replacing config.model.")
    parser.add_argument("--cache-dir", required=True, help="Prepared cache directory.")
    parser.add_argument("--output-dir", required=True, help="Run directory for checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override configured batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override configured learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override configured weight decay.")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda.")
    parser.add_argument("--train-split", default="train", help="Cache split for training.")
    parser.add_argument("--val-split", default="dev", help="Cache split for validation.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--fusion-type", choices=["cross_attention", "concat", "perceiver"], default=None)
    parser.add_argument("--dropout", type=float, default=None, help="Override encoder/fusion dropout.")
    parser.add_argument(
        "--turn-class-weighting",
        choices=["none", "sqrt_inverse", "inverse"],
        default="inverse",
        help="Turn-label imbalance correction.",
    )
    parser.add_argument(
        "--aux-class-weighting",
        choices=["none", "sqrt_inverse", "inverse"],
        default="sqrt_inverse",
        help="Imbalance correction for end-of-turn, dialog-act, and emotion targets.",
    )
    parser.add_argument("--turn-focal-gamma", type=float, default=1.5, help="Focal exponent for turn loss.")
    parser.add_argument("--turn-loss-weight", type=float, default=2.0, help="Primary turn-loss multiplier.")
    parser.add_argument("--modality-dropout", type=float, default=None, help="Training-only modality dropout.")
    parser.add_argument(
        "--schedule-epochs",
        type=int,
        default=None,
        help="Epoch span used by the LR scheduler. Keep fixed across staged/resumed training.",
    )
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision when available.")
    parser.add_argument("--resume", default="auto", help="'auto', 'none', or path to checkpoint.")
    parser.add_argument("--save-every-steps", type=int, default=0, help="Also save periodic step checkpoints.")
    parser.add_argument("--keep-step-checkpoints", type=int, default=3, help="Recent step checkpoints to retain.")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Stop after this many unimproved epochs.")
    parser.add_argument("--allow-resume-mismatch", action="store_true", help="Allow cache/config mismatch on resume.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Debug limit.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Debug limit.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    config = apply_config_overrides(load_project_config(args.config), args)
    train_cfg = config["training"]
    model_cfg = config["model"]

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    manifest = CacheManifest.load(args.cache_dir)
    batch_size = int(args.batch_size or train_cfg.get("dataloader", {}).get("batch_size", 32))
    epochs = int(args.epochs or train_cfg.get("trainer", {}).get("max_epochs", 1))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("dataloader", {}).get("num_workers", 0))
    pin_memory = device.type == "cuda"

    reference_train_loader = make_cached_dataloader(
        args.cache_dir,
        args.train_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = None
    if args.val_split in manifest.splits:
        val_loader = make_cached_dataloader(
            args.cache_dir,
            args.val_split,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    model = FacePredWorldModel.from_config(model_cfg).to(device)
    class_weights = compute_class_weights(
        reference_train_loader,
        specs=classification_weight_specs(model_cfg, args),
    )
    loss_weights = dict(train_cfg.get("loss_weights", {}))
    loss_weights["turn_taking"] = float(args.turn_loss_weight)
    loss_fn = FacePredLoss(
        loss_weights,
        class_weights=class_weights,
        focal_gammas={"turn_taking": max(0.0, float(args.turn_focal_gamma))},
    )
    optimizer_cfg = train_cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr or optimizer_cfg.get("lr", 1.0e-4)),
        weight_decay=float(args.weight_decay if args.weight_decay is not None else optimizer_cfg.get("weight_decay", 0.01)),
        betas=tuple(float(value) for value in optimizer_cfg.get("betas", [0.9, 0.999])),
    )
    schedule_epochs = max(epochs, int(args.schedule_epochs or epochs))
    total_steps = max(
        1,
        math.ceil(len(reference_train_loader) / max(1, args.grad_accum)) * schedule_epochs,
    )
    scheduler = make_scheduler(optimizer, train_cfg.get("scheduler", {}), total_steps)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = make_grad_scaler(use_amp)
    config_fingerprint = stable_fingerprint(
        {
            "config": config,
            "training_semantics": training_semantics(args, schedule_epochs, batch_size),
        }
    )
    cache_fingerprint = stable_fingerprint(manifest_fingerprint_payload(manifest))
    early_stopping_patience = int(
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else train_cfg.get("trainer", {}).get("early_stopping", {}).get("patience", 10)
    )
    modality_dropout = float(
        args.modality_dropout
        if args.modality_dropout is not None
        else train_cfg.get("augmentation", {}).get("modality_dropout", 0.0)
    )

    start_epoch = 0
    start_batch_idx = 0
    global_step = 0
    best_score = -float("inf")
    bad_epochs = 0
    resume_path = resolve_resume_path(args.resume, checkpoint_dir)
    if resume_path is not None:
        checkpoint = load_trusted_torch_artifact(resume_path, map_location=device)
        validate_resume_checkpoint(
            checkpoint,
            config_fingerprint=config_fingerprint,
            cache_fingerprint=cache_fingerprint,
            allow_mismatch=args.allow_resume_mismatch,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        restore_rng_state(checkpoint.get("rng_state"))
        start_epoch = int(checkpoint.get("resume_epoch", checkpoint.get("epoch", -1) + 1))
        start_batch_idx = int(checkpoint.get("resume_batch_idx", 0))
        global_step = int(checkpoint.get("global_step", 0))
        best_score = float(checkpoint.get("best_score", best_score))
        bad_epochs = int(checkpoint.get("bad_epochs", 0))

    write_run_metadata(output_dir, args, config, manifest, device)
    print(
        json.dumps(
            {
                "device": str(device),
                "epochs": epochs,
                "schedule_epochs": schedule_epochs,
                "start_epoch": start_epoch,
                "start_batch_idx": start_batch_idx,
                "train_batches": len(reference_train_loader),
                "val_batches": len(val_loader) if val_loader is not None else 0,
                "amp": use_amp,
                "modality_dropout": modality_dropout,
                "class_weights": {
                    name: values.tolist()
                    for name, values in class_weights.items()
                },
                "turn_focal_gamma": args.turn_focal_gamma,
                "turn_loss_weight": args.turn_loss_weight,
                "resume": str(resume_path) if resume_path else None,
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if early_stopping_patience > 0 and bad_epochs >= early_stopping_patience:
        print(json.dumps({"early_stopping": True, "resume_without_additional_epochs": True}))
        return 0

    for epoch in range(start_epoch, epochs):
        train_loader = make_cached_dataloader(
            args.cache_dir,
            args.train_split,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            global_step=global_step,
            grad_accum=max(1, args.grad_accum),
            use_amp=use_amp,
            max_batches=args.max_train_batches,
            checkpoint_dir=checkpoint_dir,
            save_every_steps=args.save_every_steps,
            keep_step_checkpoints=args.keep_step_checkpoints,
            start_batch_idx=start_batch_idx if epoch == start_epoch else 0,
            modality_dropout=modality_dropout,
            checkpoint_payload_factory=lambda epoch_value, step_value, next_batch_value, best_score_value=best_score, bad_epochs_value=bad_epochs: checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch_value,
                resume_epoch=epoch_value,
                resume_batch_idx=next_batch_value,
                global_step=step_value,
                best_score=best_score_value,
                bad_epochs=bad_epochs_value,
                config_fingerprint=config_fingerprint,
                cache_fingerprint=cache_fingerprint,
                config=config,
                args=args,
            ),
        )
        val_metrics = {}
        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                use_amp=use_amp,
                max_batches=args.max_val_batches,
            )

        score = select_best_score(val_metrics)
        is_best = score > best_score
        if is_best:
            best_score = score
            bad_epochs = 0
        else:
            bad_epochs += 1

        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
            "best_score": best_score,
            "bad_epochs": bad_epochs,
        }
        append_jsonl(metrics_path, epoch_record)
        save_checkpoint(
            checkpoint_dir / "last.pt",
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                resume_epoch=epoch + 1,
                resume_batch_idx=0,
                global_step=global_step,
                best_score=best_score,
                bad_epochs=bad_epochs,
                config_fingerprint=config_fingerprint,
                cache_fingerprint=cache_fingerprint,
                config=config,
                args=args,
            ),
        )
        if is_best:
            save_checkpoint(
                checkpoint_dir / "best.pt",
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    resume_epoch=epoch + 1,
                    resume_batch_idx=0,
                    global_step=global_step,
                    best_score=best_score,
                    bad_epochs=bad_epochs,
                    config_fingerprint=config_fingerprint,
                    cache_fingerprint=cache_fingerprint,
                    config=config,
                    args=args,
                ),
            )
        print(json.dumps(epoch_record, indent=2, sort_keys=True))
        if early_stopping_patience > 0 and bad_epochs >= early_stopping_patience:
            print(json.dumps({"early_stopping": True, "epoch": epoch, "patience": early_stopping_patience}))
            break

    return 0


def train_one_epoch(
    *,
    model: FacePredWorldModel,
    loader: Iterable[Mapping[str, Any]],
    loss_fn: FacePredLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    device: torch.device,
    epoch: int,
    global_step: int,
    grad_accum: int,
    use_amp: bool,
    max_batches: int | None,
    checkpoint_dir: Path,
    save_every_steps: int,
    keep_step_checkpoints: int,
    start_batch_idx: int,
    modality_dropout: float,
    checkpoint_payload_factory: Any,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulator = MetricAccumulator()

    loader_length = len(loader) if hasattr(loader, "__len__") else None
    effective_batches = min(loader_length, max_batches) if loader_length is not None and max_batches else loader_length
    for batch_idx, batch in enumerate(limit_iter(loader, max_batches)):
        if batch_idx < start_batch_idx:
            continue
        features, targets = move_training_batch(batch, device)
        features, modality_mask = apply_modality_dropout(features, modality_dropout)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(features, modality_mask=modality_mask)
            loss_output = loss_fn(outputs, targets)
            loss = loss_output.total / grad_accum

        scaler.scale(loss).backward()
        is_update = (batch_idx + 1) % grad_accum == 0
        is_last = effective_batches is not None and (batch_idx + 1) >= effective_batches
        if is_update or is_last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if save_every_steps > 0 and global_step % save_every_steps == 0:
                save_checkpoint(
                    checkpoint_dir / f"step_{global_step:08d}.pt",
                    checkpoint_payload_factory(epoch, global_step, batch_idx + 1),
                )
                prune_step_checkpoints(checkpoint_dir, keep_step_checkpoints)

        metrics = loss_output.metrics()
        metrics.update(world_model_metrics(outputs, targets))
        accumulator.update(metrics)

    result = accumulator.mean()
    result["epoch"] = float(epoch)
    result["lr"] = float(optimizer.param_groups[0]["lr"])
    return result, global_step


@torch.no_grad()
def evaluate(
    *,
    model: FacePredWorldModel,
    loader: Iterable[Mapping[str, Any]],
    loss_fn: FacePredLoss,
    device: torch.device,
    use_amp: bool,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    world_accumulator = WorldMetricAccumulator()
    for batch in limit_iter(loader, max_batches):
        features, targets = move_training_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(features)
            loss_output = loss_fn(outputs, targets)
        metrics = loss_output.metrics()
        accumulator.update(metrics)
        world_accumulator.update(outputs, targets)
    result = accumulator.mean()
    result.update(world_accumulator.metrics())
    return result


def move_training_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    features = {
        name: tensor.to(device, non_blocking=True).float()
        for name, tensor in batch["features"].items()
    }
    targets = {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in batch["targets"].items()
    }
    targets["mask"] = batch["mask"].to(device, non_blocking=True).bool()
    return features, targets


def world_model_metrics(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    mask = targets.get("mask")
    if "turn_taking_logits" in outputs and "turn_taking" in targets:
        labels = targets["turn_taking"]
        scores = []
        accuracies = []
        for horizon_idx in range(outputs["turn_taking_logits"].shape[-2]):
            logits = outputs["turn_taking_logits"][..., horizon_idx, :]
            horizon_labels = labels_for_horizon(labels, horizon_idx)
            valid = valid_for_horizon(horizon_labels, targets, mask, horizon_idx)
            accuracy = masked_accuracy(logits, horizon_labels, valid)
            score = masked_macro_f1(logits, horizon_labels, valid, num_classes=logits.shape[-1])
            metrics[f"turn_accuracy/h{horizon_idx}"] = accuracy
            metrics[f"turn_macro_f1/h{horizon_idx}"] = score
            accuracies.append(accuracy)
            scores.append(score)
        metrics["turn_accuracy_mean"] = float(sum(accuracies) / max(1, len(accuracies)))
        metrics["turn_macro_f1_mean"] = float(sum(scores) / max(1, len(scores)))
    if "turn_taking_entropy" in outputs:
        entropy_scores = []
        for horizon_idx in range(outputs["turn_taking_entropy"].shape[-1]):
            entropy = outputs["turn_taking_entropy"][..., horizon_idx]
            valid = valid_for_horizon(
                labels_for_horizon(targets["turn_taking"], horizon_idx),
                targets,
                mask,
                horizon_idx,
            )
            value = float(entropy[valid].mean().detach().cpu()) if valid.any() else 0.0
            metrics[f"turn_entropy/h{horizon_idx}"] = value
            entropy_scores.append(value)
        metrics["turn_entropy_mean"] = float(sum(entropy_scores) / max(1, len(entropy_scores)))
    if "dialog_act_logits" in outputs and "dialog_act" in targets:
        metrics["dialog_act_accuracy_mean"] = mean_horizon_accuracy(
            outputs["dialog_act_logits"], targets["dialog_act"], targets, mask
        )
    if "emotion_logits" in outputs and "emotion" in targets:
        metrics["emotion_accuracy_mean"] = mean_horizon_accuracy(
            outputs["emotion_logits"], targets["emotion"], targets, mask
        )
    return metrics


def labels_for_horizon(labels: torch.Tensor, horizon_idx: int) -> torch.Tensor:
    if labels.ndim >= 3:
        return labels[..., horizon_idx]
    return labels


def valid_for_horizon(
    labels: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    mask: torch.Tensor | None,
    horizon_idx: int,
) -> torch.Tensor:
    valid = labels != -100
    if mask is not None:
        valid &= mask
    horizon_mask = targets.get("horizon_mask")
    if horizon_mask is not None:
        valid &= labels_for_horizon(horizon_mask, horizon_idx).bool()
    return valid


def mean_horizon_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    mask: torch.Tensor | None,
) -> float:
    scores = []
    for horizon_idx in range(logits.shape[-2]):
        horizon_labels = labels_for_horizon(labels, horizon_idx)
        valid = valid_for_horizon(horizon_labels, targets, mask, horizon_idx)
        scores.append(masked_accuracy(logits[..., horizon_idx, :], horizon_labels, valid))
    return float(sum(scores) / max(1, len(scores)))


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> float:
    if not valid.any():
        return 0.0
    predictions = logits.argmax(dim=-1)
    return float((predictions[valid] == labels[valid]).float().mean().detach().cpu())


def masked_macro_f1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    num_classes: int,
) -> float:
    if not valid.any():
        return 0.0
    predictions = logits.argmax(dim=-1)[valid]
    labels = labels[valid]
    scores = []
    for class_id in range(num_classes):
        pred_pos = predictions == class_id
        label_pos = labels == class_id
        tp = (pred_pos & label_pos).sum().float()
        fp = (pred_pos & ~label_pos).sum().float()
        fn = (~pred_pos & label_pos).sum().float()
        denom = 2 * tp + fp + fn
        scores.append(float((2 * tp / denom.clamp_min(1.0)).detach().cpu()))
    return float(sum(scores) / len(scores))


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: Mapping[str, Any],
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    name = str(scheduler_cfg.get("name", "cosine_warmup"))
    warmup_steps = int(float(scheduler_cfg.get("warmup_fraction", 0.05)) * total_steps)
    min_lr_ratio = float(scheduler_cfg.get("min_lr", 1.0e-6)) / max(
        float(optimizer.param_groups[0]["lr"]),
        1.0e-12,
    )

    def lr_lambda(step: int) -> float:
        if name == "constant":
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return max(min_lr_ratio, float(step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_grad_scaler(enabled: bool) -> Any:
    """Use the current AMP API while remaining compatible with older PyTorch."""

    scaler_type = getattr(torch.amp, "GradScaler", None)
    if scaler_type is not None:
        return scaler_type("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_resume_path(value: str, checkpoint_dir: Path) -> Path | None:
    if value.lower() in {"none", "false", "no"}:
        return None
    if value == "auto":
        candidates = [checkpoint_dir / "last.pt", *checkpoint_dir.glob("step_*.pt")]
        existing = [path for path in candidates if path.exists()]
        return max(existing, key=lambda path: path.stat().st_mtime) if existing else None
    path = Path(value)
    return path if path.exists() else None


def checkpoint_payload(
    *,
    model: FacePredWorldModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    epoch: int,
    resume_epoch: int,
    resume_batch_idx: int,
    global_step: int,
    best_score: float,
    bad_epochs: int,
    config_fingerprint: str,
    cache_fingerprint: str,
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "resume_epoch": resume_epoch,
        "resume_batch_idx": resume_batch_idx,
        "global_step": global_step,
        "best_score": best_score,
        "bad_epochs": bad_epochs,
        "config_fingerprint": config_fingerprint,
        "cache_fingerprint": cache_fingerprint,
        "rng_state": capture_rng_state(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": dict(config),
        "args": vars(args),
    }


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp_path)
    tmp_path.replace(path)


def write_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    manifest: CacheManifest,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "args": vars(args),
        "device": str(device),
        "cache": {
            "modalities": manifest.modalities,
            "sequence_length": manifest.sequence_length,
            "step_duration_ms": manifest.step_duration_ms,
            "splits": {
                split: [shard.__dict__ for shard in shards]
                for split, shards in manifest.splits.items()
            },
        },
        "config": dict(config),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=str)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def select_best_score(val_metrics: Mapping[str, float]) -> float:
    if not val_metrics:
        return -float("inf")
    if "turn_macro_f1_mean" in val_metrics:
        return float(val_metrics["turn_macro_f1_mean"])
    if "total" in val_metrics:
        return -float(val_metrics["total"])
    return -float(val_metrics.get("loss/total", float("inf")))


def apply_config_overrides(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    if args.model_config:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required for --model-config") from exc
        with Path(args.model_config).open("r", encoding="utf-8") as handle:
            result["model"] = yaml.safe_load(handle) or {}
    if args.fusion_type:
        result["model"]["fusion"]["type"] = args.fusion_type
    if args.dropout is not None:
        result["model"]["fusion"]["dropout"] = float(args.dropout)
        for encoder in result["model"].get("encoders", {}).values():
            if isinstance(encoder, dict) and "dropout" in encoder:
                encoder["dropout"] = float(args.dropout)
    return result


def classification_weight_specs(
    model_config: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, tuple[int, str]]:
    heads = model_config.get("heads", {})
    return {
        "turn_taking": (
            int(heads.get("turn_taking", {}).get("num_classes", 4)),
            args.turn_class_weighting,
        ),
        "end_of_turn": (
            int(heads.get("end_of_turn", {}).get("num_buckets", 10)),
            args.aux_class_weighting,
        ),
        "dialog_act": (
            int(heads.get("dialog_act", {}).get("num_classes", 13)),
            args.aux_class_weighting,
        ),
        "affect": (
            int(heads.get("affect", {}).get("num_emotions", 7)),
            args.aux_class_weighting,
        ),
    }


def compute_class_weights(
    loader: Iterable[Mapping[str, Any]],
    *,
    specs: Mapping[str, tuple[int, str]],
) -> dict[str, torch.Tensor]:
    target_keys = {
        "turn_taking": "turn_taking",
        "end_of_turn": "end_of_turn",
        "dialog_act": "dialog_act",
        "affect": "emotion",
    }
    counts = {
        component: torch.zeros(num_classes, dtype=torch.float64)
        for component, (num_classes, mode) in specs.items()
        if mode != "none"
    }
    for batch in loader:
        for component, component_counts in counts.items():
            labels = batch["targets"][target_keys[component]].reshape(-1)
            valid = labels != -100
            component_counts += torch.bincount(
                labels[valid],
                minlength=component_counts.numel(),
            ).double()

    result: dict[str, torch.Tensor] = {}
    for component, component_counts in counts.items():
        mode = specs[component][1]
        exponent = 0.5 if mode == "sqrt_inverse" else 1.0
        present = component_counts > 0
        weights = torch.zeros_like(component_counts)
        weights[present] = (
            component_counts.sum().clamp_min(1.0).pow(exponent)
            / component_counts[present].pow(exponent)
        )
        weights[present] /= weights[present].mean().clamp_min(1.0e-12)
        result[component] = weights.float()
    return result


def training_semantics(
    args: argparse.Namespace,
    schedule_epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    """Fingerprint settings that must remain stable when resuming a staged run."""

    return {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": batch_size,
        "grad_accum": args.grad_accum,
        "schedule_epochs": schedule_epochs,
        "turn_class_weighting": args.turn_class_weighting,
        "aux_class_weighting": args.aux_class_weighting,
        "turn_focal_gamma": args.turn_focal_gamma,
        "turn_loss_weight": args.turn_loss_weight,
        "modality_dropout": args.modality_dropout,
    }


def apply_modality_dropout(
    features: Mapping[str, torch.Tensor],
    probability: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    if probability <= 0.0:
        return dict(features), None
    output = {name: tensor for name, tensor in features.items()}
    droppable = [name for name in output if name != "quality"]
    if not droppable:
        return output, None
    batch, steps = next(iter(output.values())).shape[:2]
    masks = {
        name: torch.ones(batch, steps, dtype=torch.bool, device=next(iter(output.values())).device)
        for name in output
    }
    keep_matrix = torch.rand(batch, len(droppable), device=next(iter(output.values())).device) >= probability
    none_kept = ~keep_matrix.any(dim=-1)
    if none_kept.any():
        keep_matrix[none_kept, 0] = True
    for idx, name in enumerate(droppable):
        keep = keep_matrix[:, idx].view(batch, 1, 1)
        output[name] = output[name] * keep
        masks[name] = keep_matrix[:, idx].view(batch, 1).expand(batch, steps)
    return output, masks


def stable_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_fingerprint_payload(manifest: CacheManifest) -> dict[str, Any]:
    return {
        "version": manifest.version,
        "modalities": manifest.modalities,
        "sequence_length": manifest.sequence_length,
        "step_duration_ms": manifest.step_duration_ms,
        "splits": {
            split: [(shard.path, shard.num_sequences) for shard in shards]
            for split, shards in manifest.splits.items()
        },
        "metadata": manifest.metadata,
    }


def validate_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    config_fingerprint: str,
    cache_fingerprint: str,
    allow_mismatch: bool,
) -> None:
    mismatches = []
    if checkpoint.get("config_fingerprint") not in {None, config_fingerprint}:
        mismatches.append("config")
    if checkpoint.get("cache_fingerprint") not in {None, cache_fingerprint}:
        mismatches.append("cache")
    if mismatches and not allow_mismatch:
        raise ValueError(
            f"Refusing to resume with mismatched {', '.join(mismatches)}. "
            "Use --allow-resume-mismatch only when this is intentional."
        )


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def prune_step_checkpoints(checkpoint_dir: Path, keep: int) -> None:
    paths = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in paths[max(0, keep) :]:
        path.unlink(missing_ok=True)


def limit_iter(iterable: Iterable[Any], max_items: int | None) -> Iterable[Any]:
    for index, item in enumerate(iterable):
        if max_items is not None and index >= max_items:
            break
        yield item


class MetricAccumulator:
    """Accumulate scalar metrics over batches."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                self.totals[key] = self.totals.get(key, 0.0) + float(value)
                self.counts[key] = self.counts.get(key, 0) + 1

    def mean(self) -> dict[str, float]:
        return {
            key: self.totals[key] / max(1, self.counts[key])
            for key in sorted(self.totals)
        }


class WorldMetricAccumulator:
    """Accumulate corpus-level classification metrics across batches."""

    def __init__(self) -> None:
        self.confusions: dict[str, list[torch.Tensor]] = {}
        self.entropy_sum: list[float] = []
        self.entropy_count: list[int] = []

    def update(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> None:
        mask = targets.get("mask")
        specs = (
            ("turn", "turn_taking_logits", "turn_taking"),
            ("dialog_act", "dialog_act_logits", "dialog_act"),
            ("emotion", "emotion_logits", "emotion"),
        )
        for metric_name, output_name, target_name in specs:
            if output_name not in outputs or target_name not in targets:
                continue
            logits = outputs[output_name]
            labels = targets[target_name]
            matrices = self.confusions.setdefault(
                metric_name,
                [
                    torch.zeros(logits.shape[-1], logits.shape[-1], dtype=torch.long)
                    for _ in range(logits.shape[-2])
                ],
            )
            for horizon_idx in range(logits.shape[-2]):
                horizon_labels = labels_for_horizon(labels, horizon_idx)
                valid = valid_for_horizon(horizon_labels, targets, mask, horizon_idx)
                predictions = logits[..., horizon_idx, :].argmax(dim=-1)
                matrices[horizon_idx] += confusion_matrix(
                    predictions[valid],
                    horizon_labels[valid],
                    logits.shape[-1],
                )

        if "turn_taking_entropy" in outputs and "turn_taking" in targets:
            entropy = outputs["turn_taking_entropy"]
            while len(self.entropy_sum) < entropy.shape[-1]:
                self.entropy_sum.append(0.0)
                self.entropy_count.append(0)
            for horizon_idx in range(entropy.shape[-1]):
                labels = labels_for_horizon(targets["turn_taking"], horizon_idx)
                valid = valid_for_horizon(labels, targets, mask, horizon_idx)
                self.entropy_sum[horizon_idx] += float(entropy[..., horizon_idx][valid].sum().cpu())
                self.entropy_count[horizon_idx] += int(valid.sum().cpu())

    def metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        turn_f1 = []
        turn_accuracy = []
        for name, matrices in self.confusions.items():
            for horizon_idx, matrix in enumerate(matrices):
                accuracy, macro_f1 = metrics_from_confusion(matrix)
                metrics[f"{name}_accuracy/h{horizon_idx}"] = accuracy
                metrics[f"{name}_macro_f1/h{horizon_idx}"] = macro_f1
                if name == "turn":
                    turn_accuracy.append(accuracy)
                    turn_f1.append(macro_f1)
                    support = matrix.sum(dim=1)
                    predicted_support = matrix.sum(dim=0)
                    true_positive = matrix.diag()
                    recall = true_positive / support.clamp_min(1)
                    precision = true_positive / predicted_support.clamp_min(1)
                    class_f1 = 2 * precision * recall / (precision + recall).clamp_min(1.0e-12)
                    majority_count = support.max() if support.numel() else torch.tensor(0)
                    majority_f1 = 2 * majority_count / (support.sum() + majority_count).clamp_min(1)
                    metrics[f"turn_active_classes/h{horizon_idx}"] = float(
                        (predicted_support > 0).sum()
                    )
                    metrics[f"turn_majority_macro_f1/h{horizon_idx}"] = float(
                        majority_f1 / max(1, matrix.shape[0])
                    )
                    metrics[f"turn_balanced_accuracy/h{horizon_idx}"] = float(recall.mean())
                    for class_idx, class_support in enumerate(support.tolist()):
                        metrics[f"turn_support/h{horizon_idx}/class{class_idx}"] = float(class_support)
                        metrics[f"turn_pred_support/h{horizon_idx}/class{class_idx}"] = float(
                            predicted_support[class_idx]
                        )
                        metrics[f"turn_recall/h{horizon_idx}/class{class_idx}"] = float(
                            recall[class_idx]
                        )
                        metrics[f"turn_f1/h{horizon_idx}/class{class_idx}"] = float(
                            class_f1[class_idx]
                        )
        if turn_accuracy:
            metrics["turn_accuracy_mean"] = float(sum(turn_accuracy) / len(turn_accuracy))
            metrics["turn_macro_f1_mean"] = float(sum(turn_f1) / len(turn_f1))
            metrics["turn_active_classes_mean"] = float(
                sum(metrics[f"turn_active_classes/h{idx}"] for idx in range(len(turn_accuracy)))
                / len(turn_accuracy)
            )
            metrics["turn_majority_macro_f1_mean"] = float(
                sum(
                    metrics[f"turn_majority_macro_f1/h{idx}"]
                    for idx in range(len(turn_accuracy))
                )
                / len(turn_accuracy)
            )
            metrics["turn_balanced_accuracy_mean"] = float(
                sum(
                    metrics[f"turn_balanced_accuracy/h{idx}"]
                    for idx in range(len(turn_accuracy))
                )
                / len(turn_accuracy)
            )
        for horizon_idx, total in enumerate(self.entropy_sum):
            metrics[f"turn_entropy/h{horizon_idx}"] = total / max(1, self.entropy_count[horizon_idx])
        if self.entropy_sum:
            metrics["turn_entropy_mean"] = float(
                sum(metrics[f"turn_entropy/h{idx}"] for idx in range(len(self.entropy_sum)))
                / len(self.entropy_sum)
            )
        return metrics

    def report(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics(),
            "confusion_matrices": {
                name: [matrix.tolist() for matrix in matrices]
                for name, matrices in self.confusions.items()
            },
        }


def confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if labels.numel() == 0:
        return torch.zeros(num_classes, num_classes, dtype=torch.long)
    indices = labels.detach().cpu().long() * num_classes + predictions.detach().cpu().long()
    return torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(matrix: torch.Tensor) -> tuple[float, float]:
    matrix = matrix.float()
    total = matrix.sum().clamp_min(1.0)
    accuracy = float(matrix.diag().sum() / total)
    tp = matrix.diag()
    fp = matrix.sum(dim=0) - tp
    fn = matrix.sum(dim=1) - tp
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1.0)
    return accuracy, float(f1.mean())


if __name__ == "__main__":
    raise SystemExit(main())
