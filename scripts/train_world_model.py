"""Train the real FacePred world model from cached sequence shards."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facepred.data import CacheManifest, make_cached_dataloader
from facepred.engine.trainer import load_project_config
from facepred.models import FacePredLoss, FacePredWorldModel
from facepred.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="Project config YAML.")
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
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision when available.")
    parser.add_argument("--resume", default="auto", help="'auto', 'none', or path to checkpoint.")
    parser.add_argument("--save-every-steps", type=int, default=0, help="Also save periodic step checkpoints.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Debug limit.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Debug limit.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    config = load_project_config(args.config)
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

    train_loader = make_cached_dataloader(
        args.cache_dir,
        args.train_split,
        batch_size=batch_size,
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
    loss_fn = FacePredLoss(train_cfg.get("loss_weights", {}))
    optimizer_cfg = train_cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr or optimizer_cfg.get("lr", 1.0e-4)),
        weight_decay=float(args.weight_decay if args.weight_decay is not None else optimizer_cfg.get("weight_decay", 0.01)),
        betas=tuple(float(value) for value in optimizer_cfg.get("betas", [0.9, 0.999])),
    )
    total_steps = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum)) * epochs)
    scheduler = make_scheduler(optimizer, train_cfg.get("scheduler", {}), total_steps)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    global_step = 0
    best_score = -float("inf")
    resume_path = resolve_resume_path(args.resume, checkpoint_dir)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_score = float(checkpoint.get("best_score", best_score))

    write_run_metadata(output_dir, args, config, manifest, device)
    print(
        json.dumps(
            {
                "device": str(device),
                "epochs": epochs,
                "start_epoch": start_epoch,
                "train_batches": len(train_loader),
                "val_batches": len(val_loader) if val_loader is not None else 0,
                "amp": use_amp,
                "resume": str(resume_path) if resume_path else None,
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )

    for epoch in range(start_epoch, epochs):
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
            checkpoint_payload_factory=lambda epoch_value, step_value, best_score_value=best_score: checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch_value,
                global_step=step_value,
                best_score=best_score_value,
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

        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
            "best_score": best_score,
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
                global_step=global_step,
                best_score=best_score,
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
                    global_step=global_step,
                    best_score=best_score,
                    config=config,
                    args=args,
                ),
            )
        print(json.dumps(epoch_record, indent=2, sort_keys=True))

    return 0


def train_one_epoch(
    *,
    model: FacePredWorldModel,
    loader: Iterable[Mapping[str, Any]],
    loss_fn: FacePredLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    global_step: int,
    grad_accum: int,
    use_amp: bool,
    max_batches: int | None,
    checkpoint_dir: Path,
    save_every_steps: int,
    checkpoint_payload_factory: Any,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulator = MetricAccumulator()

    for batch_idx, batch in enumerate(limit_iter(loader, max_batches)):
        features, targets = move_training_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(features)
            loss_output = loss_fn(outputs, targets)
            loss = loss_output.total / grad_accum

        scaler.scale(loss).backward()
        is_update = (batch_idx + 1) % grad_accum == 0
        is_last = max_batches is not None and (batch_idx + 1) >= max_batches
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
                    checkpoint_payload_factory(epoch, global_step),
                )

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
    for batch in limit_iter(loader, max_batches):
        features, targets = move_training_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(features)
            loss_output = loss_fn(outputs, targets)
        metrics = loss_output.metrics()
        metrics.update(world_model_metrics(outputs, targets))
        accumulator.update(metrics)
    return accumulator.mean()


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
        logits = outputs["turn_taking_logits"][..., 0, :]
        labels = targets["turn_taking"]
        valid = labels != -100
        if mask is not None:
            valid &= mask
        metrics["turn_accuracy"] = masked_accuracy(logits, labels, valid)
        metrics["turn_macro_f1"] = masked_macro_f1(logits, labels, valid, num_classes=logits.shape[-1])
    if "turn_taking_entropy" in outputs:
        entropy = outputs["turn_taking_entropy"][..., 0]
        if mask is not None and mask.any():
            entropy = entropy[mask]
        metrics["turn_entropy"] = float(entropy.mean().detach().cpu()) if entropy.numel() else 0.0
    if "dialog_act_logits" in outputs and "dialog_act" in targets:
        logits = outputs["dialog_act_logits"][..., 0, :]
        labels = targets["dialog_act"]
        valid = labels != -100
        if mask is not None:
            valid &= mask
        metrics["dialog_act_accuracy"] = masked_accuracy(logits, labels, valid)
    if "emotion_logits" in outputs and "emotion" in targets:
        logits = outputs["emotion_logits"][..., 0, :]
        labels = targets["emotion"]
        valid = labels != -100
        if mask is not None:
            valid &= mask
        metrics["emotion_accuracy"] = masked_accuracy(logits, labels, valid)
    return metrics


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
        path = checkpoint_dir / "last.pt"
        return path if path.exists() else None
    path = Path(value)
    return path if path.exists() else None


def checkpoint_payload(
    *,
    model: FacePredWorldModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_score: float,
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
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
    if "turn_macro_f1" in val_metrics:
        return float(val_metrics["turn_macro_f1"])
    if "total" in val_metrics:
        return -float(val_metrics["total"])
    return -float(val_metrics.get("loss/total", float("inf")))


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


if __name__ == "__main__":
    raise SystemExit(main())
