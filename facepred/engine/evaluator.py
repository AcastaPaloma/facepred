"""Evaluation utilities for the Iteration 0 FacePred scaffold."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from facepred.engine.trainer import compute_multitask_loss


@dataclass(slots=True)
class EvaluationReport:
    """Summary produced by ``FacePredEvaluator.evaluate``."""

    num_batches: int
    loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_batches": self.num_batches,
            "loss": self.loss,
            "metrics": dict(self.metrics),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)


def _flatten_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels.detach().reshape(-1).long().cpu()


def _flatten_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.detach().reshape(-1, logits.shape[-1]).cpu()


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    flat_logits = _flatten_logits(logits)
    flat_labels = _flatten_labels(labels)
    if flat_labels.numel() == 0:
        return 0.0
    return float((flat_logits.argmax(dim=-1) == flat_labels).float().mean())


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, num_classes: int | None = None) -> float:
    flat_logits = _flatten_logits(logits)
    flat_labels = _flatten_labels(labels)
    predictions = flat_logits.argmax(dim=-1)
    classes = num_classes or int(max(flat_logits.shape[-1], int(flat_labels.max().item()) + 1))

    scores = []
    for class_id in range(classes):
        pred_pos = predictions == class_id
        label_pos = flat_labels == class_id
        tp = (pred_pos & label_pos).sum().float()
        fp = (pred_pos & ~label_pos).sum().float()
        fn = (~pred_pos & label_pos).sum().float()
        denom = (2 * tp + fp + fn).clamp_min(1.0)
        scores.append(float((2 * tp / denom).item()))
    return float(sum(scores) / len(scores)) if scores else 0.0


def mean_entropy(probs: torch.Tensor) -> float:
    entropy = -(probs.detach().cpu() * torch.log(probs.detach().cpu().clamp_min(1e-8))).sum(dim=-1)
    return float(entropy.mean()) if entropy.numel() else 0.0


def evaluate_predictions(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    loss_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Compute lightweight metrics for model outputs and target tensors."""
    outputs = normalize_prediction_names(outputs)
    metrics: dict[str, float] = {}

    try:
        _, losses = compute_multitask_loss(outputs, targets, loss_weights)
        metrics.update(losses)
    except ValueError:
        pass

    if "turn_logits" in outputs and "turn_taking" in targets:
        metrics["turn/accuracy"] = classification_accuracy(outputs["turn_logits"], targets["turn_taking"])
        metrics["turn/macro_f1"] = macro_f1(outputs["turn_logits"], targets["turn_taking"])

    if "turn_probs" in outputs:
        metrics["turn/entropy"] = mean_entropy(outputs["turn_probs"])

    if "dialog_act_logits" in outputs and "dialog_act" in targets:
        metrics["dialog_act/accuracy"] = classification_accuracy(
            outputs["dialog_act_logits"], targets["dialog_act"]
        )
        metrics["dialog_act/macro_f1"] = macro_f1(outputs["dialog_act_logits"], targets["dialog_act"])

    if "emotion_logits" in outputs and "emotion" in targets:
        metrics["emotion/accuracy"] = classification_accuracy(outputs["emotion_logits"], targets["emotion"])
        metrics["emotion/macro_f1"] = macro_f1(outputs["emotion_logits"], targets["emotion"])

    if "end_of_turn" in outputs and "end_of_turn" in targets:
        pred = outputs["end_of_turn"].detach().cpu().float()
        target = targets["end_of_turn"].detach().cpu().float()
        metrics["end_of_turn/mae"] = float(torch.mean(torch.abs(pred - target)))

    if "affect" in outputs and "affect" in targets:
        pred = outputs["affect"].detach().cpu().float()
        target = targets["affect"].detach().cpu().float()
        metrics["affect/mae"] = float(torch.mean(torch.abs(pred - target)))

    return metrics


class FacePredEvaluator:
    """Callable evaluator for any model returning the scaffold output mapping."""

    def __init__(
        self,
        model: nn.Module,
        loss_weights: Mapping[str, float] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.loss_weights = dict(loss_weights or {})
        self.device = torch.device(device)
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate_batch(self, batch: Mapping[str, Any]) -> dict[str, float]:
        self.model.eval()
        moved = self._move_batch(batch)
        outputs = self.model(moved)
        targets = moved.get("targets", moved)
        return evaluate_predictions(outputs, targets, self.loss_weights)

    @torch.no_grad()
    def evaluate(self, batches: Iterable[Mapping[str, Any]]) -> EvaluationReport:
        all_metrics = [self.evaluate_batch(batch) for batch in batches]
        if not all_metrics:
            return EvaluationReport(num_batches=0, loss=None, metrics={})

        keys = sorted({key for metrics in all_metrics for key in metrics})
        averaged = {
            key: float(sum(metrics.get(key, 0.0) for metrics in all_metrics) / len(all_metrics))
            for key in keys
        }
        return EvaluationReport(
            num_batches=len(all_metrics),
            loss=averaged.get("loss/total"),
            metrics=averaged,
        )

    def _move_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            elif isinstance(value, Mapping):
                moved[key] = {
                    inner_key: inner_value.to(self.device)
                    if isinstance(inner_value, torch.Tensor)
                    else inner_value
                    for inner_key, inner_value in value.items()
                }
            else:
                moved[key] = value
        return moved


def calibration_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute multiclass Brier score for calibration smoke tests."""
    flat_probs = probs.detach().reshape(-1, probs.shape[-1]).float().cpu()
    flat_labels = labels.detach().reshape(-1).long().cpu()
    one_hot = F.one_hot(flat_labels, num_classes=flat_probs.shape[-1]).float()
    return float(torch.mean(torch.sum((flat_probs - one_hot) ** 2, dim=-1)))


def normalize_prediction_names(outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Add scaffold-compatible aliases for real world-model output names."""
    normalized = dict(outputs)
    if "turn_logits" not in normalized and "turn_taking_logits" in normalized:
        logits = normalized["turn_taking_logits"]
        if logits.ndim >= 4:
            logits = logits[..., 0, :]
        normalized["turn_logits"] = logits
    if "turn_probs" not in normalized and "turn_taking_probs" in normalized:
        probs = normalized["turn_taking_probs"]
        if probs.ndim >= 4:
            probs = probs[..., 0, :]
        normalized["turn_probs"] = probs
    if "turn_probs" not in normalized and "turn_logits" in normalized:
        normalized["turn_probs"] = normalized["turn_logits"].softmax(dim=-1)
    if "turn_entropy" not in normalized and "turn_taking_entropy" in normalized:
        entropy = normalized["turn_taking_entropy"]
        if entropy.ndim >= 3:
            entropy = entropy[..., 0]
        normalized["turn_entropy"] = entropy
    return normalized
