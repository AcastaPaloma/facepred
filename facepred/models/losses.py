"""Losses for FacePred's multi-task world model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from facepred.models.rssm import RSSMOutput


@dataclass
class LossOutput:
    """Loss result with a scalar total and detached metrics."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]

    def metrics(self) -> dict[str, float]:
        return {name: float(value.detach().cpu()) for name, value in self.components.items()}


class FacePredLoss(nn.Module):
    """Compute available FacePred losses from prediction and target dicts."""

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.weights = {
            "rssm_dynamics": 1.0,
            "turn_taking": 1.0,
            "end_of_turn": 0.5,
            "dialog_act": 0.5,
            "affect": 0.3,
            "calibration": 0.3,
        }
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})
        self.label_smoothing = label_smoothing

    def forward(
        self,
        predictions: Mapping[str, torch.Tensor | RSSMOutput],
        targets: Mapping[str, torch.Tensor],
    ) -> LossOutput:
        device = self._device_from_predictions(predictions)
        total = torch.zeros((), device=device)
        components: dict[str, torch.Tensor] = {}

        rssm_output = predictions.get("rssm")
        if isinstance(rssm_output, RSSMOutput):
            kl = rssm_kl_loss(rssm_output)
            components["rssm_dynamics"] = kl
            total = total + self.weights["rssm_dynamics"] * kl

        total = self._add_ce(
            predictions,
            targets,
            total,
            components,
            pred_key="turn_taking_logits",
            target_keys=("turn_taking", "turn_labels"),
            component="turn_taking",
        )
        total = self._add_ce(
            predictions,
            targets,
            total,
            components,
            pred_key="end_of_turn_logits",
            target_keys=("end_of_turn", "eot_bucket"),
            component="end_of_turn",
        )
        total = self._add_ce(
            predictions,
            targets,
            total,
            components,
            pred_key="dialog_act_logits",
            target_keys=("dialog_act", "dialog_acts"),
            component="dialog_act",
        )
        total = self._add_ce(
            predictions,
            targets,
            total,
            components,
            pred_key="emotion_logits",
            target_keys=("emotion", "emotions"),
            component="affect",
        )

        if "valence_arousal" in predictions and "valence_arousal" in targets:
            pred = predictions["valence_arousal"]
            target = targets["valence_arousal"].to(device=pred.device, dtype=pred.dtype)
            target = match_prediction_shape(pred, target)
            va_loss = F.mse_loss(pred, target)
            components["valence_arousal"] = va_loss
            total = total + self.weights["affect"] * va_loss

        if "turn_taking_logits" in predictions and (
            "turn_taking" in targets or "turn_labels" in targets
        ):
            target = targets.get("turn_taking", targets.get("turn_labels"))
            brier = multiclass_brier_score(predictions["turn_taking_logits"], target)
            components["calibration"] = brier
            total = total + self.weights["calibration"] * brier

        components["total"] = total
        return LossOutput(total=total, components=components)

    def _add_ce(
        self,
        predictions: Mapping[str, torch.Tensor | RSSMOutput],
        targets: Mapping[str, torch.Tensor],
        total: torch.Tensor,
        components: dict[str, torch.Tensor],
        *,
        pred_key: str,
        target_keys: tuple[str, ...],
        component: str,
    ) -> torch.Tensor:
        if pred_key not in predictions:
            return total
        target = next((targets[key] for key in target_keys if key in targets), None)
        if target is None:
            return total
        logits = predictions[pred_key]
        loss = sequence_cross_entropy(logits, target, label_smoothing=self.label_smoothing)
        components[component] = loss
        return total + self.weights[component] * loss

    def _device_from_predictions(self, predictions: Mapping[str, torch.Tensor | RSSMOutput]) -> torch.device:
        for value in predictions.values():
            if isinstance(value, torch.Tensor):
                return value.device
            if isinstance(value, RSSMOutput):
                return value.deterministic.device
        return torch.device("cpu")


def sequence_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross entropy for ``[..., classes]`` logits and matching integer targets."""
    targets = targets.to(device=logits.device, dtype=torch.long)
    targets = match_prediction_shape(logits[..., 0], targets)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


def multiclass_brier_score(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Mean multiclass Brier score for logits shaped ``[..., classes]``."""
    targets = targets.to(device=logits.device, dtype=torch.long)
    targets = match_prediction_shape(logits[..., 0], targets)
    valid = targets != ignore_index
    if not valid.any():
        return torch.zeros((), device=logits.device)
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(targets.clamp_min(0), num_classes=logits.shape[-1]).to(probs.dtype)
    squared = (probs - one_hot).pow(2).sum(dim=-1)
    return squared[valid].mean()


def match_prediction_shape(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expand targets across extra prediction axes such as horizons."""
    if target.shape == prediction.shape:
        return target
    if target.ndim > prediction.ndim:
        raise ValueError(
            f"Target has more dimensions than prediction: {target.shape} vs {prediction.shape}"
        )

    expanded = target
    while expanded.ndim < prediction.ndim:
        if expanded.ndim > 0 and expanded.shape[-1] == prediction.shape[-1]:
            expanded = expanded.unsqueeze(-2)
        else:
            expanded = expanded.unsqueeze(-1)
    try:
        return expanded.expand_as(prediction)
    except RuntimeError as exc:
        raise ValueError(
            f"Target shape {target.shape} is not compatible with prediction shape {prediction.shape}"
        ) from exc


def rssm_kl_loss(output: RSSMOutput, free_nats: float = 0.0) -> torch.Tensor:
    """KL(posterior || prior) for categorical or Gaussian RSSM latents."""
    if output.prior_logits is not None and output.posterior_logits is not None:
        p_log = F.log_softmax(output.prior_logits, dim=-1)
        q_log = F.log_softmax(output.posterior_logits, dim=-1)
        q = q_log.exp()
        kl = (q * (q_log - p_log)).sum(dim=-1).mean()
    elif output.prior_mean is not None and output.posterior_mean is not None:
        p_var = (2 * output.prior_log_std).exp()
        q_var = (2 * output.posterior_log_std).exp()
        kl = (
            output.prior_log_std
            - output.posterior_log_std
            + (q_var + (output.posterior_mean - output.prior_mean).pow(2)) / (2 * p_var)
            - 0.5
        ).sum(dim=-1).mean()
    else:
        kl = torch.zeros((), device=output.deterministic.device)
    if free_nats > 0:
        kl = torch.clamp(kl, min=free_nats)
    return kl
