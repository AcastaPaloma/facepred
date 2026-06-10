"""Losses for FacePred's multi-task world model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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
        turn_class_weights: torch.Tensor | None = None,
        class_weights: Mapping[str, torch.Tensor] | None = None,
        focal_gammas: Mapping[str, float] | None = None,
        yield_pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.weights = {
            "rssm_dynamics": 1.0,
            "turn_taking": 1.0,
            "yield": 2.0,
            "commit_safety": 0.0,
            "event_hazard": 0.0,
            "end_of_turn": 0.5,
            "dialog_act": 0.5,
            "affect": 0.3,
            "calibration": 0.3,
        }
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})
        self.label_smoothing = label_smoothing
        self.focal_gammas = {key: float(value) for key, value in (focal_gammas or {}).items()}
        if yield_pos_weight is not None:
            self.register_buffer("yield_pos_weight", yield_pos_weight.float())
        else:
            self.yield_pos_weight = None
        resolved_weights = dict(class_weights or {})
        if turn_class_weights is not None:
            resolved_weights["turn_taking"] = turn_class_weights
        self._class_weight_buffers: dict[str, str] = {}
        for component, values in resolved_weights.items():
            buffer_name = f"{component}_class_weights"
            self.register_buffer(buffer_name, values.float())
            self._class_weight_buffers[component] = buffer_name

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
        if "yield_logits" in predictions and "yield" in targets:
            yield_loss = masked_binary_cross_entropy(
                predictions["yield_logits"],
                targets["yield"],
                mask=targets.get("horizon_mask", targets.get("mask")),
                pos_weight=self.yield_pos_weight,
            )
            components["yield"] = yield_loss
            total = total + self.weights["yield"] * yield_loss
        if "commit_safety_logits" in predictions and "yield" in targets:
            commit_safety_loss = masked_binary_cross_entropy(
                predictions["commit_safety_logits"],
                targets["yield"],
                mask=targets.get("horizon_mask", targets.get("mask")),
                pos_weight=self.yield_pos_weight,
            )
            components["commit_safety"] = commit_safety_loss
            total = total + self.weights["commit_safety"] * commit_safety_loss
        total = self._add_ce(
            predictions,
            targets,
            total,
            components,
            pred_key="event_hazard_logits",
            target_keys=("event_hazard",),
            component="event_hazard",
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
            va_loss = masked_mse_loss(
                pred,
                target,
                targets.get("horizon_mask", targets.get("mask")),
            )
            components["valence_arousal"] = va_loss
            total = total + self.weights["affect"] * va_loss

        if "turn_taking_logits" in predictions and (
            "turn_taking" in targets or "turn_labels" in targets
        ):
            target = targets.get("turn_taking", targets.get("turn_labels"))
            brier = multiclass_brier_score(predictions["turn_taking_logits"], target)
            components["calibration"] = brier
            total = total + self.weights["calibration"] * brier

        if "yield_logits" in predictions and "yield" in targets:
            yield_brier = binary_brier_score(
                predictions["yield_logits"],
                targets["yield"],
                mask=targets.get("horizon_mask", targets.get("mask")),
            )
            components["yield_calibration"] = yield_brier
            total = total + self.weights["calibration"] * yield_brier
        if "commit_safety_logits" in predictions and "yield" in targets:
            safety_brier = binary_brier_score(
                predictions["commit_safety_logits"],
                targets["yield"],
                mask=targets.get("horizon_mask", targets.get("mask")),
            )
            components["commit_safety_calibration"] = safety_brier
            total = total + self.weights["calibration"] * safety_brier

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
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred_key not in predictions:
            return total
        target = next((targets[key] for key in target_keys if key in targets), None)
        if target is None:
            return total
        logits = predictions[pred_key]
        if class_weights is None:
            class_weights = self._class_weights(component)
        loss = sequence_cross_entropy(
            logits,
            target,
            label_smoothing=self.label_smoothing,
            class_weights=class_weights,
            focal_gamma=self.focal_gammas.get(component, 0.0),
        )
        components[component] = loss
        return total + self.weights[component] * loss

    def _class_weights(self, component: str) -> torch.Tensor | None:
        buffer_name = self._class_weight_buffers.get(component)
        return getattr(self, buffer_name) if buffer_name is not None else None

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
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Cross entropy for ``[..., classes]`` logits and matching integer targets."""
    targets = targets.to(device=logits.device, dtype=torch.long)
    targets = match_prediction_shape(logits[..., 0], targets)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    valid = flat_targets != ignore_index
    if not valid.any():
        return flat_logits.sum() * 0.0

    weights = class_weights.to(logits.device) if class_weights is not None else None
    losses = F.cross_entropy(
        flat_logits,
        flat_targets,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        weight=weights,
        reduction="none",
    )
    if focal_gamma > 0:
        probabilities = F.softmax(flat_logits[valid], dim=-1)
        target_probabilities = probabilities.gather(1, flat_targets[valid, None]).squeeze(1)
        losses[valid] = losses[valid] * (1.0 - target_probabilities).pow(focal_gamma)

    denominator = valid.sum().to(losses.dtype)
    if weights is not None:
        denominator = weights[flat_targets[valid]].sum()
    return losses[valid].sum() / denominator.clamp_min(1.0e-8)


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


def masked_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Binary cross entropy for horizon-shaped logits with masking."""

    targets = match_prediction_shape(logits, targets.to(device=logits.device)).to(logits.dtype)
    valid = targets != ignore_index
    if mask is not None:
        valid &= match_prediction_shape(logits, mask.to(device=logits.device, dtype=torch.bool))
    if not valid.any():
        return logits.sum() * 0.0
    weights = None
    if pos_weight is not None:
        weights = pos_weight.to(logits.device)
        while weights.ndim < logits.ndim:
            weights = weights.unsqueeze(0)
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets.clamp(0.0, 1.0),
        reduction="none",
        pos_weight=weights,
    )
    return losses[valid].mean()


def binary_brier_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    targets = match_prediction_shape(logits, targets.to(device=logits.device)).to(logits.dtype)
    valid = targets != ignore_index
    if mask is not None:
        valid &= match_prediction_shape(logits, mask.to(device=logits.device, dtype=torch.bool))
    if not valid.any():
        return logits.sum() * 0.0
    return (torch.sigmoid(logits)[valid] - targets[valid]).square().mean()


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE with optional sequence mask shaped ``[batch, steps]``."""
    squared = (prediction - target).pow(2)
    if mask is None:
        return squared.mean()
    mask = match_prediction_shape(squared[..., 0], mask.to(device=prediction.device, dtype=torch.bool))
    while mask.ndim < squared.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(squared)
    if not mask.any():
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    return squared[mask].mean()


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
