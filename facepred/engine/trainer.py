"""Lightweight training scaffolds for FacePred.

This module currently provides a tiny recurrent trainer that exercises the
expected batch and prediction contracts. The real RSSM/fusion world model exists
under ``facepred.models``; wiring that model into a production training loop is
the next implementation step.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

DEFAULT_TURN_CLASSES = ("hold", "shift", "backchannel", "overlap")
DEFAULT_MODALITY_KEYS = ("visual", "audio_prosody", "audio_ssl", "vad", "text", "quality")


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mapping values without mutating the caller's inputs."""
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), MutableMapping):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning("Config file does not exist: %s", path)
        return {}

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML is not installed; using empty config for %s", path)
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}, got {type(data).__name__}")
    return data


def load_project_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load the current YAML config tree with simple Hydra-default expansion.

    This intentionally handles only the small subset used by the scaffold:
    entries like ``- model: rssm_xs`` are loaded from ``configs/model/rssm_xs.yaml``
    and placed under the top-level ``model`` key. Unresolved OmegaConf
    interpolations are left as strings.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    root = path.parent
    raw = _load_yaml(path)
    defaults = raw.get("defaults", []) if isinstance(raw.get("defaults", []), list) else []

    expanded: dict[str, Any] = {}
    for item in defaults:
        if not isinstance(item, Mapping):
            continue
        for group, name in item.items():
            if group == "_self_" or name is None:
                continue
            child = root / str(group) / f"{name}.yaml"
            expanded[group] = _load_yaml(child)

    own_values = {key: value for key, value in raw.items() if key != "defaults"}
    return deep_update(expanded, own_values)


def get_config_section(config: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    """Return a top-level config section, accepting either full or section-only configs."""
    if not config:
        return {}
    if name in config and isinstance(config[name], Mapping):
        return config[name]
    return config


def infer_feature_dim(config: Mapping[str, Any] | None = None, fallback: int = 128) -> int:
    """Infer concatenated raw feature dimension from model encoder config."""
    model_cfg = get_config_section(config, "model")
    encoders = model_cfg.get("encoders", {}) if isinstance(model_cfg, Mapping) else {}
    if not isinstance(encoders, Mapping):
        return fallback

    total = 0
    for encoder_cfg in encoders.values():
        if isinstance(encoder_cfg, Mapping):
            total += int(encoder_cfg.get("input_dim", 0) or 0)
    return total or fallback


def infer_sequence_length(config: Mapping[str, Any] | None = None, fallback: int = 50) -> int:
    """Infer sequence length in world-model steps from training/model config."""
    training_cfg = get_config_section(config, "training")
    model_cfg = get_config_section(config, "model")
    dataloader = training_cfg.get("dataloader", {}) if isinstance(training_cfg, Mapping) else {}

    seconds = float(dataloader.get("sequence_length_s", 5.0) or 5.0)
    step_ms = int(model_cfg.get("step_duration_ms", 100) or 100)
    if step_ms <= 0:
        return fallback
    return max(1, int(round(seconds * 1000 / step_ms)))


@dataclass(slots=True)
class TrainerSettings:
    """Small set of training knobs used by the scaffold loop."""

    max_epochs: int = 1
    batch_size: int = 4
    sequence_length: int = 8
    feature_dim: int = 128
    hidden_dim: int = 128
    turn_classes: int = 4
    dialog_act_classes: int = 13
    emotion_classes: int = 7
    lr: float = 1.0e-4
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "turn_taking": 1.0,
        "end_of_turn": 0.5,
        "dialog_act": 0.5,
        "affect": 0.3,
        "emotion": 0.3,
    })

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> TrainerSettings:
        model_cfg = get_config_section(config, "model")
        training_cfg = get_config_section(config, "training")
        trainer_cfg = training_cfg.get("trainer", {}) if isinstance(training_cfg, Mapping) else {}
        dataloader_cfg = training_cfg.get("dataloader", {}) if isinstance(training_cfg, Mapping) else {}
        optimizer_cfg = training_cfg.get("optimizer", {}) if isinstance(training_cfg, Mapping) else {}
        loss_weights = training_cfg.get("loss_weights", {}) if isinstance(training_cfg, Mapping) else {}

        rssm_cfg = model_cfg.get("rssm", {}) if isinstance(model_cfg, Mapping) else {}
        heads_cfg = model_cfg.get("heads", {}) if isinstance(model_cfg, Mapping) else {}
        turn_cfg = heads_cfg.get("turn_taking", {}) if isinstance(heads_cfg, Mapping) else {}
        dialog_cfg = heads_cfg.get("dialog_act", {}) if isinstance(heads_cfg, Mapping) else {}
        affect_cfg = heads_cfg.get("affect", {}) if isinstance(heads_cfg, Mapping) else {}

        return cls(
            max_epochs=int(trainer_cfg.get("max_epochs", 1) or 1),
            batch_size=int(dataloader_cfg.get("batch_size", 4) or 4),
            sequence_length=infer_sequence_length(config, fallback=8),
            feature_dim=infer_feature_dim(config, fallback=128),
            hidden_dim=int(rssm_cfg.get("gru_hidden", 128) or 128),
            turn_classes=int(turn_cfg.get("num_classes", 4) or 4),
            dialog_act_classes=int(dialog_cfg.get("num_classes", 13) or 13),
            emotion_classes=int(affect_cfg.get("num_emotions", 7) or 7),
            lr=float(optimizer_cfg.get("lr", 1.0e-4) or 1.0e-4),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.01) or 0.01),
            gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 1.0) or 1.0),
            loss_weights={
                "turn_taking": float(loss_weights.get("turn_taking", 1.0) or 1.0),
                "end_of_turn": float(loss_weights.get("end_of_turn", 0.5) or 0.5),
                "dialog_act": float(loss_weights.get("dialog_act", 0.5) or 0.5),
                "affect": float(loss_weights.get("affect", 0.3) or 0.3),
                "emotion": float(loss_weights.get("affect", 0.3) or 0.3),
            },
        )


@dataclass(slots=True)
class TrainingResult:
    """Aggregated output from a training run."""

    epochs: int
    train_loss: float
    val_loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "metrics": dict(self.metrics),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)


def _ensure_batch_time(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.view(1, 1, -1)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor with 1, 2, or 3 dims, got {tuple(tensor.shape)}")
    return tensor


def pad_or_trim_features(features: torch.Tensor, feature_dim: int) -> torch.Tensor:
    """Pad or truncate the last dimension to match a scaffold model."""
    current_dim = int(features.shape[-1])
    if current_dim == feature_dim:
        return features
    if current_dim > feature_dim:
        return features[..., :feature_dim]
    pad = torch.zeros(*features.shape[:-1], feature_dim - current_dim, device=features.device)
    return torch.cat([features, pad.to(dtype=features.dtype)], dim=-1)


def coerce_feature_tensor(
    batch_or_features: torch.Tensor | Mapping[str, Any],
    feature_dim: int | None = None,
) -> torch.Tensor:
    """Extract a ``[batch, time, feature]`` tensor from common batch shapes."""
    if isinstance(batch_or_features, torch.Tensor):
        features = _ensure_batch_time(batch_or_features.float())
    elif "features" in batch_or_features:
        features = _ensure_batch_time(batch_or_features["features"].float())
    else:
        pieces: list[torch.Tensor] = []
        for key in DEFAULT_MODALITY_KEYS:
            value = batch_or_features.get(key)
            if isinstance(value, torch.Tensor):
                pieces.append(_ensure_batch_time(value.float()))
        if not pieces:
            raise KeyError("Batch must contain 'features' or at least one modality tensor")

        batch_size = max(piece.shape[0] for piece in pieces)
        sequence_length = max(piece.shape[1] for piece in pieces)
        aligned = []
        for piece in pieces:
            if piece.shape[0] != batch_size:
                piece = piece.expand(batch_size, -1, -1)
            if piece.shape[1] != sequence_length:
                last = piece[:, -1:, :].expand(-1, sequence_length - piece.shape[1], -1)
                piece = torch.cat([piece, last], dim=1)
            aligned.append(piece)
        features = torch.cat(aligned, dim=-1)

    if feature_dim is not None:
        features = pad_or_trim_features(features, feature_dim)
    return features


class TinyFacePredModel(nn.Module):
    """Minimal recurrent multitask model used until the real RSSM exists."""

    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        turn_classes: int = 4,
        dialog_act_classes: int = 13,
        emotion_classes: int = 7,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.turn_classes = int(turn_classes)
        self.dialog_act_classes = int(dialog_act_classes)
        self.emotion_classes = int(emotion_classes)

        self.backbone = nn.GRU(self.feature_dim, self.hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.turn_head = nn.Linear(self.hidden_dim, self.turn_classes)
        self.end_of_turn_head = nn.Linear(self.hidden_dim, 1)
        self.dialog_act_head = nn.Linear(self.hidden_dim, self.dialog_act_classes)
        self.affect_head = nn.Linear(self.hidden_dim, 2)
        self.emotion_head = nn.Linear(self.hidden_dim, self.emotion_classes)

    @classmethod
    def from_settings(cls, settings: TrainerSettings) -> TinyFacePredModel:
        return cls(
            feature_dim=settings.feature_dim,
            hidden_dim=settings.hidden_dim,
            turn_classes=settings.turn_classes,
            dialog_act_classes=settings.dialog_act_classes,
            emotion_classes=settings.emotion_classes,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> TinyFacePredModel:
        return cls.from_settings(TrainerSettings.from_config(config))

    def forward(self, batch_or_features: torch.Tensor | Mapping[str, Any]) -> dict[str, torch.Tensor]:
        features = coerce_feature_tensor(batch_or_features, self.feature_dim)
        hidden, _ = self.backbone(features)
        hidden = self.norm(hidden)

        turn_logits = self.turn_head(hidden)
        turn_probs = turn_logits.softmax(dim=-1)
        entropy = -(turn_probs * torch.log(turn_probs.clamp_min(1e-8))).sum(dim=-1)

        return {
            "turn_logits": turn_logits,
            "turn_probs": turn_probs,
            "turn_entropy": entropy,
            "end_of_turn": self.end_of_turn_head(hidden).squeeze(-1),
            "dialog_act_logits": self.dialog_act_head(hidden),
            "affect": self.affect_head(hidden),
            "emotion_logits": self.emotion_head(hidden),
            "state": hidden,
        }


def make_synthetic_batch(
    batch_size: int = 4,
    sequence_length: int = 8,
    feature_dim: int = 128,
    turn_classes: int = 4,
    dialog_act_classes: int = 13,
    emotion_classes: int = 7,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> dict[str, Any]:
    """Create a batch that exercises the scaffold prediction heads."""
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    features = torch.randn(batch_size, sequence_length, feature_dim, generator=generator)
    trend = features[..., : min(feature_dim, 8)].mean(dim=-1)
    offsets = torch.arange(sequence_length).unsqueeze(0)
    turn_taking = torch.remainder((trend > 0).long() + offsets, turn_classes).contiguous()

    batch = {
        "features": features,
        "targets": {
            "turn_taking": turn_taking.long(),
            "end_of_turn": torch.sigmoid(trend).float(),
            "dialog_act": torch.randint(
                0, dialog_act_classes, (batch_size, sequence_length), generator=generator
            ),
            "affect": torch.tanh(features[..., :2]).float(),
            "emotion": torch.randint(
                0, emotion_classes, (batch_size, sequence_length), generator=generator
            ),
        },
    }

    if str(device) != "cpu":
        batch["features"] = batch["features"].to(device)
        batch["targets"] = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch["targets"].items()
        }
    return batch


def make_synthetic_batches(
    num_batches: int = 2,
    settings: TrainerSettings | None = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> list[dict[str, Any]]:
    """Return a deterministic list of synthetic batches."""
    settings = settings or TrainerSettings()
    return [
        make_synthetic_batch(
            batch_size=settings.batch_size,
            sequence_length=settings.sequence_length,
            feature_dim=settings.feature_dim,
            turn_classes=settings.turn_classes,
            dialog_act_classes=settings.dialog_act_classes,
            emotion_classes=settings.emotion_classes,
            device=device,
            seed=seed + index,
        )
        for index in range(num_batches)
    ]


def _targets(batch: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    targets = batch.get("targets", batch)
    if not isinstance(targets, Mapping):
        raise TypeError("Batch targets must be a mapping")
    return targets


def compute_multitask_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    loss_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute scaffold losses for whichever prediction heads are present."""
    weights = loss_weights or {}
    losses: dict[str, torch.Tensor] = {}

    if "turn_logits" in outputs and "turn_taking" in targets:
        logits = outputs["turn_logits"].reshape(-1, outputs["turn_logits"].shape[-1])
        labels = targets["turn_taking"].reshape(-1).long().to(logits.device)
        losses["turn_taking"] = F.cross_entropy(logits, labels)

    if "end_of_turn" in outputs and "end_of_turn" in targets:
        pred = outputs["end_of_turn"].float()
        target = targets["end_of_turn"].float().to(pred.device)
        losses["end_of_turn"] = F.mse_loss(pred, target)

    if "dialog_act_logits" in outputs and "dialog_act" in targets:
        logits = outputs["dialog_act_logits"].reshape(-1, outputs["dialog_act_logits"].shape[-1])
        labels = targets["dialog_act"].reshape(-1).long().to(logits.device)
        losses["dialog_act"] = F.cross_entropy(logits, labels)

    if "affect" in outputs and "affect" in targets:
        pred = outputs["affect"].float()
        target = targets["affect"].float().to(pred.device)
        losses["affect"] = F.mse_loss(pred, target)

    if "emotion_logits" in outputs and "emotion" in targets:
        logits = outputs["emotion_logits"].reshape(-1, outputs["emotion_logits"].shape[-1])
        labels = targets["emotion"].reshape(-1).long().to(logits.device)
        losses["emotion"] = F.cross_entropy(logits, labels)

    if not losses:
        raise ValueError("No compatible prediction/target pairs found for loss computation")

    total = torch.zeros((), dtype=torch.float32, device=next(iter(losses.values())).device)
    metrics: dict[str, float] = {}
    for name, loss in losses.items():
        weight = float(weights.get(name, 1.0))
        total = total + weight * loss
        metrics[f"loss/{name}"] = float(loss.detach().cpu())
    metrics["loss/total"] = float(total.detach().cpu())
    return total, metrics


class FacePredTrainer:
    """Tiny trainer facade with a stable shape for later Iteration 0 growth."""

    def __init__(
        self,
        model: nn.Module | None = None,
        settings: TrainerSettings | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.settings = settings or TrainerSettings()
        self.device = torch.device(device)
        self.model = model or TinyFacePredModel.from_settings(self.settings)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.settings.lr,
            weight_decay=self.settings.weight_decay,
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        device: str | torch.device = "cpu",
    ) -> FacePredTrainer:
        settings = TrainerSettings.from_config(config)
        return cls(settings=settings, device=device)

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        self.model.train()
        batch = self._move_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(batch)
        loss, metrics = compute_multitask_loss(outputs, _targets(batch), self.settings.loss_weights)
        loss.backward()
        if self.settings.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.gradient_clip_val)
        self.optimizer.step()
        return metrics

    @torch.no_grad()
    def validate_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        self.model.eval()
        batch = self._move_batch(batch)
        outputs = self.model(batch)
        _, metrics = compute_multitask_loss(outputs, _targets(batch), self.settings.loss_weights)
        return metrics

    def fit(
        self,
        train_batches: Iterable[Mapping[str, Any]],
        val_batches: Iterable[Mapping[str, Any]] | None = None,
        max_epochs: int | None = None,
    ) -> TrainingResult:
        epochs = int(max_epochs or self.settings.max_epochs)
        last_train_metrics: dict[str, float] = {}
        last_val_metrics: dict[str, float] = {}

        cached_train = list(train_batches)
        cached_val = list(val_batches) if val_batches is not None else []
        if not cached_train:
            raise ValueError("At least one training batch is required")

        for epoch in range(epochs):
            train_metrics = [self.train_step(batch) for batch in cached_train]
            last_train_metrics = _mean_metrics(train_metrics)
            if cached_val:
                val_metrics = [self.validate_step(batch) for batch in cached_val]
                last_val_metrics = _mean_metrics(val_metrics)
            logger.info("epoch=%s train_loss=%.4f", epoch + 1, last_train_metrics["loss/total"])

        combined = {f"train/{key}": value for key, value in last_train_metrics.items()}
        combined.update({f"val/{key}": value for key, value in last_val_metrics.items()})
        return TrainingResult(
            epochs=epochs,
            train_loss=float(last_train_metrics.get("loss/total", 0.0)),
            val_loss=float(last_val_metrics["loss/total"]) if last_val_metrics else None,
            metrics=combined,
        )

    def fit_synthetic(
        self,
        num_batches: int = 2,
        num_val_batches: int = 1,
        max_epochs: int | None = None,
        seed: int = 0,
    ) -> TrainingResult:
        train_batches = make_synthetic_batches(num_batches, self.settings, seed, self.device)
        val_batches = make_synthetic_batches(num_val_batches, self.settings, seed + 1000, self.device)
        return self.fit(train_batches, val_batches, max_epochs=max_epochs)

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


def _mean_metrics(metrics: Iterable[Mapping[str, float]]) -> dict[str, float]:
    values = list(metrics)
    if not values:
        return {}
    keys = sorted({key for item in values for key in item})
    return {
        key: float(sum(float(item.get(key, 0.0)) for item in values) / len(values))
        for key in keys
    }
