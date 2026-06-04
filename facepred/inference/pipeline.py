"""Real-time inference scaffold for FacePred.

The pipeline accepts synthetic feature tensors now and can wrap a real model
later. If no model is supplied, it emits deterministic heuristic predictions so
the precompute and visualization layers remain callable.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from facepred.engine.precompute import GateDecision, PrecomputeEngine
from facepred.engine.trainer import coerce_feature_tensor
from facepred.inference.latency_profiler import LatencyProfiler


@dataclass(slots=True)
class InferencePipelineConfig:
    """Runtime knobs for the scaffold pipeline."""

    device: str = "cpu"
    feature_dim: int | None = None
    turn_classes: int = 4
    branch_count: int = 3


@dataclass(slots=True)
class InferenceResult:
    """Output from one pipeline step."""

    predictions: dict[str, torch.Tensor]
    gate_decision: GateDecision
    latency_ms: float
    latencies: dict[str, Any] = field(default_factory=dict)
    timestamp_s: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "predictions": {
                key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
                for key, value in self.predictions.items()
            },
            "gate_decision": self.gate_decision.as_dict(),
            "latency_ms": self.latency_ms,
            "latencies": self.latencies,
            "timestamp_s": self.timestamp_s,
        }


class FacePredPipeline:
    """Thin orchestration layer for feature tensors, model inference, and gating."""

    def __init__(
        self,
        model: nn.Module | None = None,
        precompute_engine: PrecomputeEngine | None = None,
        config: InferencePipelineConfig | None = None,
        profiler: LatencyProfiler | None = None,
    ) -> None:
        self.config = config or InferencePipelineConfig()
        self.device = torch.device(self.config.device)
        self.model = model
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
        self.precompute_engine = precompute_engine or PrecomputeEngine()
        self.profiler = profiler or LatencyProfiler()

    @torch.no_grad()
    def predict(self, features: torch.Tensor | Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if self.model is None:
            feature_tensor = coerce_feature_tensor(features, self.config.feature_dim).to(self.device)
            return self._heuristic_predictions(feature_tensor)

        if isinstance(features, Mapping) and self._looks_like_modality_batch(features):
            model_input = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in features.items()
            }
            outputs = self.model(model_input)
        else:
            feature_tensor = coerce_feature_tensor(features, self.config.feature_dim).to(self.device)
            outputs = self.model({"features": feature_tensor})
        predictions = dict(outputs)
        if "turn_probs" not in predictions and "turn_logits" in predictions:
            predictions["turn_probs"] = predictions["turn_logits"].softmax(dim=-1)
        if "turn_probs" not in predictions and "turn_taking_logits" in predictions:
            logits = predictions["turn_taking_logits"]
            if logits.ndim >= 4:
                logits = logits[..., 0, :]
            predictions["turn_logits"] = logits
            predictions["turn_probs"] = logits.softmax(dim=-1)
        if "turn_entropy" not in predictions and "turn_probs" in predictions:
            probs = predictions["turn_probs"]
            predictions["turn_entropy"] = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        if "turn_entropy" not in predictions and "turn_taking_entropy" in predictions:
            entropy = predictions["turn_taking_entropy"]
            predictions["turn_entropy"] = entropy[..., 0] if entropy.ndim >= 3 else entropy
        return predictions

    def run_step(
        self,
        features: torch.Tensor | Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> InferenceResult:
        self.profiler.reset()
        start = time.perf_counter()
        with self.profiler.time_block("model"):
            predictions = self.predict(features)
        with self.profiler.time_block("precompute"):
            decision = self.precompute_engine.decide(predictions, context=context)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return InferenceResult(
            predictions=predictions,
            gate_decision=decision,
            latency_ms=latency_ms,
            latencies=self.profiler.as_dict(),
        )

    def warmup(self, feature_dim: int = 128, sequence_length: int = 8) -> InferenceResult:
        features = torch.zeros(1, sequence_length, feature_dim, dtype=torch.float32)
        return self.run_step(features, context={"warmup": True})

    def _heuristic_predictions(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        latest = features[:, -1, :]
        score = torch.tanh(latest.mean(dim=-1))
        shift = torch.sigmoid(2.0 * score)
        backchannel = torch.sigmoid(-score) * 0.2
        overlap = torch.full_like(shift, 0.05)
        hold = (1.0 - shift).clamp_min(0.05)
        probs = torch.stack([hold, shift, backchannel, overlap], dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)

        branch_scores = torch.stack(
            [
                probs[:, 1],
                (1.0 - torch.abs(probs[:, 1] - probs[:, 0])).clamp_min(0.0),
                probs[:, 2],
            ],
            dim=-1,
        )
        branch_scores = branch_scores / branch_scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return {
            "turn_probs": probs.unsqueeze(1),
            "turn_entropy": entropy.unsqueeze(1),
            "branch_probs": branch_scores.unsqueeze(1),
        }

    @staticmethod
    def _looks_like_modality_batch(features: Mapping[str, Any]) -> bool:
        return "features" not in features and any(isinstance(value, torch.Tensor) for value in features.values())


def make_synthetic_pipeline(device: str = "cpu") -> FacePredPipeline:
    """Convenience constructor for smoke tests and early demos."""
    return FacePredPipeline(config=InferencePipelineConfig(device=device))
