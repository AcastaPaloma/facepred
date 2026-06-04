"""Precompute branch selection and conservative gating scaffolds."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

DEFAULT_BRANCHES = ("answer", "clarify", "backchannel")
DEFAULT_TURN_CLASSES = ("hold", "shift", "backchannel", "overlap")


@dataclass(slots=True)
class CandidateBranch:
    """A possible assistant branch prepared before final user turn completion."""

    name: str
    score: float
    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "payload": self.payload,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class GateThresholds:
    """Decision thresholds from the architecture registry D26-D27."""

    yield_threshold: float = 0.8
    entropy_max: float = 0.5
    margin_min: float = 0.3
    precompute_k: int = 3
    yield_class_index: int = 1


@dataclass(slots=True)
class GateDecision:
    """Result of applying conservative precompute gating."""

    should_commit: bool
    reason: str
    yield_probability: float
    entropy: float
    margin: float
    selected_branch: CandidateBranch | None = None
    candidates: list[CandidateBranch] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "should_commit": self.should_commit,
            "reason": self.reason,
            "yield_probability": self.yield_probability,
            "entropy": self.entropy,
            "margin": self.margin,
            "selected_branch": self.selected_branch.as_dict() if self.selected_branch else None,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "metadata": dict(self.metadata),
        }


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().softmax(dim=-1)


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -(probs.float() * torch.log(probs.float().clamp_min(1e-8))).sum(dim=-1)


def _latest_vector(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 0:
        return tensor.view(1)
    if tensor.ndim == 1:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])[-1]


def extract_turn_probs(predictions: Mapping[str, Any]) -> torch.Tensor:
    """Extract latest turn class probabilities from common output names."""
    if "turn_probs" in predictions:
        probs = _latest_vector(predictions["turn_probs"])
    elif "turn_logits" in predictions:
        probs = probabilities_from_logits(_latest_vector(predictions["turn_logits"]))
    elif "turn_taking_probs" in predictions:
        probs = _latest_vector(predictions["turn_taking_probs"])
    elif "turn_taking_logits" in predictions:
        probs = probabilities_from_logits(_latest_vector(predictions["turn_taking_logits"]))
    elif "turn_taking" in predictions:
        probs = _latest_vector(predictions["turn_taking"])
    else:
        probs = torch.tensor([0.7, 0.2, 0.05, 0.05], dtype=torch.float32)

    total = probs.sum().clamp_min(1e-8)
    return probs / total


def extract_entropy(predictions: Mapping[str, Any], turn_probs: torch.Tensor) -> float:
    if "turn_entropy" in predictions:
        tensor = predictions["turn_entropy"]
        tensor = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
        return float(tensor.detach().float().cpu().reshape(-1)[-1])
    if "turn_taking_entropy" in predictions:
        tensor = predictions["turn_taking_entropy"]
        tensor = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
        return float(tensor.detach().float().cpu().reshape(-1)[-1])
    return float(entropy_from_probs(turn_probs))


def margin(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    ordered = sorted((float(score) for score in scores), reverse=True)
    if len(ordered) == 1:
        return ordered[0]
    return float(ordered[0] - ordered[1])


def thresholds_from_config(config: Mapping[str, Any] | None = None) -> GateThresholds:
    engine_cfg = {}
    if config:
        engine_cfg = config.get("engine", config) if isinstance(config, Mapping) else {}
        if not isinstance(engine_cfg, Mapping):
            engine_cfg = {}

    return GateThresholds(
        yield_threshold=float(engine_cfg.get("gate_yield_threshold", 0.8) or 0.8),
        entropy_max=float(engine_cfg.get("gate_entropy_max", 0.5) or 0.5),
        margin_min=float(engine_cfg.get("gate_margin_min", 0.3) or 0.3),
        precompute_k=int(engine_cfg.get("precompute_k", 3) or 3),
        yield_class_index=int(engine_cfg.get("yield_class_index", 1) or 1),
    )


class PrecomputeEngine:
    """Prepare candidate branches and decide whether to commit one."""

    def __init__(
        self,
        thresholds: GateThresholds | None = None,
        branch_names: Sequence[str] = DEFAULT_BRANCHES,
        response_factory: Callable[[str, Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.thresholds = thresholds or GateThresholds()
        self.branch_names = tuple(branch_names)
        self.response_factory = response_factory or self._default_response_factory

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        response_factory: Callable[[str, Mapping[str, Any]], Any] | None = None,
    ) -> PrecomputeEngine:
        return cls(thresholds_from_config(config), response_factory=response_factory)

    def rank_candidates(
        self,
        predictions: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> list[CandidateBranch]:
        scores = self._branch_scores(predictions)
        names = self._branch_names(predictions, len(scores))

        candidates = [
            CandidateBranch(name=name, score=float(score), metadata={"rank_source": "scaffold"})
            for name, score in zip(names, scores, strict=False)
        ]
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)

        prepared = []
        for candidate in candidates[: max(1, self.thresholds.precompute_k)]:
            payload = self.response_factory(candidate.name, context or {})
            prepared.append(
                CandidateBranch(
                    name=candidate.name,
                    score=candidate.score,
                    payload=payload,
                    metadata=dict(candidate.metadata),
                )
            )
        return prepared

    def decide(
        self,
        predictions: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> GateDecision:
        turn_probs = extract_turn_probs(predictions)
        entropy = extract_entropy(predictions, turn_probs)
        yield_index = min(self.thresholds.yield_class_index, turn_probs.numel() - 1)
        yield_probability = float(turn_probs[yield_index])

        candidates = self.rank_candidates(predictions, context)
        branch_margin = margin([candidate.score for candidate in candidates])
        selected = candidates[0] if candidates else None

        checks = {
            "yield_probability": yield_probability >= self.thresholds.yield_threshold,
            "entropy": entropy <= self.thresholds.entropy_max,
            "margin": branch_margin >= self.thresholds.margin_min,
        }
        should_commit = all(checks.values())
        reason = "commit" if should_commit else self._first_failed_reason(checks)

        return GateDecision(
            should_commit=should_commit,
            reason=reason,
            yield_probability=yield_probability,
            entropy=entropy,
            margin=branch_margin,
            selected_branch=selected if should_commit else None,
            candidates=candidates,
            metadata={
                "thresholds": {
                    "yield_threshold": self.thresholds.yield_threshold,
                    "entropy_max": self.thresholds.entropy_max,
                    "margin_min": self.thresholds.margin_min,
                    "precompute_k": self.thresholds.precompute_k,
                },
                "checks": checks,
            },
        )

    def _branch_scores(self, predictions: Mapping[str, Any]) -> list[float]:
        if "branch_probs" in predictions:
            scores = _latest_vector(predictions["branch_probs"]).tolist()
            return [float(score) for score in scores]
        if "branch_scores" in predictions:
            scores = _latest_vector(predictions["branch_scores"]).tolist()
            return [float(score) for score in scores]

        turn_probs = extract_turn_probs(predictions)
        shift = float(turn_probs[min(1, turn_probs.numel() - 1)])
        backchannel = float(turn_probs[min(2, turn_probs.numel() - 1)])
        hold = float(turn_probs[0])
        clarify = max(0.0, 1.0 - abs(shift - hold))
        raw = [shift, clarify, backchannel]
        total = sum(raw)
        if total <= 0:
            return [1.0 / len(raw)] * len(raw)
        return [score / total for score in raw]

    def _branch_names(self, predictions: Mapping[str, Any], count: int) -> list[str]:
        names = predictions.get("branch_names")
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            resolved = [str(name) for name in names]
        else:
            resolved = list(self.branch_names)
        while len(resolved) < count:
            resolved.append(f"branch_{len(resolved)}")
        return resolved[:count]

    @staticmethod
    def _default_response_factory(branch_name: str, context: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "branch": branch_name,
            "status": "prepared",
            "context_keys": sorted(str(key) for key in context),
        }

    @staticmethod
    def _first_failed_reason(checks: Mapping[str, bool]) -> str:
        for name, passed in checks.items():
            if not passed:
                return f"blocked_by_{name}"
        return "blocked"


def normalized_entropy(entropy: float, num_classes: int = 4) -> float:
    """Normalize entropy to 0-1 for dashboards and diagnostics."""
    if num_classes <= 1:
        return 0.0
    return float(entropy / math.log(num_classes))
