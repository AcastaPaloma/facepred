"""Prediction heads for turn-taking and auxiliary conversational state."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HeadConfig:
    turn_classes: int = 4
    end_of_turn_buckets: int = 10
    dialog_act_classes: int = 13
    emotion_classes: int = 7
    hidden_dim: int = 128
    yield_enabled: bool = False


class MLPHead(nn.Module):
    """Two-layer prediction head preserving leading dimensions."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionHeads(nn.Module):
    """Multi-horizon heads used by the world model."""

    def __init__(
        self,
        state_dim: int,
        horizons_ms: list[int] | tuple[int, ...] = (200, 1000),
        config: HeadConfig | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.horizons_ms = list(horizons_ms)
        self.num_horizons = len(self.horizons_ms)
        self.config = config or HeadConfig()

        hidden = self.config.hidden_dim
        self.turn_taking = MLPHead(
            state_dim, hidden, self.num_horizons * self.config.turn_classes, dropout
        )
        self.yield_prediction = (
            MLPHead(state_dim, hidden, self.num_horizons, dropout)
            if self.config.yield_enabled
            else None
        )
        self.end_of_turn = MLPHead(
            state_dim, hidden, self.num_horizons * self.config.end_of_turn_buckets, dropout
        )
        self.dialog_act = MLPHead(
            state_dim, hidden, self.num_horizons * self.config.dialog_act_classes, dropout
        )
        self.valence_arousal = MLPHead(state_dim, max(32, hidden // 2), self.num_horizons * 2, dropout)
        self.emotion = MLPHead(
            state_dim, max(32, hidden // 2), self.num_horizons * self.config.emotion_classes, dropout
        )

    def forward(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return task predictions keyed by task name."""
        batch, steps, _ = state.shape
        turn_logits = self.turn_taking(state).view(
            batch, steps, self.num_horizons, self.config.turn_classes
        )
        eot_logits = self.end_of_turn(state).view(
            batch, steps, self.num_horizons, self.config.end_of_turn_buckets
        )
        dialog_logits = self.dialog_act(state).view(
            batch, steps, self.num_horizons, self.config.dialog_act_classes
        )
        valence_arousal = torch.tanh(
            self.valence_arousal(state).view(batch, steps, self.num_horizons, 2)
        )
        emotion_logits = self.emotion(state).view(
            batch, steps, self.num_horizons, self.config.emotion_classes
        )
        turn_entropy = categorical_entropy(turn_logits)
        outputs = {
            "turn_taking_logits": turn_logits,
            "end_of_turn_logits": eot_logits,
            "dialog_act_logits": dialog_logits,
            "valence_arousal": valence_arousal,
            "emotion_logits": emotion_logits,
            "turn_taking_entropy": turn_entropy,
        }
        if self.yield_prediction is not None:
            yield_logits = self.yield_prediction(state)
            yield_probs = torch.sigmoid(yield_logits)
            outputs.update(
                {
                    "yield_logits": yield_logits,
                    "yield_probs": yield_probs,
                    "yield_entropy": binary_entropy(yield_probs),
                }
            )
        return outputs


def categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute categorical entropy from logits along the last dimension."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def binary_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute Bernoulli entropy for probabilities."""

    probs = probs.clamp(1.0e-6, 1.0 - 1.0e-6)
    return -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())
