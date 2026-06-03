"""Visualization helpers for FacePred predictions and gate decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from facepred.engine.precompute import GateDecision

TURN_CLASS_NAMES = ("hold", "shift", "backchannel", "overlap")


def _latest_probs(predictions: Mapping[str, Any]) -> torch.Tensor:
    value = predictions.get(
        "turn_probs",
        predictions.get("turn_taking_probs", predictions.get("turn_taking")),
    )
    if value is None and "turn_taking_logits" in predictions:
        value = torch.as_tensor(predictions["turn_taking_logits"]).softmax(dim=-1)
    if value is None:
        return torch.tensor([0.7, 0.2, 0.05, 0.05], dtype=torch.float32)
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 1:
        probs = tensor
    else:
        probs = tensor.reshape(-1, tensor.shape[-1])[-1]
    return probs / probs.sum().clamp_min(1e-8)


def format_probability_bar(value: float, width: int = 24, fill: str = "#") -> str:
    clipped = min(1.0, max(0.0, float(value)))
    filled = int(round(clipped * width))
    return fill * filled + "." * (width - filled)


def render_prediction_table(
    predictions: Mapping[str, Any],
    class_names: Sequence[str] = TURN_CLASS_NAMES,
) -> str:
    """Render latest turn probabilities as a compact text table."""
    probs = _latest_probs(predictions)
    names = list(class_names)
    while len(names) < probs.numel():
        names.append(f"class_{len(names)}")

    rows = ["class          probability  bar"]
    for name, prob in zip(names, probs.tolist(), strict=False):
        rows.append(f"{name:<14} {prob:>10.3f}  {format_probability_bar(prob)}")
    return "\n".join(rows)


def render_gate_decision(decision: GateDecision) -> str:
    """Render a human-readable precompute gate summary."""
    selected = decision.selected_branch.name if decision.selected_branch else "-"
    lines = [
        f"commit: {decision.should_commit}",
        f"reason: {decision.reason}",
        f"selected: {selected}",
        f"yield_probability: {decision.yield_probability:.3f}",
        f"entropy: {decision.entropy:.3f}",
        f"margin: {decision.margin:.3f}",
    ]
    if decision.candidates:
        lines.append("candidates:")
        for candidate in decision.candidates:
            lines.append(f"  {candidate.name:<12} {candidate.score:.3f}")
    return "\n".join(lines)


def render_dashboard(predictions: Mapping[str, Any], decision: GateDecision) -> str:
    """Combine prediction and gate views for terminal demos."""
    return "\n\n".join([
        "Turn probabilities",
        render_prediction_table(predictions),
        "Gate decision",
        render_gate_decision(decision),
    ])


def plot_turn_probabilities(
    predictions: Mapping[str, Any],
    output_path: str | Path | None = None,
    class_names: Sequence[str] = TURN_CLASS_NAMES,
) -> Any:
    """Create a matplotlib bar chart, importing matplotlib only when called."""
    import matplotlib.pyplot as plt

    probs = _latest_probs(predictions)
    names = list(class_names)[: probs.numel()]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(names, probs.tolist(), color=["#4c78a8", "#f58518", "#54a24b", "#e45756"][: len(names)])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("probability")
    ax.set_title("FacePred turn prediction")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)
    return fig
