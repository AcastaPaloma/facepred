"""Real-time inference pipeline, visualization, and latency profiling."""

from facepred.inference.latency_profiler import LatencyProfiler, LatencySample, LatencySummary
from facepred.inference.pipeline import (
    FacePredPipeline,
    InferencePipelineConfig,
    InferenceResult,
    make_synthetic_pipeline,
)
from facepred.inference.visualizer import (
    TURN_CLASS_NAMES,
    format_probability_bar,
    plot_turn_probabilities,
    render_dashboard,
    render_gate_decision,
    render_prediction_table,
)

__all__ = [
    "FacePredPipeline",
    "InferencePipelineConfig",
    "InferenceResult",
    "LatencyProfiler",
    "LatencySample",
    "LatencySummary",
    "TURN_CLASS_NAMES",
    "format_probability_bar",
    "make_synthetic_pipeline",
    "plot_turn_probabilities",
    "render_dashboard",
    "render_gate_decision",
    "render_prediction_table",
]
