"""Training, evaluation, and precompute engine."""

from facepred.engine.evaluator import (
    EvaluationReport,
    FacePredEvaluator,
    calibration_brier_score,
    classification_accuracy,
    evaluate_predictions,
    macro_f1,
    mean_entropy,
)
from facepred.engine.precompute import (
    CandidateBranch,
    GateDecision,
    GateThresholds,
    PrecomputeEngine,
    extract_turn_probs,
    extract_yield_probs,
    normalized_entropy,
    thresholds_from_config,
)
from facepred.engine.trainer import (
    FacePredTrainer,
    TinyFacePredModel,
    TrainerSettings,
    TrainingResult,
    compute_multitask_loss,
    load_project_config,
    make_synthetic_batch,
    make_synthetic_batches,
)

__all__ = [
    "CandidateBranch",
    "EvaluationReport",
    "FacePredEvaluator",
    "FacePredTrainer",
    "GateDecision",
    "GateThresholds",
    "PrecomputeEngine",
    "TinyFacePredModel",
    "TrainerSettings",
    "TrainingResult",
    "calibration_brier_score",
    "classification_accuracy",
    "compute_multitask_loss",
    "evaluate_predictions",
    "extract_turn_probs",
    "extract_yield_probs",
    "load_project_config",
    "macro_f1",
    "make_synthetic_batch",
    "make_synthetic_batches",
    "mean_entropy",
    "normalized_entropy",
    "thresholds_from_config",
]
