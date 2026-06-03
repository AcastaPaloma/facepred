"""
FacePred — Predictive Multimodal Interaction World Model
=========================================================

A hybrid JEPA+RSSM architecture that continuously predicts near-future
conversational state from multimodal evidence (face, voice, language)
and uses those predictions to precompute candidate assistant responses
before the user finishes speaking.

Modules:
    data:       Dataset loading, label derivation, synchronization
    features:   Multimodal feature extraction (audio, visual, text, quality)
    models:     Neural network modules (encoders, fusion, RSSM, heads)
    engine:     Training, evaluation, and precompute orchestration
    inference:  Real-time pipeline and visualization
    utils:      Logging, calibration, timing, seeding utilities
"""

__version__ = "0.1.0"
