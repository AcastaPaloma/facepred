"""Multimodal feature extraction: audio, visual, text, quality signals.

Optional heavyweight audio backends are exposed lazily so importing
``facepred.features`` does not import pyannote, Whisper, or transformers.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "ASR_EMBEDDING_DIM": ".asr",
    "ASRExtractor": ".asr",
    "AUDIO_SSL_DIM": ".audio_ssl",
    "AudioSSLExtractor": ".audio_ssl",
    "FEATURE_DIMS": ".audio_prosody",
    "ProsodyExtractor": ".audio_prosody",
    "QUALITY_DIM": ".quality",
    "QualityEstimator": ".quality",
    "TranscriptSegment": ".asr",
    "TOTAL_VISUAL_DIM": ".visual",
    "VAD_FEATURE_DIM": ".vad",
    "VADExtractor": ".vad",
    "VisualExtractor": ".visual",
    "VisualFeatures": ".visual",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
