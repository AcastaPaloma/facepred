from __future__ import annotations

import torch

from facepred.features import (
    ASRExtractor,
    AudioSSLExtractor,
    ProsodyExtractor,
    QualityEstimator,
    VADExtractor,
    VisualExtractor,
)


def test_lightweight_feature_fallback_shapes() -> None:
    waveform = torch.sin(torch.linspace(0, 10, 1600))

    vad = VADExtractor(frame_ms=100).extract(waveform, sample_rate=16000, use_model=False)
    asr_segments = ASRExtractor().transcribe(waveform, sample_rate=16000, include_embeddings=True, use_model=False)
    ssl = AudioSSLExtractor(frame_ms=100).extract(waveform, sample_rate=16000, use_model=False)
    quality = QualityEstimator().compute(audio=waveform, available_modalities=["audio", "text"])

    assert vad.shape == (1, 3)
    assert len(asr_segments) == 1
    assert asr_segments[0].embedding.shape == (384,)
    assert ssl.shape == (1, 768)
    assert quality.shape == (4,)


def test_existing_extractors_expose_zero_shapes_without_heavy_init() -> None:
    assert ProsodyExtractor().get_zeros(2).shape == (2, 88)
    assert VisualExtractor().get_zeros(2).shape == (2, 1493)
