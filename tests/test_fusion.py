from __future__ import annotations

import torch

from facepred.models import ReliabilityGatedFusion


def test_reliability_gated_fusion_handles_missing_modalities() -> None:
    fusion = ReliabilityGatedFusion(
        {"audio_prosody": 8, "visual": 10, "quality": 4},
        embed_dim=12,
        num_heads=3,
        dropout=0.0,
    )
    modalities = {
        "audio_prosody": torch.randn(2, 5, 8),
        "quality": torch.ones(2, 5, 4),
    }
    quality = torch.tensor([[[0.8, 0.0, 0.6, 0.7]]]).expand(2, 5, 4)

    fused, gates = fusion(modalities, quality=quality)

    assert fused.shape == (2, 5, 12)
    assert gates.shape == (2, 5, 3)
    assert torch.all(gates[..., 1] == 0)
    assert torch.all(torch.isfinite(fused))


def test_heuristic_fusion_uses_quality_signals() -> None:
    fusion = ReliabilityGatedFusion(
        {"audio_prosody": 8, "visual": 10},
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
        reliability="heuristic",
    )
    modalities = {
        "audio_prosody": torch.randn(1, 2, 8),
        "visual": torch.randn(1, 2, 10),
    }
    quality = torch.tensor([[[0.9, 0.2, 0.5, 1.0], [0.1, 0.8, 0.5, 1.0]]])

    _, gates = fusion(modalities, quality=quality)

    assert gates[0, 0, 0] > gates[0, 0, 1]
    assert gates[0, 1, 1] > gates[0, 1, 0]
