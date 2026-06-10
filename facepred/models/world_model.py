"""Top-level FacePred world model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from facepred.models.encoders import ModalityEncoders, spec_from_config
from facepred.models.fusion import ReliabilityGatedFusion
from facepred.models.heads import HeadConfig, PredictionHeads
from facepred.models.rssm import RSSM
from facepred.models.temporal import MultiRateCausalEncoder


def _get(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


class FacePredWorldModel(nn.Module):
    """Encoders + fusion + RSSM + prediction heads."""

    def __init__(
        self,
        encoders: ModalityEncoders,
        fusion: ReliabilityGatedFusion,
        temporal: nn.Module,
        rssm: RSSM,
        heads: PredictionHeads,
    ) -> None:
        super().__init__()
        self.encoders = encoders
        self.fusion = fusion
        self.temporal = temporal
        self.rssm = rssm
        self.heads = heads

    @classmethod
    def from_config(cls, config: Any) -> FacePredWorldModel:
        """Construct the model from ``configs/model/*.yaml`` style config."""
        enc_cfg = _get(config, "encoders")
        if enc_cfg is None:
            raise ValueError("config must contain encoders")
        specs = {name: spec_from_config(value) for name, value in enc_cfg.items()}
        encoders = ModalityEncoders(specs)

        fusion_cfg = _get(config, "fusion", {})
        fusion = ReliabilityGatedFusion(
            input_dims=encoders.output_dims,
            embed_dim=int(_get(fusion_cfg, "embed_dim", 128)),
            num_heads=int(_get(fusion_cfg, "num_heads", 4)),
            dropout=float(_get(fusion_cfg, "dropout", 0.1)),
            fusion_type=str(_get(fusion_cfg, "type", "cross_attention")),
            reliability=str(_get(fusion_cfg, "reliability", "learned")),
            quality_dim=int(specs["quality"].input_dim if "quality" in specs else 4),
        )

        rssm_cfg = _get(config, "rssm", {})
        temporal_cfg = _get(config, "temporal", {})
        temporal_type = str(_get(temporal_cfg, "type", "none"))
        if temporal_type == "none":
            temporal: nn.Module = nn.Identity()
        elif temporal_type == "multi_rate_causal":
            temporal = MultiRateCausalEncoder(
                fusion.embed_dim,
                hidden_dim=int(_get(temporal_cfg, "hidden_dim", fusion.embed_dim)),
                dilations=tuple(int(value) for value in _get(temporal_cfg, "dilations", [1, 2, 4, 8])),
                kernel_size=int(_get(temporal_cfg, "kernel_size", 3)),
                dropout=float(_get(temporal_cfg, "dropout", 0.1)),
            )
        else:
            raise ValueError(f"Unsupported temporal encoder type: {temporal_type}")
        rssm = RSSM(
            input_dim=fusion.embed_dim,
            gru_hidden=int(_get(rssm_cfg, "gru_hidden", 128)),
            latent_type=str(_get(rssm_cfg, "latent_type", "categorical")),
            categorical_classes=int(_get(rssm_cfg, "categorical_classes", 16)),
            categorical_dims=int(_get(rssm_cfg, "categorical_dims", 8)),
            gaussian_dim=int(_get(rssm_cfg, "gaussian_dim", 32)),
            use_stochastic=bool(_get(rssm_cfg, "use_stochastic", True)),
        )

        heads_cfg = _get(config, "heads", {})
        turn_cfg = _get(heads_cfg, "turn_taking", {})
        eot_cfg = _get(heads_cfg, "end_of_turn", {})
        dialog_cfg = _get(heads_cfg, "dialog_act", {})
        affect_cfg = _get(heads_cfg, "affect", {})
        yield_cfg = _get(heads_cfg, "yield", {})
        hidden_dim = int(
            _get(
                turn_cfg,
                "hidden_dim",
                _get(eot_cfg, "hidden_dim", _get(dialog_cfg, "hidden_dim", 128)),
            )
        )
        head_config = HeadConfig(
            turn_classes=int(_get(turn_cfg, "num_classes", 4)),
            end_of_turn_buckets=int(_get(eot_cfg, "num_buckets", 10)),
            dialog_act_classes=int(_get(dialog_cfg, "num_classes", 13)),
            emotion_classes=int(_get(affect_cfg, "num_emotions", 7)),
            hidden_dim=hidden_dim,
            yield_enabled=bool(_get(yield_cfg, "enabled", False)),
            event_hazard_enabled=bool(_get(_get(heads_cfg, "event_hazard", {}), "enabled", False)),
            event_hazard_bins=len(_get(config, "event_hazard_bins_ms", [200, 500, 1000, 2000])),
            commit_safety_enabled=bool(
                _get(_get(heads_cfg, "commit_safety", {}), "enabled", False)
            ),
        )
        heads = PredictionHeads(
            state_dim=rssm.state_dim,
            horizons_ms=list(_get(config, "prediction_horizons_ms", [200, 1000])),
            config=head_config,
            dropout=float(_get(fusion_cfg, "dropout", 0.1)),
        )
        return cls(encoders=encoders, fusion=fusion, temporal=temporal, rssm=rssm, heads=heads)

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        *,
        quality: torch.Tensor | None = None,
        modality_mask: Mapping[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a full forward pass."""
        raw_quality = quality if quality is not None else features.get("quality")
        encoded = self.encoders(features)
        fused, reliability = self.fusion(encoded, quality=raw_quality, modality_mask=modality_mask)
        temporal = self.temporal(fused)
        rssm_output = self.rssm(temporal)
        predictions = self.heads(rssm_output.state)
        predictions.update(
            {
                "fused": fused,
                "temporal": temporal,
                "reliability": reliability,
                "rssm_state": rssm_output.state,
                "rssm": rssm_output,
            }
        )
        return predictions

    @torch.no_grad()
    def predict_step(
        self,
        features: Mapping[str, torch.Tensor],
        *,
        quality: torch.Tensor | None = None,
        modality_mask: Mapping[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self.eval()
        return self(features, quality=quality, modality_mask=modality_mask)

    def synthetic_features(
        self,
        batch_size: int = 2,
        steps: int = 50,
        *,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Create synthetic inputs matching configured encoder dimensions."""
        device = torch.device(device) if device is not None else next(self.parameters()).device
        return {
            name: torch.randn(batch_size, steps, dim, device=device)
            for name, dim in self.encoders.input_dims.items()
        }
