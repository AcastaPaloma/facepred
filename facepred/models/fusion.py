"""Multimodal fusion with reliability gating."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


class ReliabilityGatedFusion(nn.Module):
    """Fuse modality embeddings into one vector per timestep."""

    def __init__(
        self,
        input_dims: Mapping[str, int],
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        fusion_type: str = "cross_attention",
        reliability: str = "learned",
        quality_dim: int = 4,
    ) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims must contain at least one modality")
        if fusion_type not in {"cross_attention", "concat", "perceiver"}:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
        if reliability not in {"learned", "heuristic"}:
            raise ValueError(f"Unsupported reliability mode: {reliability}")

        self.modality_names = list(input_dims.keys())
        self.input_dims = dict(input_dims)
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        self.reliability = reliability
        self.quality_dim = quality_dim

        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, embed_dim) for name, dim in self.input_dims.items()}
        )
        self.modality_embeddings = nn.Parameter(torch.zeros(len(self.modality_names), embed_dim))

        if fusion_type in {"cross_attention", "perceiver"}:
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.post = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.concat = nn.Sequential(
                nn.Linear(embed_dim * len(self.modality_names), embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )

        if reliability == "learned":
            self.gate_net = nn.Sequential(
                nn.Linear(quality_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, len(self.modality_names)),
            )
        else:
            self.gate_net = None

    def forward(
        self,
        modalities: Mapping[str, torch.Tensor],
        quality: torch.Tensor | None = None,
        modality_mask: Mapping[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused features and reliability weights."""
        batch, steps, device, dtype = self._infer_shape(modalities)
        tokens: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        for idx, name in enumerate(self.modality_names):
            if name in modalities:
                tensor = modalities[name]
                if tensor.ndim != 3:
                    raise ValueError(f"{name} must be [batch, steps, dim], got {tuple(tensor.shape)}")
                projected = self.projections[name](tensor.float())
                present = torch.ones(batch, steps, device=tensor.device, dtype=torch.bool)
            else:
                projected = torch.zeros(batch, steps, self.embed_dim, device=device, dtype=dtype)
                present = torch.zeros(batch, steps, device=device, dtype=torch.bool)

            projected = projected + self.modality_embeddings[idx].view(1, 1, -1)
            tokens.append(projected)
            masks.append(self._mask_for(name, idx, modality_mask, present, device))

        token_tensor = torch.stack(tokens, dim=2)
        availability = torch.stack(masks, dim=-1).float()
        gates = self._reliability_gates(quality, availability)
        gated_tokens = token_tensor * gates.unsqueeze(-1)

        if self.fusion_type == "concat":
            fused = self.concat(gated_tokens.flatten(start_dim=2))
        else:
            flat_tokens = gated_tokens.reshape(batch * steps, len(self.modality_names), self.embed_dim)
            key_padding_mask = availability.reshape(batch * steps, len(self.modality_names)) <= 0
            all_missing = key_padding_mask.all(dim=-1)
            key_padding_mask = key_padding_mask.masked_fill(all_missing.unsqueeze(-1), False)
            attended, _ = self.attention(
                flat_tokens,
                flat_tokens,
                flat_tokens,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            attended = attended.reshape(batch, steps, len(self.modality_names), self.embed_dim)
            denom = gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            fused = (attended * gates.unsqueeze(-1)).sum(dim=2) / denom
            fused = fused + self.post(fused)

        return fused, gates

    def _infer_shape(
        self, modalities: Mapping[str, torch.Tensor]
    ) -> tuple[int, int, torch.device, torch.dtype]:
        for tensor in modalities.values():
            if tensor.ndim != 3:
                raise ValueError(f"Modality tensors must be [batch, steps, dim], got {tensor.shape}")
            return tensor.shape[0], tensor.shape[1], tensor.device, tensor.dtype
        raise ValueError("At least one modality tensor is required")

    def _mask_for(
        self,
        name: str,
        idx: int,
        modality_mask: Mapping[str, torch.Tensor] | torch.Tensor | None,
        default: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if modality_mask is None:
            return default
        if isinstance(modality_mask, Mapping):
            mask = modality_mask.get(name, default)
        else:
            mask = modality_mask[..., idx]
        return mask.to(device=device, dtype=torch.bool)

    def _reliability_gates(self, quality: torch.Tensor | None, availability: torch.Tensor) -> torch.Tensor:
        if self.reliability == "learned":
            if quality is None:
                logits = torch.zeros(
                    *availability.shape[:2],
                    len(self.modality_names),
                    device=availability.device,
                    dtype=availability.dtype,
                )
            else:
                logits = self.gate_net(quality.float())
            gates = torch.sigmoid(logits) * availability
        else:
            gates = self._heuristic_gates(quality, availability)

        fallback = availability / availability.sum(dim=-1, keepdim=True).clamp_min(1.0)
        gates = torch.where(gates.sum(dim=-1, keepdim=True) > 0, gates, fallback)
        return gates

    def _heuristic_gates(self, quality: torch.Tensor | None, availability: torch.Tensor) -> torch.Tensor:
        if quality is None:
            return availability

        snr = quality[..., 0:1].clamp(0, 1)
        face = quality[..., 1:2].clamp(0, 1)
        asr = quality[..., 2:3].clamp(0, 1)
        complete = quality[..., 3:4].clamp(0, 1)
        values: list[torch.Tensor] = []
        for name in self.modality_names:
            if name.startswith("audio") or name == "vad":
                values.append(snr)
            elif name == "visual":
                values.append(face)
            elif name == "text":
                values.append(asr)
            else:
                values.append(complete)
        return torch.cat(values, dim=-1) * availability
