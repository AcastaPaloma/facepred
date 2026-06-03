"""Quality signal estimation for modality reliability gating.

Computes per-modality quality/confidence scores that tell the fusion layer
how much to trust each modality at each timestep. Signals include:
- Audio SNR estimate
- Face detection confidence
- ASR confidence score
- Missing modality binary mask
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

QUALITY_DIM = 4  # SNR, face_confidence, ASR_confidence, missing_mask


class QualityEstimator:
    """Estimate quality signals for modality reliability gating.

    Produces a [4]-dimensional quality vector per timestep:
        [0] audio_snr_normalized: Estimated SNR in [0, 1] range
        [1] face_confidence: Face detection confidence from MediaPipe
        [2] asr_confidence: ASR hypothesis confidence from Whisper
        [3] modality_completeness: Fraction of modalities present (0-1)

    Example:
        >>> estimator = QualityEstimator()
        >>> quality = estimator.compute(
        ...     audio=waveform,
        ...     face_confidence=0.95,
        ...     asr_confidence=0.8,
        ...     available_modalities=['audio', 'visual', 'text'],
        ... )
        >>> quality.shape  # [4]
    """

    def __init__(self, total_modalities: int = 5) -> None:
        self.total_modalities = total_modalities

    def estimate_snr(self, waveform: torch.Tensor) -> float:
        """Estimate Signal-to-Noise Ratio from audio waveform.

        Uses a simple energy-based heuristic: ratio of top-percentile energy
        to bottom-percentile energy. Not a proper SNR estimator but sufficient
        as a quality signal for reliability gating.

        Args:
            waveform: Audio signal [num_samples].

        Returns:
            Normalized SNR estimate in [0, 1] range.
        """
        if waveform.numel() == 0:
            return 0.0

        energy = waveform.float() ** 2

        # Use percentile-based estimate
        sorted_energy, _ = energy.sort()
        n = len(sorted_energy)

        # Bottom 10% as "noise floor"
        noise_floor = sorted_energy[: max(1, n // 10)].mean().item()
        # Top 50% as "signal"
        signal_level = sorted_energy[n // 2 :].mean().item()

        if noise_floor < 1e-10:
            return 1.0  # Very clean signal

        snr_linear = signal_level / (noise_floor + 1e-10)
        snr_db = 10 * np.log10(max(snr_linear, 1e-10))

        # Normalize: 0dB → 0.0, 40dB+ → 1.0
        snr_normalized = float(np.clip(snr_db / 40.0, 0.0, 1.0))
        return snr_normalized

    def compute(
        self,
        audio: Optional[torch.Tensor] = None,
        face_confidence: float = 0.0,
        asr_confidence: float = 0.0,
        available_modalities: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Compute quality signals for a single timestep.

        Args:
            audio: Audio waveform for SNR estimation. None if audio unavailable.
            face_confidence: Face detection confidence from visual extractor (0-1).
            asr_confidence: ASR hypothesis confidence from Whisper (0-1).
            available_modalities: List of modality names that are available.

        Returns:
            Quality tensor of shape [4].
        """
        # SNR
        snr = self.estimate_snr(audio) if audio is not None else 0.0

        # Modality completeness
        num_available = len(available_modalities) if available_modalities else 0
        completeness = num_available / self.total_modalities

        quality = torch.tensor(
            [snr, face_confidence, asr_confidence, completeness],
            dtype=torch.float32,
        )
        return quality

    def compute_batch(
        self,
        snr_values: torch.Tensor,
        face_confidences: torch.Tensor,
        asr_confidences: torch.Tensor,
        modality_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute quality signals for a batch of timesteps.

        Args:
            snr_values: [batch, seq_len] SNR estimates.
            face_confidences: [batch, seq_len] face detection confidences.
            asr_confidences: [batch, seq_len] ASR confidences.
            modality_masks: [batch, seq_len, num_modalities] binary mask.

        Returns:
            Quality tensor [batch, seq_len, 4].
        """
        completeness = modality_masks.float().mean(dim=-1)  # [batch, seq_len]

        quality = torch.stack(
            [snr_values, face_confidences, asr_confidences, completeness],
            dim=-1,
        )  # [batch, seq_len, 4]

        return quality

    def get_feature_dim(self) -> int:
        """Return quality feature dimension."""
        return QUALITY_DIM

    def get_zeros(self, num_frames: int = 1) -> torch.Tensor:
        """Return zero quality features [num_frames, 4]."""
        return torch.zeros(num_frames, QUALITY_DIM, dtype=torch.float32)
