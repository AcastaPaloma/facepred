"""Audio prosody feature extraction using openSMILE.

Extracts eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set) features
from audio signals. These 88-dimensional features capture fundamental frequency,
energy, spectral, and cepstral characteristics relevant to prosody and paralinguistics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


# eGeMAPS feature dimension by level
FEATURE_DIMS = {
    "eGeMAPSv02": {"functionals": 88, "lld": 25},
    "ComParE_2016": {"functionals": 6373, "lld": 65},
    "IS09_emotion": {"functionals": 384, "lld": 32},
}


class ProsodyExtractor:
    """Extract prosodic features from audio using openSMILE.

    Supports eGeMAPSv02 (default, 88-dim), ComParE_2016 (6373-dim),
    and IS09_emotion (384-dim) feature sets.

    Args:
        feature_set: Name of the openSMILE feature set.
        feature_level: 'functionals' for utterance-level or 'lld' for frame-level.
        sample_rate: Expected input sample rate in Hz.

    Example:
        >>> extractor = ProsodyExtractor()
        >>> features = extractor.extract(waveform, sample_rate=16000)
        >>> features.shape  # [num_frames, 88] for eGeMAPSv02 functionals
    """

    def __init__(
        self,
        feature_set: str = "eGeMAPSv02",
        feature_level: str = "functionals",
        sample_rate: int = 16000,
    ) -> None:
        self.feature_set = feature_set
        self.feature_level = feature_level
        self.sample_rate = sample_rate
        self._smile = None

        if feature_set not in FEATURE_DIMS:
            raise ValueError(
                f"Unknown feature_set '{feature_set}'. "
                f"Choose from: {list(FEATURE_DIMS.keys())}"
            )

        self.feature_dim = FEATURE_DIMS[feature_set][feature_level]
        logger.info(
            f"ProsodyExtractor initialized: {feature_set}/{feature_level} "
            f"({self.feature_dim}-dim)"
        )

    def _init_smile(self) -> None:
        """Lazy-initialize openSMILE to avoid import overhead."""
        if self._smile is not None:
            return

        try:
            import opensmile

            self._smile = opensmile.Smile(
                feature_set=getattr(opensmile.FeatureSet, self.feature_set),
                feature_level=getattr(opensmile.FeatureLevel, self.feature_level),
            )
        except ImportError as exc:
            raise ImportError(
                "opensmile is required for prosody extraction. "
                "Install with: pip install opensmile"
            ) from exc

    def extract(
        self,
        waveform: np.ndarray | torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Extract prosody features from an audio waveform.

        Args:
            waveform: Audio signal, shape [num_samples] or [1, num_samples].
            sample_rate: Sample rate of the input waveform.

        Returns:
            Feature tensor of shape [num_frames, feature_dim].
            For functionals, num_frames=1. For LLD, num_frames depends on audio length.
        """
        self._init_smile()

        # Convert torch tensor to numpy
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()

        # Ensure 1D
        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)

        # Extract features via openSMILE
        features_df = self._smile.process_signal(waveform, sample_rate)
        features = torch.tensor(features_df.values, dtype=torch.float32)

        return features

    def extract_file(self, audio_path: str | Path) -> torch.Tensor:
        """Extract prosody features from an audio file.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.).

        Returns:
            Feature tensor of shape [num_frames, feature_dim].
        """
        self._init_smile()

        features_df = self._smile.process_file(str(audio_path))
        features = torch.tensor(features_df.values, dtype=torch.float32)

        return features

    def extract_windowed(
        self,
        waveform: np.ndarray | torch.Tensor,
        sample_rate: int = 16000,
        window_ms: int = 100,
        hop_ms: int = 100,
    ) -> torch.Tensor:
        """Extract prosody features in sliding windows for real-time use.

        Splits the audio into overlapping windows and extracts features
        for each window, producing a time series of feature vectors aligned
        to the world model's step rate.

        Args:
            waveform: Audio signal, shape [num_samples].
            sample_rate: Sample rate of input.
            window_ms: Window size in milliseconds.
            hop_ms: Hop size in milliseconds.

        Returns:
            Feature tensor of shape [num_windows, feature_dim].
        """
        self._init_smile()

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)

        window_samples = int(sample_rate * window_ms / 1000)
        hop_samples = int(sample_rate * hop_ms / 1000)

        frames = []
        for start in range(0, len(waveform) - window_samples + 1, hop_samples):
            chunk = waveform[start : start + window_samples]
            try:
                feat_df = self._smile.process_signal(chunk, sample_rate)
                frames.append(torch.tensor(feat_df.values, dtype=torch.float32).squeeze(0))
            except Exception:
                # If extraction fails on a chunk, use zeros
                frames.append(torch.zeros(self.feature_dim, dtype=torch.float32))

        if not frames:
            return torch.zeros(1, self.feature_dim, dtype=torch.float32)

        return torch.stack(frames, dim=0)

    def get_feature_dim(self) -> int:
        """Return the feature dimension for the configured feature set."""
        return self.feature_dim

    def get_zeros(self, num_frames: int = 1) -> torch.Tensor:
        """Return zero features with the correct shape (for missing data)."""
        return torch.zeros(num_frames, self.feature_dim, dtype=torch.float32)
