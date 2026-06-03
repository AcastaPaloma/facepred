"""Voice activity features with an optional pyannote backend.

The public contract is intentionally small for Iteration 0: return
``[speech_prob, overlap_prob, speaker_id_or_activity]`` per frame, matching the
model config's VAD input dimension of 3. The pyannote dependency is imported
only when extraction is requested; if it is unavailable or fails to initialize,
the extractor falls back to a deterministic energy-based activity estimate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

VAD_FEATURE_DIM = 3


class VADExtractor:
    """Extract lightweight voice activity features.

    Args:
        model: pyannote model identifier used when the optional backend is available.
        min_duration_on: Reserved for pyannote-style VAD post-processing.
        min_duration_off: Reserved for pyannote-style VAD post-processing.
        sample_rate: Target sample rate for model-backed extraction.
        frame_ms: Frame duration for the synthetic fallback.
        fallback: ``"synthetic"`` for energy VAD or ``"zeros"``.
    """

    def __init__(
        self,
        model: str = "pyannote/segmentation-3.0",
        min_duration_on: float = 0.1,
        min_duration_off: float = 0.1,
        sample_rate: int = 16000,
        frame_ms: int = 100,
        fallback: str = "synthetic",
    ) -> None:
        self.model_name = model
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.fallback = fallback

        self._inference = None
        self._backend_attempted = False
        self._backend_error: Exception | None = None

    def _init_pyannote(self) -> bool:
        """Lazy-initialize pyannote segmentation inference if available."""
        if self._inference is not None:
            return True
        if self._backend_attempted:
            return False

        self._backend_attempted = True
        try:
            from pyannote.audio import Inference, Model

            model = Model.from_pretrained(self.model_name)
            self._inference = Inference(model)
            logger.info("Initialized pyannote VAD backend: %s", self.model_name)
            return True
        except Exception as exc:  # pragma: no cover - optional backend surface
            self._backend_error = exc
            logger.warning(
                "pyannote VAD backend unavailable; using %s fallback (%s)",
                self.fallback,
                exc,
            )
            return False

    def extract(
        self,
        waveform: np.ndarray | torch.Tensor,
        sample_rate: int = 16000,
        num_frames: int | None = None,
        use_model: bool = True,
    ) -> torch.Tensor:
        """Extract VAD features with shape ``[num_frames, 3]``.

        The final channel is a normalized speaker id when a multi-speaker backend
        exposes local speaker tracks; otherwise it is the same continuous speech
        activity estimate as ``speech_prob``.
        """
        audio = self._to_mono_tensor(waveform)
        if audio.numel() == 0:
            return self.get_zeros(num_frames or 1)

        if use_model and self._init_pyannote():
            try:
                features = self._extract_with_pyannote(audio, sample_rate)
                return self._match_num_frames(features, num_frames)
            except Exception as exc:  # pragma: no cover - optional backend surface
                logger.warning("pyannote VAD extraction failed; using fallback (%s)", exc)

        return self._extract_fallback(audio, sample_rate, num_frames)

    def extract_file(
        self,
        audio_path: str | Path,
        num_frames: int | None = None,
        use_model: bool = True,
    ) -> torch.Tensor:
        """Extract VAD features from an audio file."""
        try:
            waveform, sample_rate = self._load_audio_file(audio_path)
        except Exception as exc:
            logger.warning("Could not load audio for VAD; returning zeros (%s)", exc)
            return self.get_zeros(num_frames or 1)

        return self.extract(waveform, sample_rate, num_frames, use_model=use_model)

    def _extract_with_pyannote(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        file = {"waveform": waveform.reshape(1, -1), "sample_rate": sample_rate}
        result = self._inference(file)
        data = getattr(result, "data", result)
        scores = torch.as_tensor(np.asarray(data), dtype=torch.float32)

        while scores.ndim > 2:
            if scores.shape[0] == 1:
                scores = scores.squeeze(0)
            else:
                scores = scores.reshape(scores.shape[0], -1)
                break
        if scores.ndim == 0:
            scores = scores.reshape(1, 1)
        elif scores.ndim == 1:
            scores = scores.reshape(-1, 1)

        scores = self._as_probabilities(scores)
        speech_prob = scores.max(dim=-1).values

        if scores.shape[-1] > 1:
            sorted_scores = scores.sort(dim=-1, descending=True).values
            overlap_prob = sorted_scores[:, 1]
            speaker_id = scores.argmax(dim=-1).float() + 1.0
            speaker_activity = (speaker_id / float(scores.shape[-1])) * speech_prob
        else:
            overlap_prob = torch.zeros_like(speech_prob)
            speaker_activity = speech_prob

        return torch.stack([speech_prob, overlap_prob, speaker_activity], dim=-1).clamp(0.0, 1.0)

    def _extract_fallback(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        num_frames: int | None,
    ) -> torch.Tensor:
        if self.fallback == "zeros":
            if num_frames is None:
                num_frames = self._fallback_frame_count(waveform.numel(), sample_rate)
            return self.get_zeros(num_frames)

        frames = self._frame_waveform(waveform, sample_rate, num_frames)
        rms = torch.sqrt(torch.mean(frames.float() ** 2, dim=-1) + 1e-12)

        if torch.max(rms) <= 1e-8:
            speech_prob = torch.zeros_like(rms)
        else:
            noise_floor = torch.quantile(rms, 0.2)
            high_energy = torch.quantile(rms, 0.95)
            denom = torch.clamp(high_energy - noise_floor, min=1e-8)
            speech_prob = torch.clamp((rms - noise_floor) / denom, 0.0, 1.0)

        overlap_prob = torch.zeros_like(speech_prob)
        speaker_activity = speech_prob
        return torch.stack([speech_prob, overlap_prob, speaker_activity], dim=-1)

    def _frame_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        num_frames: int | None,
    ) -> torch.Tensor:
        frame_samples = max(1, int(sample_rate * self.frame_ms / 1000))
        if num_frames is None:
            num_frames = self._fallback_frame_count(waveform.numel(), sample_rate)

        target_samples = max(frame_samples, num_frames * frame_samples)
        if waveform.numel() < target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.numel()))
        else:
            waveform = waveform[:target_samples]

        return waveform.unfold(0, frame_samples, frame_samples)[:num_frames]

    def _fallback_frame_count(self, num_samples: int, sample_rate: int) -> int:
        frame_samples = max(1, int(sample_rate * self.frame_ms / 1000))
        return max(1, int(np.ceil(num_samples / frame_samples)))

    def _match_num_frames(self, features: torch.Tensor, num_frames: int | None) -> torch.Tensor:
        if num_frames is None or features.shape[0] == num_frames:
            return features
        if features.numel() == 0:
            return self.get_zeros(num_frames)

        resized = torch.nn.functional.interpolate(
            features.T.unsqueeze(0),
            size=num_frames,
            mode="linear",
            align_corners=False,
        )
        return resized.squeeze(0).T

    @staticmethod
    def _as_probabilities(scores: torch.Tensor) -> torch.Tensor:
        if torch.any(scores < 0.0) or torch.any(scores > 1.0):
            return torch.sigmoid(scores)
        return scores.clamp(0.0, 1.0)

    @staticmethod
    def _to_mono_tensor(waveform: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(waveform, torch.Tensor):
            audio = waveform.detach().float().cpu()
        else:
            audio = torch.as_tensor(waveform, dtype=torch.float32)

        if audio.ndim == 0:
            return audio.reshape(1)
        audio = audio.squeeze()
        if audio.ndim == 1:
            return audio
        if audio.shape[0] <= 8:
            return audio.mean(dim=0)
        if audio.shape[-1] <= 8:
            return audio.mean(dim=-1)
        return audio.reshape(-1)

    @staticmethod
    def _load_audio_file(audio_path: str | Path) -> tuple[torch.Tensor, int]:
        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(str(audio_path))
            return waveform, int(sample_rate)
        except Exception:
            import soundfile as sf

            audio, sample_rate = sf.read(str(audio_path), always_2d=False)
            return torch.as_tensor(audio, dtype=torch.float32), int(sample_rate)

    def get_feature_dim(self) -> int:
        """Return VAD feature dimension."""
        return VAD_FEATURE_DIM

    def get_zeros(self, num_frames: int = 1) -> torch.Tensor:
        """Return zero VAD features with shape ``[num_frames, 3]``."""
        return torch.zeros(num_frames, VAD_FEATURE_DIM, dtype=torch.float32)
