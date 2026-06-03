"""Audio SSL embedding extraction with an optional transformers backend.

The extractor targets wav2vec2-like features with shape ``[frames, 768]``.
``transformers`` is imported only when extraction is requested. If the backend
is missing or cannot be loaded, a deterministic synthetic fallback returns
padded frame-level audio statistics with the same shape.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

AUDIO_SSL_DIM = 768


class AudioSSLExtractor:
    """Extract wav2vec2-style audio embeddings.

    Args:
        model_name: Hugging Face model identifier.
        layer: Hidden-state layer to use when available; ``-1`` means last.
        use_fp16: Use half precision only on CUDA.
        chunk_length_s: Maximum chunk size for model-backed extraction.
        sample_rate: Target model sample rate.
        frame_ms: Synthetic fallback frame duration.
        fallback: ``"synthetic"`` for padded audio stats or ``"zeros"``.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        layer: int = -1,
        use_fp16: bool = True,
        chunk_length_s: float = 10.0,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        fallback: str = "synthetic",
    ) -> None:
        self.model_name = model_name
        self.layer = layer
        self.use_fp16 = use_fp16
        self.chunk_length_s = chunk_length_s
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.fallback = fallback
        self.feature_dim = AUDIO_SSL_DIM

        self._processor = None
        self._model = None
        self._device = torch.device("cpu")
        self._backend_attempted = False
        self._backend_error: Exception | None = None

    def _init_transformers(self) -> bool:
        """Lazy-load the Hugging Face processor/model."""
        if self._model is not None and self._processor is not None:
            return True
        if self._backend_attempted:
            return False

        self._backend_attempted = True
        try:
            from transformers import AutoModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            if self.use_fp16 and self._device.type == "cuda":
                self._model.half()
            self._model.eval()
            logger.info("Initialized audio SSL backend: %s", self.model_name)
            return True
        except Exception as exc:  # pragma: no cover - optional backend surface
            self._backend_error = exc
            logger.warning(
                "Audio SSL backend unavailable; using %s fallback (%s)",
                self.fallback,
                exc,
            )
            return False

    def extract(
        self,
        waveform: np.ndarray | torch.Tensor,
        sample_rate: int = 16000,
        use_model: bool = True,
    ) -> torch.Tensor:
        """Extract embeddings with shape ``[frames, 768]``."""
        audio = self._to_mono_tensor(waveform)
        if audio.numel() == 0:
            return self.get_zeros(1)

        if sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)
            sample_rate = self.sample_rate

        if use_model and self._init_transformers():
            try:
                return self._extract_with_transformers(audio, sample_rate)
            except Exception as exc:  # pragma: no cover - optional backend surface
                logger.warning("Audio SSL extraction failed; using fallback (%s)", exc)

        return self._extract_fallback(audio, sample_rate)

    def extract_file(
        self,
        audio_path: str | Path,
        use_model: bool = True,
    ) -> torch.Tensor:
        """Extract SSL features from an audio file."""
        try:
            waveform, sample_rate = self._load_audio_file(audio_path)
        except Exception as exc:
            logger.warning("Could not load audio for audio SSL; returning zeros (%s)", exc)
            return self.get_zeros(1)

        return self.extract(waveform, sample_rate, use_model=use_model)

    def _extract_with_transformers(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        chunk_samples = max(1, int(self.chunk_length_s * sample_rate))
        chunks = []

        for start in range(0, waveform.numel(), chunk_samples):
            chunk = waveform[start : start + chunk_samples]
            if chunk.numel() == 0:
                continue

            inputs = self._processor(
                chunk.detach().cpu().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=False,
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True)

            if getattr(outputs, "hidden_states", None) is not None:
                hidden = outputs.hidden_states[self.layer]
            else:
                hidden = outputs.last_hidden_state
            chunks.append(hidden.squeeze(0).detach().float().cpu())

        if not chunks:
            return self.get_zeros(1)
        return self._fit_feature_dim(torch.cat(chunks, dim=0))

    def _extract_fallback(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        frame_count = self._fallback_frame_count(waveform.numel(), sample_rate)
        if self.fallback == "zeros":
            return self.get_zeros(frame_count)

        frames = self._frame_waveform(waveform, sample_rate, frame_count)
        stats = self._frame_stats(frames)
        features = torch.zeros(frame_count, self.feature_dim, dtype=torch.float32)
        features[:, : stats.shape[-1]] = stats
        return features

    def _frame_stats(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.float()
        rms = torch.sqrt(torch.mean(frames**2, dim=-1) + 1e-12)
        mean = frames.mean(dim=-1)
        std = frames.std(dim=-1, unbiased=False)
        max_abs = frames.abs().max(dim=-1).values
        min_value = frames.min(dim=-1).values
        max_value = frames.max(dim=-1).values
        if frames.shape[-1] > 1:
            zcr = (frames[:, 1:] * frames[:, :-1] < 0).float().mean(dim=-1)
        else:
            zcr = torch.zeros(frames.shape[0], dtype=torch.float32)

        stats = torch.stack([rms, mean, std, max_abs, min_value, max_value, zcr], dim=-1)
        normalizer = torch.clamp(stats.abs().amax(dim=0, keepdim=True), min=1e-8)
        return torch.clamp(stats / normalizer, -1.0, 1.0)

    def _frame_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        frame_count: int,
    ) -> torch.Tensor:
        frame_samples = max(1, int(sample_rate * self.frame_ms / 1000))
        target_samples = max(frame_samples, frame_count * frame_samples)
        if waveform.numel() < target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.numel()))
        else:
            waveform = waveform[:target_samples]
        return waveform.unfold(0, frame_samples, frame_samples)[:frame_count]

    def _fallback_frame_count(self, num_samples: int, sample_rate: int) -> int:
        frame_samples = max(1, int(sample_rate * self.frame_ms / 1000))
        return max(1, int(np.ceil(num_samples / frame_samples)))

    def _fit_feature_dim(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] == self.feature_dim:
            return features
        if features.shape[-1] > self.feature_dim:
            return features[:, : self.feature_dim]
        pad = self.feature_dim - features.shape[-1]
        return torch.nn.functional.pad(features, (0, pad))

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
    def _resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
        try:
            import torchaudio.functional as F

            return F.resample(waveform, source_rate, target_rate)
        except Exception:
            duration = waveform.numel() / max(1, source_rate)
            target_samples = max(1, int(round(duration * target_rate)))
            resized = torch.nn.functional.interpolate(
                waveform.reshape(1, 1, -1),
                size=target_samples,
                mode="linear",
                align_corners=False,
            )
            return resized.reshape(-1)

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
        """Return audio SSL embedding dimension."""
        return self.feature_dim

    def get_zeros(self, num_frames: int = 1) -> torch.Tensor:
        """Return zero SSL embeddings with shape ``[num_frames, 768]``."""
        return torch.zeros(num_frames, self.feature_dim, dtype=torch.float32)
