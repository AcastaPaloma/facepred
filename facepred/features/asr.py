"""ASR transcript wrapper with optional Whisper backend.

Whisper is imported and loaded lazily. When the backend is unavailable, callers
still receive a well-formed silent transcript segment with confidence ``0.0``
and, when requested, a zero embedding placeholder matching the text encoder
config dimension.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

ASR_EMBEDDING_DIM = 384


@dataclass
class TranscriptSegment:
    """A timestamped ASR hypothesis.

    Attributes:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        text: Transcript text for the segment.
        confidence: Heuristic confidence in ``[0, 1]``.
        embedding: Optional placeholder or text embedding of shape ``[384]``.
    """

    start: float
    end: float
    text: str
    confidence: float
    embedding: torch.Tensor | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable segment dictionary."""
        data: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "confidence": self.confidence,
        }
        if self.avg_logprob is not None:
            data["avg_logprob"] = self.avg_logprob
        if self.no_speech_prob is not None:
            data["no_speech_prob"] = self.no_speech_prob
        if self.embedding is not None:
            data["embedding"] = self.embedding.detach().cpu().tolist()
        return data


class ASRExtractor:
    """Transcribe audio with Whisper when available, otherwise return silence.

    Args:
        model: Whisper model size/name, e.g. ``"small"``.
        language: Optional transcription language hint.
        streaming: Reserved flag for streaming callers.
        chunk_length_s: Reserved chunk size for future streaming extraction.
        sample_rate: Whisper input sample rate.
        embedding_dim: Placeholder embedding dimension for text encoder inputs.
    """

    def __init__(
        self,
        model: str = "small",
        language: str | None = "en",
        streaming: bool = True,
        chunk_length_s: float = 5.0,
        sample_rate: int = 16000,
        embedding_dim: int = ASR_EMBEDDING_DIM,
    ) -> None:
        self.model_name = model
        self.language = language
        self.streaming = streaming
        self.chunk_length_s = chunk_length_s
        self.sample_rate = sample_rate
        self.embedding_dim = embedding_dim

        self._model = None
        self._backend_attempted = False
        self._backend_error: Exception | None = None

    def _init_whisper(self) -> bool:
        """Lazy-load Whisper on first real transcription request."""
        if self._model is not None:
            return True
        if self._backend_attempted:
            return False

        self._backend_attempted = True
        try:
            import whisper

            self._model = whisper.load_model(self.model_name)
            logger.info("Initialized Whisper ASR backend: %s", self.model_name)
            return True
        except Exception as exc:  # pragma: no cover - optional backend surface
            self._backend_error = exc
            logger.warning("Whisper ASR backend unavailable; using silent fallback (%s)", exc)
            return False

    def transcribe(
        self,
        waveform: np.ndarray | torch.Tensor,
        sample_rate: int = 16000,
        include_embeddings: bool = False,
        use_model: bool = True,
    ) -> list[TranscriptSegment]:
        """Return transcript segments with confidence scores."""
        audio = self._to_mono_tensor(waveform)
        duration_s = self._duration_seconds(audio, sample_rate)
        if audio.numel() == 0:
            return []

        if use_model and self._init_whisper():
            try:
                audio_np = self._prepare_whisper_audio(audio, sample_rate)
                result = self._model.transcribe(
                    audio_np,
                    language=self.language,
                    fp16=False,
                )
                return self._parse_result(result, include_embeddings, duration_s)
            except Exception as exc:  # pragma: no cover - optional backend surface
                logger.warning("Whisper transcription failed; using fallback (%s)", exc)

        return self._fallback_segments(duration_s, include_embeddings)

    def transcribe_file(
        self,
        audio_path: str | Path,
        include_embeddings: bool = False,
        use_model: bool = True,
    ) -> list[TranscriptSegment]:
        """Transcribe an audio file into timestamped segments."""
        if use_model and self._init_whisper():
            try:
                result = self._model.transcribe(
                    str(audio_path),
                    language=self.language,
                    fp16=False,
                )
                return self._parse_result(result, include_embeddings, duration_s=0.0)
            except Exception as exc:  # pragma: no cover - optional backend surface
                logger.warning("Whisper file transcription failed; using fallback (%s)", exc)

        try:
            waveform, sample_rate = self._load_audio_file(audio_path)
            duration_s = self._duration_seconds(self._to_mono_tensor(waveform), sample_rate)
        except Exception as exc:
            logger.warning("Could not load audio for ASR fallback (%s)", exc)
            duration_s = 0.0
        return self._fallback_segments(duration_s, include_embeddings)

    def segment_embeddings(self, segments: list[TranscriptSegment]) -> torch.Tensor:
        """Stack segment embeddings, using zeros where a segment has no embedding."""
        if not segments:
            return self.get_zeros(1)

        embeddings = []
        for segment in segments:
            if segment.embedding is None:
                embeddings.append(torch.zeros(self.embedding_dim, dtype=torch.float32))
            else:
                embeddings.append(self._fit_embedding_dim(segment.embedding))
        return torch.stack(embeddings, dim=0)

    def mean_confidence(self, segments: list[TranscriptSegment]) -> float:
        """Return duration-weighted mean confidence for quality features."""
        if not segments:
            return 0.0

        total_duration = sum(max(0.0, segment.end - segment.start) for segment in segments)
        if total_duration <= 0.0:
            return float(np.mean([segment.confidence for segment in segments]))

        weighted = sum(
            segment.confidence * max(0.0, segment.end - segment.start) for segment in segments
        )
        return float(weighted / total_duration)

    def _parse_result(
        self,
        result: dict[str, Any],
        include_embeddings: bool,
        duration_s: float,
    ) -> list[TranscriptSegment]:
        parsed = [
            self._segment_from_whisper(segment, include_embeddings)
            for segment in result.get("segments", [])
        ]
        if parsed:
            return parsed

        text = str(result.get("text", "")).strip()
        if text:
            return [
                TranscriptSegment(
                    start=0.0,
                    end=duration_s,
                    text=text,
                    confidence=0.0,
                    embedding=self._embedding_placeholder() if include_embeddings else None,
                )
            ]
        return self._fallback_segments(duration_s, include_embeddings)

    def _segment_from_whisper(
        self,
        segment: dict[str, Any],
        include_embeddings: bool,
    ) -> TranscriptSegment:
        avg_logprob = self._optional_float(segment.get("avg_logprob"))
        no_speech_prob = self._optional_float(segment.get("no_speech_prob"))
        confidence = self._confidence_from_whisper(segment)
        return TranscriptSegment(
            start=float(segment.get("start", 0.0)),
            end=float(segment.get("end", 0.0)),
            text=str(segment.get("text", "")).strip(),
            confidence=confidence,
            embedding=self._embedding_placeholder() if include_embeddings else None,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
        )

    def _confidence_from_whisper(self, segment: dict[str, Any]) -> float:
        words = segment.get("words") or []
        word_probs = [
            float(word["probability"])
            for word in words
            if isinstance(word, dict) and word.get("probability") is not None
        ]
        if word_probs:
            return float(np.clip(np.mean(word_probs), 0.0, 1.0))

        avg_logprob = self._optional_float(segment.get("avg_logprob"))
        no_speech_prob = self._optional_float(segment.get("no_speech_prob")) or 0.0
        if avg_logprob is None:
            return float(np.clip(1.0 - no_speech_prob, 0.0, 1.0))

        confidence = math.exp(max(-20.0, min(0.0, avg_logprob)))
        confidence *= 1.0 - np.clip(no_speech_prob, 0.0, 1.0)
        return float(np.clip(confidence, 0.0, 1.0))

    def _fallback_segments(
        self,
        duration_s: float,
        include_embeddings: bool,
    ) -> list[TranscriptSegment]:
        if duration_s <= 0.0:
            return []
        return [
            TranscriptSegment(
                start=0.0,
                end=duration_s,
                text="",
                confidence=0.0,
                embedding=self._embedding_placeholder() if include_embeddings else None,
                avg_logprob=None,
                no_speech_prob=1.0,
            )
        ]

    def _prepare_whisper_audio(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        if sample_rate != self.sample_rate:
            waveform = self._resample(waveform, sample_rate, self.sample_rate)
        return waveform.detach().cpu().numpy().astype(np.float32)

    def _embedding_placeholder(self) -> torch.Tensor:
        return torch.zeros(self.embedding_dim, dtype=torch.float32)

    def _fit_embedding_dim(self, embedding: torch.Tensor) -> torch.Tensor:
        embedding = embedding.detach().float().cpu().flatten()
        if embedding.numel() == self.embedding_dim:
            return embedding
        if embedding.numel() > self.embedding_dim:
            return embedding[: self.embedding_dim]
        return torch.nn.functional.pad(embedding, (0, self.embedding_dim - embedding.numel()))

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _duration_seconds(waveform: torch.Tensor, sample_rate: int) -> float:
        if sample_rate <= 0:
            return 0.0
        return float(waveform.numel() / sample_rate)

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
        """Return ASR/text embedding feature dimension."""
        return self.embedding_dim

    def get_zeros(self, num_frames: int = 1) -> torch.Tensor:
        """Return zero text embeddings with shape ``[num_frames, 384]`` by default."""
        return torch.zeros(num_frames, self.embedding_dim, dtype=torch.float32)
