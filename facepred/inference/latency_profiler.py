"""Small latency profiler for inference smoke tests and demos."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable, Iterator


@dataclass(slots=True)
class LatencySample:
    """Single timing sample in milliseconds."""

    name: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class LatencySummary:
    """Aggregate latency statistics for one timed block."""

    name: str
    count: int
    mean_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


class LatencyProfiler:
    """Collect simple per-stage timings with a context-manager API."""

    def __init__(self) -> None:
        self.samples: list[LatencySample] = []

    @contextmanager
    def time_block(self, name: str, **metadata: Any) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1000.0, **metadata)

    def record(self, name: str, duration_ms: float, **metadata: Any) -> LatencySample:
        sample = LatencySample(name=name, duration_ms=float(duration_ms), metadata=dict(metadata))
        self.samples.append(sample)
        return sample

    def profile_callable(self, name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self.time_block(name):
            return func(*args, **kwargs)

    def summary(self, name: str | None = None) -> list[LatencySummary]:
        selected = [sample for sample in self.samples if name is None or sample.name == name]
        by_name: dict[str, list[float]] = {}
        for sample in selected:
            by_name.setdefault(sample.name, []).append(sample.duration_ms)

        summaries = []
        for sample_name, durations in sorted(by_name.items()):
            summaries.append(
                LatencySummary(
                    name=sample_name,
                    count=len(durations),
                    mean_ms=float(mean(durations)),
                    min_ms=float(min(durations)),
                    max_ms=float(max(durations)),
                )
            )
        return summaries

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": [sample.as_dict() for sample in self.samples],
            "summary": [summary.as_dict() for summary in self.summary()],
            "total_ms": self.total_ms(),
        }

    def total_ms(self) -> float:
        return float(sum(sample.duration_ms for sample in self.samples))

    def reset(self) -> None:
        self.samples.clear()

