"""Small timing primitives for training loops and smoke tests."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Timer:
    """A reusable monotonic stopwatch.

    ``Timer`` starts immediately by default, can be used as a context manager,
    and accumulates elapsed time across start/stop cycles until reset.
    """

    name: str | None = None
    auto_start: bool = True
    clock: Callable[[], float] = field(default=time.perf_counter, repr=False)
    _started_at: float | None = field(default=None, init=False, repr=False)
    _elapsed: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.auto_start:
            self.start()

    def __enter__(self) -> Timer:
        if not self.running:
            self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        """Whether the timer is currently running."""
        return self._started_at is not None

    @property
    def elapsed(self) -> float:
        """Elapsed seconds, including the current running interval."""
        if self._started_at is None:
            return self._elapsed
        return self._elapsed + (self.clock() - self._started_at)

    def start(self) -> Timer:
        """Start or resume the timer."""
        if self._started_at is None:
            self._started_at = self.clock()
        return self

    def stop(self) -> float:
        """Stop the timer and return total elapsed seconds."""
        if self._started_at is not None:
            now = self.clock()
            self._elapsed += now - self._started_at
            self._started_at = None
        return self._elapsed

    def reset(self, *, start: bool | None = None) -> Timer:
        """Reset elapsed time and optionally start immediately."""
        should_start = self.auto_start if start is None else start
        self._elapsed = 0.0
        self._started_at = None
        if should_start:
            self.start()
        return self

    def lap(self) -> float:
        """Return elapsed seconds and restart the timer from zero."""
        elapsed = self.elapsed
        self.reset(start=self.running)
        return elapsed


@contextmanager
def time_block(name: str | None = None) -> Iterator[Timer]:
    """Context manager that yields a stopped ``Timer`` after the block exits."""
    timer = Timer(name=name)
    try:
        yield timer
    finally:
        timer.stop()


def format_seconds(seconds: float, *, precision: int = 3) -> str:
    """Format seconds using readable units from microseconds to hours."""
    if seconds < 0:
        raise ValueError("Duration cannot be negative.")
    if seconds < 1e-3:
        return f"{seconds * 1_000_000:.{precision}f}us"
    if seconds < 1:
        return f"{seconds * 1_000:.{precision}f}ms"
    if seconds < 60:
        return f"{seconds:.{precision}f}s"
    if seconds < 3600:
        minutes, remainder = divmod(seconds, 60)
        return f"{int(minutes)}m {remainder:.{precision}f}s"
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {remainder:.{precision}f}s"
