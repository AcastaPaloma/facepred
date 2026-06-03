"""Deterministic seeding helpers for reproducible smoke tests and training."""

from __future__ import annotations

import os
import random
from typing import Any

DEFAULT_SEED = 42
MAX_NUMPY_SEED = 2**32 - 1


def resolve_seed(
    seed: int | str | None = None,
    *,
    env_var: str = "FACEPRED_SEED",
    default: int = DEFAULT_SEED,
) -> int:
    """Resolve and validate a seed from an explicit value, environment, or default."""
    raw_seed: int | str | None = seed
    if raw_seed is None:
        raw_seed = os.environ.get(env_var, default)

    try:
        resolved = int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Seed must be an integer-compatible value, got {raw_seed!r}.") from exc

    if not 0 <= resolved <= MAX_NUMPY_SEED:
        raise ValueError(f"Seed must be in [0, {MAX_NUMPY_SEED}], got {resolved}.")
    return resolved


def seed_everything(
    seed: int | str | None = None,
    *,
    deterministic: bool = True,
    env_var: str = "FACEPRED_SEED",
) -> int:
    """Seed Python, NumPy, and PyTorch when available.

    Returns the resolved seed so callers can log the exact value used.
    """
    resolved = resolve_seed(seed, env_var=env_var)
    os.environ["PYTHONHASHSEED"] = str(resolved)

    random.seed(resolved)

    try:
        import numpy as np

        np.random.seed(resolved)
    except ImportError:
        pass

    try:
        import torch
    except ImportError:
        return resolved

    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    return resolved


def seed_worker(worker_id: int, base_seed: int | str | None = None) -> int:
    """Seed a dataloader worker's Python and NumPy RNGs.

    When ``base_seed`` is omitted and PyTorch is installed, this follows the
    standard ``torch.initial_seed() % 2**32`` pattern.
    """
    if base_seed is None:
        try:
            import torch

            base_seed = torch.initial_seed()
        except ImportError:
            base_seed = DEFAULT_SEED

    try:
        base_seed_int = int(base_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Base seed must be integer-compatible, got {base_seed!r}.") from exc

    worker_seed = (base_seed_int + int(worker_id)) % (MAX_NUMPY_SEED + 1)
    random.seed(worker_seed)

    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass

    return worker_seed


def make_torch_generator(
    seed: int | str | None = None,
    *,
    device: str | Any | None = None,
) -> Any:
    """Create a seeded ``torch.Generator`` without importing torch at module import time."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("make_torch_generator requires PyTorch to be installed.") from exc

    generator = torch.Generator(device=device) if device is not None else torch.Generator()
    generator.manual_seed(resolve_seed(seed))
    return generator
