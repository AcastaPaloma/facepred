"""Create lightweight synthetic feature tensors for scaffold development."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facepred.engine.trainer import DEFAULT_MODALITY_KEYS, load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="Project config YAML.")
    parser.add_argument("--output", default=None, help="Optional .pt file for generated features.")
    parser.add_argument("--num-sequences", type=int, default=2, help="Synthetic sequence count.")
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length in world-model steps.")
    parser.add_argument("--modalities", default="all", help="Comma-separated modalities or 'all'.")
    parser.add_argument("--zeros", action="store_true", help="Generate zeros instead of random features.")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic data seed.")
    return parser.parse_args()


def modality_dims(config: Mapping[str, Any]) -> dict[str, int]:
    model_cfg = config.get("model", config)
    encoders = model_cfg.get("encoders", {}) if isinstance(model_cfg, Mapping) else {}
    dims: dict[str, int] = {}
    if isinstance(encoders, Mapping):
        for key, value in encoders.items():
            if isinstance(value, Mapping):
                dims[str(key)] = int(value.get("input_dim", value.get("output_dim", 0)) or 0)
    return {key: dim for key, dim in dims.items() if dim > 0}


def selected_modalities(requested: str, dims: Mapping[str, int]) -> list[str]:
    if requested == "all":
        return [key for key in DEFAULT_MODALITY_KEYS if key in dims]
    return [item.strip() for item in requested.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    config = load_project_config(args.config)
    dims = modality_dims(config)
    if not dims:
        dims = {"audio_prosody": 88, "vad": 3, "quality": 4}

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    features: dict[str, torch.Tensor] = {}
    for modality in selected_modalities(args.modalities, dims):
        dim = dims.get(modality)
        if dim is None:
            raise KeyError(f"Unknown modality '{modality}'. Available: {sorted(dims)}")
        shape = (args.num_sequences, args.seq_len, dim)
        features[modality] = torch.zeros(shape) if args.zeros else torch.randn(shape, generator=generator)

    concatenated = torch.cat([features[key] for key in features], dim=-1)
    payload = {
        "features": features,
        "concatenated": concatenated,
        "metadata": {
            "num_sequences": args.num_sequences,
            "seq_len": args.seq_len,
            "modalities": list(features),
            "dims": {key: int(value.shape[-1]) for key, value in features.items()},
        },
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)

    summary = dict(payload["metadata"])
    summary["concatenated_dim"] = int(concatenated.shape[-1])
    summary["output"] = args.output
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

