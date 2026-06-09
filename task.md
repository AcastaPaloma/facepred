# FacePred - Iteration 0 Build Tracker

## Phase 1: Foundation
- [x] `environment.yml` - conda setup scaffold
- [x] `.gitignore` - ML project ignores
- [x] `pyproject.toml` - package config
- [x] `README.md` - basic project overview
- [x] Directory structure + `__init__.py` files
- [x] Hydra configs (`config.yaml`, `model/`, `data/`, `feature/`, `training/`)

## Phase 2: Core Modules
- [x] Data module: `meld.py`, `label_derivation.py`, `synchronizer.py`, `augmentations.py`
- [x] Feature extraction: `audio_prosody.py`, `vad.py`, `asr.py`, `audio_ssl.py`, `visual.py`, `quality.py`
- [x] Models: `encoders.py`, `fusion.py`, `rssm.py`, `heads.py`, `losses.py`, `world_model.py`
- [x] Utils: `seeding.py`, `timing.py`, `logging_utils.py`, `calibration.py`

## Phase 3: Integration
- [x] Engine: `trainer.py`, `evaluator.py`, `precompute.py`
- [x] Scripts: `train.py`, `evaluate.py`, `extract_features.py`, `derive_labels.py`
- [x] Inference: `pipeline.py`, `visualizer.py`, `latency_profiler.py`
- [x] Demo: `demo.py`

## Phase 4: Validation
- [x] Unit tests (`test_models.py`, `test_data.py`, `test_features.py`, `test_fusion.py`)
- [x] Smoke test: forward pass through full pipeline
- [x] Verify training loop runs on CPU with synthetic data

## Phase 5: Colab Training Lane
- [x] Cached sequence dataset and manifest/shard format
- [x] MELD cheap-feature cache preparation script
- [x] Real `FacePredWorldModel` cache training script
- [x] Checkpoint/resume support (`last.pt`, `best.pt`, step checkpoints)
- [x] Cached checkpoint evaluation script
- [x] Colab runbook and quickstart notebook

## Phase 6: First Real Causal Campaign
- [x] Real MELD archive staging for Colab Free
- [x] Resumable per-dialogue real-audio cache extraction
- [x] Causal features with no transcript leakage
- [x] Future-shifted 200 ms and 1000 ms targets
- [x] Class-balanced loss and training modality dropout
- [x] Intra-epoch newest-checkpoint recovery
- [x] Dev-only successive-halving tuning and 75-epoch winner continuation
- [x] Per-horizon corpus metrics and final test confusion matrices
- [x] Colab Free real-training notebook and runbook

## Phase 7: Live Campaign Hardening
- [x] Detect and diagnose majority-class collapse
- [x] Inverse-frequency + focal primary turn objective
- [x] Auxiliary categorical class balancing
- [x] Shared long-horizon scheduler across successive-halving stages
- [x] Predicted-support, per-class, balanced-accuracy, and majority-baseline metrics
- [x] Automatic collapse gate before expensive continuation
- [x] Cache signal inspection script and campaign-v2 Colab instructions
