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
