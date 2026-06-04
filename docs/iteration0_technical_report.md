# FacePred Iteration 0 Technical Report

## Executive Summary

FacePred Iteration 0 is a validated code scaffold for a predictive multimodal interaction world model. The repository now has working package structure, configuration, lazy feature extraction interfaces, MELD-style data utilities, model components, synthetic training/evaluation scripts, inference/precompute scaffolds, and tests.

The current build is not yet a real corpus training run. It proves the contracts that real training will rely on: fixed-rate multimodal tensors, derived turn-taking labels, reliability-gated fusion, RSSM state propagation, multi-horizon prediction heads, multitask losses, conservative precompute gating, and CPU-safe smoke workflows.

The next-stage Colab training lane is now implemented as a cache-first real training pipeline rather than a raw audio/video training loop. On a tight schedule and with weak local hardware, the fastest credible path is:

1. Create fixed-rate MELD feature/label caches.
2. Train the real `FacePredWorldModel` from those caches.
3. Run feature precomputation and training on Colab or another cloud runtime.
4. Keep the local machine for lint, tests, small smoke runs, and report/demo generation.

## Repository State

### Foundation

- `environment.yml` defines a Windows/conda-friendly Python 3.11 environment.
- Dependency pins were tightened after the first real conda smoke run exposed version drift:
  - `numpy==1.26.4` with `torch==2.1.2`
  - `transformers==4.36.2`, `datasets==2.16.1`
  - `peft==0.7.1`, `accelerate==0.25.0`
  - NumPy-1-compatible OpenCV and pyannote subpackages
- `pyproject.toml` defines packaging, pytest, and ruff settings.
- `configs/` contains Hydra-style config groups for model, data, feature extraction, training, engine gates, inference, and evaluation.
- `README.md` and `task.md` document the project purpose and iteration-0 checklist.

### Data Layer

Implemented under `facepred/data/`:

- `meld.py`
  - MELD split discovery and CSV normalization.
  - `MELDDataset` grouped by dialogue.
  - `MELDConfig`, `MELDRecord`.
  - Synthetic MELD-like dataframe and feature generation.
  - Padded dialogue collation.
- `label_derivation.py`
  - Four-class turn labels: `hold`, `shift`, `backchannel`, `overlap`.
  - End-of-turn buckets and horizon labels.
  - Emotion, sentiment, coarse dialog-act, and valence/arousal targets.
  - Utterance-level and fixed-timestep label projection utilities.
- `synchronizer.py`
  - Fixed-rate time grids.
  - Nearest, previous, next, and linear modality alignment.
  - `SynchronizedBatch` export to torch tensors.
- `augmentations.py`
  - Additive Gaussian noise.
  - Modality dropout.
  - Temporal jitter.
  - Composed `MultimodalAugmentor`.
- `cached.py`
  - `manifest.json` + `.pt` shard cache format.
  - `CachedSequenceDataset`.
  - Cached sequence collation and DataLoader helper.
  - Intended for Colab/local-runtime training reads.

Current limitation: the cache path creates cheap metadata-derived features. Expensive media-derived modalities are still planned as later cache passes.

### Feature Layer

Implemented under `facepred/features/`:

- `audio_prosody.py`
  - Lazy openSMILE prosody extraction.
  - eGeMAPS/ComParE/IS09 dimensions.
  - Windowed extraction and zero fallback tensors.
- `visual.py`
  - Lazy MediaPipe Face Landmarker wrapper.
  - Visual feature shape: 1493 dims.
  - Landmarks, blendshapes, head pose, confidence.
- `quality.py`
  - Quality vector: SNR, face confidence, ASR confidence, modality completeness.
- `vad.py`
  - Lazy pyannote VAD wrapper.
  - Energy fallback.
  - Output shape: `[frames, 3]`.
- `asr.py`
  - Lazy Whisper wrapper.
  - `TranscriptSegment` with confidence and optional 384-d placeholder embedding.
- `audio_ssl.py`
  - Lazy Hugging Face wav2vec2-style wrapper.
  - Synthetic/zero fallback with output shape `[frames, 768]`.
- `features/__init__.py`
  - Lazy exports so optional heavy dependencies are not imported at package import time.

Current limitation: these extractors exist, but no batch precompute script has yet run them over MELD and saved cache files.

### Model Layer

Implemented under `facepred/models/`:

- `encoders.py`
  - Per-modality MLP encoders.
  - Config-derived encoder specs.
  - `ModalityEncoders` container.
- `fusion.py`
  - `ReliabilityGatedFusion`.
  - Cross-attention fusion, concat baseline, heuristic or learned reliability gates.
  - Missing modality support.
- `rssm.py`
  - Dreamer-style recurrent state-space model.
  - Categorical and Gaussian latent support.
  - Deterministic GRU state plus stochastic latent state.
- `heads.py`
  - Multi-horizon prediction heads.
  - Turn taking, end-of-turn bucket, dialog act, valence/arousal, emotion, entropy.
- `losses.py`
  - `FacePredLoss`.
  - RSSM KL, sequence cross-entropy, Brier calibration, valence/arousal MSE.
  - Supports timestep targets expanded across prediction horizons.
- `world_model.py`
  - `FacePredWorldModel` top-level wrapper.
  - Encoders -> fusion -> RSSM -> prediction heads.
  - Synthetic feature generator for contract tests.

Current limitation: the real world model is validated by tests and smoke calls, but the default trainer still uses a tiny scaffold model. Wiring `FacePredWorldModel` into a production trainer is the next code step.

### Engine Layer

Implemented under `facepred/engine/`:

- `trainer.py`
  - Lightweight YAML config loader.
  - `TinyFacePredModel` scaffold.
  - Synthetic batch generation.
  - Basic train/validation loop.
- `evaluator.py`
  - Loss, accuracy, macro-F1, entropy, MAE, Brier helpers.
  - Prediction-name normalization between scaffold and world-model outputs.
- `precompute.py`
  - Candidate response branch ranking.
  - Conservative gate decision using yield probability, entropy, and branch margin.

Current limitation: there is no checkpoint/resume loop for real cached data yet.

### Inference Layer

Implemented under `facepred/inference/`:

- `pipeline.py`
  - Model-or-heuristic inference step.
  - Precompute gate integration.
  - Latency profiling per step.
- `visualizer.py`
  - Terminal probability table.
  - Gate decision rendering.
  - Lazy matplotlib plotting.
- `latency_profiler.py`
  - Timing samples and summaries.

Current limitation: the demo is terminal/synthetic. Live camera/microphone ingestion is not built yet.

### Scripts

Implemented under `scripts/`:

- `train.py`
  - Synthetic scaffold training.
- `evaluate.py`
  - Synthetic scaffold evaluation.
- `extract_features.py`
  - Synthetic feature tensor generation.
- `derive_labels.py`
  - Synthetic or CSV/JSON turn-label derivation.
- `demo.py`
  - Terminal inference/precompute gate demo.
- `prepare_meld_cache.py`
  - Builds Colab-friendly cache shards from synthetic or real MELD metadata.
  - Generates timing-derived VAD, hashed text features, and quality features.
- `train_world_model.py`
  - Trains the real `FacePredWorldModel` from cached shards.
  - Saves `last.pt`, `best.pt`, optional step checkpoints, `metrics.jsonl`, and `run_config.json`.
  - Supports `--resume auto`, CUDA device auto-detection, and AMP.
- `evaluate_world_model.py`
  - Loads a world-model checkpoint and reports cached-split metrics.

Current limitation: scripts do not yet include:

- real MELD download automation
- expensive media-derived feature cache generation

### Tests

Implemented under `tests/`:

- `test_data.py`
- `test_features.py`
- `test_fusion.py`
- `test_models.py`
- `test_utils.py`
- `test_cached.py`

Latest validation in the `facepred` conda environment:

```text
python -m pip check
No broken requirements found.

python -m ruff check .
All checks passed.

python -m pytest -q
17 passed.

python scripts/train.py --epochs 1 --batches 1 --val-batches 1 --batch-size 2 --seq-len 4 --feature-dim 32
Passed with no NumPy/Torch/Transformers compatibility warnings.
```

## Implemented Tensor Contracts

Default model config dimensions:

- `visual`: 1493
- `audio_prosody`: 88
- `audio_ssl`: 768
- `vad`: 3
- `text`: 384
- `quality`: 4

World model output contract:

- `turn_taking_logits`: `[batch, steps, horizons, classes]`
- `end_of_turn_logits`: `[batch, steps, horizons, buckets]`
- `dialog_act_logits`: `[batch, steps, horizons, classes]`
- `valence_arousal`: `[batch, steps, horizons, 2]`
- `emotion_logits`: `[batch, steps, horizons, emotions]`
- `turn_taking_entropy`: `[batch, steps, horizons]`
- `reliability`: `[batch, steps, modalities]`
- `rssm_state`: `[batch, steps, state_dim]`

Default horizons:

- 200 ms
- 1000 ms

Default turn classes:

- hold
- shift
- backchannel
- overlap

## What Is Real Versus Scaffolded

Real and usable now:

- Package structure and dependency pins.
- Config-driven model construction.
- RSSM/fusion/head/loss modules.
- Label derivation logic.
- Modality synchronization utilities.
- Lazy feature extractor interfaces.
- Synthetic train/eval/demo scripts.
- Tests and lint.

Scaffolded but not yet production training:

- `FacePredTrainer` trains `TinyFacePredModel`, not `FacePredWorldModel`.
- `extract_features.py` creates synthetic tensors, not real media-derived caches.
- Inference pipeline can wrap a real model, but there is no trained checkpoint.

Not built yet:

- Real MELD raw media preparation.
- Feature cache manifest format.
- Cached dataset and dataloader for real training.
- Checkpoint/resume/logging for long cloud runs.
- W&B or TensorBoard integration.
- Real evaluation reports over train/dev/test.
- Latency benchmark over actual model outputs.

## Colab And Cloud Training Notes

Google's Colab FAQ states that Colab resources are not guaranteed or unlimited, usage limits fluctuate, and GPU/TPU availability varies over time. It also notes that VMs are private to the account but are deleted after idle periods and have enforced maximum lifetimes. For this project, that means training must checkpoint often and must not assume a long uninterrupted runtime. Source: https://research.google.com/colaboratory/faq.html

The same FAQ warns that Google Drive I/O can fail or slow down when folders contain many files, and recommends avoiding many small reads from Drive by copying archive files into the runtime and unpacking locally. For FacePred, that argues for writing feature caches as chunked `.pt`/`.npz` shards and copying active shards to local Colab disk before training. Source: https://research.google.com/colaboratory/faq.html

Colab Enterprise documentation lists GPU default runtimes around L4/T4 availability depending on region, and notes that GPU default runtimes are region-dependent. Even if using consumer Colab rather than Enterprise, the practical planning lesson is the same: target T4/L4-class constraints first, not a guaranteed A100. Source: https://docs.cloud.google.com/colab/docs/default-runtimes-with-gpus

## Recommended Next Implementation Plan

### Step 1: Commit Current Colab Training Lane

The current state is green and worth preserving. It now includes the cache-first real world-model training lane.

### Step 2: Run A Synthetic Cache Smoke On Colab

Use `docs/colab_training.md` or `notebooks/facepred_colab_quickstart.ipynb` to run:

1. `prepare_meld_cache.py --synthetic`
2. `train_world_model.py --resume auto`
3. `evaluate_world_model.py`

### Step 3: Run Real MELD Metadata Cache

Point `--data-root` at raw MELD in Drive and prepare `meld_cheap_v0`.

### Step 4: Train The First Real Checkpoint

Train `world_xs_cheap_v0` from the local Colab runtime cache while writing checkpoints and metrics to Drive.

### Step 5: Add Expensive Features Incrementally

After the cache trainer works:

1. Add openSMILE prosody caches.
2. Add Whisper transcript/confidence caches if needed.
3. Add MediaPipe visual caches.
4. Add wav2vec2/audio SSL caches last.

Do not run these inside the training loop. Precompute them once, shard them, and train from disk.

### Step 5: First Meaningful Baselines

Run these in order:

1. `timing_vad_text_quality`
2. `audio_prosody_vad_text_quality`
3. `visual_vad_text_quality`
4. `full_multimodal_small`

The first baseline should be weak but quick. Its job is to make sure labels, dataloading, loss, checkpointing, and evaluation are real.

## Risk Register

High risk:

- Feature extraction over MELD media may take longer than model training.
- Colab runtime interruptions can kill long extraction jobs.
- MELD file layout may differ by source.
- Label derivation is heuristic and needs audit against examples.

Medium risk:

- Full multimodal tensors are large.
- wav2vec2 on CPU is too slow for local iteration.
- Visual extraction on Windows may be fragile.
- Class imbalance may make turn labels look deceptively good or bad.

Low risk:

- The model forward/loss contract is already tested.
- Synthetic training/eval path is green.
- Dependency graph is currently clean.

## Decision Recommendation

Start the real training loop through the new cache-first scripts. Do not train directly from raw audio/video.

The fastest pushable milestone is:

> Real MELD metadata + fixed-rate heuristic labels + cheap cached features + `FacePredWorldModel` training + checkpoint + evaluation report.

That milestone is credible, demoable, and aligned with the architecture. It also sets up Colab for the expensive feature passes without forcing local hardware to do work it is bad at.
