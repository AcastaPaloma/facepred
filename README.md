# FacePred

FacePred is an iteration-0 scaffold for a predictive multimodal interaction world model. The goal is to predict near-future conversational state from face, voice, text, and quality signals so an assistant can prepare candidate responses before the user fully yields the turn.

The architecture registry lives in [`.references/architectural_decisions.md`](.references/architectural_decisions.md). The current default is a small RSSM with categorical latents, multimodal reliability-gated fusion, and prediction heads for turn taking, end-of-turn timing, dialog act, affect, and uncertainty.

## What Is Included

- Hydra-style configs under [`configs/`](configs/)
- Lazy feature extractors for prosody, visual landmarks, quality, VAD, ASR, and audio SSL embeddings
- MELD-style dataset normalization, synthetic fixtures, label derivation, synchronization, and augmentations
- A PyTorch world-model stack under [`facepred/models/`](facepred/models/)
- Lightweight training, evaluation, precompute gating, and inference scaffolds
- Synthetic smoke scripts under [`scripts/`](scripts/)

Heavy media dependencies such as MediaPipe, openSMILE, Whisper, pyannote, and transformers are imported lazily. You can run the synthetic smoke path before installing every optional backend.

## Setup

From a conda shell:

```bash
conda env create -f environment.yml
conda activate facepred
```

For editable development:

```bash
pip install -e ".[dev]"
```

The environment targets Python 3.11. If you run the repo with a different system Python, synthetic imports may still work, but the project dependencies are pinned for the conda environment.

## Smoke Commands

Run a synthetic training loop:

```bash
python scripts/train.py --epochs 1 --batches 1 --val-batches 1 --batch-size 2 --seq-len 4 --feature-dim 32
```

Evaluate the lightweight scaffold:

```bash
python scripts/evaluate.py --batches 1 --batch-size 2 --seq-len 4 --feature-dim 32
```

Generate synthetic modality features:

```bash
python scripts/extract_features.py --num-sequences 2 --seq-len 5 --modalities audio_prosody,vad,quality
```

Derive synthetic turn labels:

```bash
python scripts/derive_labels.py --synthetic-count 6
```

Run the terminal demo:

```bash
python scripts/demo.py
```

## Iteration 0 Status

The current build is designed to prove tensor contracts and control flow on CPU. Real MELD media extraction, full training recipes, and calibrated latency experiments come after the synthetic path is stable.

See [`task.md`](task.md) for the current build tracker.

For a fuller build log, codebase inventory, and training-readiness notes, see
[`docs/iteration0_technical_report.md`](docs/iteration0_technical_report.md).

For the Colab cache/training workflow, see
[`docs/colab_training.md`](docs/colab_training.md) and
[`notebooks/facepred_colab_quickstart.ipynb`](notebooks/facepred_colab_quickstart.ipynb).
