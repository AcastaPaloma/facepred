# Google Colab Training Runbook

This runbook assumes real training happens on Colab while local development stays focused on tests, lint, and small smoke commands.

## Drive Layout

Recommended Drive root:

```text
MyDrive/facepred/
  data/
    MELD.Raw/                  # raw MELD files, optional for first synthetic smoke
  cache/
    meld_cheap_v0/             # prepared tensor cache
  runs/
    world_xs_cheap_v0/         # checkpoints + metrics
```

During Colab training, copy the cache from Drive to local runtime storage:

```bash
rsync -a /content/drive/MyDrive/facepred/cache/meld_cheap_v0/ /content/facepred_cache/
```

Train from `/content/facepred_cache` and write checkpoints to Drive. This avoids reading many small files directly from Drive during every training step.

## Colab Setup

In a Colab notebook:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Clone or update the repo:

```bash
cd /content
git clone https://github.com/YOUR_USER/YOUR_REPO.git facepred || true
cd /content/facepred
git pull
```

Install the minimal cheap-training dependencies. Do not install CPU-only torch over Colab's CUDA torch.

```bash
pip install -e .
pip install numpy==1.26.4 pandas scipy pyyaml hydra-core omegaconf rich pytest ruff
```

Optional feature extraction dependencies can be installed later:

```bash
pip install opensmile soundfile librosa opencv-python==4.9.0.80 mediapipe==0.10.9
pip install transformers==4.36.2 datasets==2.16.1 accelerate==0.25.0 peft==0.7.1
```

## Smoke The Repo On Colab

```bash
python -m ruff check .
python -m pytest -q
```

## Prepare A Cheap Cache

Synthetic smoke cache:

```bash
python scripts/prepare_meld_cache.py \
  --config configs/config.yaml \
  --output-dir /content/drive/MyDrive/facepred/cache/synthetic_cheap_v0 \
  --synthetic \
  --synthetic-dialogues 16 \
  --utterances-per-dialogue 8 \
  --splits train dev test
```

Real MELD metadata cache:

```bash
python scripts/prepare_meld_cache.py \
  --config configs/config.yaml \
  --data-root /content/drive/MyDrive/facepred/data/MELD.Raw \
  --output-dir /content/drive/MyDrive/facepred/cache/meld_cheap_v0 \
  --splits train dev test
```

The cheap cache uses timing-derived VAD, hashed text features, and quality features. It does not require MediaPipe, openSMILE, Whisper, or wav2vec2.

## Copy Cache Locally

```bash
rm -rf /content/facepred_cache
mkdir -p /content/facepred_cache
rsync -a /content/drive/MyDrive/facepred/cache/meld_cheap_v0/ /content/facepred_cache/
```

For synthetic smoke, replace `meld_cheap_v0` with `synthetic_cheap_v0`.

## Train

```bash
python scripts/train_world_model.py \
  --config configs/config.yaml \
  --cache-dir /content/facepred_cache \
  --output-dir /content/drive/MyDrive/facepred/runs/world_xs_cheap_v0 \
  --epochs 10 \
  --batch-size 32 \
  --device auto \
  --amp \
  --resume auto \
  --save-every-steps 250
```

Checkpoint behavior:

- `checkpoints/last.pt` is written every epoch.
- `checkpoints/best.pt` is written when validation turn macro-F1 improves.
- `checkpoints/step_XXXXXXXX.pt` is written when `--save-every-steps` is set.
- `metrics.jsonl` appends one record per epoch.
- `run_config.json` snapshots args, cache manifest, and config.

If Colab disconnects, rerun the same command with `--resume auto`; it will restore from `last.pt`.

## Evaluate

```bash
python scripts/evaluate_world_model.py \
  --config configs/config.yaml \
  --cache-dir /content/facepred_cache \
  --checkpoint /content/drive/MyDrive/facepred/runs/world_xs_cheap_v0/checkpoints/best.pt \
  --split test \
  --device auto \
  --output /content/drive/MyDrive/facepred/runs/world_xs_cheap_v0/test_metrics.json
```

## Tight-Schedule First Milestone

The first code-push training milestone should be:

1. `prepare_meld_cache.py` runs over synthetic and/or real MELD metadata.
2. `train_world_model.py` trains `FacePredWorldModel` from cache.
3. `best.pt`, `last.pt`, `metrics.jsonl`, and `test_metrics.json` exist in Drive.
4. `pytest` and `ruff` pass on the same commit.

After this milestone, add expensive cached modalities one at a time:

1. openSMILE prosody
2. MediaPipe visual
3. Whisper confidence/transcripts
4. wav2vec2/audio SSL
