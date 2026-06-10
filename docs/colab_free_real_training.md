# Colab Real-Training Runbook

## Classification

This is the campaign-v5 FacePred safe-yield bridge experiment:

- real MELD audiovisual source files
- real decoded audio
- causal 100 ms audio statistics including pitch, voicing, and energy slopes
- 20-second causal context with a local-TCN plus long-context GRU ablation
- discrete event-time and learned commit-safety ablations
- no ground-truth transcript input
- sparse earliest-event and explicit safe-yield targets at 200 ms and 1000 ms
- deterministic concat baseline with controlled temporal/head ablations
- dev-fitted per-horizon temperature and conservative commit thresholds
- dev-only successive-halving selection
- one final test evaluation after model selection

It is an audio-first baseline. Visual features and wav2vec2 remain later ablations.

## Colab Persistence

Google Drive is persistent. `/content`, installed packages, Python state, and the
active VM are not persistent.

The pipeline writes these persistent artifacts:

```text
MyDrive/facepred/
  data/MELD.Raw.tar.gz
  cache/meld_audio_yield_v3/
    .progress/                 # one resumable file per completed dialogue
    manifest.json
    train/*.pt
    dev/*.pt
    test/*.pt
  runs/audio_campaign_v5/
    selection.json
    silence_baseline.json
    <candidate>/
      checkpoints/
        best.pt
        last.pt
        step_XXXXXXXX.pt
      metrics.jsonl
      run_config.json
      yield_calibration.json
```

After any disconnect, remount Drive, clone/install the repository, rerun data
staging, copy the finished cache to `/content`, and rerun the tuning/training
command. Extraction skips completed dialogues. Training resumes from the newest
`last.pt` or step checkpoint.

Colab cannot automatically allocate and start a replacement VM. Reconnecting and
running the resume cells is manual.

## Before Starting

1. In Colab, select a CPU runtime for download and feature extraction.
2. Ensure Drive has at least 13 GB free for the persistent MELD archive, plus
   room for cache and checkpoints.
3. Ensure the Colab runtime has roughly 30 GB free for the local archive copy and
   extracted media.
4. Use a GPU runtime only after the cache is complete.

If Drive cannot hold the archive, pass `--archive /content/MELD.Raw.tar.gz` to
`stage_meld_colab.py`. The archive will then need to be downloaded again after a
runtime reset, while completed feature progress remains on Drive.

## 1. Mount Drive And Install

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
cd /content
git clone YOUR_REPO_URL facepred || true
cd /content/facepred
git pull
apt-get -qq update && apt-get -qq install -y ffmpeg
pip install -e .
pip install numpy==1.26.4 pandas scipy pyyaml huggingface_hub pytest ruff
```

Do not replace Colab's CUDA-enabled PyTorch installation.

## 2. Download And Stage MELD

This downloads the official `declare-lab/MELD` raw archive to Drive once, copies
it to local runtime storage, extracts it locally, and downloads the official
train/dev/test annotation CSVs from the MELD GitHub repository:

```bash
python scripts/stage_meld_colab.py \
  --archive /content/drive/MyDrive/facepred/data/MELD.Raw.tar.gz \
  --local-archive /content/MELD.Raw.tar.gz \
  --extract-dir /content/facepred_data
```

The archive is approximately 10.9 GB. This is a download and local extraction,
not training-time streaming.

The staging command must finish with JSON containing three `csvs` entries and a
nonzero `media_files` count before running cache preparation. It validates that
the Drive archive is approximately 10.9 GB and recursively extracts MELD's
nested split archives when present.

If staging reports that the archive is undersized, remove the bad Drive copy
and rerun:

```bash
rm -f /content/drive/MyDrive/facepred/data/MELD.Raw.tar.gz
rm -f /content/MELD.Raw.tar.gz
python scripts/stage_meld_colab.py \
  --archive /content/drive/MyDrive/facepred/data/MELD.Raw.tar.gz \
  --local-archive /content/MELD.Raw.tar.gz \
  --extract-dir /content/facepred_data
```

## 3. Build The Real Audio Cache

First run a two-dialogue validation:

```bash
python scripts/prepare_meld_audio_cache.py \
  --data-root /content/facepred_data \
  --output-dir /content/drive/MyDrive/facepred/cache/meld_audio_yield_v3_smoke \
  --sequence-length-s 20 --train-stride-s 5 --eval-stride-s 20 \
  --max-dialogues 2
```

Then build the full persistent cache:

```bash
python scripts/prepare_meld_audio_cache.py \
  --data-root /content/facepred_data \
  --output-dir /content/drive/MyDrive/facepred/cache/meld_audio_yield_v3 \
  --sequence-length-s 20 --train-stride-s 5 --eval-stride-s 20
```

If Colab disconnects, rerun the staging command and the same cache command.
Completed dialogue files under `.progress/` are reused.

Validate the persistent Drive cache before switching runtimes or copying it:

```bash
python scripts/validate_campaign_v5_cache.py \
  --cache-dir /content/drive/MyDrive/facepred/cache/meld_audio_yield_v3
```

Do not use `--allow-config-mismatch` to reuse v2 progress. If a directory named
`meld_audio_yield_v3` contains a v2 manifest or extraction config, rename that
directory and rebuild v3 from an empty destination. A valid interrupted v3
build has a matching `extraction_config.json` and may be resumed with the same
cache command.

## 4. Switch To GPU And Copy Cache Locally

Change the Colab runtime to GPU. Then rerun mount, clone/install, and:

```bash
rm -rf /content/facepred_cache
mkdir -p /content/facepred_cache
rsync -a --exclude '.progress/' /content/drive/MyDrive/facepred/cache/meld_audio_yield_v3/ /content/facepred_cache/
python scripts/validate_campaign_v5_cache.py --cache-dir /content/facepred_cache
python scripts/inspect_training_cache.py --cache-dir /content/facepred_cache --split train
python scripts/inspect_training_cache.py --cache-dir /content/facepred_cache --split dev
```

Training reads the cache from local runtime disk and writes checkpoints to Drive.
The inspection output must include `yield` and `event_hazard` targets, 32
audio-prosody features, and manifest target schema
`earliest_event_safe_yield_v3`. If it does not, the wrong Drive cache was copied.

## 5. Tune Then Train

Keep earlier campaign artifacts for historical comparison. Campaign v5 requires
the rebuilt `meld_audio_yield_v3` cache because it adds richer causal audio
features, event-time targets, and 20-second windows.

```bash
python scripts/tune_and_train_colab_v5.py \
  --cache-dir /content/facepred_cache \
  --output-root /content/drive/MyDrive/facepred/runs/audio_campaign_v5 \
  --device cuda \
  --batch-size 16 \
  --stage1-epochs 5 \
  --stage2-epochs 15 \
  --max-epochs 40 \
  --save-every-steps 100 \
  --early-stopping-patience 10
```

The sweep keeps deterministic GRU-S, concat fusion, learning rate, cache,
natural sampling, capped-at-3 safe-yield weighting, and seed fixed. It isolates
multi-rate causal context, the event-hazard auxiliary, and the commit verifier.

The four candidates train for five epochs. The top two continue to fifteen.
The winner continues toward 40 epochs with early stopping.

Rerunning the same command is the recovery procedure. Completed candidates and
epochs are resumed rather than restarted.

The campaign first records a sustained-silence endpointing baseline. Validation
selects on mean safe-yield average precision and reports precision, recall, F1,
Brier score, ECE, false commits, late responses, prediction lead time,
floor-transfer-gap groups, sparse event confusion matrices, and prevalence.
After selecting the winner, `yield_calibration.json` is fitted on dev at a
minimum precision of 90% and minimum coverage of 25 dev commits. If that policy
is infeasible, the artifact explicitly abstains instead of silently accepting a
lower-precision threshold. Calibration and gating use commit-safety scores when
the winner has the verifier, otherwise they use direct yield. See
[`campaign_v5.md`](campaign_v5.md) for rationale and the dataset decision.

## 6. Evaluate The Selected Winner

Only after tuning and long training complete:

```bash
python scripts/evaluate_selected_run.py \
  --selection /content/drive/MyDrive/facepred/runs/audio_campaign_v5/selection.json \
  --cache-dir /content/facepred_cache \
  --split test \
  --device cuda
```

The resulting JSON applies the dev-fitted temperatures and operating thresholds,
then includes safe-yield metrics, commit counts, lead-time and gap diagnostics,
auxiliary event confusion matrices, and the calibration artifact path.

## Recovery Cell

After a GPU training disconnect, run:

```bash
cd /content
git clone YOUR_REPO_URL facepred || true
cd /content/facepred
git pull
pip install -e .
pip install numpy==1.26.4 pandas scipy pyyaml
rm -rf /content/facepred_cache
mkdir -p /content/facepred_cache
rsync -a --exclude '.progress/' /content/drive/MyDrive/facepred/cache/meld_audio_yield_v3/ /content/facepred_cache/
python scripts/inspect_training_cache.py --cache-dir /content/facepred_cache --split dev
python scripts/tune_and_train_colab_v5.py \
  --cache-dir /content/facepred_cache \
  --output-root /content/drive/MyDrive/facepred/runs/audio_campaign_v5 \
  --device cuda --batch-size 16 --stage1-epochs 5 --stage2-epochs 15 \
  --max-epochs 40 --save-every-steps 100 --early-stopping-patience 10
```
