# FacePred: Architectural Decisions Registry

> Every design choice in this project is documented here. If you're contributing or experimenting, this is your guide to what knobs you can turn and what happens when you do.

---

## World Model Architecture

### D01: Latent Variable Type — Categorical vs. Gaussian

| | Details |
|---|---|
| **Default** | **Categorical** (32 classes × 16 one-hot dims = 512-dim) |
| **Alternative** | Gaussian (mean + variance, 64-dim per component) |
| **Rationale** | DreamerV3 showed categorical latents avoid posterior collapse and have better gradient flow. Gaussian requires careful KL balancing. |
| **Effect of switching to Gaussian** | ⚠️ May need KL annealing to prevent posterior collapse. Could be slightly more expressive for continuous dynamics. Gaussian is simpler to interpret (mean = "best guess," variance = "uncertainty"). Historically weaker in Dreamer experiments. |
| **Config key** | `model.rssm.latent_type: categorical \| gaussian` |
| **Config keys** | `model.rssm.categorical_classes: 32`, `model.rssm.categorical_dims: 16` or `model.rssm.gaussian_dim: 64` |

### D02: RSSM Size

| | Details |
|---|---|
| **Default** | **XS** for iter 0 (GRU=128), **S** for iter 1 (GRU=256) |
| **Alternatives** | M (GRU=512, ~5M params), L (GRU=1024, ~15M params) |
| **Rationale** | Start small, prove the concept works, then scale. Conversation dynamics are arguably simpler than Atari/robotics, so smaller models may suffice. |
| **Effect of scaling up** | ✅ More capacity to capture complex dialogue patterns, longer-range dependencies. ⚠️ Slower training and inference. May overfit on small datasets (MELD=13K utterances). 🔴 Large models (L+) may not hit 100ms inference budget on CPU. |
| **Config key** | `model.rssm.gru_hidden: 128 \| 256 \| 512 \| 1024` |

### D03: Deterministic-Only vs. Stochastic+Deterministic State

| | Details |
|---|---|
| **Default** | **Both** (deterministic GRU h_t + stochastic categorical z_t) |
| **Alternative** | Deterministic only (just GRU, no stochastic latent) |
| **Rationale** | The stochastic component captures uncertainty about the current state. Without it, the model can't express "I'm not sure if they're finishing or pausing." |
| **Effect of removing stochastic** | ⚠️ Loses ability to represent multi-modal uncertainty (e.g., "50% chance they're asking a question, 50% chance they're making a statement"). Simpler and faster, but fundamentally less capable for our gating mechanism. |
| **Config key** | `model.rssm.use_stochastic: true \| false` |

### D04: World Model Update Rate

| | Details |
|---|---|
| **Default** | **100ms (10 Hz)** |
| **Alternatives** | 50ms (20 Hz), 80ms (12.5 Hz), 200ms (5 Hz) |
| **Rationale** | Visual head start is ~80-150ms. 100ms gives us enough temporal resolution to catch pre-acoustic visual cues while keeping compute manageable. |
| **Effect of faster (50ms)** | ✅ Finer-grained predictions, better temporal resolution for fast cues. ⚠️ 2× more compute per second. May not add meaningful information at sub-100ms for face landmarks (30fps camera = 33ms between frames anyway). |
| **Effect of slower (200ms)** | ✅ 2× less compute, easier to hit latency budgets. ⚠️ May miss fast turn-taking cues. 200ms is close to the human reaction time floor, so we'd lose the "head start" advantage. |
| **Config key** | `model.step_duration_ms: 100` |

### D05: Prediction Horizons

| | Details |
|---|---|
| **Default** | **2 horizons: 200ms, 1000ms** (MVP), expandable to 4: 200, 500, 1000, 2000ms |
| **Rationale** | 200ms = "imminent" (is the turn ending right now?), 1000ms = "soon" (should I start preparing a response?). More horizons add paper granularity. |
| **Effect of more horizons** | ✅ Richer temporal prediction, better for paper ablations. ⚠️ More output dimensions per head, slightly more compute, more loss terms to balance. |
| **Effect of fewer (just 200ms)** | ⚠️ Can't do forward-looking precomputation. System becomes purely reactive, defeating the purpose. |
| **Config key** | `model.prediction_horizons_ms: [200, 1000]` |

### D06: JEPA Pretraining Objective

| | Details |
|---|---|
| **Default** | **Deferred to iteration 4.** Iter 0-3 use frozen pretrained encoders without JEPA. |
| **Alternatives** | Add JEPA loss from iter 1, joint JEPA+RSSM from the start |
| **Rationale** | JEPA pretraining needs large unlabeled video data and GPU time. Frozen encoders (wav2vec2, MediaPipe) already provide good representations. JEPA is the "level up" once the basic pipeline works. |
| **Effect of adding JEPA early** | ✅ Better representations, potentially stronger predictions. 🔴 Needs cloud GPU, large video data, and significantly more engineering. Risk of overcomplicating before basic pipeline is validated. |
| **Config key** | `training.use_jepa_loss: false`, `training.jepa_lambda: 1.0` |

---

## Feature Extraction

### D07: Visual Feature Source — MediaPipe vs. OpenFace

| | Details |
|---|---|
| **Default** | **MediaPipe Face Landmarker** |
| **Alternative** | OpenFace 2.0 |
| **Rationale** | MediaPipe runs natively on CPU at 30+ FPS, has Python bindings, easy install. OpenFace provides calibrated Action Units (FACS-based) which are more standardized, but requires cmake + dlib + OpenCV build from source. |
| **Effect of switching to OpenFace** | ✅ Calibrated AUs are more interpretable and comparable to affect literature. ⚠️ Much harder to install (especially on Windows). Slower. Build issues could cost a full day. 📝 Consider supporting both behind a common interface. |
| **Config key** | `feature.visual.extractor: mediapipe \| openface` |

### D08: Audio SSL Model — wav2vec2-base vs. -large vs. HuBERT

| | Details |
|---|---|
| **Default** | **wav2vec2-base** (95M params, 768-dim output) |
| **Alternatives** | wav2vec2-large (317M, 1024-dim), HuBERT-base (95M), HuBERT-large (317M), WavLM |
| **Rationale** | Base is 3× smaller than large, fits comfortably on T4, and wav2vec2 has the strongest LoRA fine-tuning precedent. |
| **Effect of using large** | ✅ Better speech representations, especially for emotion/affect. ⚠️ 3× more VRAM (10-12 GB in FP16 for inference). Slower. Marginal improvement for turn-taking may not justify the cost. |
| **Effect of HuBERT instead** | ✅ Slightly better for some speech tasks. ⚠️ Less LoRA precedent in literature. Otherwise very similar. |
| **Config key** | `feature.audio.ssl_model: facebook/wav2vec2-base-960h \| facebook/wav2vec2-large-960h-lv60-self \| facebook/hubert-base-ls960` |

### D09: ASR Model — Whisper small vs. medium vs. faster-whisper

| | Details |
|---|---|
| **Default** | **Whisper small** (244M params) |
| **Alternatives** | Whisper tiny (39M), Whisper medium (769M), faster-whisper (CTranslate2 INT8) |
| **Rationale** | Small balances quality and speed. Medium is better but 3× larger. faster-whisper is 4× faster with INT8 quantization. |
| **Effect of faster-whisper** | ✅ 4× faster inference, INT8 → ~1 GB VRAM. ⚠️ Requires CTranslate2 dependency. Slightly different API. |
| **Effect of medium** | ✅ Better ASR accuracy, especially in noise. ⚠️ 5 GB VRAM, noticeably slower. |
| **Config key** | `feature.asr.model: whisper-small \| whisper-medium \| faster-whisper-small` |

### D10: Prosody Features — eGeMAPS vs. ComParE vs. Raw Mel

| | Details |
|---|---|
| **Default** | **eGeMAPS** (88-dim, via openSMILE) |
| **Alternatives** | ComParE 2016 (6373-dim), IS09 emotion (384-dim), raw log-mel spectrogram (80-dim per frame) |
| **Rationale** | eGeMAPS is the standard for paralinguistic research — compact, interpretable, well-validated. ComParE is exhaustive but huge. Raw mel requires the model to learn everything. |
| **Effect of ComParE** | ✅ More features, potentially captures subtle cues eGeMAPS misses. ⚠️ 6373 dims is massive — needs PCA or a larger encoder. Slower to extract. |
| **Effect of raw mel** | ✅ No information bottleneck, model learns what matters. ⚠️ Needs more data and compute to learn what eGeMAPS gives you for free. |
| **Config key** | `feature.audio.prosody: egemaps \| compare \| mel` |

### D11: Camera Frame Rate

| | Details |
|---|---|
| **Default** | **30 FPS** (standard webcam) |
| **Alternatives** | 15 FPS, 60 FPS |
| **Rationale** | Visual head start is ~80-150ms. At 30 FPS, frames are 33ms apart, giving 2-4 frames of pre-acoustic visual information. Sufficient for landmark-based features. |
| **Effect of 60 FPS** | ✅ Double temporal resolution for mouth dynamics. ⚠️ 2× more frames to process. MediaPipe can handle it on modern CPUs but your i5-8400T may struggle. Diminishing returns for landmark-based features (face changes slowly relative to 60fps). More useful if doing raw pixel analysis. |
| **Effect of 15 FPS** | ⚠️ Only 1-2 frames of visual head start. May lose temporal precision. Acceptable as a degraded-quality test. |
| **Config key** | `feature.visual.fps: 30` |

### D12: Face Feature Representation

| | Details |
|---|---|
| **Default** | **Landmarks (478×3) + Blendshapes (52) + Head Pose (6) + Confidence (1) = 1493-dim** |
| **Alternative A** | Landmarks only (1434-dim) |
| **Alternative B** | Mouth ROI only (lips region landmarks, ~60 points × 3 = 180-dim) |
| **Alternative C** | Blendshapes only (52-dim) — much smaller, but loses spatial detail |
| **Rationale** | Full feature set captures everything; the model learns what to ignore. Blendshapes are already a compressed representation of facial motion. |
| **Effect of mouth-only** | ✅ Much smaller input, faster processing, focused on speech-relevant motion. ⚠️ Loses gaze, eyebrows, head nods — all important for turn-taking according to literature. |
| **Effect of blendshapes-only** | ✅ 52-dim is extremely compact. Captures AU-like movements. ⚠️ Loses fine spatial detail of lip shape. May be enough for turn-taking but weaker for affect. |
| **Config key** | `feature.visual.components: [landmarks, blendshapes, head_pose, confidence]` |

---

## Fusion

### D13: Fusion Strategy — Cross-Attention vs. Concatenation vs. Perceiver

| | Details |
|---|---|
| **Default** | **Cross-attention with reliability gating** |
| **Alternative A** | Simple concatenation + MLP |
| **Alternative B** | Perceiver-style (learned queries attend to all modalities) |
| **Rationale** | Cross-attention lets each modality attend to others selectively. Reliability gating lets the model downweight degraded modalities. Simple concat is a valid ablation baseline. |
| **Effect of simple concat** | ✅ Dead simple, fewer params, harder to overfit. ⚠️ No selective modality weighting. All modalities treated equally regardless of quality. Good ablation baseline. |
| **Effect of Perceiver** | ✅ More flexible, handles variable-length sequences naturally, supports missing modalities. ⚠️ More params (~2-5×), harder to train on small data, overkill for 4-5 fixed modalities. |
| **Config key** | `model.fusion.type: cross_attention \| concat \| perceiver` |

### D14: Reliability Gating — Learned vs. Heuristic

| | Details |
|---|---|
| **Default** | **Learned** (small network predicts reliability from quality signals) |
| **Alternative** | Heuristic (SNR > threshold → audio reliable, face_confidence > 0.8 → visual reliable) |
| **Rationale** | Learned gating can discover non-obvious reliability patterns. Heuristic is interpretable and doesn't need training. |
| **Effect of heuristic** | ✅ Interpretable, no extra params, works from day 1 without training. ⚠️ Requires manual threshold tuning. Won't discover subtle reliability patterns. |
| **Config key** | `model.fusion.reliability: learned \| heuristic` |

### D15: Modality Dropout Augmentation Rate

| | Details |
|---|---|
| **Default** | **20% per modality per sample** during training |
| **Alternatives** | 0% (no dropout), 10%, 30%, 50% |
| **Rationale** | Forces the model to work even when a modality is missing (camera blocked, mic noise). Mimics real deployment conditions. |
| **Effect of higher dropout (50%)** | ✅ Very robust to missing modalities. ⚠️ Model sees less multimodal data, may underperform when all modalities ARE available. |
| **Effect of no dropout** | ✅ Best performance when all modalities present. 🔴 Catastrophic degradation when any modality drops out at inference. Not deployment-safe. |
| **Config key** | `training.modality_dropout: 0.2` |

---

## Prediction Heads

### D16: Turn-Taking Taxonomy — 4-class vs. 3-class vs. Binary

| | Details |
|---|---|
| **Default** | **4 classes: hold, shift, backchannel, overlap** |
| **Alternative A** | 3 classes: hold, shift, backchannel (merge overlap into hold) |
| **Alternative B** | Binary: hold vs. shift (simplest) |
| **Rationale** | 4-class matches the VAP literature and captures the most nuanced behavior. Binary is the common simplification. |
| **Effect of binary** | ✅ Simpler, easier to train, higher accuracy numbers. ⚠️ Can't distinguish backchannels from real turn-takes. Can't detect overlaps. Less publishable — reviewers will ask "why not model backchannels?" |
| **Config key** | `model.heads.turn_taking.num_classes: 4 \| 3 \| 2` |

### D17: Dialog Act Taxonomy

| | Details |
|---|---|
| **Default** | **13-class reduced DAMSL** (from SwDA: statement, question, backchannel, opinion, etc.) |
| **Alternatives** | Full 43-class DAMSL, 6-class DailyDialog (inform, question, directive, commissive, etc.) |
| **Rationale** | 43 classes is too fine-grained for our data size. 13-class is the standard collapsed set used in most turn-taking papers. |
| **Effect of full 43-class** | ✅ More expressive. ⚠️ Many classes have <100 examples. Model will struggle on rare acts. |
| **Config key** | `model.heads.dialog_act.num_classes: 13 \| 43 \| 6` |

### D18: Affect Representation — Continuous vs. Categorical vs. Both

| | Details |
|---|---|
| **Default** | **Both** — Valence/Arousal continuous (2-dim) + MELD emotion categorical (7-class) as auxiliary |
| **Alternative A** | Continuous only (valence + arousal) |
| **Alternative B** | Categorical only (7 emotions from MELD: anger, disgust, fear, joy, neutral, sadness, surprise) |
| **Rationale** | Continuous is more theoretically sound (emotion isn't discrete). Categorical provides strong supervision signal from MELD labels. Using both gives the model two complementary views. |
| **Effect of continuous only** | ✅ More principled, avoids discrete emotion pitfalls. ⚠️ Harder to supervise — need valence/arousal annotations (MELD doesn't directly provide these, must map from categorical). |
| **Config key** | `model.heads.affect.type: both \| continuous \| categorical` |

### D19: Uncertainty Head — Entropy vs. Ensemble vs. MC Dropout

| | Details |
|---|---|
| **Default** | **Single model + entropy** (simplest: measure entropy of prediction distributions) |
| **Alternative A** | Deep ensembles (train 3-5 models, measure disagreement) |
| **Alternative B** | MC Dropout (run inference N times with dropout, measure variance) |
| **Alternative C** | Conformal prediction (distribution-free coverage guarantees) |
| **Rationale** | Entropy is free — just compute it from the model's own outputs. Ensembles are the gold standard but 3-5× training cost. MC Dropout is a cheap approximation. |
| **Effect of ensembles** | ✅ Best calibration, strongest uncertainty estimates. 🔴 3-5× training and inference cost. Needs separate model storage. |
| **Effect of MC Dropout** | ✅ Better than entropy, only ~5-10× inference cost (run model 5-10 times per step). ⚠️ Questionable theoretical grounding for structured prediction. |
| **Config key** | `model.uncertainty.method: entropy \| ensemble \| mc_dropout \| conformal` |

---

## Training

### D20: Loss Weighting Strategy — Fixed vs. Uncertainty-Based vs. GradNorm

| | Details |
|---|---|
| **Default** | **Fixed lambdas** (manually tuned per Hydra config) |
| **Alternative A** | Uncertainty-weighted (Kendall & Gal, 2018 — learn task weights via homoscedastic uncertainty) |
| **Alternative B** | GradNorm (balance gradient magnitudes across tasks) |
| **Rationale** | Fixed lambdas are simplest and most interpretable. Automatic weighting is elegant but adds complexity and can be unstable. |
| **Effect of uncertainty-weighted** | ✅ Automatically balances tasks. ⚠️ Adds learnable parameters (one σ per task). Can be unstable early in training. Less controllable for ablations. |
| **Config key** | `training.loss_weighting: fixed \| uncertainty \| gradnorm` |

### D21: Optimizer — AdamW vs. Adam vs. SGD

| | Details |
|---|---|
| **Default** | **AdamW** (lr=1e-4, weight_decay=0.01) |
| **Rationale** | AdamW is standard for transformer-based models. Weight decay helps regularization on small datasets. |
| **Config key** | `training.optimizer: adamw`, `training.lr: 1e-4`, `training.weight_decay: 0.01` |

### D22: Learning Rate Schedule

| | Details |
|---|---|
| **Default** | **Cosine annealing with linear warmup** (warmup=5% of steps) |
| **Alternatives** | Constant LR, step decay, one-cycle |
| **Config key** | `training.scheduler: cosine_warmup \| constant \| step \| one_cycle` |

### D23: Batch Size

| | Details |
|---|---|
| **Default** | **32** (sequences of conversation windows) |
| **Rationale** | Small enough for T4 16GB with frozen encoders + small trainable model. Large enough for stable gradient estimates. |
| **Effect of larger (64-128)** | ✅ More stable gradients, potentially better convergence. ⚠️ May not fit in T4 VRAM depending on sequence length. |
| **Config key** | `training.batch_size: 32` |

### D24: Sequence Length (Conversation Window)

| | Details |
|---|---|
| **Default** | **5 seconds** (50 steps at 100ms) |
| **Alternatives** | 2s (20 steps), 10s (100 steps), 30s (300 steps) |
| **Rationale** | 5s captures typical turn exchanges (most turns are 1-3s) with enough context for the RSSM to build up state. |
| **Effect of longer (30s)** | ✅ More context, better for tracking long-term patterns. ⚠️ GRU may struggle with very long sequences. More memory. Diminishing returns for turn-taking (most cues are within 2-3s). |
| **Config key** | `training.sequence_length_s: 5.0` |

### D25: Data Augmentation — Noise Types

| | Details |
|---|---|
| **Default** | Additive noise (Gaussian, babble), modality dropout (20%), temporal jitter (±1 frame) |
| **Alternatives** | Reverberation, codec compression, frame dropping, lighting changes, motion blur |
| **Rationale** | Start with the augmentations that most directly affect deployment (noise, missing modalities). Add exotic augmentations in later iterations. |
| **Config key** | `training.augmentation.*` |

---

## Precompute Engine

### D26: Number of Precompute Branches (k)

| | Details |
|---|---|
| **Default** | **k=3** (e.g., answer, clarify, backchannel) |
| **Alternatives** | k=1 (always precompute the most likely response), k=5 (more branches) |
| **Rationale** | k=3 covers the most common response strategies without excessive compute. |
| **Effect of k=1** | ✅ Simplest. Least compute waste. ⚠️ If the top prediction is wrong, no fallback. Higher latency when prediction is incorrect. |
| **Effect of k=5** | ✅ More fallback options. ⚠️ 5× LLM precompute cost. Most branches will be discarded. Diminishing returns past k=3 in our preliminary analysis. |
| **Config key** | `engine.precompute_k: 3` |

### D27: Gating Threshold

| | Details |
|---|---|
| **Default** | **Commit when: P(yield) > 0.8 AND entropy < 0.5 AND top-branch margin > 0.3** |
| **Rationale** | Conservative gating prevents "fast but wrong" failures. Thresholds are tunable per deployment context. |
| **Effect of more aggressive gating** (lower thresholds) | ✅ System responds faster. 🔴 More false commitments — speaks when it shouldn't. Users perceive as "jumpy" or interrupting. |
| **Effect of more conservative** (higher thresholds) | ✅ Almost never wrong when it does precommit. ⚠️ Falls back to conventional (slow) response more often. Fewer speed gains. |
| **Config key** | `engine.gate_yield_threshold: 0.8`, `engine.gate_entropy_max: 0.5`, `engine.gate_margin_min: 0.3` |

---

## Inference & Demo

### D28: Inference Device

| | Details |
|---|---|
| **Default** | **CPU** for local demo (your i5-8400T) |
| **Alternative** | GPU on cloud (for benchmarking real-time performance) |
| **Concern** | i5-8400T at 1.7GHz is slow. MediaPipe should be fine (30fps). openSMILE is fine. But wav2vec2 on CPU will be slow (~500ms per 1s of audio). For live demo, may need to skip wav2vec2 and use only lightweight features (MediaPipe + openSMILE + partial ASR). |
| **Config key** | `inference.device: cpu \| cuda` |

### D29: Demo Mode — Features Used

| | Details |
|---|---|
| **Default for local demo** | **MediaPipe + openSMILE + Whisper-tiny** (all CPU-friendly) |
| **Full mode** | Add wav2vec2, pyannote (needs GPU or accepts higher latency) |
| **Rationale** | Your CPU can handle MediaPipe (30fps) + openSMILE (real-time) + Whisper tiny (39M, near-real-time). Adding wav2vec2-base on your i5 would add ~500ms latency per chunk, breaking real-time. |
| **Config key** | `inference.features: [mediapipe, opensmile, whisper_tiny]` or `inference.features: [mediapipe, opensmile, whisper_small, wav2vec2, pyannote]` |

---

## Evaluation

### D30: Primary Evaluation Metric

| | Details |
|---|---|
| **Default** | **Effective response latency** (TRP-to-audio-onset) as primary; hold/shift F1 as secondary |
| **Rationale** | The project's thesis is about latency reduction. Turn-taking accuracy is a means to that end, not the end itself. |
| **Config key** | `evaluation.primary_metric: effective_latency \| turn_f1` |

### D31: Baseline Comparisons

| | Details |
|---|---|
| **Default** | 3 baselines: (1) 500ms silence threshold, (2) audio-only RSSM (no vision/text), (3) ungated precompute (always commit top branch) |
| **Rationale** | (1) is the straw man. (2) proves vision helps. (3) proves gating is necessary (without it, you're fast but wrong). |

### D32: Statistical Testing

| | Details |
|---|---|
| **Default** | Paired bootstrap resampling (10K samples) for corpus metrics, McNemar's test for paired outcomes |
| **Rationale** | Modern NLP/ML best practice. Avoids the paired t-test assumption of normality. |
| **Config key** | `evaluation.bootstrap_samples: 10000` |

---

## Summary: What to Experiment With First

If you're a contributor, these are the highest-impact knobs:

| Knob | Why It Matters Most |
|---|---|
| **D02: RSSM size** | Directly trades capacity vs. speed |
| **D05: Prediction horizons** | Controls how far ahead the system plans |
| **D08: Audio SSL model** | base vs. large is a key quality/cost tradeoff |
| **D13: Fusion strategy** | Cross-attention vs concat is a fundamental architecture choice |
| **D15: Modality dropout rate** | Controls robustness vs. peak performance |
| **D19: Uncertainty method** | Determines quality of the gating decision |
| **D26: Number of branches** | Controls precompute cost vs. coverage |
| **D27: Gating thresholds** | The core speed-vs-quality tradeoff |
