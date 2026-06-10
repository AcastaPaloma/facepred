# Campaign v5: Causal Context, Event Hazard, And Commit Safety

## Motivation

Campaign v4 modestly improved safe-yield ranking, but the selected model still
could not satisfy the `>=90%` precision and minimum-coverage commit policy.
Calibration correctly abstained. The limiting problem is therefore
discrimination and missing conversational evidence, not temperature scaling.

Campaign v5 tests a stronger but still causal MELD bridge architecture before
moving to continuous dyadic data:

- 32-dimensional causal audio statistics with F0, voicing confidence, log
  energy, and their slopes
- fixed 100 ms sample windows, so feature boundaries do not depend on the
  utterance's future duration
- 20-second windows instead of five-second windows
- a local causal dilated TCN combined with a full-context causal GRU
- a discrete competing-risk event-time head
- a separate learned commit-safety verifier

The direct yield head remains the proposal model. The verifier sees the latent
state plus detached proposal probability, entropy, temporal delta, and
stability. Calibration and the runtime gate prefer verifier scores when the
head exists, while retaining the old yield fallback for legacy checkpoints.

## Controlled Candidate Matrix

All candidates use the same rich-feature schema-v3 cache, 20-second context,
natural training-window distribution, capped-at-3 positive weighting, learning
rate, scheduler, and seed.

| Candidate | Multi-rate context | Event hazard | Commit verifier |
|---|---:|---:|---:|
| `rich_gru_direct` | no | no | no |
| `multirate_direct` | yes | no | no |
| `multirate_hazard` | yes | yes | no |
| `multirate_hazard_safety` | yes | yes | yes |

The event-hazard target is one categorical competing-risk distribution:

- class `0`: no event inside 2000 ms
- remaining classes: earliest `{shift, backchannel, overlap}` event crossed
  with `{200, 500, 1000, 2000}` ms time bins

This gives the model explicit event type and timing supervision without
pretending MELD provides two continuous speaker channels.

## Dataset Decision

MELD remains useful only as a bridge benchmark. It is composed of television
utterance clips and cannot provide the honest continuous two-speaker activity
target used by Voice Activity Projection.

The recommended dataset path is:

1. **Switchboard** for the first true VAP experiment. The original
   [VAP paper](https://www.isca-archive.org/interspeech_2022/ekstedt22_interspeech.pdf)
   started from 2,438 Switchboard dialogues, excluded 98 sessions containing
   more than two speakers, and used 2,205/135 dialogues for train/test. It
   provides continuous two-channel telephone conversations and directly
   matches the target formulation. The official release is
   [LDC97S62](https://catalog.ldc.upenn.edu/LDC97S62).
2. **Switchboard + Fisher** for scale. The
   [VAP prosody study](https://arxiv.org/abs/2209.05161) trained on 8,288
   dialogues from the two corpora. Both are distributed through the
   Linguistic Data Consortium and require licensing
   ([Part 1](https://catalog.ldc.upenn.edu/LDC2004S13),
   [Part 2](https://catalog.ldc.upenn.edu/LDC2005S13)).
3. **CANDOR** before the visual campaign. The
   [CANDOR corpus](https://www.science.org/doi/10.1126/sciadv.adf3197)
   contains 1,656 natural two-person videoconference conversations totaling
   more than 850 hours of video, and
   [MM-VAP](https://arxiv.org/abs/2505.21043) uses it for audiovisual VAP.
   The authors provide the dataset from the
   [CANDOR data page](https://guscooney.com/candor-dataset/).

Other relevant dataset choices clarify the tradeoffs:

| Work | Dataset choice | Relevance |
|---|---|---|
| [TurnGPT](https://aclanthology.org/2020.findings-emnlp.268/) | Switchboard, MapTask, and text-dialogue corpora | Strong linguistic turn-completion baseline; not a replacement for continuous audio |
| [Multilingual VAP](https://arxiv.org/abs/2403.06487) | Switchboard plus Mandarin and Japanese dyadic corpora | Evidence that the VAP objective transfers across languages |
| [VAP with multimodal encoders](https://arxiv.org/abs/2506.03980) | NoXi French subset | Useful small audiovisual dyadic benchmark; only about seven hours in the reported subset |
| [General continuous turn-taking](https://aclanthology.org/W17-5527/) | HCRC Map Task | Classic, smaller task-oriented continuous-conversation benchmark |

The repository now includes `build_vap_state_targets`, which encodes future
activity for both speakers into the categorical VAP state contract. Dataset
ingestion remains separate because Switchboard/Fisher files cannot be bundled
or automatically downloaded without the user's LDC credentials and licenses.

## Interpretation Policy

Campaign v5 can answer whether richer causal evidence, event timing, and a
specialized verifier improve the MELD proxy. It cannot establish a true VAP
result. If v5 still cannot achieve useful high-precision coverage, the next
experiment is Switchboard, not a larger MELD model.
