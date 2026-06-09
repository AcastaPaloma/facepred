# Campaign v4: Yield Imbalance And Operating-Point Study

## Why This Campaign Exists

Campaign v3 established a useful corrected MELD audio baseline. Its selected
GRU-S concat model achieved strong ranking relative to prevalence, but it did
not produce a deployable commit operating point:

- selected dev mean yield AP: `0.25898`
- selected test mean yield AP: `0.20305`
- raw threshold `0.5`: no test yield commits
- requested dev precision policy: `>= 90%`
- best fitted dev policy result: infeasible at useful coverage

The v3 model therefore learned a meaningful ordering of likely yields, while
its probability scale and precision/coverage frontier remained inadequate for
safe commits. Campaign v4 targets that failure without changing model capacity,
fusion, labels, or audio features.

## Literature-Guided Decisions

- [Voice Activity Projection](https://arxiv.org/abs/2205.09812) motivates
  modeling future voice-activity events and evaluating shifts and
  backchannels. MELD still cannot provide VAP's honest continuous two-speaker
  objective, so v4 retains the corrected safe-yield proxy.
- [Focal Loss](https://arxiv.org/abs/1708.02002) shows why a large population
  of easy negatives can dominate rare-event learning. V4 tests useful-window
  exposure and mild positive weighting instead of the aggressive full
  negative/positive ratio that distorted v3 probabilities.
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
  supports dev-fitted temperature scaling. V4 additionally treats an
  infeasible precision requirement as an explicit abstention result.
- [Investigating Speech Features for Continuous Turn-Taking Prediction Using
  LSTMs](https://arxiv.org/abs/1806.11461) supports later acoustic/prosodic
  feature ablations. Pitch, voicing, SSL features, and longer context are
  deliberately deferred so this campaign isolates imbalance treatment.

## Controlled Candidate Matrix

Every candidate uses the same cache schema v2, GRU-S deterministic state,
concat fusion, `3e-4` learning rate, targets, scheduler, and seed.

| Candidate | Yield weighting | Window sampling |
|---|---|---|
| `gru_s_concat_unweighted` | none | natural |
| `gru_s_concat_sqrt_weight` | sqrt(negative / positive) | natural |
| `gru_s_concat_cap3_weight` | negative / positive capped at 3 | natural |
| `gru_s_concat_event_balanced_unweighted` | none | event-balanced |
| `gru_s_concat_event_balanced_cap3` | capped at 3 | event-balanced |

Event-balanced sampling assigns training-window probability mass to:

- `50%` background windows
- `25%` non-yield event windows, such as overlap and backchannel windows
- `25%` safe-yield windows

Labels inside sampled windows are unchanged. Validation and test loading remain
natural and unweighted.

## Calibration And Evaluation Policy

Calibration fits one temperature and one threshold per horizon on dev. A
threshold is deployable only when it:

- reaches at least `90%` precision
- produces at least `25` dev commits

If no threshold meets both constraints, the calibration artifact records
`infeasible_abstain` and installs a threshold above one. The gate then makes no
yield commits instead of silently violating policy. Selected-run evaluation
applies the fitted temperatures and thresholds before reporting commit metrics.

Average precision remains the selection metric because it evaluates ranking
independently of a single threshold. Precision, recall, false-commit rate,
late-response rate, commit count, Brier score, and ECE describe the selected
operating point.

## Evaluation Window Audit

The current dev and test cache uses non-overlapping five-second windows:
evaluation stride equals sequence length. Padding is masked. Therefore current
evaluation does not duplicate central dialogue timesteps. Training windows
overlap intentionally. Any future context experiment must preserve
non-overlapping evaluation stride or add dialogue/timestep deduplication.

## Deferred Ablations

After v4, run the top weighting/sampling policy across three seeds. Then isolate:

1. causal pitch, voicing probability, energy slope, and pitch slope
2. 5-second versus 10-second context
3. frozen causal speech representations
4. manually audited transition labels

Continuous dyadic data with a true VAP objective remains the highest-value
dataset change before adding vision.
