# BCI IV 2a algorithm validation

## What changed

The QC algorithm does not read expert artifact labels while it processes EEG.
Labels are loaded only after all QC decisions have been written.

Two signal and statistical corrections were retained:

1. The benchmark converter exports the full 0 to 6 second trial interval. The earlier converter exported only 2 to 6 seconds, although the expert artifact flag applies at trial level.
2. Adaptive median and MAD thresholds are calibrated across all windows for the same subject. The earlier implementation calibrated each trial independently, leaving too few windows to estimate a stable outlier distribution. If a subject has fewer than 20 windows, the algorithm falls back to session calibration.
3. The production window duration is two seconds. A six-second BCI IV 2a trial produces three QC windows.

The old behavior can still be reproduced with `--session-calibration`.

## Subject 1 calibration A/B

The same full 0 to 6 second input was processed twice.

| Method | Sensitivity | Specificity | Precision | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Session calibration | 33.3% | 92.7% | 20.0% | 25.0% | 63.0% |
| Subject-wide calibration | 73.3% | 95.6% | 47.8% | 57.9% | 84.5% |

This comparison supports the calibration-scope correction, but Subject 1 was used while diagnosing the problem and must not be treated as an unbiased final test.

## Frozen two-second nine-subject validation

After the algorithm was frozen, official training data for Subjects 2 to 9 was downloaded and processed without changing thresholds.

Across Subjects 1 to 9:

| Metric | Result |
|---|---:|
| Trials | 2,592 |
| Expert artifact trials | 264 |
| Windows | 7,776 |
| Sensitivity | 31.1% |
| Specificity | 91.6% |
| Precision | 29.6% |
| F1 | 30.3% |
| Balanced accuracy | 61.3% |

For held-out Subjects 2 to 9 only, sensitivity was 28.5%, specificity 91.1%, precision 28.0%, F1 28.2%, and balanced accuracy 59.8%.

The subject-level breakdown is stored in `docs/bciiv2a_subject_metrics.csv`.

## Interpretation

The two-second configuration is the production baseline. A one-second A/B run detected more artifact trials, but it also produced substantially more false positives because each trial had six chances to trigger the any-window aggregation rule. The two-second configuration therefore retains the higher F1, precision, and specificity. The current result should still be described as an initial signal-only baseline, not a validated production QC method.

The benchmark includes three EOG channels. Naive EOG outlier rules produced
many false positives, so EOG is not merged into automatic KEEP/WARN/REJECT.
Instead, fixed subject-relative robust-z thresholds confirm and order trials
already in the precision-primary review lane. The confirmed tier measures
61/119 = 51.3% precision and the strong tier measures 41/61 = 67.2% precision
on this training benchmark. The complete safety net is unchanged.

## Acceptance decision

Expert labels provide the reference acceptance set, not the algorithm threshold:

- Expert-clean trials may enter the benchmark clean feature set.
- Expert-artifact trials should be excluded or sent to manual review.
- `accepted_eeg_features.csv` is an expert-filtered reference output for benchmark analysis. It must not be presented as the algorithm's independent prediction.
- `benchmark_trial_results.csv` preserves both the expert decision and the independent pipeline decision so disagreements remain visible.

## Remaining uncertainties

The artifact annotation is trial-level and does not identify the artifact type or exact onset. Exporting 0 to 6 seconds improves coverage, but an artifact later in the approximately 7.5 second trial cycle may still be missed. TUAR remains a better future benchmark for artifact-type and temporal evaluation, but it requires authorized access.

## Qwen review queue

The separate offline Qwen review reaches 91.3% artifact recall on this labelled
training set (241 of 264 artifact trials), but only 12.4% specificity and 10.6%
precision. The 91.3% value must not be described as accuracy.

The integration preserves this recall by sending every algorithm or Qwen flag
to `combined_review_queue`. It does not automatically delete all queued data.
The requested union-KEEP output remains available as `retained`, while
`clean_eligible` requires both reviewers to keep every window.

For human review, P0 and P1 form a 273-trial precision-primary queue with 82
artifact trials, or 30.0% measured precision. P2 through P4 form a 2,011-trial
recall safety net that contributes the remaining 159 detected artifacts. This
raises the yield of the first review pass but does not change the full queue's
10.6% precision or 91.3% recall.

EOG confirmation further orders the first-pass queue. A fixed EOG median robust
z threshold of 4 yields 119 trials with 61 artifacts (51.3% precision); a
threshold of 7 yields 61 trials with 41 artifacts (67.2% precision). These are
post-hoc measurements. Processing uses no expert labels, and the EOG tiers do
not automatically delete data.

These Qwen results use the same BCI IV 2a training subjects and derived EEG
features. They are useful for internal triage design, but they do not establish
external generalization.
