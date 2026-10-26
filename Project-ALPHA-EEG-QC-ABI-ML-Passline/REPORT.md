# Machine-Learning EEG Quality Control Plugin

> This supplied text is an illustrative passline draft. Students must expand it to meet every required length, evidence, and reference requirement in the project description.

## Abstract

_Required length: 800 to 1100 characters._

This project implements a machine-learning Python plugin for the Adaptive EEG QC Bash/CSV ABI. The supplied host creates EEG windows and stable window IDs. The student code receives reconstructed windows, extracts numeric features, and runs a frozen classifier to return validated QC decisions. This draft demonstrates the required hooks, reproducible model inference, and valid ABI output on the supplied examples.

## Introduction

_Required length: 600 to 1000 words._

EEG recordings are numeric time series collected from multiple channels. In a quality-control workflow, each recording is split into short windows so that poor-quality segments can be rejected without discarding the whole file. This project uses the supplied ABI host and CSV layout, but places the feature and decision logic in a student-written Python plugin.

The objective of this passline implementation is to satisfy the required ABI hooks while demonstrating basic ML inference in Python. The implementation uses only the standard library so that it remains easy to run in a basic student environment.

## Related Works

_Required length: 600 to 1400 words._

Discuss relevant EEG quality-control and artifact-detection research, introduce suitable machine-learning approaches, and compare them with the feature and model choices used in this project. This passline draft does not supply the literature review.

## Implementation

_Required length: 1400 to 2000 words. Include design justifications and critical code snippets._

### ABI Input and Artifact Loading

The ABI provides window metadata in `windows_v1.csv` and sample values in `eeg_samples_v1.csv`. The runner reconstructs a Python mapping for each window. Important fields include `window_id`, `sample_rate_hz`, and `eeg_samples`. The plugin preserves `window_id` so each output row can be linked to the original host-generated window.

### Feature Construction

The plugin extracts numeric features including channel count, sample count, maximum amplitude range, mean range, flatline ratio, mean absolute amplitude, RMS amplitude, peak absolute amplitude, and zero-crossing rate. These values are transformed into model inputs such as amplitude gates, flatline gates, RMS scale, and peak scale.

### ML Decision Logic

A frozen multiclass linear model scores `KEEP`, `WARN`, and `REJECT`. Softmax converts the scores into probabilities, and the highest-probability class becomes the ABI status. A second set of frozen linear heads predicts reason labels such as `AMP_RANGE` and `FLATLINE`. Empty windows become `ABSTAIN` because there is no usable signal for inference.

### Output Validation and Provenance

The implementation is in `student_ml_plugin/plugin.py`. `qc_decision_v1` returns the ABI QC status, quality score, and reason codes. `feature_extractor_v1` returns stable feature names and finite numeric values. Inference parameters are stored in `student_ml_plugin/params.json`. The model is embedded as Python constants, so no model download or external service is required.

## Benchmark

_Required length: 1000 to 1400 words. Include methodology, results, failure cases, tables, and graphs._

The tests cover clean, noisy, flatline, feature, malformed, and ABI integration cases. Run them with:

```bash
python3 -m unittest discover -s tests
```

Run the supplied passline benchmark with:

```bash
python3 scripts/benchmark_complex.py
```

Passing the passline benchmark means the code reaches the baseline for the code component. It does not establish performance on a real dataset. Where the BCI IV 2a benchmark is completed, report sensitivity, specificity, precision, F1, balanced accuracy, and representative failure cases.

## Discussion

_Required length: at least 1000 words._

This lightweight implementation demonstrates the ABI shape and model-scored inference, but it remains intentionally small. A fuller discussion should interpret the results, examine feature and threshold trade-offs, address data leakage and other limitations, compare alternative approaches, and explain lessons learned about Python design and reproducible evaluation.

## Conclusion and Future Work

_Required length: at least 600 words._

The project implements the two required Python hooks and produces valid ABI output rows. Future work could train and validate a model on a larger labeled EEG corpus, preserve richer model metadata, improve calibration, and evaluate performance across subjects and recording devices.

## References

_Requirement: at least 20 references in APA style._

Include Python and ABI documentation, EEG quality-control literature, and relevant machine-learning sources. The following categories are starting points, not a complete reference list:

- Python documentation for `csv`, `math`, `unittest`, and `pathlib`
- Scikit-learn documentation for linear classifiers
- MNE-Python tutorials for EEG background
