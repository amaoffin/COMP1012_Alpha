# EEG QC ABI Template

Use this file as the alignment checklist for the Python ABI. The runtime loads
student code by exact function name, so spelling, argument order, and return
keys must match.

This project is intended to match the workload and difficulty of a Year 1
semester project built around a modular applied Python system. The supplied
passline example is deliberately the simplest acceptable code implementation.
Passing the passline benchmark means the code can load through the ABI, return
valid output, and perform an initial identification of EEG data quality on the
supplied examples. In this package, passline corresponds to the 60% baseline for
the code component. Higher marks should come from clearer design, better
features, stronger tests, better explanations, justified extensions, and BCI IV
2a benchmark analysis, not from changing the ABI. The BCI IV 2a benchmark
grading points are sensitivity, specificity, precision, F1, and balanced
accuracy.

The intended module-level difficulty is comparable to a project with separate
question-bank, response-tracking, adaptive-learning, feedback, visualization,
and learner-modeling modules. In this EEG QC project, the corresponding modules
are ABI CSV inputs, per-window QC outputs and feature rows, model thresholds
and confidence mapping, reason codes and false keep/false reject discussion,
report plots, and window-quality profiling with benchmark cases.

## ABI Identity

- ABI version in `sample_abi/windows_v1.csv`: `eeg_qc_bash_csv_abi_v1`
- ABI runner: `abi_runtime/eeg_qc_abi.py`
- Student plugin file for this project: `student_ml_plugin/plugin.py`
- Optional parameter file for this project: `student_ml_plugin/params.json`
- Model manifest for this project: `student_ml_plugin/model_manifest.json`
- Feature manifest for this project: `student_ml_plugin/feature_manifest.json`
- Output file shape: `sample_output/plugin_output_v1.csv`

## Required Hook Names

The runner looks for these exact top-level function names in the plugin module:

- `qc_decision_v1`
- `feature_extractor_v1`

Do not rename them, nest them inside a class, or expose only differently named
wrappers. Helper functions may use any private names.

## Function Signatures

```python
from typing import Any, Mapping


def qc_decision_v1(
    window: Mapping[str, Any], params: Mapping[str, Any]
) -> Mapping[str, Any]:
    ...


def feature_extractor_v1(
    window: Mapping[str, Any], params: Mapping[str, Any]
) -> Mapping[str, float]:
    ...
```

The ABI runner passes one reconstructed window at a time. `params` contains
values from CLI `--param key=value` arguments when using the ABI runner
directly. Project tests may also pass values loaded from `params.json`.

## `qc_decision_v1` Return Contract

Return a mapping with these keys:

- `status`: required string. Must be one of `KEEP`, `WARN`, `REJECT`,
  `ABSTAIN`.
- `quality_score`: optional number. If present, it must be finite and in
  `[0, 100]`.
- `reason_codes`: optional list of strings. Common values in this project are
  `NONE`, `AMP_RANGE`, `FLATLINE`, and `EMPTY_WINDOW`.

Example:

```python
return {
    "status": "KEEP",
    "quality_score": 95.0,
    "reason_codes": ["NONE"],
}
```

## `feature_extractor_v1` Return Contract

Return a mapping from feature names to numeric values. Each value must be
convertible to `float`.

Example:

```python
return {
    "range_max_uv": 42.0,
    "flatline_ratio": 0.0,
}
```

The runner writes one `feature_extractor_v1` output row per returned feature.

## Window Mapping

Each `window` contains metadata from `windows_v1.csv` plus attached sample
arrays:

- `window_id`
- `window_index`
- `start_sec`
- `duration_sec`
- `sample_rate_hz`
- `start_sample`
- `end_sample`
- `eeg_channel_count`
- `sample_count`
- `eog_channel_count`
- `eeg_samples`: list of EEG channels, each a list of numeric microvolt values
- `eeg_channel_names`: list of EEG channel names
- `eog_samples`: list of EOG channels, each a list of numeric microvolt values
- `eog_channel_names`: list of EOG channel names

The plugin should preserve `window_id` implicitly by returning only hook output;
the runner attaches the `window_id` to each output row.

## Minimal Plugin Skeleton

```python
from __future__ import annotations

from typing import Any, Mapping


def _float_param(params: Mapping[str, Any], name: str, default: float) -> float:
    try:
        return float(params.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric parameter: {name}") from exc


def qc_decision_v1(
    window: Mapping[str, Any], params: Mapping[str, Any]
) -> Mapping[str, Any]:
    eeg_samples = window.get("eeg_samples", [])
    if not eeg_samples:
        return {
            "status": "ABSTAIN",
            "quality_score": 0.0,
            "reason_codes": ["EMPTY_WINDOW"],
        }

    # Replace this with ML inference logic for Project Alpha.
    return {
        "status": "KEEP",
        "quality_score": 95.0,
        "reason_codes": ["NONE"],
    }


def feature_extractor_v1(
    window: Mapping[str, Any], params: Mapping[str, Any]
) -> Mapping[str, float]:
    eeg_samples = window.get("eeg_samples", [])
    channel_count = len(eeg_samples) if isinstance(eeg_samples, list) else 0
    return {
        "channel_count": float(channel_count),
    }
```

## Optional High-Score Directions

These are suggestions for stronger submissions, not mandatory requirements:

- Add at least five useful signal-quality features instead of relying only on
  the minimal range/flatline examples.
- Emit richer reason codes, such as `AMP_RANGE`, `FLATLINE`, `LOW_VARIANCE`,
  and `EOG_ARTIFACT`, when supported by the implementation.
- Handle malformed, empty, partial, or unusual windows clearly and
  reproducibly.
- Submit additional student-designed test cases beyond the bundled examples.
- Include failure analysis in the report, explaining where the approach works
  and where it can fail.
- For ML submissions, document the training/evaluation split or explain the
  frozen model rationale and calibration choices.

## ABI Usage

From this project folder, run the full passline check:

```bash
python3 scripts/benchmark_complex.py
```

Run only unit tests:

```bash
python3 -m unittest discover -s tests
```

Run the ABI runner directly:

```bash
python3 abi_runtime/eeg_qc_abi.py \
  --windows sample_abi/windows_v1.csv \
  --eeg-samples sample_abi/eeg_samples_v1.csv \
  --eog-samples sample_abi/eog_samples_v1.csv \
  --plugin student_ml_plugin/plugin.py \
  --out sample_output/plugin_output_v1.csv
```

Pass runtime parameters directly:

```bash
python3 abi_runtime/eeg_qc_abi.py \
  --windows sample_abi/windows_v1.csv \
  --eeg-samples sample_abi/eeg_samples_v1.csv \
  --eog-samples sample_abi/eog_samples_v1.csv \
  --plugin student_ml_plugin/plugin.py \
  --out sample_output/plugin_output_v1.csv \
  --param decision_temperature=1.0 \
  --param reason_probability_cutoff=0.5
```

## Output CSV Columns

The ABI runner writes these columns:

- `hook`
- `window_id`
- `status`
- `quality_score`
- `reason_codes`
- `feature_name`
- `feature_value`
- `error`

Rows from `qc_decision_v1` fill `status`, `quality_score`, and `reason_codes`.
Rows from `feature_extractor_v1` fill `feature_name` and `feature_value`.
