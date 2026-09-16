# Student ML Plugin

This folder contains a minimal machine-learning EEG QC plugin for Project Alpha.
It implements both required ABI hooks:

- `qc_decision_v1(window, params)`
- `feature_extractor_v1(window, params)`

The ABI is shared with the non-ML version. The runner still passes one window
mapping into the plugin and expects the same output schema.

## Model

The plugin uses a frozen multiclass linear classifier embedded directly in
`plugin.py`:

- `MODEL_WEIGHTS` predicts `KEEP`, `WARN`, or `REJECT`.
- `REASON_WEIGHTS` predicts explanation labels such as `AMP_RANGE` and
  `FLATLINE`.
- `decision_temperature` controls softmax confidence calibration.
- `reason_probability_cutoff` controls which reason labels are emitted.
- `model_manifest.json` records model provenance and label metadata.
- `feature_manifest.json` records feature names and model feature order.

No third-party package, model download, random decision, or external service is
required.

## Benchmark Boundary

The passline benchmark checks that this code can load through the ABI, return
valid output, and perform an initial identification of EEG data quality on the
supplied examples. That passline result corresponds to the 60% baseline for the
code component.

BCI IV 2a benchmark credit is based on sensitivity, specificity, precision, F1, and balanced accuracy from `benchmark_evaluation.csv`.
