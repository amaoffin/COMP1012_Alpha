# BCI IV 2a Benchmark Data

This directory was copied from:

`/Users/marc/Desktop/Task_2_Adaptive_EEG_QC_4Week_Courseware`

It contains the BCI IV 2a benchmark data used by the courseware. The actual
dataset in this package is BCI Competition IV Dataset 2a:

- `bciiv2a_csv/`: 2,592 trial CSV files, 22 EEG channels, 6 seconds per trial.
- `bciiv2a_ground_truth.csv`: trial-level expert artifact labels.
- `bciiv2a_manifest.json`: dataset provenance and subject summary.
- `evaluate_bciiv2a_qc.py`: courseware evaluator for sensitivity, specificity, precision, F1, and balanced accuracy.
- `BCIIV2A_ALGORITHM_VALIDATION.md`: copied validation note from the courseware.

The ALPHA and BETA passline projects use this shared directory through:

```bash
python3 scripts/run_bciiv2a_plugin_benchmark.py
```

Use `--max-trials 12` for a quick smoke test, or omit it for the full benchmark.
Passline verification is the 60% baseline for the code component: the code must
load through the ABI, return valid output, and perform an initial identification
of EEG data quality on the supplied examples. Higher benchmark credit requires
running this BCI IV 2a benchmark and reporting the evaluator outputs.

The BCI IV 2a benchmark grading points are:

- sensitivity
- specificity
- precision
- F1
- balanced accuracy

These values are reported in `benchmark_evaluation.csv`.
