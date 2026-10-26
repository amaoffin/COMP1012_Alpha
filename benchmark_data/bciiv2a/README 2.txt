• ## BCI IV 2a Benchmark Setup Guide

  The setup procedure is the same for both Project Alpha and Project Beta.

  You will receive the BCI IV 2a benchmark data as a ZIP file. Download and extract the ZIP file, then place the
  extracted benchmark_data folder in the root directory of your student package.

  The required folder structure is:

  COMP1012-2027-STUDENT-PACKAGE/
  ├── benchmark_data/
  │   └── bciiv2a/
  │       ├── bciiv2a_csv/
  │       ├── bciiv2a_ground_truth.csv
  │       ├── bciiv2a_manifest.json
  │       └── evaluate_bciiv2a_qc.py
  │
  ├── Project-ALPHA-EEG-QC-ABI-ML-Passline/
  │
  └── Project-BETA-EEG-QC-ABI-NonML-Passline/

  Important:

  - Place benchmark_data in the same root directory as the Project Alpha and Project Beta folders.
  - Do not place benchmark_data inside either project folder.
  - Extract the ZIP file before running the benchmark.
  - Do not run the benchmark directly from inside the ZIP file.
  - Students only need to run the benchmark for their chosen project.

  ### Project Alpha

  Short smoke test:

  cd Project-ALPHA-EEG-QC-ABI-ML-Passline
  python3 scripts/run_bciiv2a_plugin_benchmark.py --max-trials 12

  Full benchmark:

  python3 scripts/run_bciiv2a_plugin_benchmark.py

  ### Project Beta

  Short smoke test:

  cd Project-BETA-EEG-QC-ABI-NonML-Passline
  python3 scripts/run_bciiv2a_plugin_benchmark.py --max-trials 12

  Full benchmark:

  python3 scripts/run_bciiv2a_plugin_benchmark.py

  ### Results

  The benchmark results will be written inside the chosen project:

  benchmark_output/bciiv2a_plugin/

  The main benchmark evidence is:

  benchmark_evaluation.csv

  This file reports sensitivity, specificity, precision, F1, and balanced accuracy. Keep it for submission and discuss
  the results and relevant failure cases in your report.