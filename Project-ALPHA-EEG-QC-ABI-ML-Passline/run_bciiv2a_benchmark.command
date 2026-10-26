#!/bin/bash

set -e

cd "$(dirname "$0")"

echo "Running Project Alpha EEG QC ML plugin on BCI IV 2a benchmark..."
echo
python3 scripts/run_bciiv2a_plugin_benchmark.py
echo
echo "BCI IV 2a benchmark complete."
echo "You can close this window."
