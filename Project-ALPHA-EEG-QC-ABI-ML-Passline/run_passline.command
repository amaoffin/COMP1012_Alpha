#!/bin/bash

set -e

cd "$(dirname "$0")"

echo "Running Project Alpha EEG QC ML benchmark package..."
echo
python3 scripts/benchmark_complex.py
echo
echo "All checks passed."
echo "You can close this window."
