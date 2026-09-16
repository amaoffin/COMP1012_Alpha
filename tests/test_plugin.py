from __future__ import annotations

import csv
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student_ml_plugin"))

import plugin  # noqa: E402


PARAMS = {
    "model_version": "frozen_linear_eeg_qc_v1",
    "decision_temperature": "1.0",
    "reason_probability_cutoff": "0.5",
    "flatline_range_epsilon": "0.000001",
}


class PluginUnitTests(unittest.TestCase):
    def test_clean_window_is_keep(self) -> None:
        window = {"window_id": "w_clean", "eeg_samples": [[1.0, 2.0, 3.0], [-2.0, 0.0, 2.0]]}
        result = plugin.qc_decision_v1(window, PARAMS)
        self.assertEqual(result["status"], "KEEP")
        self.assertEqual(result["reason_codes"], ["NONE"])

    def test_large_range_is_reject(self) -> None:
        window = {"window_id": "w_noisy", "eeg_samples": [[0.0, 250.0, -260.0, 10.0]]}
        result = plugin.qc_decision_v1(window, PARAMS)
        self.assertEqual(result["status"], "REJECT")
        self.assertIn("AMP_RANGE", result["reason_codes"])

    def test_flatline_window_is_reject(self) -> None:
        window = {"window_id": "w_flat", "eeg_samples": [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]}
        result = plugin.qc_decision_v1(window, PARAMS)
        self.assertEqual(result["status"], "REJECT")
        self.assertIn("FLATLINE", result["reason_codes"])

    def test_feature_values_are_finite(self) -> None:
        window = {"window_id": "w_features", "eeg_samples": [[-1.0, 0.0, 1.0]]}
        features = plugin.feature_extractor_v1(window, PARAMS)
        self.assertIn("range_max_uv", features)
        self.assertIn("ml_range_gate", features)
        for value in features.values():
            self.assertTrue(math.isfinite(float(value)))

    def test_non_finite_sample_fails(self) -> None:
        window = {"window_id": "w_bad", "eeg_samples": [[float("nan")]]}
        with self.assertRaises(ValueError):
            plugin.qc_decision_v1(window, PARAMS)


class AbiIntegrationTest(unittest.TestCase):
    def test_runner_writes_valid_output(self) -> None:
        runner = ROOT / "abi_runtime" / "eeg_qc_abi.py"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            windows = temp / "windows_v1.csv"
            eeg = temp / "eeg_samples_v1.csv"
            eog = temp / "eog_samples_v1.csv"
            out = temp / "plugin_output_v1.csv"

            windows.write_text(
                "abi_version,run_id,session_id,subject_id,window_id,window_index,"
                "start_sec,duration_sec,sample_rate_hz,start_sample,end_sample,"
                "eeg_channel_count,sample_count,eog_channel_count\n"
                "eeg_qc_bash_csv_abi_v1,run_demo,sess_demo,subj_demo,w1,0,"
                "0.0,2.0,128.0,0,3,1,3,0\n",
                encoding="utf-8",
            )
            eeg.write_text(
                "window_id,channel_index,channel_name,sample_index,value_uv\n"
                "w1,0,C3,0,1.0\n"
                "w1,0,C3,1,2.0\n"
                "w1,0,C3,2,3.0\n",
                encoding="utf-8",
            )
            eog.write_text(
                "window_id,channel_index,channel_name,sample_index,value_uv\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "--windows",
                    str(windows),
                    "--eeg-samples",
                    str(eeg),
                    "--eog-samples",
                    str(eog),
                    "--plugin",
                    str(ROOT / "student_ml_plugin" / "plugin.py"),
                    "--out",
                    str(out),
                ],
                check=True,
            )

            with out.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 2)
            self.assertEqual(rows[0]["hook"], "qc_decision_v1")
            self.assertEqual(rows[0]["window_id"], "w1")
            self.assertEqual(rows[0]["status"], "KEEP")


if __name__ == "__main__":
    unittest.main()
