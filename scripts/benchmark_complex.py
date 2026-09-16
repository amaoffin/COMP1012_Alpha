"""Complex benchmark runner for the Project Alpha ML EEG QC plugin."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "student_ml_plugin" / "plugin.py"
ABI_RUNTIME = ROOT / "abi_runtime" / "eeg_qc_abi.py"
PARAMS_PATH = ROOT / "student_ml_plugin" / "params.json"

sys.path.insert(0, str(PLUGIN.parent))
import plugin  # noqa: E402


@dataclass(frozen=True)
class BenchmarkCase:
    window_id: str
    eeg_samples: list[list[float]]
    expected_status: str
    expected_reason_codes: frozenset[str]


def run_unit_tests() -> int:
    print("Running unit tests first...")
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT
    )
    if completed.returncode != 0:
        print("Unit tests failed.")
        return 1
    return 0


def run_sample_validation(params: dict[str, str]) -> int:
    print("Running sample ABI regression check...")
    sample_windows = ROOT / "sample_abi" / "windows_v1.csv"
    sample_eeg = ROOT / "sample_abi" / "eeg_samples_v1.csv"
    sample_eog = ROOT / "sample_abi" / "eog_samples_v1.csv"
    sample_output = ROOT / "sample_output" / "plugin_output_v1.csv"

    cmd = [
        sys.executable,
        str(ABI_RUNTIME),
        "--windows",
        str(sample_windows),
        "--eeg-samples",
        str(sample_eeg),
        "--eog-samples",
        str(sample_eog),
        "--plugin",
        str(PLUGIN),
        "--out",
        str(sample_output),
    ]
    param_flags = []
    for key, value in params.items():
        param_flags.extend(["--param", f"{key}={value}"])

    run = subprocess.run(cmd + param_flags, cwd=ROOT)
    if run.returncode != 0:
        print("Sample ABI regression check failed.")
        return 1
    return 0


def _load_params() -> dict[str, str]:
    with PARAMS_PATH.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    return {key: str(value) for key, value in values.items()}


def _cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            "w_keep_balanced",
            [[1.0, 2.0, 3.0, 4.0], [-2.0, -1.0, -0.5, 0.5]],
            "KEEP",
            frozenset({"NONE"}),
        ),
        BenchmarkCase(
            "w_warn_range_near_border",
            [[0.0, 99.9, 50.0, 99.8], [1.0, 2.0, 3.0, 2.0]],
            "WARN",
            frozenset({"AMP_RANGE"}),
        ),
        BenchmarkCase(
            "w_warn_range_exact",
            [[0.0, 100.0, 50.0], [1.0, 2.0, 3.0]],
            "WARN",
            frozenset({"AMP_RANGE"}),
        ),
        BenchmarkCase(
            "w_reject_range_border",
            [[0.0, 180.0, 90.0], [1.0, -1.0, 2.0]],
            "REJECT",
            frozenset({"AMP_RANGE"}),
        ),
        BenchmarkCase(
            "w_reject_range_clear",
            [[-250.0, 0.0, 180.1], [2.0, 3.0, 2.5]],
            "REJECT",
            frozenset({"AMP_RANGE"}),
        ),
        BenchmarkCase(
            "w_warn_flat_ratio",
            [[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [5.0, 6.0, 7.0]],
            "WARN",
            frozenset({"FLATLINE"}),
        ),
        BenchmarkCase(
            "w_reject_flat_ratio",
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            "REJECT",
            frozenset({"FLATLINE"}),
        ),
        BenchmarkCase(
            "w_warn_amp_and_flatline",
            [[-200.0, 0.0], [2.0, 2.0], [4.0, 4.0], [6.0, 7.0]],
            "REJECT",
            frozenset({"AMP_RANGE", "FLATLINE"}),
        ),
        BenchmarkCase(
            "w_empty_window",
            [],
            "ABSTAIN",
            frozenset({"EMPTY_WINDOW"}),
        ),
    ]


def run_direct_benchmark(
    params: dict[str, str], cases: list[BenchmarkCase]
) -> tuple[int, int, int]:
    correct_status = 0
    correct_reason = 0
    total = len(cases)
    status_misses: list[str] = []
    reason_misses = []

    for case in cases:
        result = plugin.qc_decision_v1(
            {"window_id": case.window_id, "eeg_samples": case.eeg_samples},
            params,
        )
        status = str(result["status"]).upper()
        reasons = frozenset(str(code) for code in result.get("reason_codes", []))

        status_ok = status == case.expected_status
        reason_ok = reasons == case.expected_reason_codes

        if status_ok:
            correct_status += 1
        else:
            status_misses.append(f"{case.window_id}: got {status}, expected {case.expected_status}")
        if reason_ok:
            correct_reason += 1
        else:
            reason_misses.append(
                f"{case.window_id}: got {sorted(reasons)}, expected {sorted(case.expected_reason_codes)}"
            )

        print(
            f"[DIRECT] {case.window_id}: status={status} reasons={sorted(reasons)} score={result['quality_score']}"
        )

    print(f"\nDIRECT benchmark accuracy: {correct_status}/{total} status matches")
    print(f"DIRECT reason accuracy: {correct_reason}/{total} reason matches")
    if status_misses:
        print("DIRECT status mismatches:")
        for msg in status_misses:
            print(" -", msg)
    if reason_misses:
        print("DIRECT reason mismatches:")
        for msg in reason_misses:
            print(" -", msg)

    return correct_status, correct_reason, total


def _write_windows_case(
    case: BenchmarkCase, index: int
) -> tuple[dict[str, str], list[tuple[int, str, int, float]]]:
    rows: list[tuple[int, str, int, float]] = []
    if not case.eeg_samples:
        sample_count = 0
        channel_count = 0
    else:
        sample_count = max(len(channel) for channel in case.eeg_samples)
        channel_count = len(case.eeg_samples)

    for channel_index, channel in enumerate(case.eeg_samples):
        channel_name = f"C{channel_index + 1}"
        for sample_index, value in enumerate(channel):
            rows.append((channel_index, channel_name, sample_index, float(value)))

    window_row = {
        "abi_version": "eeg_qc_bash_csv_abi_v1",
        "run_id": "run_benchmark",
        "session_id": "session_benchmark",
        "subject_id": "sub_benchmark",
        "window_id": case.window_id,
        "window_index": str(index),
        "start_sec": f"{index * 2}.0",
        "duration_sec": "2.0",
        "sample_rate_hz": "128.0",
        "start_sample": str(index * 256),
        "end_sample": str(index * 256 + max(sample_count, 0)),
        "eeg_channel_count": str(channel_count),
        "sample_count": str(sample_count),
        "eog_channel_count": "0",
    }
    return window_row, rows


def run_integration_benchmark(
    params: dict[str, str], cases: list[BenchmarkCase]
) -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        windows_csv = temp / "windows.csv"
        eeg_csv = temp / "eeg_samples.csv"
        eog_csv = temp / "eog_samples.csv"
        out_csv = temp / "benchmark_output.csv"

        windows_rows: list[dict[str, str]] = []
        eeg_rows: list[str] = []
        for index, case in enumerate(cases):
            window_row, rows = _write_windows_case(case, index)
            windows_rows.append(window_row)
            for row in rows:
                eeg_rows.append(
                    f"{case.window_id},{row[0]},{row[1]},{row[2]},{row[3]}"
                )

        with windows_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(windows_rows[0].keys()))
            writer.writeheader()
            for row in windows_rows:
                writer.writerow(row)

        with eeg_csv.open("w", encoding="utf-8", newline="") as handle:
            handle.write("window_id,channel_index,channel_name,sample_index,value_uv\n")
            if eeg_rows:
                handle.write("\n".join(eeg_rows))
                handle.write("\n")

        with eog_csv.open("w", encoding="utf-8", newline="") as handle:
            handle.write("window_id,channel_index,channel_name,sample_index,value_uv\n")

        param_flags = []
        for key, value in params.items():
            param_flags.extend(["--param", f"{key}={value}"])
        command = [
            sys.executable,
            str(ABI_RUNTIME),
            "--windows",
            str(windows_csv),
            "--eeg-samples",
            str(eeg_csv),
            "--eog-samples",
            str(eog_csv),
            "--plugin",
            str(PLUGIN),
            "--out",
            str(out_csv),
        ] + param_flags
        subprocess.run(command, check=True, cwd=ROOT)

        expected = {case.window_id: case.expected_status for case in cases}
        actual: dict[str, str] = {}
        with out_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["hook"] == "qc_decision_v1":
                    actual[row["window_id"]] = row["status"]

        correct = 0
        mismatches = []
        for window_id, exp_status in expected.items():
            pred_status = actual.get(window_id)
            if pred_status == exp_status:
                correct += 1
            else:
                mismatches.append(f"{window_id}: got {pred_status}, expected {exp_status}")

        print(f"\nINTEGRATION benchmark accuracy: {correct}/{len(expected)} status matches")
        if mismatches:
            print("INTEGRATION status mismatches:")
            for msg in mismatches:
                print(" -", msg)

        return correct


def main() -> int:
    print("Running Project Alpha EEG QC ML benchmark package...")
    if run_unit_tests() != 0:
        return 1

    params = _load_params()
    if run_sample_validation(params) != 0:
        return 1

    params = _load_params()
    cases = _cases()
    print(f"Loaded {len(cases)} benchmark cases from script-level plan.")

    correct_status, correct_reason, total = run_direct_benchmark(params, cases)
    integration_correct = run_integration_benchmark(params, cases)
    if correct_status != total or correct_reason != total or integration_correct != total:
        print("Complex benchmark did not match expected behavior.")
        return 1
    print("Complex benchmark passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
