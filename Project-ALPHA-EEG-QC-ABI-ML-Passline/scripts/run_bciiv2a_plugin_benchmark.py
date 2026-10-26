"""Run the student Python ABI plugin on the BCI IV 2a benchmark.

The benchmark data is copied from Task_2_Adaptive_EEG_QC_4Week_Courseware into:

    ../../benchmark_data/bciiv2a/

This script keeps the student-facing ABI hooks unchanged. It reads each BCI
trial CSV, reconstructs 2-second ABI windows, calls qc_decision_v1 and
feature_extractor_v1, then writes courseware-compatible benchmark outputs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = PROJECT_ROOT.parent
BENCHMARK_ROOT = SUITE_ROOT / "benchmark_data" / "bciiv2a"
DEFAULT_RAW_DIR = BENCHMARK_ROOT / "bciiv2a_csv"
DEFAULT_GROUND_TRUTH = BENCHMARK_ROOT / "bciiv2a_ground_truth.csv"
DEFAULT_EVALUATOR = BENCHMARK_ROOT / "evaluate_bciiv2a_qc.py"
DEFAULT_OUT = PROJECT_ROOT / "benchmark_output" / "bciiv2a_plugin"
EEG_CHANNELS = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]


def _plugin_path() -> Path:
    ml_plugin = PROJECT_ROOT / "student_ml_plugin" / "plugin.py"
    if ml_plugin.exists():
        return ml_plugin
    return PROJECT_ROOT / "student_non_ml_plugin" / "plugin.py"


def _params_path() -> Path:
    ml_params = PROJECT_ROOT / "student_ml_plugin" / "params.json"
    if ml_params.exists():
        return ml_params
    return PROJECT_ROOT / "student_non_ml_plugin" / "params.json"


def load_plugin(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("student_plugin", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load plugin module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_params(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key: str(value) for key, value in payload.items()}


def parse_subjects(value: str) -> set[str]:
    if not value:
        return set()
    subjects: set[str] = set()
    for part in value.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            for number in range(int(start), int(end) + 1):
                subjects.add(f"a{number:02d}")
        elif part.startswith("a"):
            subjects.add(part)
        else:
            subjects.add(f"a{int(part):02d}")
    return subjects


def trial_to_channels(path: Path) -> tuple[list[list[float]], str, str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        channels = [[] for _ in EEG_CHANNELS]
        subject_id = ""
        session_id = ""
        condition = ""
        for row in reader:
            subject_id = row["subject_id"]
            session_id = row["session_id"]
            condition = row["condition"]
            for index, channel_name in enumerate(EEG_CHANNELS):
                channels[index].append(float(row[channel_name]))
    return channels, subject_id, session_id, condition


def iter_windows(
    channels: list[list[float]],
    session_id: str,
    sample_rate_hz: float,
    window_seconds: float,
) -> list[dict[str, object]]:
    window_size = int(round(sample_rate_hz * window_seconds))
    if window_size <= 0:
        raise ValueError("window size must be positive")
    sample_count = max((len(channel) for channel in channels), default=0)
    windows: list[dict[str, object]] = []
    for window_index, start in enumerate(range(0, sample_count, window_size)):
        end = min(start + window_size, sample_count)
        if start >= end:
            continue
        windows.append(
            {
                "abi_version": "eeg_qc_bciiv2a_plugin_benchmark_v1",
                "window_id": f"{session_id}_w{window_index:03d}",
                "window_index": window_index,
                "start_sec": start / sample_rate_hz,
                "duration_sec": (end - start) / sample_rate_hz,
                "sample_rate_hz": sample_rate_hz,
                "start_sample": start,
                "end_sample": end,
                "eeg_channel_count": len(channels),
                "sample_count": end - start,
                "eog_channel_count": 0,
                "eeg_samples": [channel[start:end] for channel in channels],
                "eeg_channel_names": EEG_CHANNELS,
                "eog_samples": [],
                "eog_channel_names": [],
            }
        )
    return windows


def run_courseware_evaluator(
    evaluator: Path,
    ground_truth: Path,
    output_dir: Path,
) -> None:
    if not evaluator.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(evaluator),
            "--ground-truth",
            str(ground_truth),
            "--pipeline-output",
            str(output_dir),
        ],
        check=True,
        cwd=SUITE_ROOT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Python ABI plugin on the BCI IV 2a benchmark."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plugin", type=Path, default=_plugin_path())
    parser.add_argument("--params", type=Path, default=_params_path())
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--subjects", default="", help="Example: 1,2,5-9")
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--sample-rate", type=float, default=250.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument(
        "--skip-courseware-evaluator",
        action="store_true",
        help="Write plugin outputs without running evaluate_bciiv2a_qc.py.",
    )
    args = parser.parse_args()

    if not args.raw_dir.exists():
        parser.error(f"raw benchmark directory not found: {args.raw_dir}")
    if not args.ground_truth.exists():
        parser.error(f"ground truth not found: {args.ground_truth}")

    plugin = load_plugin(args.plugin)
    params = load_params(args.params)
    selected_subjects = parse_subjects(args.subjects)
    truth_rows = read_csv(args.ground_truth)
    if selected_subjects:
        truth_rows = [
            row for row in truth_rows
            if row["subject_id"].lower() in selected_subjects
        ]
    if args.max_trials > 0:
        truth_rows = truth_rows[: args.max_trials]
    if not truth_rows:
        raise SystemExit("no benchmark trials selected")

    output_dir = args.out
    output_dir.mkdir(parents=True, exist_ok=True)
    window_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    missing_files: list[str] = []

    for trial_index, truth in enumerate(truth_rows, start=1):
        csv_name = Path(truth["csv_file"]).name
        trial_path = args.raw_dir / csv_name
        if not trial_path.exists():
            missing_files.append(str(trial_path))
            continue

        channels, subject_id, session_id, condition = trial_to_channels(trial_path)
        for window in iter_windows(
            channels,
            session_id,
            args.sample_rate,
            args.window_seconds,
        ):
            decision = plugin.qc_decision_v1(window, params)
            status = str(decision.get("status", "")).upper()
            reasons = ";".join(str(code) for code in decision.get("reason_codes", []))
            quality_score = decision.get("quality_score", "")
            window_rows.append(
                {
                    "window_id": window["window_id"],
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "condition": condition,
                    "window_index": window["window_index"],
                    "start_sec": f"{float(window['start_sec']):.6f}",
                    "duration_sec": f"{float(window['duration_sec']):.6f}",
                    "qc_status": status,
                    "fixed_qc_status": status,
                    "quality_score": quality_score,
                    "reason_codes": reasons,
                    "expert_artifact": truth["expert_artifact"],
                }
            )
            decision_rows.append(
                {
                    "trial_index": trial_index,
                    "window_id": window["window_id"],
                    "session_id": session_id,
                    "status": status,
                    "quality_score": quality_score,
                    "reason_codes": reasons,
                }
            )
            features = plugin.feature_extractor_v1(window, params)
            for name, value in features.items():
                feature_rows.append(
                    {
                        "window_id": window["window_id"],
                        "session_id": session_id,
                        "feature_name": name,
                        "feature_value": float(value),
                    }
                )

    if missing_files:
        preview = "\n".join(missing_files[:5])
        raise SystemExit(f"{len(missing_files)} benchmark files missing:\n{preview}")

    write_csv(
        output_dir / "eeg_windows.csv",
        window_rows,
        [
            "window_id",
            "subject_id",
            "session_id",
            "condition",
            "window_index",
            "start_sec",
            "duration_sec",
            "qc_status",
            "fixed_qc_status",
            "quality_score",
            "reason_codes",
            "expert_artifact",
        ],
    )
    write_csv(
        output_dir / "plugin_window_decisions.csv",
        decision_rows,
        [
            "trial_index",
            "window_id",
            "session_id",
            "status",
            "quality_score",
            "reason_codes",
        ],
    )
    write_csv(
        output_dir / "plugin_feature_rows.csv",
        feature_rows,
        ["window_id", "session_id", "feature_name", "feature_value"],
    )
    write_csv(
        output_dir / "eeg_features.csv",
        feature_rows,
        ["window_id", "session_id", "feature_name", "feature_value"],
    )
    subset_truth_path = output_dir / "selected_ground_truth.csv"
    write_csv(subset_truth_path, truth_rows, list(truth_rows[0]))

    if not args.skip_courseware_evaluator:
        run_courseware_evaluator(args.evaluator, subset_truth_path, output_dir)

    print(f"Processed trials: {len(truth_rows)}")
    print(f"Processed windows: {len(window_rows)}")
    metrics_path = output_dir / "benchmark_evaluation.csv"
    if metrics_path.exists():
        for row in read_csv(metrics_path):
            if row["method"] == "adaptive_warn_or_reject":
                print(
                    "adaptive_warn_or_reject: "
                    f"sensitivity={row['sensitivity']} "
                    f"specificity={row['specificity']} "
                    f"precision={row['precision']} "
                    f"f1={row['f1']} "
                    f"balanced_accuracy={row['balanced_accuracy']}"
                )
    print(f"Wrote: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
