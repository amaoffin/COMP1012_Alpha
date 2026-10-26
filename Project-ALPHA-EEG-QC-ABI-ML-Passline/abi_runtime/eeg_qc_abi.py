"""Self-contained CSV runner for the EEG QC Python ABI.

This module mirrors the courseware Bash/CSV ABI host, but it is bundled inside
the student passline package so verification does not require Rust or cargo.
It only loads a student plugin, reconstructs windows from CSV files, validates
hook outputs, and writes the stable ABI output table.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

ABI_VERSION = "eeg_qc_bash_csv_abi_v1"
QC_HOOK = "qc_decision_v1"
FEATURE_HOOK = "feature_extractor_v1"
QC_STATUSES = {"KEEP", "WARN", "REJECT", "ABSTAIN"}
OUTPUT_COLUMNS = [
    "hook",
    "window_id",
    "status",
    "quality_score",
    "reason_codes",
    "feature_name",
    "feature_value",
    "error",
]


class QcDecisionHook(Protocol):
    def __call__(
        self, window: Mapping[str, Any], params: Mapping[str, str]
    ) -> Mapping[str, Any]:
        """Return status, optional quality_score, and optional reason_codes."""


class FeatureExtractorHook(Protocol):
    def __call__(
        self, window: Mapping[str, Any], params: Mapping[str, str]
    ) -> Mapping[str, float]:
        """Return named numeric features for one ABI window."""


@dataclass(frozen=True)
class PluginHooks:
    qc_decision_v1: Optional[QcDecisionHook] = None
    feature_extractor_v1: Optional[FeatureExtractorHook] = None


def read_windows(path: Path) -> dict[str, dict[str, Any]]:
    windows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("abi_version") != ABI_VERSION:
                raise ValueError(f"{path}: unsupported abi_version")
            window_id = require_cell(row, "window_id")
            if window_id in windows:
                raise ValueError(f"duplicate window_id in ABI input: {window_id}")
            windows[window_id] = {
                **row,
                "window_index": int(require_cell(row, "window_index")),
                "start_sec": float(require_cell(row, "start_sec")),
                "duration_sec": float(require_cell(row, "duration_sec")),
                "sample_rate_hz": float(require_cell(row, "sample_rate_hz")),
                "start_sample": int(require_cell(row, "start_sample")),
                "end_sample": int(require_cell(row, "end_sample")),
                "eeg_channel_count": int(require_cell(row, "eeg_channel_count")),
                "sample_count": int(require_cell(row, "sample_count")),
                "eog_channel_count": int(require_cell(row, "eog_channel_count")),
                "eeg_samples": [],
                "eeg_channel_names": [],
                "eog_samples": [],
                "eog_channel_names": [],
            }
    return windows


def attach_samples(path: Path, windows: dict[str, dict[str, Any]], key: str) -> None:
    names_key = {
        "eeg_samples": "eeg_channel_names",
        "eog_samples": "eog_channel_names",
    }[key]
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            window_id = require_cell(row, "window_id")
            if window_id not in windows:
                raise ValueError(f"{path}: unknown window_id {window_id}")
            channel_index = int(require_cell(row, "channel_index"))
            sample_index = int(require_cell(row, "sample_index"))
            value = float(require_cell(row, "value_uv"))
            matrix: list[list[float]] = windows[window_id][key]
            names: list[str] = windows[window_id][names_key]
            while len(matrix) <= channel_index:
                matrix.append([])
                names.append("")
            while len(matrix[channel_index]) <= sample_index:
                matrix[channel_index].append(0.0)
            matrix[channel_index][sample_index] = value
            name = row.get("channel_name", "")
            if name:
                names[channel_index] = name


def load_abi_windows(windows_path: Path, eeg_path: Path, eog_path: Path) -> list[dict[str, Any]]:
    windows = read_windows(windows_path)
    attach_samples(eeg_path, windows, "eeg_samples")
    if eog_path.exists():
        attach_samples(eog_path, windows, "eog_samples")
    return list(windows.values())


def validate_qc_result(window_id: str, result: Mapping[str, Any]) -> dict[str, str]:
    status = str(result.get("status", "")).upper()
    if status not in QC_STATUSES:
        raise ValueError(f"{window_id}: status must be one of {sorted(QC_STATUSES)}")
    output = empty_output_row(QC_HOOK, window_id)
    output["status"] = status
    if "quality_score" in result:
        score = float(result["quality_score"])
        if not 0.0 <= score <= 100.0:
            raise ValueError(f"{window_id}: quality_score must be in [0, 100]")
        output["quality_score"] = str(score)
    reason_codes = result.get("reason_codes", [])
    if reason_codes:
        output["reason_codes"] = ";".join(str(code) for code in reason_codes)
    return output


def validate_feature_result(window_id: str, result: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, value in result.items():
        output = empty_output_row(FEATURE_HOOK, window_id)
        output["feature_name"] = str(name)
        output["feature_value"] = str(float(value))
        rows.append(output)
    return rows


def run_plugin(
    windows_path: Path,
    eeg_path: Path,
    eog_path: Path,
    output_path: Path,
    hooks: PluginHooks,
    params: Optional[Mapping[str, str]] = None,
) -> None:
    params = params or {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for window in load_abi_windows(windows_path, eeg_path, eog_path):
            window_id = str(window["window_id"])
            if hooks.qc_decision_v1 is not None:
                writer.writerow(validate_qc_result(window_id, hooks.qc_decision_v1(window, params)))
            if hooks.feature_extractor_v1 is not None:
                for row in validate_feature_result(
                    window_id, hooks.feature_extractor_v1(window, params)
                ):
                    writer.writerow(row)


def load_plugin(path: Path) -> PluginHooks:
    spec = importlib.util.spec_from_file_location("student_plugin", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load plugin module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return PluginHooks(
        qc_decision_v1=getattr(module, QC_HOOK, None),
        feature_extractor_v1=getattr(module, FEATURE_HOOK, None),
    )


def parse_params(values: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--param must use key=value form: {value}")
        key, raw = value.split("=", 1)
        params[key] = raw
    return params


def empty_output_row(hook: str, window_id: str) -> dict[str, str]:
    return {
        "hook": hook,
        "window_id": window_id,
        "status": "",
        "quality_score": "",
        "reason_codes": "",
        "feature_name": "",
        "feature_value": "",
        "error": "",
    }


def require_cell(row: Mapping[str, Optional[str]], name: str) -> str:
    value = row.get(name)
    if value is None or value == "":
        raise ValueError(f"missing required CSV column value: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an EEG QC Bash/CSV ABI plugin")
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--eeg-samples", required=True, type=Path)
    parser.add_argument("--eog-samples", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--param", action="append", default=[])
    args = parser.parse_args()
    run_plugin(
        args.windows,
        args.eeg_samples,
        args.eog_samples,
        args.out,
        load_plugin(args.plugin),
        parse_params(args.param),
    )


if __name__ == "__main__":
    main()
