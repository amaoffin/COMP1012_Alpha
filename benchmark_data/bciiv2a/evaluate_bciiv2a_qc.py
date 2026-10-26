#!/usr/bin/env python3
"""Evaluate adaptive and fixed QC against BCI IV 2a artifact flags."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import shutil


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metrics(
    method: str,
    truth: dict[str, int],
    predictions: dict[str, int],
) -> dict[str, object]:
    sessions = sorted(truth)
    tp = sum(truth[s] == 1 and predictions.get(s, 0) == 1 for s in sessions)
    fp = sum(truth[s] == 0 and predictions.get(s, 0) == 1 for s in sessions)
    tn = sum(truth[s] == 0 and predictions.get(s, 0) == 0 for s in sessions)
    fn = sum(truth[s] == 1 and predictions.get(s, 0) == 0 for s in sessions)
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * precision * sensitivity, precision + sensitivity)
    balanced_accuracy = (sensitivity + specificity) / 2
    accuracy = safe_div(tp + tn, len(sessions))
    return {
        "method": method,
        "trials": len(sessions),
        "artifact_trials": sum(truth.values()),
        "clean_trials": len(sessions) - sum(truth.values()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": f"{sensitivity:.6f}",
        "specificity": f"{specificity:.6f}",
        "precision": f"{precision:.6f}",
        "f1": f"{f1:.6f}",
        "balanced_accuracy": f"{balanced_accuracy:.6f}",
        "accuracy": f"{accuracy:.6f}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score EEG QC against BCI IV 2a expert artifact flags."
    )
    parser.add_argument(
        "--ground-truth",
        default="data/bciiv2a_ground_truth.csv",
    )
    parser.add_argument(
        "--pipeline-output",
        default="out/bciiv2a",
    )
    parser.add_argument(
        "--subject-metrics-output",
        default="",
        help=(
            "Optional subject-level metrics CSV path. "
            "Defaults to docs/bciiv2a_subject_metrics.csv for out/bciiv2a."
        ),
    )
    args = parser.parse_args()

    ground_truth_path = Path(args.ground_truth)
    output_dir = Path(args.pipeline_output)
    window_path = output_dir / "eeg_windows.csv"
    if not ground_truth_path.exists():
        parser.error(f"ground truth not found: {ground_truth_path}")
    if not window_path.exists():
        parser.error(f"pipeline output not found: {window_path}")

    truth_rows = read_csv(ground_truth_path)
    window_rows = read_csv(window_path)
    truth = {
        row["session_id"].lower(): int(row["expert_artifact"])
        for row in truth_rows
    }

    by_session: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in window_rows:
        by_session[row["session_id"].lower()].append(row)

    missing = sorted(set(truth) - set(by_session))
    unknown = sorted(set(by_session) - set(truth))
    if missing:
        raise SystemExit(
            f"{len(missing)} ground-truth sessions are missing from pipeline output"
        )
    if unknown:
        raise SystemExit(
            f"{len(unknown)} pipeline sessions have no ground-truth label"
        )

    predictions: dict[str, dict[str, int]] = {
        "adaptive_warn_or_reject": {},
        "adaptive_reject_only": {},
        "fixed_any_channel_threshold": {},
    }
    trial_rows: list[dict[str, object]] = []
    truth_by_session = {
        row["session_id"].lower(): row for row in truth_rows
    }

    for session_id in sorted(truth):
        windows = sorted(
            by_session[session_id],
            key=lambda row: int(row["window_index"]),
        )
        adaptive_statuses = [row["qc_status"] for row in windows]
        fixed_statuses = [row["fixed_qc_status"] for row in windows]
        adaptive_warn_or_reject = int(
            any(status in {"WARN", "REJECT"} for status in adaptive_statuses)
        )
        adaptive_reject_only = int(
            any(status == "REJECT" for status in adaptive_statuses)
        )
        fixed_positive = int(any(status == "REJECT" for status in fixed_statuses))

        predictions["adaptive_warn_or_reject"][session_id] = (
            adaptive_warn_or_reject
        )
        predictions["adaptive_reject_only"][session_id] = adaptive_reject_only
        predictions["fixed_any_channel_threshold"][session_id] = fixed_positive

        source = truth_by_session[session_id]
        trial_rows.append(
            {
                "subject_id": source["subject_id"],
                "session_id": session_id,
                "run_index": source["run_index"],
                "trial_number": source["trial_number"],
                "class_name": source["class_name"],
                "expert_artifact": truth[session_id],
                "expert_recommended_status": (
                    "REJECT" if truth[session_id] else "ACCEPT"
                ),
                "window_count": len(windows),
                "adaptive_statuses": ";".join(adaptive_statuses),
                "fixed_statuses": ";".join(fixed_statuses),
                "adaptive_warn_or_reject": adaptive_warn_or_reject,
                "adaptive_reject_only": adaptive_reject_only,
                "fixed_any_channel_threshold": fixed_positive,
                "pipeline_recommended_status": (
                    "FLAG" if adaptive_warn_or_reject else "ACCEPT"
                ),
                "expert_pipeline_agree": int(
                    truth[session_id] == adaptive_warn_or_reject
                ),
            }
        )

    metric_rows = [
        metrics(method, truth, method_predictions)
        for method, method_predictions in predictions.items()
    ]

    subject_rows: list[dict[str, object]] = []
    subject_ids = sorted({row["subject_id"] for row in truth_rows})
    for subject_id in subject_ids:
        subject_sessions = {
            row["session_id"].lower()
            for row in truth_rows
            if row["subject_id"] == subject_id
        }
        subject_truth = {
            session_id: truth[session_id]
            for session_id in sorted(subject_sessions)
        }
        subject_predictions = {
            session_id: predictions["adaptive_warn_or_reject"][session_id]
            for session_id in sorted(subject_sessions)
        }
        row = metrics(
            "adaptive_warn_or_reject",
            subject_truth,
            subject_predictions,
        )
        subject_rows.append(
            {
                "subject_id": subject_id,
                "trials": row["trials"],
                "artifact_trials": row["artifact_trials"],
                "tp": row["tp"],
                "fp": row["fp"],
                "tn": row["tn"],
                "fn": row["fn"],
                "sensitivity": row["sensitivity"],
                "specificity": row["specificity"],
                "precision": row["precision"],
                "f1": row["f1"],
                "balanced_accuracy": row["balanced_accuracy"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "benchmark_evaluation.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    subject_metrics_path: Path | None = None
    if args.subject_metrics_output:
        subject_metrics_path = Path(args.subject_metrics_output)
    elif output_dir == Path("out/bciiv2a"):
        subject_metrics_path = Path("docs/bciiv2a_subject_metrics.csv")
    if subject_metrics_path is not None:
        subject_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with subject_metrics_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(subject_rows[0]),
            )
            writer.writeheader()
            writer.writerows(subject_rows)

    trial_path = output_dir / "benchmark_trial_results.csv"
    with trial_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)

    normalized_truth_path = output_dir / "benchmark_ground_truth.csv"
    shutil.copyfile(ground_truth_path, normalized_truth_path)

    acceptance_summary = [
        {
            "decision_source": "expert_ground_truth",
            "accept_trials": len(truth) - sum(truth.values()),
            "flag_or_reject_trials": sum(truth.values()),
            "total_trials": len(truth),
        },
        {
            "decision_source": "adaptive_warn_or_reject",
            "accept_trials": sum(
                prediction == 0
                for prediction in predictions["adaptive_warn_or_reject"].values()
            ),
            "flag_or_reject_trials": sum(
                predictions["adaptive_warn_or_reject"].values()
            ),
            "total_trials": len(truth),
        },
    ]
    acceptance_path = output_dir / "benchmark_acceptance_summary.csv"
    with acceptance_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "decision_source",
                "accept_trials",
                "flag_or_reject_trials",
                "total_trials",
            ],
        )
        writer.writeheader()
        writer.writerows(acceptance_summary)

    clean_sessions = {
        session_id for session_id, expert_artifact in truth.items()
        if expert_artifact == 0
    }
    artifact_sessions = set(truth) - clean_sessions

    window_rows = read_csv(output_dir / "eeg_windows.csv")
    feature_rows = read_csv(output_dir / "eeg_features.csv")

    accepted_window_rows = [
        row for row in window_rows
        if row["session_id"].lower() in clean_sessions
    ]
    excluded_window_rows = [
        row for row in window_rows
        if row["session_id"].lower() in artifact_sessions
    ]
    accepted_feature_rows = [
        row for row in feature_rows
        if row["session_id"].lower() in clean_sessions
    ]
    excluded_feature_rows = [
        row for row in feature_rows
        if row["session_id"].lower() in artifact_sessions
    ]

    def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    accepted_windows_path = output_dir / "accepted_eeg_windows.csv"
    excluded_windows_path = output_dir / "expert_artifact_eeg_windows.csv"
    accepted_features_path = output_dir / "accepted_eeg_features.csv"
    excluded_features_path = output_dir / "expert_artifact_eeg_features.csv"
    write_rows(accepted_windows_path, accepted_window_rows)
    write_rows(excluded_windows_path, excluded_window_rows)
    write_rows(accepted_features_path, accepted_feature_rows)
    write_rows(excluded_features_path, excluded_feature_rows)

    print("BCI Competition IV Dataset 2a QC benchmark")
    print("============================================")
    print(f"Trials: {len(truth)}")
    print(f"Expert artifact trials: {sum(truth.values())}")
    print(f"Clean trials: {len(truth) - sum(truth.values())}")
    print()
    for row in metric_rows:
        print(row["method"])
        print(
            f"  TP={row['tp']} FP={row['fp']} "
            f"TN={row['tn']} FN={row['fn']}"
        )
        print(
            f"  sensitivity={row['sensitivity']} "
            f"specificity={row['specificity']} "
            f"precision={row['precision']} "
            f"F1={row['f1']} "
            f"balanced_accuracy={row['balanced_accuracy']}"
        )
    print()
    print(f"Metrics: {metrics_path}")
    if subject_metrics_path is not None:
        print(f"Subject metrics: {subject_metrics_path}")
    print(f"Per-trial results: {trial_path}")
    print(f"Ground truth copy: {normalized_truth_path}")
    print(f"Acceptance summary: {acceptance_path}")
    print(
        f"Accepted clean feature rows: {accepted_features_path} "
        f"({len(accepted_feature_rows)})"
    )
    print(
        f"Excluded expert-artifact feature rows: {excluded_features_path} "
        f"({len(excluded_feature_rows)})"
    )

    truth_counts = Counter(truth.values())
    if truth_counts[1] < 20:
        print(
            "Caution: this subject subset has few artifact-positive trials. "
            "Use subjects 1-9 for a more stable estimate."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
