#integrated random forest; thinking of adding SVM - wont hit full integration benchmark accuracy :(
#consider trimodal approach
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import math
from scipy.stats import skew, kurtosis

model_path =  "C:\\Users\\ayish\\Documents\\COMP1012_Alpha\\Project-ALPHA-EEG-QC-ABI-ML-Passline\\student_ml_plugin\\trained\\random_forest.joblib"

feature_names = [
    "channel_count",
    "non_empty_channel_count",
    "sample_count_total",
    "range_max_uv",
    "range_mean_uv",
    "flatline_ratio",
    "mean_abs_uv",
    "rms_uv",
    "peak_abs_uv",
    "peak_to_peak_uv",
    "line_length_uv",
    "zero_crossing_rate",
    "kurtosis",
    "skewness",
    "std_uv",
]
current_dir = Path(__file__).resolve().parent
model_path = (current_dir / "trained" / "random_forest.joblib")
_model = None

#loads the trained Random Forest model from the specified path and checks that the number of features matches the expected number of features
def _load_model():
    global _model
    if _model is None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Random Forest model not found: {model_path}"
            )
        _model = joblib.load(model_path)

        if hasattr(_model, "n_features_in_"):
            if _model.n_features_in_ != len(feature_names):
                raise ValueError(
                    "Model expects "
                    f"{_model.n_features_in_} features, "
                    f"but plugin provides {len(feature_names)}."
                )

    return _model

def _model_features(summary: Mapping[str, float]) -> list[float]:
    return [float(summary[name]) for name in feature_names]

def _predict_status(summary: Mapping[str, float], params: Mapping[str, Any],) -> tuple[str, float]:
    model = _load_model()
    features = np.asarray(
        [_model_features(summary)],
        dtype=float,
    )
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)

    if 1 not in classes or 0 not in classes:
        raise ValueError(
            f"Expected model classes 0 and 1; got {classes}"
        )

    p_clean = float(probabilities[classes.index(0)])
    p_artifact = float(probabilities[classes.index(1)])

    warn_threshold = float(params.get("warn_probability", 0.60))
    reject_threshold = float(params.get("reject_probability", 0.90))

    if p_artifact >= reject_threshold:
        status = "REJECT"
    elif p_artifact >= warn_threshold:
        status = "WARN"
    else:
        status = "KEEP"

    confidence = max(p_clean, p_artifact)

    return status, confidence

def _channels(window: Mapping[str, Any]) -> list[list[float]]:
    raw = window.get("eeg_samples", [])
    if not isinstance(raw, list):
        raise ValueError("window['eeg_samples'] must be a list")

    channels: list[list[float]] = []
    for channel in raw:
        if not isinstance(channel, list):
            raise ValueError("each EEG channel must be a list")
        clean_channel: list[float] = []
        for value in channel:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("EEG samples must be finite numbers")
            clean_channel.append(number)
        channels.append(clean_channel)
    return channels

def _summary(window: Mapping[str, Any], params: Mapping[str, Any],) -> dict[str, float]:
    channels = _channels(window)
    non_empty_channels = [
        np.asarray(channel, dtype=float)
        for channel in channels
        if len(channel) > 0
    ]
    if not non_empty_channels:
        #makes everything 0 -> qc_decision_v1 can ABSTAIN
        return {
            "channel_count": float(len(channels)),
            "non_empty_channel_count": 0.0,
            "sample_count_total": 0.0,
            "range_max_uv": 0.0,
            "range_mean_uv": 0.0,
            "flatline_ratio": 0.0,
            "mean_abs_uv": 0.0,
            "rms_uv": 0.0,
            "peak_abs_uv": 0.0,
            "peak_to_peak_uv": 0.0,
            "line_length_uv": 0.0,
            "zero_crossing_rate": 0.0,
            "kurtosis": 0.0,
            "skewness": 0.0,
            "std_uv": 0.0,
        }

    signal = np.concatenate(non_empty_channels)
    differences = np.diff(signal)
    ranges = [float(np.ptp(channel)) for channel in non_empty_channels]

    return {
        "channel_count": float(len(channels)),
        "non_empty_channel_count": float(len(non_empty_channels)),
        "sample_count_total": float(len(signal)),
        "range_max_uv": max(ranges) if ranges else 0.0,
        "range_mean_uv": float(np.mean(ranges)) if ranges else 0.0,
        "flatline_ratio": (float(np.mean(np.abs(differences) < 1e-8)) if len(differences) else 0.0),
        "mean_abs_uv": float(np.mean(np.abs(signal))),
        "rms_uv": float(np.sqrt(np.mean(signal ** 2))),
        "peak_abs_uv": float(np.max(np.abs(signal))),
        "peak_to_peak_uv": float(np.ptp(signal)),
        "line_length_uv": (float(np.mean(np.abs(differences))) if len(differences) else 0.0),
        "zero_crossing_rate": (float(np.mean(signal[:-1] * signal[1:] < 0)) if len(signal) > 1 else 0.0),
        "kurtosis": (float(kurtosis(signal)) if len(signal) > 3 else 0.0),
        "skewness": (float(skew(signal)) if len(signal) > 2 else 0.0),
        "std_uv": float(np.std(signal)),
    }
    
def _predict_reasons(summary: Mapping[str, float], params: Mapping[str, Any], status: str,) -> list[str]:
    reasons = []

    if status == "KEEP":
        reasons.append("MODEL_CLEAN")
    elif status == "WARN":
        reasons.append("MODEL_ARTIFACT_SUSPECTED")
    elif status == "REJECT":
        reasons.append("MODEL_ARTIFACT_DETECTED")

    #add interpretable reasons when the summary supports them.
    flatline_limit = float(params.get("flatline_ratio_threshold", 0.10))

    if summary.get("flatline_ratio", 0.0) >= flatline_limit:
        reasons.append("FLATLINE_SUSPECTED")

    if summary.get("range_max_uv", 0.0) == 0.0:
        reasons.append("ZERO_SIGNAL_RANGE")
    
    #try to add more reasons later

    return reasons

def qc_decision_v1(window: Mapping[str, Any], params: Mapping[str, Any],) -> Mapping[str, Any]:
    summary = _summary(window, params)

    if (summary["sample_count_total"] <= 0  or summary["non_empty_channel_count"] <= 0):
        return {
            "status": "ABSTAIN",
            "quality_score": 0.0,
            "reason_codes": ["EMPTY_WINDOW"],
        }

    status, confidence = _predict_status(summary, params)
    reasons = _predict_reasons(summary, params, status)

    quality_score = round(max(0.0, min(100.0, confidence * 100.0)), 3,)

    return {
        "status": status,
        "quality_score": quality_score,
        "reason_codes": reasons,
    }

def feature_extractor_v1(window: Mapping[str, Any], params: Mapping[str, Any],) -> Mapping[str, float]:
    summary = _summary(window, params)
    model_features = _model_features(summary)
    return {
        **summary,
        "ml_range_gate": model_features[0],
        "ml_range_reject_gate": model_features[1],
        "ml_flatline_gate": model_features[2],
        "ml_flatline_reject_gate": model_features[3],
        "ml_rms_gate": model_features[4],
        "ml_peak_gate": model_features[5],
    }

def main():
    qc_decision_v1("Project-ALPHA-EEG-QC-ABI-ML-Passline\\eeg_formatting\\processed_data\\Control1129.csv", "Project-ALPHA-EEG-QC-ABI-ML-Passline\\student_ml_plugin\\params.json")

if __name__ == "__main__":
    main()