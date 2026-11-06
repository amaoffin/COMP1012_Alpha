#integrated random forest
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import math
from scipy.stats import skew, kurtosis

current_dir = Path(__file__).resolve().parent
rf_model_path =  "C:\\Users\\ayish\\Documents\\COMP1012_Alpha\\Project-ALPHA-EEG-QC-ABI-ML-Passline\\student_ml_plugin\\trained\\random_forest.joblib"
_rf_model = None

LABELS = ("KEEP", "WARN", "REJECT")
REASON_LABELS = ("AMP_RANGE", "FLATLINE", "ABNORMAL KURTOSIS", "ABNORMAL SKEWNESS", "EXCESSIVE LINE LENGTH","EXCESSIVE ZERO CROSSINGS","INSUFFICIENT CHANNELS")

MODEL_WEIGHTS: dict[str, tuple[float, ...]] = {
    "KEEP": (3.25, -3.85, -5.25, -4.05, -5.15, -1.15, -0.65),
    "WARN": (-0.80, 3.60, -3.70, 3.45, -3.40, 0.30, 0.10),
    "REJECT": (-2.25, 1.40, 5.40, 1.20, 5.70, 0.75, 0.60),
}

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

def _float_param(params: Mapping[str, Any], name: str, default: float) -> float:
    try:
        return float(params.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric parameter: {name}") from exc

def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _dot(weights: tuple[float, ...], values: tuple[float, ...]) -> float:
    return weights[0] + sum(w * v for w, v in zip(weights[1:], values))


def _softmax(scores: Mapping[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("decision_temperature must be positive")
    scaled = {label: score / temperature for label, score in scores.items()}
    offset = max(scaled.values())
    exp_scores = {label: math.exp(score - offset) for label, score in scaled.items()}
    total = sum(exp_scores.values())
    return {label: value / total for label, value in exp_scores.items()}

def _frozen_features(summary: Mapping[str, float]) -> tuple[float, ...]:
    rmax = summary["range_max_uv"]
    flat = summary["flatline_ratio"]
    return (
        _sigmoid((rmax - 100.0) / 10.0),
        _sigmoid((rmax - 180.0) / 16.0),
        _sigmoid((flat - 0.25) / 0.06),
        _sigmoid((flat - 0.50) / 0.06),
        math.log1p(summary["rms_uv"]) / 6.0,
        math.log1p(summary["peak_abs_uv"]) / 6.0,
    )


def _frozen_predict(summary: Mapping[str, float], params: Mapping[str, Any]):
    features = _frozen_features(summary)
    scores = {label: _dot(MODEL_WEIGHTS[label], features) for label in LABELS}
    probabilities = _softmax(scores, _float_param(params, "decision_temperature", 1.0))
    status = max(LABELS, key=lambda label: probabilities[label])
    return status, float(probabilities[status]), probabilities, features
    
#loads the trained Random Forest model from the specified path and checks that the number of features matches the expected number of features
def _load_rf_model(params: Mapping[str, Any]):
    global _rf_model
    if _rf_model is None:
        path = Path(params.get("rf_model_path", rf_model_path))
        if not path.is_absolute():
            path = current_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"Random Forest model not found: {path}")
        _rf_model = joblib.load(path)
        if hasattr(_rf_model, "n_features_in_") and _rf_model.n_features_in_ != len(feature_names):
            raise ValueError(
                f"RF expects {_rf_model.n_features_in_} features, but plugin provides "
                f"{len(feature_names)}."
            )
    return _rf_model

def _model_features(summary: Mapping[str, float]) -> list[float]:
    return [float(summary[name]) for name in feature_names]

def _rf_predict(summary: Mapping[str, float], params: Mapping[str, Any],) -> tuple[str, float]:
    model = _load_rf_model(params)
    x = np.asarray([[float(summary[name]) for name in feature_names]], dtype=float)
    probs = model.predict_proba(x)[0]
    classes = list(model.classes_)
    if 0 not in classes or 1 not in classes:
        raise ValueError(f"RF must use classes 0=clean and 1=artifact; got {classes}")
    p_artifact = float(probs[classes.index(1)])
    warn_t = _float_param(params, "warn_probability", 0.7)
    reject_t = _float_param(params, "reject_probability", 0.8)
    status = "REJECT" if p_artifact >= reject_t else ("WARN" if p_artifact >= warn_t else "KEEP")
    return status, p_artifact, max(float(probs[classes.index(0)]), p_artifact)


def _fuse(predictions: Mapping[str, Mapping[str, Any]]) -> str:
    statuses = [item["status"] for item in predictions.values()]
    artifact_votes = sum(status != "KEEP" for status in statuses)
    reject_votes = sum(status == "REJECT" for status in statuses)
    if reject_votes == 2:
        return "REJECT"
    if artifact_votes <= 1:
        return "KEEP"
    return "WARN"

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
    non_empty_channels = [np.asarray(ch, dtype=float) for ch in channels if len(ch) > 0]
    flat_eps = 1e-6
    
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
        "flatline_ratio": sum(r <= flat_eps for r in ranges) / len(non_empty_channels) if non_empty_channels else 1.0,
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
    
def _predict_reasons(summary: Mapping[str, float], status: str,) -> list[str]:
    if status == "KEEP":
        return ["NONE"]

    reasons: list[str] = []

    #amplitude range
    if summary["range_max_uv"] >= 500.0:
        reasons.append("AMP_RANGE")

    #flatline ratio
    if summary["flatline_ratio"] >= 0.1:
        reasons.append("FLATLINE")

    #abnormal kurtosis
    if abs(summary["kurtosis"]) >= 3.0:
        reasons.append("ABNORMAL KURTOSIS")

    #abnormal skewness
    if abs(summary["skewness"]) >= 10.0:
        reasons.append("ABNORMAL SKEWNESS")

    #excessive line length
    if summary["line_length_uv"] >= 20.0:
        reasons.append("EXCESSIVE LINE LENGTH")

    #excessive zero crossings
    if summary["zero_crossing_rate"] >= 0.4:
        reasons.append("EXCESSIVE ZERO CROSSINGS")

    #insufficient channels
    if summary["non_empty_channel_count"] <= 1.0:
        reasons.append("INSUFFICIENT CHANNELS")

    return reasons if reasons else ["NONE"]

def qc_decision_v1(window: Mapping[str, Any], params: Mapping[str, Any]) -> Mapping[str, Any]:
    #gets summary of the eeg window using the _summary functions
    summary = _summary(window, params)
    
    #prelim checking to make sure it isnt empty
    if summary["sample_count_total"] <= 0 or summary["non_empty_channel_count"] <= 0:
        return {"status": "ABSTAIN", "quality_score": 0.0, "reason_codes": ["EMPTY_WINDOW"]}

    #predicts status of eeg window with frozen and RF model
    frozen_status, frozen_conf, frozen_probs = _frozen_predict(summary, params)
    rf_status, rf_p_artifact, rf_conf = _rf_predict(summary, params)

    #combines predictions into a single dictionary
    predictions = {
        "FROZEN": {"status": frozen_status, "confidence": frozen_conf},
        "RF": {"status": rf_status, "confidence": rf_conf, "p_artifact": rf_p_artifact},
    }

    #fuses predictions into a final status with the _fuse function
    final_status = _fuse(predictions)

    #predicts reasons for the final status with the _predict_reasons function
    reasons = _predict_reasons(summary, final_status)
    
    #approximate ensemble quality score from probabilities (KEEP for frozen, 1-p_artifact for RF), averaged and scaled to 0-100
    clean_scores = [frozen_probs["KEEP"], 1.0 - rf_p_artifact]
    quality_score = round(float(np.mean(clean_scores)) * 100.0, 3)

    #final returns dictionary for the plugin with final status, quality of eeg window, and reasons for status
    return {
        "status": final_status,
        "quality_score": max(0.0, min(100.0, quality_score)),
        "reason_codes": reasons,
    }


def feature_extractor_v1(window: Mapping[str, Any], params: Mapping[str, Any],) -> Mapping[str, float]:
    #gets the summary of the EEG window and returns a dictionary of features, defined in the _summary function
    summary = _summary(window, params)
    return {
        **summary
    }