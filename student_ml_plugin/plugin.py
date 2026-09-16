#taken from sample for now, annotating to find features

from __future__ import annotations

import math
from typing import Any, Mapping


LABELS = ("KEEP", "WARN", "REJECT")
REASON_LABELS = ("AMP_RANGE", "FLATLINE")

# Feature order used by the frozen classifier:
# range_gate, range_reject_gate, flat_gate, flat_reject_gate, rms_gate, peak_gate
#what does gate mean???
MODEL_WEIGHTS: dict[str, tuple[float, ...]] = {
    "KEEP": (3.25, -3.85, -5.25, -4.05, -5.15, -1.15, -0.65),
    "WARN": (-0.80, 3.60, -3.70, 3.45, -3.40, 0.30, 0.10),
    "REJECT": (-2.25, 1.40, 5.40, 1.20, 5.70, 0.75, 0.60),
}

REASON_WEIGHTS: dict[str, tuple[float, ...]] = {
    #amplitude range
    "AMP_RANGE": (-1.15, 4.40, 5.10, -1.25, -1.10, 0.60, 0.90),
    #flatline lol
    "FLATLINE": (-1.20, -1.35, -1.20, 4.80, 5.25, -0.35, -0.25),
}


def _float_param(params: Mapping[str, Any], name: str, default: float) -> float:
    try:
        return float(params.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric parameter: {name}") from exc


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


def _range(channel: list[float]) -> float:
    if not channel:
        return 0.0
    return max(channel) - min(channel)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _dot(weights: tuple[float, ...], values: tuple[float, ...]) -> float:
    return weights[0] + sum(weight * value for weight, value in zip(weights[1:], values))

#softmax function that computes the softmax probabilities for a set of scores, given a temperature parameter, returning a dictionary of probabilities
def _softmax(scores: Mapping[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0.0:
        raise ValueError("decision_temperature must be positive")
    scaled = {label: score / temperature for label, score in scores.items()}
    offset = max(scaled.values())
    exp_scores = {label: math.exp(score - offset) for label, score in scaled.items()}
    total = sum(exp_scores.values())
    return {label: value / total for label, value in exp_scores.items()}

#summary function that computes a summary of the EEG window to be used for the model, maybe have to rewrite this to include more features and work with other models
def _summary(window: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, float]:
    channels = _channels(window)
    ranges = [_range(channel) for channel in channels if channel]
    non_empty_channels = len(ranges)
    total_samples = sum(len(channel) for channel in channels)

    flatline_eps = _float_param(params, "flatline_range_epsilon", 1e-6)
    flat_channels = sum(1 for value in ranges if value <= flatline_eps)
    flatline_ratio = flat_channels / non_empty_channels if non_empty_channels else 1.0
    range_max = max(ranges) if ranges else 0.0
    range_mean = sum(ranges) / len(ranges) if ranges else 0.0

    abs_sum = 0.0
    square_sum = 0.0
    abs_count = 0
    peak_abs = 0.0
    zero_crossings = 0
    zero_crossing_possible = 0
    for channel in channels:
        previous = None
        for value in channel:
            abs_value = abs(value)
            abs_sum += abs_value
            square_sum += value * value
            peak_abs = max(peak_abs, abs_value)
            abs_count += 1
            if previous is not None and value != 0.0 and previous != 0.0:
                zero_crossing_possible += 1
                if (value > 0.0) != (previous > 0.0):
                    zero_crossings += 1
            previous = value

    mean_abs_uv = abs_sum / abs_count if abs_count else 0.0
    rms_uv = math.sqrt(square_sum / abs_count) if abs_count else 0.0
    zero_crossing_rate = (
        zero_crossings / zero_crossing_possible if zero_crossing_possible else 0.0
    )

    return {
        "channel_count": float(len(channels)),
        "non_empty_channel_count": float(non_empty_channels),
        "sample_count_total": float(total_samples),
        "range_max_uv": float(range_max),
        "range_mean_uv": float(range_mean),
        "flatline_ratio": float(flatline_ratio),
        "mean_abs_uv": float(mean_abs_uv),
        "rms_uv": float(rms_uv),
        "peak_abs_uv": float(peak_abs),
        "zero_crossing_rate": float(zero_crossing_rate),
    }

#takes feature summary and returns a tuple of features for the model
def _model_features(summary: Mapping[str, float]) -> tuple[float, ...]:
    range_max = summary["range_max_uv"]
    flatline_ratio = summary["flatline_ratio"]
    rms_uv = summary["rms_uv"]
    peak_abs_uv = summary["peak_abs_uv"]
    return (
        _sigmoid((range_max - 100.0) / 10.0),
        _sigmoid((range_max - 180.0) / 16.0),
        _sigmoid((flatline_ratio - 0.25) / 0.06),
        _sigmoid((flatline_ratio - 0.50) / 0.06),
        math.log1p(rms_uv) / 6.0,
        math.log1p(peak_abs_uv) / 6.0,
    )

#predicts the status of the EEG window based on the summary and parameters, returning the predicted label and its associated probability
def _predict_status(summary: Mapping[str, float], params: Mapping[str, Any]) -> tuple[str, float]:
    features = _model_features(summary)
    scores = {label: _dot(MODEL_WEIGHTS[label], features) for label in LABELS}
    temperature = _float_param(params, "decision_temperature", 1.0)
    probabilities = _softmax(scores, temperature)
    label = max(LABELS, key=lambda item: probabilities[item])
    return label, probabilities[label]

#predicts the reasons for the predicted status based on the summary, parameters, and predicted status, returning a list of reason codes
def _predict_reasons(
    summary: Mapping[str, float], params: Mapping[str, Any], status: str
) -> list[str]:
    if status == "KEEP":
        return ["NONE"]

    features = _model_features(summary)
    cutoff = _float_param(params, "reason_probability_cutoff", 0.50)
    reasons: list[str] = []
    scored_reasons: list[tuple[float, str]] = []
    for label in REASON_LABELS:
        probability = _sigmoid(_dot(REASON_WEIGHTS[label], features))
        scored_reasons.append((probability, label))
        if probability >= cutoff:
            reasons.append(label)

    if reasons:
        return reasons
    return [max(scored_reasons)[1]]


def qc_decision_v1(
    window: Mapping[str, Any], params: Mapping[str, Any]
) -> Mapping[str, Any]:
    summary = _summary(window, params)
#checks if the window is empty or has no non-empty channels, returning an abstain decision if so
    if summary["sample_count_total"] <= 0 or summary["non_empty_channel_count"] <= 0:
        return {
            "status": "ABSTAIN",
            "quality_score": 0.0,
            "reason_codes": ["EMPTY_WINDOW"],
        }

    status, confidence = _predict_status(summary, params)
    reasons = _predict_reasons(summary, params, status)
    quality_score = round(max(0.0, min(100.0, confidence * 100.0)), 3)

    return {
        "status": status,
        "quality_score": quality_score,
        "reason_codes": reasons,
    }

#feature extractor function that computes a summary of the EEG window and extracts model features, returning a dictionary of features
def feature_extractor_v1(
    window: Mapping[str, Any], params: Mapping[str, Any]
) -> Mapping[str, float]:
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
