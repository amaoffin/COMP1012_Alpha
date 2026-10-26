#WIP
from typing import Any, Mapping

#take a window of EEG data and parameters, and return a summary of features for the window using same features as training_features.py
def _summary(window: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, float]:

def _predict_status(summary: Mapping[str, float], params: Mapping[str, Any]) -> tuple[str, float]:

def _predict_reasons(summary: Mapping[str, float], params: Mapping[str, Any], status: str) -> list[str]:
    
def _model_features(summary: Mapping[str, float]) -> list[float]:
    
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

def feature_extractor_v1(window: Mapping[str, Any], params: Mapping[str, Any]) -> Mapping[str, float]:
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