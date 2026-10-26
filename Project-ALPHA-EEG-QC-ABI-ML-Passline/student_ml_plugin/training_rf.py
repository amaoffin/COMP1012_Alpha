#1st model: RF
import argparse
import json

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--features", default="Project-ALPHA-EEG-QC-ABI-ML-Passline\\student_ml_plugin\\features.csv")
    parser.add_argument("--model-dir", default="Project-ALPHA-EEG-QC-ABI-ML-Passline\\student_ml_plugin\\trained")
    parser.add_argument("--feature-manifest", default="Project-ALPHA-EEG-QC-ABI-ML-Passline\\feature_manifest.json")
    parser.add_argument("--model-manifest", default="Project-ALPHA-EEG-QC-ABI-ML-Passline\\student_ml_plugin\\model_manifest.json")
    parser.add_argument("--params", default="Project-ALPHA-EEG-QC-ABI-ML-Passline\\student_ml_plugin\\params.json")
    return parser.parse_args()

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def training(model_key, estimator, args):
    feature_manifest = load_json(args.feature_manifest)
    model_manifest = load_json(args.model_manifest)
    params = load_json(args.params)

    feature_names = feature_manifest.get("interpretable_features", [])
    if not feature_names:
        raise ValueError("No interpretable features found in feature manifest")
    
    