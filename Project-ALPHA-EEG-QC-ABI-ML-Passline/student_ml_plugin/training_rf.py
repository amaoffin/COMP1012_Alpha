#1st model: RF
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--features", default="Project-ALPHA-EEG-QC-ABI-ML-Passline\\student_ml_plugin\\features.csv")
    parser.add_argument("--model-dir", default="Project-ALPHA-EEG-QC-ABI-ML-Passline\\student_ml_plugin\\trained")
    parser.add_argument("--feature-manifest", default="Project-ALPHA-EEG-QC-ABI-ML-Passline\\student_ml_plugin\\feature_manifest.json")
    parser.add_argument("--model-manifest", default="Project-ALPHA-EEG-QC-ABI-ML-Passline\\student_ml_plugin\\model_manifest.json")
    parser.add_argument("--params", default="Project-ALPHA-EEG-QC-ABI-ML-Passline\\student_ml_plugin\\params.json")
    return parser.parse_args()

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def training(model_key, estimator, args):
    #loads all the jsons
    feature_manifest = load_json(args.feature_manifest)
    model_manifest = load_json(args.model_manifest)
    params = load_json(args.params)

    #checks for interpretable features in the feature manifest and gets the list of feature names
    feature_names = feature_manifest.get("interpretable_features", [])
    if not feature_names:
        raise ValueError("No interpretable features found in feature manifest.")

    #checks for the model configuration in the model manifest
    model_config = model_manifest.get("training_models", {}).get(model_key)
    if not model_config:
        raise ValueError(f"model_manifest.json has no training_models.{model_key} configuration.")

    #loads the features CSV and checks for required columns and splits; returns the data as a pd.DataFrame
    training_df = load_data(args.features, feature_names, params)
    
    model_dir = Path(args.model_dir)
    #makes the model directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)

    #gets the split configuration for train, validation, and test splits
    split_config = params.get("evaluation", {})
    train_split = split_config.get("train_split", "train")
    val_split = split_config.get("validation_split", "val")
    test_split = split_config.get("test_split", "test")

    #gets the training, validation, and test data based on the splits defined in the params file
    #X and y are the features and labels, respectively; _[split] indicates the split (train, val, test)
    X_train = training_df.loc[training_df.split == train_split, feature_names].to_numpy(dtype=float)
    y_train = training_df.loc[training_df.split == train_split, "binary_label"].to_numpy(dtype=int)
    X_val = training_df.loc[training_df.split == val_split, feature_names].to_numpy(dtype=float)
    y_val = training_df.loc[training_df.split == val_split, "binary_label"].to_numpy(dtype=int)
    X_test = training_df.loc[training_df.split == test_split, feature_names].to_numpy(dtype=float)
    y_test = training_df.loc[training_df.split == test_split, "binary_label"].to_numpy(dtype=int)

    #checks if the training split contains both clean and artifact examples; raises an error if not
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training split must contain both clean and artifact examples.")
    #estimator is a sklearn pipeline that imputes missing values and trains a random forest classifier
    estimator.fit(X_train, y_train)
    #predicts the labels for the validation and test splits using the trained estimator
    test_pred = estimator.predict(X_test)
    val_pred = estimator.predict(X_val)

    #saves the trained model to the model directory with the specified artifact filename (or default to {model_key}.joblib)
    artifact_filename = model_config.get("artifact_filename", f"{model_key}.joblib")
    model_path = model_dir / artifact_filename
    joblib.dump(estimator, model_path)

    #gets the model version from the model manifest (or defaults to "unspecified")
    model_version = model_manifest.get("model_version", "unspecified")
    results = {
        "model": model_config.get("model_name", model_key),
        "model_family": model_config.get("model_family", "unspecified"),
        "model_version": model_version,
        "feature_manifest": str(Path(args.feature_manifest)),
        "model_manifest": str(Path(args.model_manifest)),
        "params": str(Path(args.params)),
        "features": feature_names,
        "label_mapping": {"0": "clean", "1": "artifact (EOG or EMG or flatline)"},
    }
    #writes the evaluation results to a JSON file in the model directory with the specified evaluation filename (or default to {model_key}_evaluation.json)
    report_path = model_dir / f"{model_key}_evaluation.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nFIN. Model saved to: {model_path.resolve()}")
    print(f"Evaluation saved to: {report_path.resolve()}")
    
#checks if data contains both clean and artifact examples; returns a binary label (0 for clean, 1 for artifact (any kind))
def to_binary_labels(v):
    if isinstance(v, str):
        v = v.strip().lower()
        if v in {"clean", "good", "0"}:
            return 0
        if v in {"artifact", "contaminated", "bad", "1", "2", "3", "eog", "emg", "flatline"}:
            return 1
        raise ValueError(f"Unrecognized label: {v}")
    iv = int(v)
    if iv == 0:
        return 0
    if iv in (1, 2,3):
        return 1
    raise ValueError(f"Expected label 0, 1, or 2; got {v}")

#loads data from features CSV and checks for required columns and splits; returns the data as a pd.DataFrame
def load_data(features_path, feature_names, params):
    data = pd.read_csv(features_path)
    #needed columns: label, split, and all feature names from the feature manifest
    required = {"label", "split", *feature_names}
    #finds any missing columns in the features CSV
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Features CSV is missing columns: {sorted(missing)}. ")
    #applies the to_binary_labels function to the label column to create a new binary_label column (for binary classification)
    data["binary_label"] = data["label"].apply(to_binary_labels)
    #gets the split column and checks if the splits defined in the params file are present in the features CSV
    data["split"] = data["split"]
    split_config = params.get("evaluation", {})
    for split_key, name in [("train_split", "train"), ("validation_split", "val"), ("test_split", "test")]:
        #just to be safe, get the split name from the params file or use the default name
        split_name = split_config.get(split_key, name)
        if split_name not in set(data["split"]):
            raise ValueError(f"No '{split_name}' rows found in features CSV")
    return data

def main():
    #uses parse_args to get command line arguments and load the params JSON file
    args = parse_args("Train a Random Forest model for EEG quality control.")
    params = load_json(args.params)
    #gets the random forest configuration from the params file (or defaults to an empty dict)
    cfg = params.get("random_forest", {})
    #defines a sklearn pipeline that imputes missing values and trains a random forest classifier with the specified hyperparameters from the params file (or defaults)
    model = Pipeline([
        #imputer step uses the strategy defined in the params file (or defaults to median) (imputer is a sklearn SimpleImputer)
        ("imputer", SimpleImputer(strategy=cfg.get("imputer_strategy", "median"))),
        #classifier step uses the hyperparameters defined in the params file (or defaults) (classifier is a sklearn RandomForestClassifier)
        ("classifier", RandomForestClassifier(
            n_estimators=int(cfg.get("n_estimators", 200)),
            class_weight=cfg.get("class_weight", "balanced"),
            random_state=int(cfg.get("random_state", 42)),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
            n_jobs=int(cfg.get("n_jobs", -1))
        ))
    ])
    #calls the training function with the model key, estimator, and command line arguments
    training("random_forest", model, args)

#runs
if __name__ == "__main__":
    main()
