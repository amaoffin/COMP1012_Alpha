
#3rd model: MLP
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score,)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

METADATA_COLUMNS = {"window_id", "label", "split", "artifact_type","source_file", "file_name", "csv_path",}
EXCLUDED_FEATURES = {""}


def get_features(df, manifest_path=None):
    if manifest_path:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        names = manifest.get("interpretable_features") or manifest.get("features")
        if not names:
            raise ValueError("Manifest has no 'interpretable_features' or 'features' list.")
        #exclude any features that are in the EXCLUDED_FEATURES set
        names = [name for name in names if name not in EXCLUDED_FEATURES]
        missing = [name for name in names if name not in df.columns]
        if missing:
            raise ValueError(f"Features missing from CSV: {missing}")
    else:
        names = [
            col for col in df.select_dtypes(include="number").columns
            if col not in METADATA_COLUMNS
        ]

    names = [
        name for name in names
        if name not in EXCLUDED_FEATURES and name not in METADATA_COLUMNS
    ]
    if not names:
        raise ValueError("No usable feature columns found.")
    return names


def to_binary_labels(labels):
    clean = {"clean", "good", "0"}
    artifact = {"artifact", "contaminated", "bad", "1", "2", "3", "eog", "emg", "flatline"}
    output = []
    for value in labels.astype(str).str.strip().str.lower():
        if value in clean:
            output.append(0)
        elif value in artifact:
            output.append(1)
        else:
            raise ValueError(f"Unrecognized label {value!r}. Use 0/clean or a nonzero numeric/artifact label.")
    return np.asarray(output, dtype=int)


def get_artifact_probability(model, X):
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    if 0 not in classes or 1 not in classes:
        raise ValueError(f"MLP must use classes 0=clean and 1=artifact; got {classes}")
    return probs[:, classes.index(1)]


def select_threshold(y_true, probabilities, target_specificity=0.9):
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    results = []

    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        balanced_acc = (sensitivity + specificity) / 2

        if specificity >= target_specificity:
            results.append((balanced_acc, sensitivity, specificity, threshold))

    if not results:
        raise ValueError("No threshold satisfies the target specificity.")

    #maximize balanced accuracy, then sensitivity, then prefer the higher threshold
    best = max(results, key=lambda x: (x[0], x[1], x[3]))
    return float(best[3])

def report_metrics(y_true, y_pred, title="Test results"):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    print(f"\n{title} (artifact is the positive class)")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Sensitivity:       {sensitivity:.4f}")
    print(f"  Specificity:       {specificity:.4f}")
    print(f"  Precision:         {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1 score:          {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"  Accuracy:          {accuracy_score(y_true, y_pred):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train an MLP EEG QC classifier.")
    parser.add_argument("--features", default="student_ml_plugin\\features.csv")
    parser.add_argument("--manifest", default="student_ml_plugin\\feature_manifest.json")
    parser.add_argument("--out", default="student_ml_plugin\\trained\\mlp.joblib")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--hidden-layers", nargs="+", type=int, default=[32, 16])
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-specificity", type=float, default=0.9)
    args = parser.parse_args()

    if not 0 < args.target_specificity <= 1:
        raise ValueError("--target-specificity must be between 0 and 1.")

    df = pd.read_csv(args.features)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    features = get_features(df, args.manifest)
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = to_binary_labels(df["label"])

    #use predefined train, validation, and test splits when available
    if "split" in df.columns and df["split"].notna().any():
        splits = df["split"].astype(str).str.strip().str.lower()
        test_mask = splits == args.test_split.strip().lower()
        val_mask = splits == "val"
        train_mask = splits == "train"

        if not test_mask.any():
            raise ValueError(f"No rows have split={args.test_split!r}. Available: {sorted(splits.unique())}")
        if not train_mask.any() or not val_mask.any():
            raise ValueError("Dataset must contain separate train, val, and test splits.")

        X_train, y_train = X.loc[train_mask], y[train_mask.to_numpy()]
        X_val, y_val = X.loc[val_mask], y[val_mask.to_numpy()]
        X_test, y_test = X.loc[test_mask], y[test_mask.to_numpy()]

    else:
        #create separate training, validation, and test sets
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.15, random_state=args.random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.15, random_state=args.random_state, stratify=y_trainval
        )

    if len(np.unique(y_train)) < 2:
        raise ValueError("Training data must contain both clean and artifact examples.")
    if len(np.unique(y_val)) < 2:
        raise ValueError("Validation data must contain both clean and artifact examples.")

    basemodel = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=tuple(args.hidden_layers),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate="adaptive",
            max_iter=args.max_iter,
            early_stopping=True,
            random_state=args.random_state,
        )),
    ])

    #train model
    basemodel.fit(X_train, y_train)
    
    model = CalibratedClassifierCV(estimator=basemodel, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    #select probability threshold using validation data only
    val_probabilities = get_artifact_probability(model, X_val)
    threshold = select_threshold(y_val, val_probabilities, args.target_specificity)
    val_predictions = (val_probabilities >= threshold).astype(float)

    #evaluate once on the untouched test set
    test_probabilities = get_artifact_probability(model, X_test)
    test_predictions = (test_probabilities >= threshold).astype(float)

    print(f"Training windows:   {len(X_train)}")
    print(f"Validation windows: {len(X_val)}")
    print(f"Testing windows:    {len(X_test)}")
    print(f"Features ({len(features)}): {', '.join(features)}")
    print(f"Selected threshold: {threshold:.4f}")
    print(f"Target specificity: {args.target_specificity:.4f}")

    report_metrics(y_val, val_predictions, "Validation results")
    report_metrics(y_test, test_predictions, "Test results")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": features,
        "label_mapping": {"clean": 0, "artifact": 1},
        "model_type": "MLPClassifier",
        "decision_threshold": threshold,
        "target_specificity": args.target_specificity,
    }, output)

    print(f"\nSaved model to: {output}")

if __name__ == "__main__":
    main()
