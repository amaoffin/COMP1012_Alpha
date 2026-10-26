#todo: trains model
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

#docu: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--features", default="student_ml_plugin/features.csv")
    parser.add_argument("--model-dir", default="student_ml_plugin/trained")
    parser.add_argument("--feature-manifest", default="student_ml_plugin/feature_manifest.json")
    parser.add_argument("--model-manifest", default="student_ml_plugin/model_manifest.json")
    parser.add_argument("--params", default="student_ml_plugin/params.json")
    return parser.parse_args()