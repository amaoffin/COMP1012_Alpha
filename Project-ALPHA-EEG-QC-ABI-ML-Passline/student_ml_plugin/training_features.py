#done (?): extracts features from the EEG window and returns a dictionary of features for the model to train on for every window and then saves all to csv for tranging
from pathlib import Path
from unicodedata import name

from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd

#function to extract features from a CSV file containing EEG data (like the samples) and returns a dictionary of features for the model to train on
def extract_features(csv_path) -> dict:
    window = pd.read_csv(csv_path)
    #makes sure only uv values are used for feature extraction
    if "value (uv)" in window.columns:
        window["value (uv)"] = pd.to_numeric(
            window["value (uv)"], errors="coerce"
        )
        #pivot basically makes the data into a matrix with channels as columns and samples as rows, with the values being the EEG values in uv
        window = window.pivot(
            index="sample index",
            columns="channel name",
            values="value (uv)"
        )
    #makes SURE only numeric values are used for feature extraction + drops any columns that are all NaN
    window = window.select_dtypes(include="number").dropna(axis=1, how="all")
    features = {}
    #channel count
    features['channel_count'] = window.shape[1]
    #non-empty channel count (channels with at least one non-NaN value)
    non_empty_channels = window.notna().any().sum()
    features["non_empty_channel_count"] = non_empty_channels
    #total sample count (# of rows * # of columns)
    features['sample_count_total'] = window.shape[0] * window.shape[1]
    #range max and mean
    ranges = window.max() - window.min()
    features["range_max_uv"] = ranges.max() if not ranges.empty else 0.0
    features["range_mean_uv"] = ranges.mean() if not ranges.empty else 0.0
    #range based flatline doesnt work need to check consec samples if theyre same
    flatline_eps = 1e-6
    min_flatline_samples = 4
    flat_channels = 0

    for col in window.columns:
        values = window[col].dropna()
        unchanged = values.diff().abs() <= flatline_eps

        #detects consec unchanged samples
        groups = (~unchanged).cumsum()
        if unchanged.groupby(groups).sum().max() >= min_flatline_samples - 1:
            flat_channels += 1

    features["flatline_ratio"] = (flat_channels / non_empty_channels if non_empty_channels else 1.0)
    #mean absolute value and root mean square (RMS) are calculated across all channels
    features['mean_abs_uv'] = np.mean(np.abs(window.values))
    features['rms_uv'] = np.sqrt(np.mean(window.values ** 2))
    #peak absolute value is calculated as the maximum absolute value across all channels
    features['peak_abs_uv'] = np.max(np.abs(window.values))
    #peak-to-peak amplitude for each channel
    features['peak_to_peak_uv'] = window.max().max() - window.min().min()
    #line length for each channel (the sum of the absolute differences between consecutive samples)
    features['line_length_uv'] = np.sum(np.abs(np.diff(window.values, axis=0)))
    #zero crossing rate for each channel (the number of times the signal changes sign divided by the total number of samples)
    zero_crossings = np.sum(np.diff(np.sign(window.values), axis=0) != 0)
    features['zero_crossing_rate'] = zero_crossings / (window.shape[0] - 1)
    #kurtosis for each channel (the fourth standardized moment, which measures the "tailedness" of the distribution)
    features['kurtosis'] = kurtosis(window.values, axis=0).mean()
    #skewness for each channel (the third standardized moment, which measures the asymmetry of the distribution)
    features['skewness'] = skew(window.values, axis=0).mean()
    #standard deviation for each channel
    features['std_uv'] = np.std(window.values, axis=0).mean()

    return features

def features_to_csv(features: dict, csv_path: str):
    window = pd.DataFrame([features])
    window.to_csv(csv_path, index=False)

def main():
    #example usage
    csv_path = 'eeg_formatting/eegdenoisenet/train/window_000000.csv'
    features = extract_features(csv_path)
    data_dir = Path("eeg_formatting/eegdenoisenet/train")
    metadata_path = Path("eeg_formatting/eegdenoisenet/train/metadata.csv")
    metadata = pd.read_csv(metadata_path)
    rows = []
    for record in metadata.to_dict(orient="records"):
        window_id = str(record["window_id"])
        csv_path = data_dir / f"{window_id}.csv"
        all_features = extract_features(csv_path)
        missing_features = [name for name in features if name not in all_features]
        if missing_features:
            raise ValueError(f"missing features: {missing_features}")
        row = {
            "window_id": window_id,
            "label": record["label"],
            "split": str(record["split"]).strip().lower(),
        }
        if "artifact_type" in record:
            row["artifact_type"] = record["artifact_type"]
        row.update({name: all_features[name] for name in features})
        rows.append(row)
    out = Path("student_ml_plugin/features.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"extracted features for {len(rows)} windows")
    
if __name__ == "__main__":
    main()