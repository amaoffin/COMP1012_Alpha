#creating training dataset from EEGdenoisenet datasets; heavy inspiration from https://es.mathworks.com/help/signal/ug/denoise-eeg-signals-using-differentiable-signal-processing-layers.html (but that's for matlab)
import numpy as np
import os
import csv

eeg_data = np.load('eeg_formatting/eegdenoisenet/EEG_all_epochs.npy')
eog_data = np.load('eeg_formatting/eegdenoisenet/EOG_all_epochs.npy')
emg_data = np.load('eeg_formatting/eegdenoisenet/EMG_all_epochs.npy')
Fs = 256
x = 0
data_no = 100
epoch_length = 512
rng = np.random.default_rng()
output_dir = 'eeg_formatting/eegdenoisenet/train'

#addsan artifact to clean EEG at the specified SNR in dB
def mix_artifact(clean, artifact, snr_db):
    clean = np.asarray(clean, dtype=np.float64)
    artifact = np.asarray(artifact, dtype=np.float64)

    clean_rms = np.sqrt(np.mean(clean ** 2))
    artifact_rms = np.sqrt(np.mean(artifact ** 2))

    if clean_rms == 0 or artifact_rms == 0:
        raise ValueError("cannot mix signals with zero RMS")

    target_artifact_rms = clean_rms / (10 ** (snr_db / 20)) #from standard SNR equation
    scaled_artifact = artifact * (
        target_artifact_rms / artifact_rms
    )

    return clean + scaled_artifact

#make clean and artifact-contaminated EEG windows for training
#0 - clean; 1 - synthetic EOG contamination; 2 - synthetic EMG contamination
def make_training_dataset():
    for name, data in [
        ("EEG", eeg_data),
        ("EOG", eog_data),
        ("EMG", emg_data)
    ]:
        if data.ndim != 2 or data.shape[1] != epoch_length:
            raise ValueError(
                f"{name} must have shape (epochs, {epoch_length}), got {data.shape}"
            )

    #balancing with mod3
    labels = np.arange(data_no) % 3
    rng.shuffle(labels)

    #snr levels for contamination (dB)
    snr_levels = [-10, -5, 0, 5]

    metadata_path = os.path.join(output_dir, "metadata.csv")

    with open(
        metadata_path, "w", newline="", encoding="utf-8"
    ) as metadata_file:
        metadata_writer = csv.writer(metadata_file)
        metadata_writer.writerow([
            "window_id",
            "label",
            "artifact_type",
            "snr_db",
            "eeg_source_index",
            "artifact_source_index"
        ])

        for x, label in enumerate(labels):
            window_id = f"training_{x:04d}"

            #selects clean eeg
            eeg_idx = int(rng.integers(len(eeg_data)))
            clean = eeg_data[eeg_idx].astype(np.float64).copy()

            artifact_type = "none"
            artifact_idx = ""
            snr_db = ""
            window = clean

            if label == 1:
                #eog contam
                artifact_type = "EOG"
                artifact_idx = int(rng.integers(len(eog_data)))
                snr_db = int(rng.choice(snr_levels))

                artifact = eog_data[artifact_idx]
                window = mix_artifact(clean, artifact, snr_db)

            elif label == 2:
                #emg contam
                artifact_type = "EMG"
                artifact_idx = int(rng.integers(len(emg_data)))
                snr_db = int(rng.choice(snr_levels))

                artifact = emg_data[artifact_idx]
                window = mix_artifact(clean, artifact, snr_db)

            #saves as csv
            output_csv = os.path.join(
                output_dir, f"{window_id}.csv"
            )

            with open(
                output_csv, "w", newline="", encoding="utf-8"
            ) as csvfile:

                writer = csv.writer(csvfile)

                writer.writerow([
                    "window_id",
                    "channel index",
                    "channel name",
                    "sample index",
                    "value (uv)"
                ])

                for sample_idx, value in enumerate(window):
                    writer.writerow([
                        window_id,
                        0,
                        "EEG",
                        sample_idx,
                        float(value)
                    ])

            #ground truth metadata
            metadata_writer.writerow([
                window_id,
                int(label),
                artifact_type,
                snr_db,
                eeg_idx,
                artifact_idx
            ])

    print("done")


if __name__ == "__main__":
    make_training_dataset()