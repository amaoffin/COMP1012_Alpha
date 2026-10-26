#creating training dataset from EEGdenoisenet datasets; heavy inspiration from https://es.mathworks.com/help/signal/ug/denoise-eeg-signals-using-differentiable-signal-processing-layers.html (but that's for matlab)
import numpy as np
import os
import csv

eeg_data = np.load('Project-ALPHA-EEG-QC-ABI-ML-Passline/eeg_formatting/eegdenoisenet/EEG_all_epochs.npy')
eog_data = np.load('Project-ALPHA-EEG-QC-ABI-ML-Passline/eeg_formatting/eegdenoisenet/EOG_all_epochs.npy')
emg_data = np.load('Project-ALPHA-EEG-QC-ABI-ML-Passline/eeg_formatting/eegdenoisenet/EMG_all_epochs.npy')
Fs = 256
x = 0
data_no = 500
epoch_length = 512
rng = np.random.default_rng()
output_dir = 'Project-ALPHA-EEG-QC-ABI-ML-Passline/eeg_formatting/eegdenoisenet/train'
#normal eeg channel names
channels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "C3", "Cz", "C4", "T7", "T8", "P7", "P8", "P3", "Pz", "P4", "O1", "O2"]
#dict with split ratios for train, validation, and test sets
split = {"train": 0.70, "val": 0.15,"test": 0.15}
#snr levels (dB) for adding artifacts to clean EEG, picks one to have diff contam levels
snr = [-10, -5, 0, 5]

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
#0 - clean; 1 - synthetic EOG contamination; 2 - synthetic EMG contamination; 3 - flatline
def make_training_dataset():
    source_data = {
        "EEG": eeg_data,
        "EOG": eog_data,
        "EMG": emg_data
    }

    for name, data in source_data.items():
        if data.ndim != 2 or data.shape[1] != epoch_length:
            raise ValueError(
                f"{name} must have shape (epochs, {epoch_length}), got {data.shape}"
            )

        if len(data) == 0 or not np.isfinite(data).all():
            raise ValueError(
                f"{name} has no epochs or has non-finite values"
            )

    os.makedirs(output_dir, exist_ok=True)

    #split source epochs b4 generating windows, so that each split has its own source epochs
    source_indices = {}

    for name, data in source_data.items():
        indices = rng.permutation(len(data))

        n_train = int(len(indices) * split["train"])
        n_val = int(len(indices) * split["val"])

        source_indices[name] = {
            "train": indices[:n_train],
            "val": indices[n_train:n_train + n_val],
            "test": indices[n_train + n_val:]
        }

    #allocation
    split_names = list(split.keys())

    n_train = int(data_no * split["train"])
    n_val = int(data_no * split["val"])

    split_counts = {"train": n_train, "val": n_val, "test": data_no - n_train - n_val
    }

    metadata_path = os.path.join(output_dir, "metadata.csv")

    with open(metadata_path, "w", newline="", encoding="utf-8") as metadata_file:
        metadata_writer = csv.writer(metadata_file)
        metadata_writer.writerow(["window_id", "label", "artifact_type", "snr_db", "eeg_source_index", "artifact_source_index", "split"])

        window_number = 0

        #generate windows of each split
        for splitting in split_names:
            count = split_counts[splitting]

            #balance the labels across clean, EOG, EMG, and flatline (basically 0, 1, 2, 3)
            labels = np.arange(count) % 4
            rng.shuffle(labels)

            eeg_pool = source_indices["EEG"][splitting]
            eog_pool = source_indices["EOG"][splitting]
            emg_pool = source_indices["EMG"][splitting]

            for label in labels:
                #makes unique window ID for each window as "window_000001", "window_000002", etc.(06d means 6 digits with leading zeros)
                window_id = f"window_{window_number:06d}"
                window_number += 1

                #choose a clean EEG source from this split only
                eeg_idx = int(rng.choice(eeg_pool))
                clean = eeg_data[eeg_idx].astype(np.float64).copy()

                artifact_type = "none"
                artifact_idx = ""
                snr_db = ""
                window = clean
                #1 has eog contam
                if label == 1:
                    artifact_type = "EOG"
                    artifact_idx = int(rng.choice(eog_pool))
                    snr_db = int(rng.choice(snr))

                    artifact = eog_data[artifact_idx]
                    window = window = mix_artifact(clean, artifact, snr_db)

                #2 has emg contam
                elif label == 2:
                    artifact_type = "EMG"
                    artifact_idx = int(rng.choice(emg_pool))
                    snr_db = int(rng.choice(snr))

                    artifact = emg_data[artifact_idx]
                    window = mix_artifact(clean, artifact, snr_db)
                
                #flatline ish
                elif label == 3:
                    artifact_type = "flatline"
                    artifact_idx = ""
                    snr_db = ""
                    
                    #chooses a random start point and length for the flatline segment, which is between 0.5 and 2 seconds long (Fs is 256 Hz, so Fs//2 is 0.5 seconds and Fs*2 is 2 seconds)
                    #then replaces that segment of the clean EEG with zeros to simulate a flatline
                    #not sure if this is the best way to simulate a flatline, but it should be good enough for training the model i hope
                    start = rng.integers(0, len(clean) - Fs // 2)
                    length = rng.integers(Fs // 2, Fs * 2)
                    window = clean.copy()
                    window[start:min(start + length, len(clean))] = 0
                    

                #save 'multichannel' window to CSV, multichannela
                output_csv = os.path.join(output_dir, f"{window_id}.csv")

                with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["window_id", "channel_index", "channel_name", "sample_index", "value_uv"])

                    for channel_idx, channel_name in enumerate(channels):
                        for sample_idx, value in enumerate(window):
                            writer.writerow([window_id, channel_idx,channel_name, sample_idx, float(value)])

                #save ground-truth metadata
                metadata_writer.writerow([window_id, int(label), artifact_type, snr_db, eeg_idx, artifact_idx, splitting])
    print("done")

if __name__ == "__main__":
    make_training_dataset()