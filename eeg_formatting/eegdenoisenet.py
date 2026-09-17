import numpy as np

eeg_data = np.load('eeg_formatting/eegdenoisenet/EEG_all_epochs.npy')
eog_data = np.load('eeg_formatting/eegdenoisenet/EOG_all_epochs.npy')
emg_data = np.load('eeg_formatting/eegdenoisenet/EMG_all_epochs.npy')
Fs = 256
x = 0
data_no = 100

#combining to make training dataset
def make_training_dataset():
    output_dir = 'eeg_formatting/eegdenoisenet/train'
    for i in range(data_no):
        output_csv = os.path.join(output_dir, f'training_{x}.csv')
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["window_id", "channel index", "channel name", "sample index", "value (uv)"]) #initial headers
        x += 1