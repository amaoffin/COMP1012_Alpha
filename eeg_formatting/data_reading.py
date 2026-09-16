import mne
import os

import matplotlib.pyplot as plt
import numpy as np
import csv

##From MNE documentation: https://mne.tools/stable/auto_tutorials/raw/10_raw_overview.html#extracting-data-from-raw-objects
raw = mne.io.read_raw_brainvision('eeg_formatting/sample_data/Control1129.vhdr', preload=True)
'''
n_time_samps = raw.n_times
time_secs = raw.times
ch_names = raw.ch_names
n_chan = len(ch_names)  # note: there is no raw.n_channels attribute
print(
    f"the (cropped) sample data object has {n_time_samps} time samples and "
    f"{n_chan} channels."
)
print(f"The last time sample is at {time_secs[-1]} seconds.")
print("The first few channel names are {}.".format(", ".join(ch_names[:3])))
print()  # insert a blank line in the output

# some examples of raw.info:
print("bad channels:", raw.info["bads"])  # chs marked "bad" during acquisition
print(raw.info["sfreq"], "Hz")  # sampling frequency
print(raw.info["description"], "\n")  # miscellaneous acquisition info

print(raw.info)
'''

sampling_freq = raw.info["sfreq"]
start_stop_seconds = np.array([11, 13])
start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)

channel_names = raw.info["ch_names"]

name = "Control1129"
output_dir = "eeg_formatting/processed_data"
output_csv = os.path.join(output_dir, f"{name}.csv")
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["window_id", "channel index", "channel name", "sample index", "value (uv)"])
    for channel_index, channel_name in enumerate(channel_names):
        for sample_index in range(start_sample, stop_sample):
            value = raw.get_data(picks=channel_index, start=sample_index, stop=sample_index + 1)[0, 0]
            writer.writerow([f"{name}", channel_index, channel_name, sample_index, value])
print(f"CSV written to {output_csv}")
##SUCCESS!! will make functions for diff formats next and also check if uv is valid bc very small values