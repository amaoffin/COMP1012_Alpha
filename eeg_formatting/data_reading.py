import mne #MNE documentation: https://mne.tools/stable/auto_tutorials/raw/10_raw_overview.html#extracting-data-from-raw-objects
import os
import matplotlib.pyplot as plt
import numpy as np
import csv

def check_extension(file_path, output_dir, name):
    #checks the file extension and calls the appropriate reading function
    _, ext = os.path.splitext(file_path)
    if ext == ".vhdr":
        return read_vhdr_to_csv(file_path, output_dir, name)
    elif ext == ".edf":
        return read_edf_to_csv(file_path, output_dir, name)
    else:
        raise ValueError("Unsupported file format")

def read_vhdr_to_csv(vhdr_file_path, output_dir, name):
    #from mne docs, reads the raw data with vhdr file path and preloads it into memory
    raw = mne.io.read_raw_brainvision(vhdr_file_path, preload=True)
    
    sampling_freq = raw.info["sfreq"]
    #just taking a small window of data for testing
    start_stop_seconds = np.array([11, 13])
    start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)
    
    #channel names are stored in the raw.info dictionary under the key "ch_names"
    channel_names = raw.info["ch_names"]

    output_csv = os.path.join(output_dir, f"{name}.csv") #name for csv
    
    #writing csv
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["window_id", "channel index", "channel name", "sample index", "value (uv)"]) #initial headers
        #looping through each channel and sample to write the data to the csv
        for channel_index, channel_name in enumerate(channel_names):
            for sample_index in range(start_sample, stop_sample):
                value = raw.get_data(picks=channel_index, start=sample_index, stop=sample_index + 1)[0, 0]
                writer.writerow([f"{name}", channel_index, channel_name, sample_index, value*1000000])
    print(f"CSV written to {output_csv}") #success message

def read_edf_to_csv(edf_file_path, output_dir, name):
    #from mne docs, reads the raw data with edf file path and preloads it into memory
    raw = mne.io.read_raw_edf(edf_file_path, preload=True)
    
    sampling_freq = raw.info["sfreq"]
    #just taking a small window of data for testing
    start_stop_seconds = np.array([11, 13])
    start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)
    
    #channel names are stored in the raw.info dictionary under the key "ch_names"
    channel_names = raw.info["ch_names"]

    output_csv = os.path.join(output_dir, f"{name}.csv") #name for csv
    
    #writing csv
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["window_id", "channel index", "channel name", "sample index", "value (uv)"]) #initial headers
        #looping through each channel and sample to write the data to the csv
        for channel_index, channel_name in enumerate(channel_names):
            for sample_index in range(start_sample, stop_sample):
                value = raw.get_data(picks=channel_index, start=sample_index, stop=sample_index + 1)[0, 0]
                writer.writerow([f"{name}", channel_index, channel_name, sample_index, value*1000000])
    print(f"CSV written to {output_csv}") #success message
def main() -> None:
    #example usage of the functions
    file_path = "sample_data/Control1129.vhdr" #path to vhdr file
    output_dir = "processed_data" #output directory for the csv
    name = file_path.split("/")[-1].split(".")[0]  #extracts the name from the file path

    check_extension(file_path, output_dir, name) #calls the function to read the file and write to csv

if __name__ == "__main__":
    main()