import os
import pandas as pd
import numpy as np
from scipy.signal import resample_poly

ORIGINAL_FS = 500
TARGET_FS = 128

def downsample_signal(signal):
    return resample_poly(signal, TARGET_FS, ORIGINAL_FS)

def is_healthy(metadata_path):
    meta = pd.read_csv(metadata_path)
    if "group_code" in meta.columns:
        return str(meta["group_code"].iloc[0]).strip().upper() == "C"
    return False

def process_subject(subject_folder, output_root):
    exports_folder = os.path.join(subject_folder, "exports")

    if not os.path.exists(exports_folder):
        return False

    eeg_file = None
    metadata_file = None

    for f in os.listdir(exports_folder):
        if f.endswith("_channels_timeseries.csv"):
            eeg_file = os.path.join(exports_folder, f)
        elif f.endswith("_metadata.csv"):
            metadata_file = os.path.join(exports_folder, f)

    if not eeg_file or not metadata_file:
        return False

    # Check if healthy
    if not is_healthy(metadata_file):
        return False

    # Process EEG
    subject_name = os.path.basename(subject_folder).replace("_eeg", "")
    output_folder = os.path.join(output_root, subject_name)
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(eeg_file)

    for channel in df.columns:
        signal = df[channel].values
        downsampled = downsample_signal(signal)

        output_file = os.path.join(output_folder, f"{channel}.txt")
        np.savetxt(output_file, downsampled, fmt="%.6f")

    print(f"Processed healthy subject: {subject_name}")
    return True

def main(dataset_root, output_root):
    healthy_count = 0

    for folder in os.listdir(dataset_root):
        subject_path = os.path.join(dataset_root, folder)

        if os.path.isdir(subject_path):
            if process_subject(subject_path, output_root):
                healthy_count += 1

    print(f"\nTotal healthy subjects processed: {healthy_count}")

# -------- RUN HERE --------
dataset_root = "dataset-processed"
output_root = "healthy_output"

main(dataset_root, output_root)
