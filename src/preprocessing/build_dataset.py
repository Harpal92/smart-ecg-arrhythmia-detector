"""
Build the processed dataset from raw MIT-BIH records in data/mitdb/.

For each record:
    1. Load signal + annotations with wfdb
    2. Bandpass filter the signal
    3. Segment fixed-length windows around each annotated beat
    4. Map annotation symbols to Normal(0)/Arrhythmic(1) labels
    5. Extract handcrafted features (for RF/SVM)

Outputs, written to data/processed/:
    segments.npy   (n_beats, window_len)  raw waveform windows, for the CNN
    features.npy   (n_beats, 5)           handcrafted features, for RF/SVM
    labels.npy     (n_beats,)             0 = Normal, 1 = Arrhythmic
    record_ids.npy (n_beats,)             source record per beat (for patient-level splits)

Usage:
    python src/preprocessing/build_dataset.py
"""

import os
import sys
import glob
import numpy as np
import wfdb

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features"))

from filtering import bandpass_filter
from segmentation import segment_beats
from feature_extraction import extract_features

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mitdb")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
FS = 360  # MIT-BIH sampling frequency


def list_available_records(raw_dir=RAW_DIR):
    hea_files = glob.glob(os.path.join(raw_dir, "*.hea"))
    return sorted(os.path.splitext(os.path.basename(f))[0] for f in hea_files)


def process_record(record_id, raw_dir=RAW_DIR, window_sec=0.72):
    record = wfdb.rdrecord(os.path.join(raw_dir, record_id))
    annotation = wfdb.rdann(os.path.join(raw_dir, record_id), "atr")

    raw_signal = record.p_signal[:, 0]  # lead 0 (MLII in most MIT-BIH records)
    filtered = bandpass_filter(raw_signal, FS)

    segments, labels, centers = segment_beats(
        filtered, annotation.sample, annotation.symbol, FS, window_sec=window_sec
    )
    return segments, labels, centers


def build_dataset(records=None, out_dir=OUT_DIR):
    records = records or list_available_records()
    if not records:
        raise FileNotFoundError(
            f"No MIT-BIH records found in {RAW_DIR}. "
            f"Run src/preprocessing/download_mitdb.py first."
        )

    all_segments, all_labels, all_centers, all_record_ids = [], [], [], []

    for rec in records:
        print(f"Processing record {rec} ...")
        try:
            segments, labels, centers = process_record(rec)
        except Exception as e:
            print(f"  Skipping {rec}: {e}")
            continue
        all_segments.append(segments)
        all_labels.append(labels)
        all_centers.append(centers)
        all_record_ids.extend([rec] * len(labels))

    segments = np.concatenate(all_segments, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    centers = np.concatenate(all_centers, axis=0)
    record_ids = np.array(all_record_ids)

    features = extract_features(segments, centers, FS)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "segments.npy"), segments)
    np.save(os.path.join(out_dir, "features.npy"), features)
    np.save(os.path.join(out_dir, "labels.npy"), labels)
    np.save(os.path.join(out_dir, "record_ids.npy"), record_ids)

    print(f"\nDataset built: {len(labels)} beats "
          f"({np.sum(labels == 0)} Normal, {np.sum(labels == 1)} Arrhythmic)")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    build_dataset()
