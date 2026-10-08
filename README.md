# ECG-Based Arrhythmia Detection Using Machine Learning

An end-to-end system for detecting cardiac arrhythmias from ECG signals using classical machine learning and deep learning, validated with a low-cost real-time hardware acquisition module.

## Overview

This project detects arrhythmic heartbeats from ECG signals through a two-part pipeline:

1. **Software Pipeline** — Trained and evaluated on the MIT-BIH Arrhythmia Database, using signal processing (R-peak detection) and both classical ML (Random Forest, SVM) and deep learning (1D CNN) models.
2. **Hardware-Based Live Validation** — A low-cost AD8232 + Arduino UNO ECG acquisition module captures a real, live heartbeat signal, which is passed through the same filtering and R-peak detection pipeline for real-time, real-world validation beyond the training dataset.

## Problem Statement

Manual ECG interpretation is time-consuming, requires specialized expertise, and doesn't scale well for continuous or remote patient monitoring. This project builds an automated, accurate system for classifying heartbeats as **Normal** or **Arrhythmic**, and validates the detection pipeline on real, physically captured signals — not just pre-recorded datasets.

## Objectives

- Study and preprocess ECG signals from a standard, expert-annotated clinical dataset
- Detect individual heartbeats using R-peak detection
- Extract features and train ML/DL models to classify heartbeats as Normal or Arrhythmic
- Evaluate model performance against cardiologist-labeled ground truth
- Design a low-cost hardware module for live ECG acquisition and validate the trained pipeline on real, self-captured data

## Hardware Used

| Component | Function |
|---|---|
| AD8232 ECG Sensor Module | Acquires the analog electrocardiographic signal from surface electrodes |
| Arduino UNO | Digitizes the analog signal and transmits it via USB serial |
| Disposable ECG Electrode Pads (3-lead) | Interface with the body to detect electrical activity |
| Connecting Wires | Electrical connections between sensor module and microcontroller |

The AD8232 interfaces with the Arduino UNO via analog input (A0) and two digital pins (LO+, LO-) for lead-off detection. Three electrodes follow a standard 3-lead configuration (right arm, left arm, right leg).

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python, Arduino C/C++ |
| Data Handling | `wfdb`, `numpy`, `pandas` |
| Signal Processing | `scipy` (bandpass filtering, Pan-Tompkins R-peak detection) |
| Machine Learning | `scikit-learn` (Random Forest, SVM) |
| Deep Learning | TensorFlow / Keras (1D CNN) |
| Hardware Interfacing | `pyserial` |
| Visualization | `matplotlib` |

## Dataset

- **Primary (real):** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb) — 48 half-hour ECG recordings from 47 subjects, sampled at 360 Hz, with beat-by-beat cardiologist annotations (Normal, PVC, AFib, etc.). Download it with `python src/preprocessing/download_mitdb.py` (requires internet access to physionet.org).
- **Bundled demo dataset (`data/sample_demo/`):** a small **synthetic** ECG dataset in the same WFDB format (3 records, ~1,200 beats), included so the full pipeline runs out of the box for testing without needing to fetch anything first. It is **not real clinical data** — swap it for the real MIT-BIH data before drawing any actual conclusions or writing results into your report.
- **Secondary:** Live ECG signal captured via your own AD8232 hardware module — used for qualitative validation of preprocessing/R-peak detection, not for model training (unlabeled).

## Methodology

1. **Data Acquisition** — Download MIT-BIH records using `wfdb`
2. **Preprocessing** — Bandpass filtering to remove baseline wander and high-frequency noise
3. **R-Peak Detection** — Pan-Tompkins algorithm
4. **Segmentation** — Fixed-length windows around each R-peak, labeled via PhysioNet annotations
5. **Feature Extraction** — RR-interval, QRS duration, peak amplitude, signal energy
6. **Model Training** — Random Forest and SVM baselines on handcrafted features; 1D CNN on raw waveform segments
7. **Evaluation** — Accuracy, precision, recall, F1-score, confusion matrix
8. **Hardware Integration** — Capture live ECG via AD8232 + Arduino, stream over serial
9. **Live Validation** — Apply the same filtering/R-peak pipeline to the live signal, visualize in real time

## Repository Structure

```
├── data/
│   ├── mitdb/               # real MIT-BIH data goes here (run download_mitdb.py)
│   └── sample_demo/         # bundled synthetic demo dataset (WFDB format)
├── src/
│   ├── preprocessing/       # download, filtering, R-peak detection, segmentation, build_dataset
│   ├── features/            # handcrafted feature extraction
│   ├── models/              # RF/SVM baseline, CNN, evaluation
│   └── hardware/            # live serial ECG reader + real-time visualization
├── arduino/
│   └── ecg_acquisition/     # AD8232 + Arduino UNO acquisition sketch
├── results/                 # generated models/, figures/ land here after running
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.9+
- Arduino IDE (only needed for the hardware module)
- AD8232 + Arduino UNO setup (optional, for live validation)

### Installation
```bash
git clone https://github.com/<your-username>/ecg-arrhythmia-detection-ml.git
cd ecg-arrhythmia-detection-ml
pip install -r requirements.txt
```

### 1. Get data
Either use the bundled synthetic demo data (already in `data/sample_demo/`, no action needed), or fetch the real dataset:
```bash
python src/preprocessing/download_mitdb.py
```

### 2. Build the processed dataset (segments + features + labels)
```bash
cd src/preprocessing
python -c "from build_dataset import build_dataset, list_available_records; build_dataset(records=list_available_records('../../data/sample_demo'))"
```
(Swap the path to `../../data/mitdb` once you've downloaded the real dataset — records are auto-discovered from whichever folder you point at.)

### 3. Train models
```bash
cd ../..
python src/models/train_baseline.py --model rf
python src/models/train_baseline.py --model svm
python src/models/train_cnn.py --epochs 20
```

### 4. Evaluate
```bash
python src/models/evaluate.py --model all
```
Prints accuracy/precision/recall/F1 for each model and saves confusion matrix plots to `results/figures/`.

### 5. Live hardware validation (optional)
1. Wire the AD8232 to the Arduino UNO as described in `arduino/ecg_acquisition/ecg_acquisition.ino`.
2. Flash the sketch via the Arduino IDE.
3. Run:
```bash
python src/hardware/live_ecg.py --port /dev/ttyUSB0   # or COM3, /dev/tty.usbmodemXXXX, etc.
```
This streams the live signal, applies the same bandpass filter + Pan-Tompkins detector as the training pipeline, and plots it in real time with detected R-peaks marked.

## Expected Outcome

A machine learning system capable of classifying ECG heartbeats as Normal or Arrhythmic with high recall, benchmarked against MIT-BIH, alongside a functioning low-cost hardware prototype demonstrating live ECG acquisition and real-time signal processing. Includes a comparative analysis of classical ML vs. deep learning approaches.

## Applications

- Wearable ECG monitoring devices with on-device or cloud-based arrhythmia alerts
- Remote patient monitoring in areas with limited access to cardiologists
- Triage / second-opinion support tool for physicians reviewing large volumes of ECG data
- Educational demonstration of signal processing and ML applied to biomedical signals

## References

- Moody, G.B., Mark, R.G. "The impact of the MIT-BIH Arrhythmia Database." *IEEE Engineering in Medicine and Biology Magazine*, 2001.
- [PhysioNet: MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb)
- Pan, J., Tompkins, W.J. "A Real-Time QRS Detection Algorithm." *IEEE Transactions on Biomedical Engineering*, 1985.
- Analog Devices, AD8232 Single-Lead Heart Rate Monitor Front End — Datasheet

## Authors

- Jatin Kamboj — UE235056
- Harpal Singh — UE235048

7th Semester ECE Project, UIET, Panjab University
