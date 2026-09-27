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
| Deep Learning | TensorFlow / PyTorch (1D CNN) |
| Hardware Interfacing | `pyserial` |
| Visualization | `matplotlib` |
| Environment | Jupyter Notebook / Google Colab, Arduino IDE |

## Dataset

- **Primary:** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb) — 48 half-hour ECG recordings from 47 subjects, sampled at 360 Hz, with beat-by-beat cardiologist annotations (Normal, PVC, AFib, etc.)
- **Secondary:** Live, self-captured ECG signal via AD8232 — used only for qualitative validation of preprocessing/R-peak detection, not for model training (unlabeled).

## Methodology

1. **Data Acquisition** — Download MIT-BIH records using `wfdb`
2. **Preprocessing** — Bandpass filtering to remove baseline wander and high-frequency noise
3. **R-Peak Detection** — Pan-Tompkins algorithm
4. **Segmentation** — Fixed-length windows around each R-peak, labeled via PhysioNet annotations
5. **Feature Extraction** — RR-interval, QRS duration, peak amplitude
6. **Model Training** — Random Forest and SVM baselines; 1D CNN on raw waveform segments
7. **Evaluation** — Accuracy, precision, recall, F1-score, confusion matrix
8. **Hardware Integration** — Capture live ECG via AD8232 + Arduino, stream over serial
9. **Live Validation** — Apply the same filtering/R-peak pipeline to the live signal, visualize in real time

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
