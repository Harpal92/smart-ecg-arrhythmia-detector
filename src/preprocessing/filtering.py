"""
Bandpass filtering for raw ECG signals.

Removes baseline wander (low-frequency drift from respiration/movement)
and high-frequency noise (muscle artifact, powerline interference),
isolating the 0.5-40 Hz band where the clinically relevant QRS/ECG
morphology lives.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    """
    Apply a zero-phase Butterworth bandpass filter.

    Args:
        signal: 1D array of raw ECG samples.
        fs: sampling frequency in Hz (360 Hz for MIT-BIH).
        lowcut: lower cutoff frequency in Hz.
        highcut: upper cutoff frequency in Hz.
        order: filter order.

    Returns:
        Filtered 1D array, same length as input.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def remove_baseline_wander(signal, fs, cutoff=0.5, order=2):
    """High-pass filter alone, useful if you only want wander removed."""
    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = butter(order, high, btype="high")
    return filtfilt(b, a, signal)


def normalize(signal):
    """Zero-mean, unit-variance normalization."""
    signal = np.asarray(signal, dtype=float)
    std = signal.std()
    if std == 0:
        return signal - signal.mean()
    return (signal - signal.mean()) / std
