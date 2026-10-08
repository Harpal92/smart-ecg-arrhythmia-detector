"""
Handcrafted feature extraction for the classical ML baselines
(Random Forest, SVM): RR-interval, QRS duration, and peak amplitude,
computed per beat segment.
"""

import numpy as np


def _prev_rr(centers, fs):
    """RR interval (sec) to the previous beat; first beat gets the median."""
    rr = np.diff(centers) / fs
    rr = np.concatenate([[np.median(rr) if len(rr) else 0.0], rr])
    return rr


def _next_rr(centers, fs):
    """RR interval (sec) to the next beat; last beat gets the median."""
    rr = np.diff(centers) / fs
    rr = np.concatenate([rr, [np.median(rr) if len(rr) else 0.0]])
    return rr


def estimate_qrs_duration(segment, fs, amplitude_thresh_ratio=0.3):
    """
    Rough QRS duration estimate: width of the region around the segment's
    peak that exceeds `amplitude_thresh_ratio` of the peak amplitude.
    """
    peak_idx = np.argmax(np.abs(segment))
    peak_val = np.abs(segment[peak_idx])
    thresh = amplitude_thresh_ratio * peak_val

    above = np.abs(segment) >= thresh
    # walk outward from the peak to find contiguous region above threshold
    start = peak_idx
    while start > 0 and above[start - 1]:
        start -= 1
    end = peak_idx
    while end < len(segment) - 1 and above[end + 1]:
        end += 1

    return (end - start) / fs  # seconds


def extract_features(segments, centers, fs):
    """
    Build a feature matrix from beat segments.

    Args:
        segments: (n_beats, window_len) array of raw waveform windows.
        centers: (n_beats,) array of R-peak sample indices (for RR intervals).
        fs: sampling frequency in Hz.

    Returns:
        (n_beats, 5) feature matrix:
            [prev_rr, next_rr, qrs_duration, peak_amplitude, signal_energy]
    """
    prev_rr = _prev_rr(centers, fs)
    next_rr = _next_rr(centers, fs)

    qrs_durations = np.array([estimate_qrs_duration(seg, fs) for seg in segments])
    peak_amplitudes = np.array([np.max(np.abs(seg)) for seg in segments])
    signal_energy = np.array([np.sum(seg ** 2) for seg in segments])

    features = np.column_stack([
        prev_rr, next_rr, qrs_durations, peak_amplitudes, signal_energy
    ])
    return features


FEATURE_NAMES = ["prev_rr", "next_rr", "qrs_duration", "peak_amplitude", "signal_energy"]
