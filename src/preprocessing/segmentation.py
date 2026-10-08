"""
Segment a continuous ECG signal into fixed-length beat windows around
each R-peak, and map PhysioNet MIT-BIH annotation symbols to a binary
Normal / Arrhythmic label.

MIT-BIH annotation symbol reference (subset used here):
    N, L, R, e, j            -> Normal (0)
    A, a, J, S, V, F, E, /, f, Q  -> Arrhythmic (1)
Any symbol not in this map is treated as non-beat (skipped).
"""

import numpy as np

NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}
ARRHYTHMIC_SYMBOLS = {"A", "a", "J", "S", "V", "F", "E", "/", "f", "Q"}


def symbol_to_label(symbol):
    if symbol in NORMAL_SYMBOLS:
        return 0
    if symbol in ARRHYTHMIC_SYMBOLS:
        return 1
    return None  # non-beat annotation (e.g. rhythm change marker), skip


def segment_beats(signal, ann_samples, ann_symbols, fs, window_sec=0.72):
    """
    Extract a fixed-length window around each annotated beat.

    Args:
        signal: 1D bandpass-filtered ECG array.
        ann_samples: sample indices of annotated beats (from .atr file).
        ann_symbols: annotation symbol for each index.
        fs: sampling frequency in Hz.
        window_sec: total window length in seconds, centered on the
            R-peak (0.72 s ~ 260 samples at 360 Hz, enough to cover
            one full QRS-T complex).

    Returns:
        segments: (n_beats, window_len) float array of raw waveform windows.
        labels: (n_beats,) int array, 0 = Normal, 1 = Arrhythmic.
        centers: (n_beats,) int array of the R-peak sample index used per beat.
    """
    half_win = int((window_sec / 2) * fs)
    segments, labels, centers = [], [], []

    for idx, sym in zip(ann_samples, ann_symbols):
        label = symbol_to_label(sym)
        if label is None:
            continue
        lo, hi = idx - half_win, idx + half_win
        if lo < 0 or hi > len(signal):
            continue  # drop incomplete edge beats
        segments.append(signal[lo:hi])
        labels.append(label)
        centers.append(idx)

    return np.array(segments), np.array(labels), np.array(centers)


def match_detected_to_annotated(detected_peaks, ann_samples, ann_symbols, fs, tolerance_sec=0.1):
    """
    For beats found by our own R-peak detector (rather than the ground-truth
    annotation index), match each detected peak to the nearest annotation
    within `tolerance_sec` to assign it a label. Useful for evaluating the
    detector itself (sensitivity / positive predictivity) rather than just
    training on annotation-centered windows.

    Returns:
        matched_peaks, labels, unmatched_count
    """
    tol = int(tolerance_sec * fs)
    ann_samples = np.asarray(ann_samples)
    matched_peaks, labels = [], []
    unmatched = 0

    for peak in detected_peaks:
        diffs = np.abs(ann_samples - peak)
        j = np.argmin(diffs)
        if diffs[j] <= tol:
            label = symbol_to_label(ann_symbols[j])
            if label is not None:
                matched_peaks.append(peak)
                labels.append(label)
                continue
        unmatched += 1

    return np.array(matched_peaks), np.array(labels), unmatched
