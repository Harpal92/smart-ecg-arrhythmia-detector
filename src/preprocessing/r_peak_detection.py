"""
R-peak detection using a simplified Pan-Tompkins algorithm.

Pipeline: derivative -> squaring -> moving-window integration ->
adaptive thresholding -> peak refinement on the original filtered signal.

Reference:
    Pan, J., Tompkins, W.J. "A Real-Time QRS Detection Algorithm."
    IEEE Transactions on Biomedical Engineering, 1985.
"""

import numpy as np
from scipy.signal import find_peaks


def pan_tompkins_r_peaks(signal, fs, refractory_ms=200):
    """
    Detect R-peak sample indices in a bandpass-filtered ECG signal.

    Args:
        signal: 1D bandpass-filtered ECG array.
        fs: sampling frequency in Hz.
        refractory_ms: minimum physiological distance between beats
            (200 ms ~ 300 bpm upper bound) to suppress double-detections.

    Returns:
        1D array of integer sample indices corresponding to R-peaks.
    """
    signal = np.asarray(signal, dtype=float)

    # 1. Derivative (highlights QRS slope)
    derivative = np.diff(signal, prepend=signal[0])

    # 2. Squaring (nonlinear amplification, all-positive)
    squared = derivative ** 2

    # 3. Moving-window integration (~150 ms window, smooths into QRS "hump")
    window_size = max(1, int(0.150 * fs))
    integrated = np.convolve(squared, np.ones(window_size) / window_size, mode="same")

    # 4. Adaptive threshold based on signal statistics
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    min_distance = int((refractory_ms / 1000.0) * fs)

    peaks, _ = find_peaks(integrated, height=threshold, distance=min_distance)

    # 5. Refine each peak to the true local maximum in the filtered signal
    #    within a small search window (integration stage introduces lag/smear)
    search_radius = int(0.075 * fs)
    refined_peaks = []
    for p in peaks:
        lo = max(0, p - search_radius)
        hi = min(len(signal), p + search_radius)
        if hi <= lo:
            continue
        local_max = lo + np.argmax(signal[lo:hi])
        refined_peaks.append(local_max)

    return np.unique(np.array(refined_peaks, dtype=int))


def compute_heart_rate(r_peaks, fs):
    """Instantaneous heart rate (bpm) from consecutive R-peak intervals."""
    if len(r_peaks) < 2:
        return np.array([])
    rr_intervals_sec = np.diff(r_peaks) / fs
    return 60.0 / rr_intervals_sec
