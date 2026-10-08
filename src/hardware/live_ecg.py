"""
Read live ECG data streamed from the AD8232 + Arduino UNO module over
USB serial, apply the same bandpass filtering and Pan-Tompkins R-peak
detection used on the MIT-BIH pipeline, and plot it in real time.

Usage:
    python src/hardware/live_ecg.py --port COM3          (Windows)
    python src/hardware/live_ecg.py --port /dev/ttyUSB0   (Linux)
    python src/hardware/live_ecg.py --port /dev/tty.usbmodem14101  (macOS)

Requires the Arduino to be running arduino/ecg_acquisition/ecg_acquisition.ino
at the matching baud rate (115200) and sample rate (250 Hz).
"""

import os
import sys
import argparse
import collections
import numpy as np
import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))
from filtering import bandpass_filter          # noqa: E402
from r_peak_detection import pan_tompkins_r_peaks  # noqa: E402

FS = 250            # must match SAMPLE_RATE_HZ in the Arduino sketch
BUFFER_SECONDS = 6  # rolling window shown on screen
BUFFER_LEN = FS * BUFFER_SECONDS


def read_serial_value(ser):
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line == "LO" or line == "":
            return None
        return float(line)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM3 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    buffer = collections.deque([0.0] * BUFFER_LEN, maxlen=BUFFER_LEN)

    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot([], [], lw=1.2, color="crimson")
    peak_scatter = ax.scatter([], [], color="black", zorder=5, s=20)
    ax.set_xlim(0, BUFFER_SECONDS)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Filtered amplitude (normalized)")
    ax.set_title("Live ECG — filtered signal with detected R-peaks")

    def update(_frame):
        # Drain everything currently waiting in the serial buffer
        while ser.in_waiting:
            val = read_serial_value(ser)
            if val is not None:
                buffer.append(val)

        raw = np.array(buffer)
        if raw.std() == 0:
            return line, peak_scatter

        filtered = bandpass_filter(raw, FS, lowcut=0.5, highcut=40.0)
        filtered = (filtered - filtered.mean()) / (filtered.std() + 1e-8)
        peaks = pan_tompkins_r_peaks(filtered, FS)

        t = np.linspace(0, BUFFER_SECONDS, len(filtered))
        line.set_data(t, filtered)
        if len(peaks):
            peak_scatter.set_offsets(np.column_stack([t[peaks], filtered[peaks]]))
        else:
            peak_scatter.set_offsets(np.empty((0, 2)))
        return line, peak_scatter

    ani = FuncAnimation(fig, update, interval=100, blit=True)  # noqa: F841
    plt.tight_layout()
    plt.show()

    ser.close()


if __name__ == "__main__":
    main()
