import wfdb
import matplotlib.pyplot as plt

record = wfdb.rdrecord('data/mitdb/100')
signal = record.p_signal[:, 0]

# Zoom into a window around the arrhythmic beat at sample 2044
center = 2044
window = 400
segment = signal[center - window : center + window]

plt.figure(figsize=(10, 4))
plt.plot(segment)
plt.axvline(x=window, color='red', linestyle='--', label='Arrhythmic (A) beat')
plt.title("Zoomed view: Arrhythmic beat vs. neighbors")
plt.xlabel("Sample number (relative)")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.grid(True)
plt.savefig("arrhythmic_beat_zoom.png", dpi=120)
plt.show()
print("Saved to arrhythmic_beat_zoom.png")