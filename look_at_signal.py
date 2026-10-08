import wfdb
import matplotlib.pyplot as plt

record = wfdb.rdrecord('data/mitdb/100')
signal = record.p_signal[:, 0]   # first lead: MLII
fs = record.fs

# Look at the first 10 seconds only (3600 samples at 360 Hz) - full 30 min would be unreadable
seconds_to_show = 10
n_samples = seconds_to_show * fs
segment = signal[:n_samples]

print("Signal stats over this segment:")
print("  Min:", segment.min())
print("  Max:", segment.max())
print("  Mean:", segment.mean())

plt.figure(figsize=(12, 4))
plt.plot(segment)
plt.title(f"Raw, UNFILTERED ECG signal - Record 100 - First {seconds_to_show} seconds")
plt.xlabel("Sample number")
plt.ylabel("Amplitude (mV)")
plt.grid(True)
plt.savefig("raw_signal_look.png", dpi=120)
print("Saved plot to raw_signal_look.png")
plt.show()