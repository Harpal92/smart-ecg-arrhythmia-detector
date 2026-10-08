import wfdb

record = wfdb.rdrecord('data/mitdb/100')
annotation = wfdb.rdann('data/mitdb/100', 'atr')

fs = record.fs

# Look at all annotated beats within the first 10 seconds (3600 samples)
print("Beats in the first 10 seconds:")
for sample, symbol in zip(annotation.sample, annotation.symbol):
    if sample < 10 * fs:
        time_sec = sample / fs
        print(f"  Sample {sample} (t={time_sec:.2f}s) -> symbol: '{symbol}'")