"""
Download the MIT-BIH Arrhythmia Database from PhysioNet.

Run this once on a machine with internet access to physionet.org:
    python src/preprocessing/download_mitdb.py

This populates data/mitdb/ with the .dat, .hea and .atr files for
each of the 48 records (100-124, 200-234, subject to availability).

NOTE: PhysioNet is sometimes unreachable from restricted/sandboxed
networks (corporate proxies, CI runners, some cloud dev environments).
If this script fails with a 403/connection error, download the dataset
manually from https://physionet.org/content/mitdb/1.0.0/ instead.
"""

import os
import wfdb

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mitdb")

# Standard 48-record MIT-BIH Arrhythmia Database record list
RECORD_LIST = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def download_all(records=None, out_dir=DATA_DIR):
    records = records or RECORD_LIST
    os.makedirs(out_dir, exist_ok=True)
    ok, failed = [], []

    for rec in records:
        try:
            print(f"Downloading record {rec} ...")
            wfdb.dl_database("mitdb", dl_dir=out_dir, records=[rec])
            ok.append(rec)
        except Exception as e:
            print(f"  Failed: {rec} -> {e}")
            failed.append(rec)

    print(f"\nDone. Downloaded: {len(ok)}/{len(records)}")
    if failed:
        print(f"Failed records: {failed}")
    return ok, failed


if __name__ == "__main__":
    download_all()
