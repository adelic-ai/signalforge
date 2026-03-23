"""
edf_to_rms_csv.py — Convert an EDF file to a SignalForge-ready RMS CSV.

Computes per-sample RMS across all EEG channels and writes:
    t_sec, eeg_rms

Usage
-----
    uv run python examples/edf_to_rms_csv.py path/to/recording.edf
    uv run python examples/edf_to_rms_csv.py path/to/recording.edf --out my_output.csv

Requires mne:
    uv add mne
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def edf_to_rms_csv(edf_path: str | Path, out_path: str | Path | None = None) -> Path:
    import mne

    edf_path = Path(edf_path)
    if out_path is None:
        out_path = edf_path.with_suffix("").name + "_eeg_rms.csv"
    out_path = Path(out_path)

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    X = raw.get_data()          # (n_channels, n_samples)
    sfreq = float(raw.info["sfreq"])
    nchan, n = X.shape

    eeg_rms = np.sqrt(np.mean(X ** 2, axis=0))
    t_sec   = np.arange(n) / sfreq

    pd.DataFrame({"t_sec": t_sec, "eeg_rms": eeg_rms}).to_csv(out_path, index=False)

    print(f"wrote {out_path}")
    print(f"  rows={n}  sfreq={sfreq} Hz  channels={nchan}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDF → SignalForge RMS CSV")
    parser.add_argument("edf", help="Path to .edf file")
    parser.add_argument("--out", default=None, help="Output CSV path (default: <name>_eeg_rms.csv)")
    args = parser.parse_args()
    edf_to_rms_csv(args.edf, args.out)
