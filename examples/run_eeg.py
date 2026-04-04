#!/usr/bin/env python3
"""
SignalForge end-to-end pipeline — CHB-MIT EEG (seizure detection).

Input:  docs/eeg_chbmit/chb01_03_eeg_rms.csv
        chb01_03.edf → RMS across 23 channels at 256 Hz.
        Columns: t_sec, eeg_rms
        Seizure window: 2996–3036 seconds (minutes 49–50).

        Use eeg_chbmit/edf_to_helix_signal.py to regenerate from raw EDF.

Output: printed summary + optional CSV export (pass --out <dir>)

Run from repo root:
    python docs/run_eeg.py
    python docs/run_eeg.py --out docs/runs/eeg

Primary ordering uses sample index (sample_idx = round(t_sec * 256)).
Grain = 256 samples = 1 second. Bins are 1-second RMS epochs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from signalforge.signal import CanonicalRecord, OrderType
from signalforge.pipeline.binned import materialize
from signalforge.pipeline.surface import measure
from signalforge.pipeline.feature import engineer
from signalforge.pipeline.bundle import assemble
from signalforge.domains import eeg

SFREQ = 256
SEIZURE_START_S = 2996
SEIZURE_END_S = 3036


def load_records(csv_path: Path) -> list[CanonicalRecord]:
    df = pd.read_csv(csv_path)

    t_secs = df["t_sec"].to_numpy(dtype=np.float64)
    rms_values = df["eeg_rms"].to_numpy(dtype=np.float64)
    sample_indices = np.round(t_secs * SFREQ).astype(np.int64)

    records = [
        CanonicalRecord(
            primary_order=int(si),
            order_type=OrderType.SEQUENCE,
            channel="eeg",
            metric="rms",
            value=float(v),
            seq_order=int(si),
        )
        for si, v in zip(sample_indices, rms_values)
    ]
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "eeg_chbmit" / "chb01_03_eeg_rms.csv"))
    ap.add_argument("--out", default=None, help="Export directory for CSV output")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    print(f"Dataset : {csv_path.name}")
    print(f"Seizure : {SEIZURE_START_S}–{SEIZURE_END_S} s")

    # Stage 0 — ingest
    t0 = time.perf_counter()
    records = load_records(csv_path)
    t_ingest = time.perf_counter() - t0
    print(f"\nIngest  : {len(records):,} records at {SFREQ} Hz  ({t_ingest:.2f}s)")

    # Stage 1 — SamplingPlan (horizon = 921600 samples = 1 hr at 256 Hz)
    plan = eeg.sampling_plan()
    windows_s = tuple(w // SFREQ for w in plan.windows)
    print(f"\nPlan    : horizon={plan.horizon} samples ({plan.horizon // SFREQ}s)")
    print(f"          grain={plan.grain} samples (1s)  cbin={plan.cbin}")
    print(f"          windows (seconds): {windows_s}")
    print(f"          prime_basis={plan.prime_basis}")

    # Stage 2 — materialize bins (1-second bins, aggregated by mean RMS)
    t0 = time.perf_counter()
    agg_funcs = {"eeg": {"rms": "mean"}}
    binned = materialize(records, plan, agg_funcs=agg_funcs)
    t_mat = time.perf_counter() - t0
    print(f"\nBinned  : {len(binned):,} 1-second bins  ({t_mat:.2f}s)")

    # Stage 3 — measure surfaces
    t0 = time.perf_counter()
    surfaces = measure(binned, plan, profile="continuous")
    t_meas = time.perf_counter() - t0
    print(f"\nSurfaces: {len(surfaces)}")
    for s in surfaces:
        finite = np.isfinite(list(s.values.values())[0])
        print(f"  {s.channel}  shape={s.shape}  coverage={finite.mean():.1%}")
    print(f"  ({t_meas:.2f}s)")

    # Stage 4 — engineer features
    t0 = time.perf_counter()
    tensors = [engineer(s, plan) for s in surfaces]
    t_eng = time.perf_counter() - t0
    print(f"\nTensors : {len(tensors)}, {len(tensors[0].feature_names)} features each")
    print(f"  features: {tensors[0].feature_names}")
    print(f"  ({t_eng:.2f}s)")

    # Stage 5 — assemble bundle
    t0 = time.perf_counter()
    bundle = assemble(tensors)
    t_asm = time.perf_counter() - t0
    print(f"\nBundle  : {bundle}")
    print(f"  ({t_asm:.2f}s)")

    # Seizure zone check: z-scores around the seizure window.
    # bin_index = sample_index // cbin. Since primary_order = sample_index:
    #   seizure bins ≈ [SEIZURE_START_S, SEIZURE_END_S]
    # time_axis holds bin indices. Convert seconds → bin index using min_bin offset.
    ft = tensors[0]
    if "mean_zscore" in ft.values:
        time_axis = np.array(ft.time_axis)
        min_bin = int(time_axis[0])

        # bin_index for a time in seconds (sample_index = t_s * 256, bin = sample // cbin)
        seiz_bin_start = (SEIZURE_START_S * SFREQ) // plan.cbin
        seiz_bin_end   = (SEIZURE_END_S   * SFREQ) // plan.cbin

        t_start_idx = np.searchsorted(time_axis, seiz_bin_start)
        t_end_idx   = np.searchsorted(time_axis, seiz_bin_end)

        zscore_mat = ft.values["mean_zscore"]  # (n_scales, n_time)

        print(f"\nSeizure zone (bins {seiz_bin_start}–{seiz_bin_end}):")
        windows_s_list = list(windows_s)
        for scale_idx, ws in enumerate(windows_s_list):
            row = zscore_mat[scale_idx, t_start_idx:t_end_idx]
            valid = row[np.isfinite(row)]
            if len(valid) > 0:
                print(f"  scale={ws:5d}s  "
                      f"z_max={valid.max():.2f}  "
                      f"z_mean={valid.mean():.2f}")

    if args.out:
        out_dir = Path(args.out)
        bundle.export_csv(str(out_dir))
        print(f"\nExported to {out_dir}/")


if __name__ == "__main__":
    main()
