#!/usr/bin/env python3
"""
SignalForge end-to-end pipeline — generic time series.

Input:  Any two-column CSV (date/timestamp, value).
Output: printed z-score summary per scale + optional CSV export.

Examples
--------
uv run python examples/run_timeseries.py --csv data/vix_2005_2012.csv
uv run python examples/run_timeseries.py --csv data/vix_2005_2012.csv --out runs/vix
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from signalforge.pipeline.binned import materialize
from signalforge.pipeline.surface import measure
from signalforge.pipeline.feature import engineer
from signalforge.pipeline.bundle import assemble
from signalforge.domains import timeseries


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SignalForge on a generic time series.")
    ap.add_argument("--csv", required=True, help="Path to two-column CSV (date, value)")
    ap.add_argument("--out", default=None, help="Export directory for CSV output")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    print(f"Dataset : {csv_path.name}")

    # Stage 0 — ingest
    t0 = time.perf_counter()
    records = timeseries.ingest(str(csv_path))
    t_ingest = time.perf_counter() - t0
    print(f"Ingest  : {len(records):,} records  ({t_ingest:.2f}s)")

    # Stage 1 — SamplingPlan
    plan = timeseries.sampling_plan()
    print(f"\nPlan    : horizon={plan.horizon}  grain={plan.grain}")
    print(f"          windows={plan.windows}")
    print(f"          prime_basis={plan.prime_basis}")

    # Stage 2 — materialize bins
    t0 = time.perf_counter()
    agg_funcs = {"value": {"value": "mean"}}
    binned = materialize(records, plan, agg_funcs=agg_funcs)
    t_mat = time.perf_counter() - t0
    print(f"\nBinned  : {len(binned):,} records  ({t_mat:.2f}s)")

    # Stage 3 — measure surfaces
    t0 = time.perf_counter()
    surfaces = measure(binned, plan, profile="continuous")
    t_meas = time.perf_counter() - t0
    print(f"\nSurfaces: {len(surfaces)}")
    for s in surfaces:
        finite = np.isfinite(list(s.values.values())[0])
        print(f"  {s.channel:12s}  shape={s.shape}  coverage={finite.mean():.1%}")
    print(f"  ({t_meas:.2f}s)")

    # Stage 4 — engineer features
    t0 = time.perf_counter()
    tensors = [engineer(s, plan) for s in surfaces]
    t_eng = time.perf_counter() - t0
    print(f"\nTensors : {len(tensors)}, {len(tensors[0].feature_names)} features each")
    print(f"  ({t_eng:.2f}s)")

    # Stage 5 — assemble bundle
    t0 = time.perf_counter()
    bundle = assemble(tensors)
    t_asm = time.perf_counter() - t0
    print(f"\nBundle  : {bundle}")

    # Anomaly summary — z-scores computed from mean surface
    print("\nAnomaly summary (peak z-score per scale):")
    for s in surfaces:
        means = s.values.get("mean")
        if means is None:
            continue
        print(f"\n  {s.channel}")
        scale_peaks = []
        for i, w in enumerate(plan.windows):
            row = means[i]
            finite = row[np.isfinite(row)]
            if len(finite) == 0:
                continue
            mu = np.mean(finite)
            sigma = np.std(finite)
            if sigma == 0:
                continue
            z = (finite - mu) / sigma
            peak = float(np.max(np.abs(z)))
            scale_peaks.append((peak, w))
        scale_peaks.sort(reverse=True)
        for peak, w in scale_peaks[:5]:
            print(f"    scale={w:>4d}  peak |z| = {peak:.2f}σ")

    if args.out:
        out_dir = Path(args.out)
        bundle.export_csv(str(out_dir))
        print(f"\nExported to {out_dir}/")


if __name__ == "__main__":
    main()
