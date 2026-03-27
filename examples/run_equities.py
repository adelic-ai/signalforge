#!/usr/bin/env python3
"""
SignalForge end-to-end pipeline — equity price data.

Input:  CSV produced by examples/yfinance_to_csv.py
        Columns: timestamp, ticker, metric, value

Output: printed z-score summary per scale + optional CSV export

Run from repo root
------------------
# Download GME daily data (includes 2021 squeeze) then run:
uv run python examples/yfinance_to_csv.py --ticker GME --interval 1d --period 2y --metrics Close,Volume --out data/gme_daily.csv
uv run python examples/run_equities.py --csv data/gme_daily.csv --plan daily

# Download recent 1-minute data and run:
uv run python examples/yfinance_to_csv.py --ticker GME --interval 1m --period 5d --out data/gme_1m.csv
uv run python examples/run_equities.py --csv data/gme_1m.csv

# Export results
uv run python examples/run_equities.py --csv data/gme_daily.csv --plan daily --out runs/equities
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

from signalforge.pipeline.canonical import CanonicalRecord, OrderType
from signalforge.pipeline.binned import materialize
from signalforge.pipeline.surface import measure
from signalforge.pipeline.feature import engineer
from signalforge.pipeline.bundle import assemble
from signalforge.domains import equities


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SignalForge on equity data.")
    ap.add_argument("--csv",  required=True, help="Path to CSV from yfinance_to_csv.py")
    ap.add_argument("--plan", default="intraday", choices=["intraday", "daily"],
                    help="Sampling plan: intraday (1-min bars) or daily (default: intraday)")
    ap.add_argument("--out",  default=None, help="Export directory for CSV output")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    print(f"Dataset : {csv_path.name}")

    # Stage 0 — ingest
    t0 = time.perf_counter()
    records = equities.ingest(str(csv_path))
    t_ingest = time.perf_counter() - t0

    channels = sorted({r.channel for r in records})
    tickers  = sorted({r.keys.get("ticker", "?") for r in records})
    print(f"Ingest  : {len(records):,} records  channels={channels}  tickers={tickers}")
    print(f"          ({t_ingest:.2f}s)")

    # Stage 1 — SamplingPlan
    if args.plan == "daily":
        plan = equities.sampling_plan_daily()
        plan_label = f"daily  horizon={plan.horizon}d  grain={plan.grain}d"
    else:
        plan = equities.sampling_plan()
        plan_label = f"intraday  horizon={plan.horizon} bars  grain={plan.grain} bar"

    print(f"\nPlan    : {plan_label}")
    print(f"          windows={plan.windows}")
    print(f"          prime_basis={plan.prime_basis}")

    # Stage 2 — materialize bins
    t0 = time.perf_counter()
    agg_funcs = {ch: {"value": "mean"} for ch in channels}
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

    # Anomaly summary — top z-scores per channel per scale
    print("\nAnomaly summary (peak z-score per scale):")
    for s in surfaces:
        if "z_score" not in s.values:
            continue
        zs = s.values["z_score"]
        print(f"\n  {s.channel}")
        scale_peaks = []
        for i, w in enumerate(plan.windows):
            row = zs[i]
            finite = row[np.isfinite(row)]
            if len(finite) == 0:
                continue
            peak = float(np.max(np.abs(finite)))
            scale_peaks.append((peak, w))
        scale_peaks.sort(reverse=True)
        for peak, w in scale_peaks[:5]:
            label = f"{w}d" if args.plan == "daily" else f"{w}bar"
            print(f"    scale={label:>8s}  peak |z| = {peak:.2f}σ")

    if args.out:
        out_dir = Path(args.out)
        bundle.export_csv(str(out_dir))
        print(f"\nExported to {out_dir}/")


if __name__ == "__main__":
    main()
