#!/usr/bin/env python3
"""
SignalForge end-to-end pipeline — Intermagnet geomagnetic data.

Input:  docs/yellowknife_2025_01_01_to_09.csv
        Yellowknife Observatory (YKC), Jan 1–9 2025, 1-minute cadence.
        Columns: timestamp, station, component, value

Output: printed summary + optional CSV export (pass --out <dir>)

Run from repo root:
    python docs/run_intermagnet.py
    python docs/run_intermagnet.py --out docs/runs/intermagnet
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

from signalforge.pipeline.canonical import CanonicalRecord, OrderType
from signalforge.pipeline.binned import materialize
from signalforge.pipeline.surface import measure
from signalforge.pipeline.feature import engineer
from signalforge.pipeline.bundle import assemble
from signalforge.domains import intermagnet


def _to_epoch_seconds(ts: pd.Series) -> pd.Series:
    """Unix epoch seconds from a tz-aware datetime series, any pandas resolution."""
    raw = ts.astype("int64")
    dtype = str(ts.dtype)  # e.g. "datetime64[us, UTC]" or "datetime64[ns, UTC]"
    if "[s" in dtype and "[us" not in dtype and "[ns" not in dtype:
        return raw                   # already seconds
    if "[ms" in dtype:
        return raw // 1_000
    if "[us" in dtype:
        return raw // 1_000_000
    return raw // 1_000_000_000      # nanoseconds (default)


def load_records(csv_path: Path) -> list[CanonicalRecord]:
    df = pd.read_csv(csv_path)

    # Parse timestamps → unix epoch seconds (integer), pandas-version-safe.
    df["epoch"] = _to_epoch_seconds(pd.to_datetime(df["timestamp"], utc=True))

    records = []
    for row in df.itertuples(index=False):
        epoch = int(row.epoch)
        records.append(CanonicalRecord(
            primary_order=epoch,
            order_type=OrderType.TIME,
            channel=str(row.component),
            metric="value",
            value=float(row.value),
            keys={"station": str(row.station)},
            time_order=epoch,
        ))

    # Pipeline expects non-decreasing primary_order.
    records.sort(key=lambda r: r.primary_order)
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "yellowknife_2025_01_01_to_09.csv"))
    ap.add_argument("--out", default=None, help="Export directory for CSV output")
    ap.add_argument(
        "--grain", type=int, default=None,
        help="Bin size in seconds (default: 60 for short datasets, 86400 for year-scale). "
             "Use 86400 for daily-resolution year analysis.",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    print(f"Dataset : {csv_path.name}")

    # Stage 0 — ingest
    t0 = time.perf_counter()
    records = load_records(csv_path)
    t_ingest = time.perf_counter() - t0
    channels = sorted({r.channel for r in records})
    print(f"Ingest  : {len(records):,} records, {len(channels)} channels: {channels}")
    print(f"          ({t_ingest:.2f}s)")

    # Auto-select grain: if more than 30 days of data and grain not specified, use daily.
    if args.grain is not None:
        grain = args.grain
    else:
        span_s = records[-1].primary_order - records[0].primary_order
        grain = 86400 if span_s > 30 * 86400 else 60
        if grain == 86400:
            print(f"\nNote    : large dataset ({span_s // 86400} days) — "
                  f"using daily bins (grain=86400). Pass --grain 60 for minute resolution.")

    # Stage 1 — SamplingPlan
    if grain == 86400:
        plan = intermagnet.sampling_plan_yearly()
    else:
        plan = intermagnet.sampling_plan()
    print(f"\nPlan    : horizon={plan.horizon}s  grain={plan.grain}s  cbin={plan.cbin}s")
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
    print(f"  features: {tensors[0].feature_names}")
    print(f"  ({t_eng:.2f}s)")

    # Stage 5 — assemble bundle
    t0 = time.perf_counter()
    bundle = assemble(tensors)
    t_asm = time.perf_counter() - t0
    print(f"\nBundle  : {bundle}")
    print(f"  entities : {bundle.entities}")
    print(f"  ({t_asm:.2f}s)")

    # Spot check: mean value for YKCF at finest scale
    for s in surfaces:
        if "mean" in s.values:
            row0 = s.values["mean"][0]
            finite_vals = row0[np.isfinite(row0)]
            if len(finite_vals) > 0:
                print(f"\n  {s.channel} mean[scale=0]: "
                      f"min={finite_vals.min():.3f}  "
                      f"max={finite_vals.max():.3f}  "
                      f"mean={finite_vals.mean():.3f}")

    if args.out:
        out_dir = Path(args.out)
        bundle.export_csv(str(out_dir))
        print(f"\nExported to {out_dir}/")


if __name__ == "__main__":
    main()
