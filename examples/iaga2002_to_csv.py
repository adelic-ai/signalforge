"""
iaga2002_to_csv.py — Convert an IAGA-2002 .min file to a SignalForge-ready CSV.

IAGA-2002 is the standard format for INTERMAGNET geomagnetic observatory data.
Files are available from https://intermagnet.org

Outputs a long-format CSV:
    timestamp, station, component, value

Usage
-----
    uv run python examples/iaga2002_to_csv.py ykc20250101vmin.min
    uv run python examples/iaga2002_to_csv.py ykc20250101vmin.min --out ykc_jan2025.csv

Multiple files (e.g. one per day) can be concatenated:
    cat ykc2025*.min | uv run python examples/iaga2002_to_csv.py - --out ykc_2025.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


_MISSING = 99999.0   # IAGA-2002 missing value sentinel


def _parse_iaga2002(lines: list[str]) -> tuple[str, list[str], list[dict]]:
    """Parse IAGA-2002 lines. Returns (station_code, component_names, records)."""
    station = ""
    components: list[str] = []
    records: list[dict] = []

    in_header = True
    for line in lines:
        if in_header:
            if line.startswith("IAGA CODE"):
                station = line.split()[2].strip()
            if line.startswith(" # DECBASZ") or (line.startswith("DATE") and "TIME" in line):
                # Component names are the last 4 tokens of the column header line
                # e.g. "DATE       TIME         DOY   YKCX      YKCY      YKCZ      YKCF   |"
                parts = line.split()
                components = [p.rstrip("|") for p in parts[-5:-1]]
                in_header = False
                continue
            if line.strip().startswith("DATE") and "TIME" in line:
                parts = line.split()
                components = [p.rstrip("|") for p in parts[-5:-1]]
                in_header = False
            continue

        # Data line: DATE TIME DOY V1 V2 V3 V4 |
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            ts = pd.Timestamp(f"{parts[0]}T{parts[1]}+00:00")
            values = [float(v) for v in parts[3:7]]
        except (ValueError, IndexError):
            continue

        for comp, val in zip(components, values):
            if abs(val - _MISSING) < 1.0:   # treat as missing
                continue
            records.append({"timestamp": ts, "station": station, "component": comp, "value": val})

    return station, components, records


def iaga2002_to_csv(src: str | Path, out_path: str | Path | None = None) -> Path:
    if str(src) == "-":
        lines = sys.stdin.readlines()
        stem = "iaga"
    else:
        src = Path(src)
        lines = src.read_text(errors="replace").splitlines()
        stem = src.stem

    if out_path is None:
        out_path = Path(f"{stem}.csv")
    out_path = Path(out_path)

    station, components, records = _parse_iaga2002(lines)
    if not records:
        raise ValueError("No data records found — check that the file is IAGA-2002 format.")

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)

    print(f"wrote {out_path}")
    print(f"  station={station}  components={components}  rows={len(df)}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IAGA-2002 → SignalForge CSV")
    parser.add_argument("src", help="Path to .min file, or - to read from stdin")
    parser.add_argument("--out", default=None, help="Output CSV path (default: <stem>.csv)")
    args = parser.parse_args()
    iaga2002_to_csv(args.src, args.out)
