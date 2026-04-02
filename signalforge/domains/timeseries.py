"""
signalforge.domains.timeseries

Generic domain for any two-column time series CSV: date/timestamp + value.

Accepts CSVs with exactly two columns (any names). First column is parsed as
a date, second as a numeric value. Rows with missing values are dropped.

SamplingPlan uses horizon=360, grain=1 (same lattice as equities-daily).
"""

from __future__ import annotations

from ..lattice.coordinates import lattice_members, smallest_divisor_gte
from ..lattice.sampling import SamplingPlan


def sampling_plan(
    horizon: int = 360,
    grain: int = 1,
) -> SamplingPlan:
    """
    SamplingPlan for a generic daily time series.

    Horizon of 360 = 2³ × 3² × 5, giving 24 divisors and clean windows.
    """
    cbin = smallest_divisor_gte(horizon, grain)
    valid = set(lattice_members(horizon, cbin))

    anchors = {1, 5, 10, 20, 60, 180, horizon}
    fine_cutoff = 20

    selected = sorted(
        w for w in valid
        if w in anchors or w <= fine_cutoff or w == horizon
    )

    if not selected:
        selected = sorted(valid)

    return SamplingPlan(horizon, grain, windows=selected)


def ingest(path: str) -> list:
    """
    Load a two-column CSV (date, value) into CanonicalRecords.

    Auto-detects column names. First column is parsed as a date,
    second as a float. Rows with missing or non-numeric values are skipped.
    """
    import pandas as pd
    from ..pipeline.canonical import CanonicalRecord, OrderType

    df = pd.read_csv(path)
    date_col, value_col = df.columns[0], df.columns[1]

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])

    records = [
        CanonicalRecord(
            primary_order=i,
            order_type=OrderType.SEQUENCE,
            channel="value",
            metric="value",
            value=float(row[value_col]),
            seq_order=i,
        )
        for i, row in df.iterrows()
    ]
    return records
