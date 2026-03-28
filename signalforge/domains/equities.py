"""
signalforge.domains.equities

SamplingPlan factory for intraday and daily equity price data.

Two plans are provided:

    sampling_plan()        — intraday, 1-minute bars, horizon=360 bars (6 hours)
    sampling_plan_daily()  — daily bars, horizon=360 days

Horizon choice
--------------
A full NYSE session is 390 minutes (9:30–16:00). 390 = 2 × 3 × 5 × 13 — a
thin lattice. 360 = 2³ × 3² × 5 gives 24 divisors and clean standard windows
(1, 5, 15, 30, 60, 360 minutes). Trim or offset data to fit the 360-bar window.

Standard intraday windows (minutes): 1, 5, 15, 30, 60, 360.
Standard daily windows (days): 1, 5, 10, 20, 60, 180, 360.

Data source
-----------
yfinance is the recommended free source. See examples/yfinance_to_csv.py.
1-minute data is available for the trailing ~7 days. Daily data goes back years.

For historical 1-minute data (e.g. GME squeeze, January 2021), use a paid
provider (Polygon.io, Alpaca) or fall back to daily resolution.

References
----------
NYSE trading hours: https://www.nyse.com/markets/hours-calendars
"""

from __future__ import annotations

from ..lattice.coordinates import lattice_members, smallest_divisor_gte
from ..lattice.sampling import SamplingPlan

# Intraday constants (minutes)
_ONE_BAR     = 1
_FIVE_BARS   = 5
_FIFTEEN     = 15
_THIRTY      = 30
_ONE_HOUR    = 60
_SESSION     = 360    # 6 hours — 360 = 2³ × 3² × 5, 24 divisors

# Daily constants (days)
_ONE_WEEK    = 5      # trading days
_TWO_WEEKS   = 10
_ONE_MONTH   = 20
_QUARTER     = 60
_HALF_YEAR   = 180
_ONE_YEAR    = 360


def sampling_plan(
    horizon: int = _SESSION,
    grain: int = _ONE_BAR,
) -> SamplingPlan:
    """
    SamplingPlan for intraday equity data at 1-minute bar resolution.

    Horizon of 360 bars covers 6 hours (6:30 open through ~12:30, or
    trimmed from either end of the 9:30–16:00 session). Standard anchor
    windows: 1, 5, 15, 30, 60, and 360 minutes.

    Parameters
    ----------
    horizon : int
        Session length in bars. Default: 360.
    grain : int
        Bars per bin. Default: 1 (one bar = one minute).

    Returns
    -------
    SamplingPlan

    Examples
    --------
    >>> from signalforge.domains import equities
    >>> plan = equities.sampling_plan()
    >>> plan.windows
    (1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360)
    """
    cbin = smallest_divisor_gte(horizon, grain)
    valid = set(lattice_members(horizon, cbin))

    anchors = {_ONE_BAR, _FIVE_BARS, _FIFTEEN, _THIRTY, _ONE_HOUR, horizon}
    fine_cutoff = _ONE_HOUR

    selected = sorted(
        w for w in valid
        if w in anchors or w <= fine_cutoff or w == horizon
    )

    if not selected:
        selected = sorted(valid)

    return SamplingPlan(horizon, grain, windows=selected)


def sampling_plan_daily(
    horizon: int = _ONE_YEAR,
    grain: int = _ONE_BAR,
) -> SamplingPlan:
    """
    SamplingPlan for daily equity bar data.

    Horizon of 360 trading days = 2³ × 3² × 5, giving clean windows at
    standard lookback periods: 1d, 5d (1w), 10d (2w), 20d (1mo), 60d (1q),
    180d (6mo), 360d (1yr).

    Parameters
    ----------
    horizon : int
        Lookback in trading days. Default: 360.
    grain : int
        Days per bin. Default: 1.

    Returns
    -------
    SamplingPlan

    Examples
    --------
    >>> from signalforge.domains import equities
    >>> plan = equities.sampling_plan_daily()
    >>> plan.prime_basis
    {2: 3, 3: 2, 5: 1}
    """
    cbin = smallest_divisor_gte(horizon, grain)
    valid = set(lattice_members(horizon, cbin))

    anchors = {_ONE_BAR, _ONE_WEEK, _TWO_WEEKS, _ONE_MONTH, _QUARTER, _HALF_YEAR, horizon}
    fine_cutoff = _ONE_MONTH

    selected = sorted(
        w for w in valid
        if w in anchors or w <= fine_cutoff or w == horizon
    )

    if not selected:
        selected = sorted(valid)

    return SamplingPlan(horizon, grain, windows=selected)


def ingest(path: str) -> list:
    """
    Load a preprocessed equity CSV into CanonicalRecords.

    Expected columns: timestamp, ticker, metric, value
      - timestamp : ISO datetime string, tz-aware preferred
      - ticker    : equity symbol (e.g. "GME")
      - metric    : one of Open, High, Low, Close, Volume
      - value     : float

    primary_order is unix epoch seconds for time-ordered data, or
    sequential bar index for sequence-ordered data (set seq_order accordingly).

    Parameters
    ----------
    path : str
        Path to CSV file produced by examples/yfinance_to_csv.py.
    """
    import pandas as pd
    from ..pipeline.canonical import CanonicalRecord, OrderType

    df = pd.read_csv(path)

    ts = pd.to_datetime(df["timestamp"], utc=True)
    raw = ts.astype("int64")
    dtype = str(ts.dtype)
    if "[s" in dtype and "[us" not in dtype and "[ns" not in dtype:
        epochs = raw
    elif "[ms" in dtype:
        epochs = raw // 1_000
    elif "[us" in dtype:
        epochs = raw // 1_000_000
    else:
        epochs = raw // 1_000_000_000

    records = [
        CanonicalRecord(
            primary_order=int(epoch),
            order_type=OrderType.TIME,
            channel=str(row.metric),
            metric="value",
            value=float(row.value),
            keys={"ticker": str(row.ticker)},
            time_order=int(epoch),
        )
        for epoch, row in zip(epochs, df.itertuples(index=False))
        if str(row.value).replace(".", "").replace("-", "").isdigit()
    ]
    records.sort(key=lambda r: r.primary_order)
    return records
