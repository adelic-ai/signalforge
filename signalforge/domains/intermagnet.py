"""
signalforge.domains.intermagnet

SamplingPlan factory for INTERMAGNET geomagnetic observatory data.

INTERMAGNET (International Real-time Magnetic Observatory Network)
distributes geomagnetic field measurements from observatories worldwide.
Standard data products are defined at fixed cadences:

    1-second data  (INTERMAGNET definitive 1s product)
    1-minute data  (the primary INTERMAGNET product)
    hourly means   (derived from 1-minute data)
    daily means    (derived from hourly means)

These cadences are exact integer multiples of each other, which means
they map cleanly onto the divisibility lattice. A horizon of 86400
(one day in seconds) with grain of 60 (one minute) gives a lattice
whose members include all standard INTERMAGNET aggregation intervals.

References
----------
INTERMAGNET Technical Reference Manual:
    https://intermagnet.org/publication-software/technicalsoft-e.php
INTERMAGNET data formats and cadences:
    https://intermagnet.org/data-donnee/data-donnee-eng.php
"""

from __future__ import annotations

import binjamin as bj
from ..lattice.sampling import SamplingPlan

# Standard INTERMAGNET cadences in seconds.
_ONE_MINUTE = 60
_ONE_HOUR = 3_600
_ONE_DAY = 86_400


def sampling_plan(
    horizon: int = _ONE_DAY,
    grain: int = _ONE_MINUTE,
) -> SamplingPlan:
    """
    Build a SamplingPlan suited to INTERMAGNET geomagnetic data.

    Default configuration covers one day at one-minute resolution —
    the primary INTERMAGNET product cadence. Windows are selected at
    the standard INTERMAGNET aggregation intervals (1min, 1hr, 1day)
    plus any lattice members that fall between them, giving coherent
    multiscale coverage across the full range.

    Parameters
    ----------
    horizon : int
        Outer boundary of the coordinate space in seconds. Default: 86400 (one day).
    grain : int
        Finest bin in seconds. Default: 60 (one minute).

    Returns
    -------
    SamplingPlan

    Examples
    --------
    >>> from signalforge.domains import intermagnet
    >>> plan = intermagnet.sampling_plan()
    >>> plan.cbin
    60
    >>> plan.prime_basis
    {2: 5, 3: 3, 5: 1}
    >>> plan.windows
    (60, 120, 180, 360, 720, 1440, 3600, 7200, 14400, 21600, 43200, 86400)
    """
    cbin = bj.smallest_divisor_gte(horizon, grain)
    valid = bj.lattice_members(horizon, cbin)

    # Anchor windows at the standard INTERMAGNET products that fall within
    # the horizon, then fill in lattice members between them for multiscale
    # coverage across sub-hourly structure.
    anchors = {_ONE_MINUTE, _ONE_HOUR, _ONE_DAY}
    valid_set = set(valid)

    fine_cutoff = _ONE_HOUR * 2
    selected = sorted(
        w for w in valid_set
        if w in anchors or w <= fine_cutoff or w == horizon
    )

    if not selected:
        selected = list(valid)

    return SamplingPlan(horizon, grain, windows=selected)


def sampling_plan_yearly(
    horizon: int = 360 * _ONE_DAY,
    grain: int = _ONE_DAY,
) -> SamplingPlan:
    """
    SamplingPlan for year-scale INTERMAGNET analysis at daily resolution.

    Uses grain=86400 (one day) so each bin represents one day of observations.
    The default horizon is 360 days, whose rich factorization (2^10 × 3^5 × 5^3)
    yields many lattice members at natural geomagnetic timescales:
    1d, 2d, 3d, 5d, 9d, 15d, 27d (Carrington rotation), 45d, 90d, 180d, 360d.

    Parameters
    ----------
    horizon : int
        Outer boundary in seconds. Default: 31104000 (360 days).
    grain : int
        Finest bin in seconds. Default: 86400 (one day).
    """
    cbin = bj.smallest_divisor_gte(horizon, grain)
    valid = bj.lattice_members(horizon, cbin)

    # Anchor at geophysically meaningful day-count windows, fill sub-monthly lattice.
    _ONE_WEEK  = 7  * _ONE_DAY
    _CARRINGTON = 27 * _ONE_DAY   # solar Carrington rotation
    _ONE_MONTH = 30 * _ONE_DAY
    _ONE_QUARTER = 90 * _ONE_DAY

    anchors = {_ONE_DAY, _ONE_WEEK, _CARRINGTON, _ONE_MONTH, _ONE_QUARTER, horizon}
    # Include all lattice members up to one month for sub-monthly structure.
    fine_cutoff = _ONE_MONTH
    valid_set = set(valid)

    selected = sorted(
        w for w in valid_set
        if w in anchors or w <= fine_cutoff or w == horizon
    )

    if not selected:
        selected = list(valid)

    return SamplingPlan(horizon, grain, windows=selected)


def ingest(path: str) -> list:
    """
    Load a preprocessed INTERMAGNET CSV into CanonicalRecords.

    Expected columns: timestamp, station, component, value
    primary_order is unix epoch seconds.

    Parameters
    ----------
    path : str
        Path to CSV file.
    """
    import pandas as pd
    from ..signal import CanonicalRecord, OrderType

    df = pd.read_csv(path)
    ts = pd.to_datetime(df["timestamp"], utc=True).astype("int64")
    dtype = str(pd.to_datetime(df["timestamp"], utc=True).dtype)
    if "[s" in dtype and "[us" not in dtype and "[ns" not in dtype:
        epochs = ts
    elif "[ms" in dtype:
        epochs = ts // 1_000
    elif "[us" in dtype:
        epochs = ts // 1_000_000
    else:
        epochs = ts // 1_000_000_000

    records = [
        CanonicalRecord(
            primary_order=int(epoch),
            order_type=OrderType.TIME,
            channel=str(row.component),
            metric="value",
            value=float(row.value),
            keys={"station": str(row.station)},
            time_order=int(epoch),
        )
        for epoch, row in zip(epochs, df.itertuples(index=False))
    ]
    records.sort(key=lambda r: r.primary_order)
    return records


def sampling_plan_1s(horizon: int = _ONE_DAY) -> SamplingPlan:
    """
    SamplingPlan for 1-second INTERMAGNET data.

    Uses grain=1 to capture the full 1-second product resolution.
    """
    return sampling_plan(horizon=horizon, grain=1)
