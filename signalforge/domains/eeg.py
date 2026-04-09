"""
signalforge.domains.eeg

SamplingPlan factory for EEG (electroencephalogram) data.

Standard clinical EEG sampling rates are 256 Hz, 512 Hz, and 1024 Hz.
This module defaults to 256 Hz, which is the CHB-MIT Scalp EEG Database cadence.

Using sample index as primary_order (not wall clock), grain is expressed
in samples. One second of 256 Hz data = 256 samples.

Horizon: one hour in samples = 256 × 3600 = 921600
    921600 = 2^12 × 3^2 × 5^2

This factorization yields a rich lattice of clinically meaningful windows:

    Samples    Seconds    Clinical relevance
    -------    -------    ------------------
    256        1 s        EEG epoch, ictal onset detection
    512        2 s        Short-term spectral analysis
    1024       4 s        Standard clinical epoch
    2048       8 s        Extended epoch
    4096       16 s       Pre-ictal window
    7680       30 s       Seizure classification minimum
    15360      60 s       Post-ictal monitoring
    76800      300 s      5-minute baseline
    460800     1800 s     30-minute session
    921600     3600 s     Full-hour session

References
----------
CHB-MIT Scalp EEG Database:
    https://physionet.org/content/chbmit/1.0.0/
"""

from __future__ import annotations

import binjamin as bj
from ..lattice.sampling import SamplingPlan

# 256 Hz — CHB-MIT and standard clinical EEG
_SFREQ = 256

# Horizon: one hour at 256 Hz
_ONE_HOUR_SAMPLES = _SFREQ * 3600       # 921600

# Grain: one second at 256 Hz
_ONE_SECOND_SAMPLES = _SFREQ            # 256

# Clinically meaningful windows in samples
_EPOCH_1S   = _SFREQ * 1               # 256
_EPOCH_4S   = _SFREQ * 4               # 1024
_EPOCH_16S  = _SFREQ * 16              # 4096
_EPOCH_30S  = _SFREQ * 30              # 7680
_EPOCH_1MIN = _SFREQ * 60              # 15360
_EPOCH_5MIN = _SFREQ * 300             # 76800


def sampling_plan(
    horizon: int = _ONE_HOUR_SAMPLES,
    grain: int = _ONE_SECOND_SAMPLES,
    sfreq: int = _SFREQ,
) -> SamplingPlan:
    """
    Build a SamplingPlan suited to EEG data at 256 Hz.

    Primary ordering is sample index (not wall clock). One bin = one second
    of samples (grain=256). Windows are selected at clinically standard epoch
    lengths plus intermediate lattice members for full multiscale coverage.

    Parameters
    ----------
    horizon : int
        Total samples in the coordinate space. Default: 921600 (1 hour at 256 Hz).
    grain : int
        Samples per bin. Default: 256 (1 second).
    sfreq : int
        Sampling frequency in Hz. Used only to compute default grain/horizon.
        Override horizon/grain directly for non-standard rates.

    Returns
    -------
    SamplingPlan

    Examples
    --------
    >>> from signalforge.domains import eeg
    >>> plan = eeg.sampling_plan()
    >>> plan.cbin
    256
    >>> plan.windows[:5]
    (256, 512, 1024, 2048, 4096)
    """
    cbin = bj.smallest_divisor_gte(horizon, grain)
    valid = set(bj.lattice_members(horizon, cbin))

    # Anchor at clinical epoch lengths, include all sub-1-minute lattice members.
    anchors = {_EPOCH_1S, _EPOCH_4S, _EPOCH_16S, _EPOCH_30S, _EPOCH_1MIN, _EPOCH_5MIN, horizon}
    fine_cutoff = _EPOCH_1MIN  # include every lattice member up to 1 minute

    selected = sorted(
        w for w in valid
        if w in anchors or w <= fine_cutoff or w == horizon
    )

    if not selected:
        selected = sorted(valid)

    return SamplingPlan(horizon, grain, windows=selected)


def ingest(path: str, sfreq: int = _SFREQ) -> list:
    """
    Load a preprocessed EEG CSV into Records.

    Expected columns: t_sec, eeg_rms
    primary_order is sample index (round(t_sec * sfreq)).

    Parameters
    ----------
    path : str
        Path to CSV file.
    sfreq : int
        Sampling frequency in Hz. Default: 256.
    """
    import numpy as np
    import pandas as pd
    from ..signal import Schema, Axis, AxisType, Record

    df = pd.read_csv(path)
    t_secs = df["t_sec"].to_numpy(dtype=np.float64)
    rms_values = df["eeg_rms"].to_numpy(dtype=np.float64)
    sample_indices = np.round(t_secs * sfreq).astype(np.int64)

    schema = Schema(
        name="eeg",
        axes=[
            Axis("sample", AxisType.ORDERED),
            Axis("rms", AxisType.NUMERIC),
        ],
        primary_order="sample",
        value_axis="rms",
    )

    return [
        Record(schema, {"sample": int(si), "rms": float(v)})
        for si, v in zip(sample_indices, rms_values)
    ]


def sampling_plan_for_sfreq(sfreq: int, duration_s: int = 3600) -> SamplingPlan:
    """
    Build a SamplingPlan for an arbitrary EEG sampling frequency.

    Parameters
    ----------
    sfreq : int
        Sampling frequency in Hz (e.g. 256, 512, 1024).
    duration_s : int
        Session duration in seconds. Default: 3600 (one hour).

    Returns
    -------
    SamplingPlan
    """
    horizon = sfreq * duration_s
    grain = sfreq  # one second per bin
    return sampling_plan(horizon=horizon, grain=grain, sfreq=sfreq)
