"""
signalforge.signal._measure

measure_signal: LatticeSignal + SamplingPlan → Surface

Takes a signal (indexed by integers, carrying values) and produces a
Surface by binning and windowed aggregation. This is the signal-centric
entry point — no CanonicalRecords or BinnedRecords needed.

Vectorized: binning uses np.add.at, windowed aggregation uses prefix sums.
Both steps are O(N) with no Python loops over data.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..lattice.sampling import SamplingPlan
from ._signal import LatticeSignal
from ._surface import Surface


def measure_signal(
    signal: LatticeSignal,
    plan: SamplingPlan,
    agg: str = "mean",
) -> Surface:
    """Measure a LatticeSignal into a Surface using a SamplingPlan.

    Bins the signal's values by floor(index / cbin), then slides
    windows across the binned data to produce a 2D (scale x time) grid.

    Fully vectorized — no Python loops over data points.

    Parameters
    ----------
    signal : LatticeSignal
        Any signal — RealSignal, ComplexSignal, or Surface row.
    plan : SamplingPlan
        Defines windows, hops, and cbin.
    agg : str
        Aggregation within each bin: "mean" (default), "sum".

    Returns
    -------
    Surface
    """
    idx = signal.index
    vals = signal.values
    cbin = plan.cbin

    is_complex = np.issubdtype(vals.dtype, np.complexfloating)
    dtype = np.complex128 if is_complex else np.float64

    if agg not in ("mean", "sum"):
        raise ValueError(f"Unknown agg {agg!r}. Use: 'mean', 'sum'")

    # --- Step 1: Bin (vectorized) ---
    bin_indices = idx // cbin
    min_bin = int(bin_indices.min())
    max_bin = int(bin_indices.max())
    n_time = max_bin - min_bin + 1

    # Relative bin positions
    rel_bins = (bin_indices - min_bin).astype(np.intp)

    # Accumulate values and counts into dense arrays
    dense_sum = np.zeros(n_time, dtype=dtype)
    dense_count = np.zeros(n_time, dtype=np.intp)

    np.add.at(dense_sum, rel_bins, vals)
    np.add.at(dense_count, rel_bins, 1)

    # Compute dense binned values
    if agg == "mean":
        with np.errstate(divide='ignore', invalid='ignore'):
            dense = np.where(dense_count > 0, dense_sum / dense_count, np.nan)
    else:  # sum
        dense = np.where(dense_count > 0, dense_sum, np.nan)

    # --- Step 2: Windowed measurement (prefix sums) ---
    time_axis = np.arange(min_bin, max_bin + 1, dtype=np.int64)
    n_scales = len(plan.windows)

    values_arr = np.full((n_scales, n_time), np.nan, dtype=dtype)
    n_events_arr = np.zeros((n_scales, n_time), dtype=np.intp)
    coverage_arr = np.zeros((n_scales, n_time), dtype=np.float64)

    # Prefix sums for O(1) windowed aggregation
    valid_mask = (dense_count > 0).astype(np.float64)
    zero = np.array([0 + 0j]) if is_complex else np.array([0.0])
    dense_for_sum = np.where(valid_mask.astype(bool), dense, 0)

    cs_val = np.concatenate([zero, dense_for_sum.cumsum()])
    cs_cnt = np.concatenate([[0.0], valid_mask.cumsum()])
    cs_ne = np.concatenate([[0], dense_count.cumsum()])

    for scale_idx, (window, hop) in enumerate(zip(plan.windows, plan.hops)):
        w = window // cbin
        h = hop // cbin

        starts = np.arange(0, n_time, h)
        ends = np.minimum(starts + w, n_time)

        cnt_w = cs_cnt[ends] - cs_cnt[starts]
        coverage_arr[scale_idx, starts] = cnt_w / w
        n_events_arr[scale_idx, starts] = (cs_ne[ends] - cs_ne[starts]).astype(np.intp)

        s_val = cs_val[ends] - cs_val[starts]

        if agg == "mean":
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.where(cnt_w > 0, s_val / cnt_w, np.nan)
            values_arr[scale_idx, starts] = result
        else:  # sum
            values_arr[scale_idx, starts] = np.where(cnt_w > 0, s_val, np.nan)

    scale_axis = tuple(w // cbin for w in plan.windows)

    return Surface(
        time_axis=time_axis,
        scale_axis=scale_axis,
        data={agg: values_arr},
        channel=signal.channel,
        plan=plan,
        keys=signal.keys,
        metric=agg,
        profile=agg,
        coordinates=plan.coordinates,
        n_events=n_events_arr,
        coverage=coverage_arr,
    )
