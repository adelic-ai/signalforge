"""
signalforge.signal._measure

measure_signal: LatticeSignal + SamplingPlan → Surface

Takes a signal (indexed by integers, carrying values) and produces a
Surface by binning and windowed aggregation. This is the signal-centric
entry point — no CanonicalRecords or BinnedRecords needed.

For the full pipeline with profiles and aggregation functions, use
signalforge.pipeline.surface.measure(). This function provides the
direct signal → surface path.
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

    Parameters
    ----------
    signal : LatticeSignal
        Any signal — RealSignal, ComplexSignal, or Surface row.
    plan : SamplingPlan
        Defines windows, hops, and cbin.
    agg : str
        Aggregation within each bin: "mean" (default), "sum", "max", "min".

    Returns
    -------
    Surface
    """
    idx = signal.index
    vals = signal.values
    cbin = plan.cbin

    is_complex = np.issubdtype(vals.dtype, np.complexfloating)

    # --- Step 1: Bin ---
    bin_indices = idx // cbin
    min_bin = int(bin_indices.min())
    max_bin = int(bin_indices.max())
    n_time = max_bin - min_bin + 1

    if is_complex:
        dense = np.full(n_time, np.nan + 0j, dtype=np.complex128)
    else:
        dense = np.full(n_time, np.nan, dtype=np.float64)
    dense_count = np.zeros(n_time, dtype=np.intp)

    # Accumulate into bins
    _AGG = {
        "mean": "mean",
        "sum": "sum",
        "max": "max",
        "min": "min",
    }
    if agg not in _AGG:
        raise ValueError(f"Unknown agg {agg!r}. Use: {sorted(_AGG)}")

    # Group values by bin
    from collections import defaultdict
    bins: Dict[int, list] = defaultdict(list)
    for i in range(len(idx)):
        b = int(bin_indices[i]) - min_bin
        bins[b].append(vals[i])

    for b, bvals in bins.items():
        arr = np.array(bvals)
        dense_count[b] = len(bvals)
        if agg == "mean":
            dense[b] = np.mean(arr)
        elif agg == "sum":
            dense[b] = np.sum(arr)
        elif agg == "max":
            if is_complex:
                dense[b] = arr[np.argmax(np.abs(arr))]
            else:
                dense[b] = np.max(arr)
        elif agg == "min":
            if is_complex:
                dense[b] = arr[np.argmin(np.abs(arr))]
            else:
                dense[b] = np.min(arr)

    # --- Step 2: Windowed measurement ---
    time_axis = np.arange(min_bin, max_bin + 1, dtype=np.int64)
    n_scales = len(plan.windows)

    if is_complex:
        values_arr = np.full((n_scales, n_time), np.nan + 0j, dtype=np.complex128)
    else:
        values_arr = np.full((n_scales, n_time), np.nan, dtype=np.float64)
    n_events_arr = np.zeros((n_scales, n_time), dtype=np.intp)
    coverage_arr = np.zeros((n_scales, n_time), dtype=np.float64)

    # Prefix sums for fast windowed aggregation
    valid_mask = np.isfinite(dense.real if is_complex else dense).astype(np.float64)
    cs_val = np.concatenate([[0 + 0j if is_complex else 0.0],
                              np.where(valid_mask.astype(bool), dense, 0).cumsum()])
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
        elif agg == "sum":
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
