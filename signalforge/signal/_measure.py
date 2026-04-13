"""
signalforge.signal._measure

measure_signal: LatticeSignal + SamplingPlan → Surface

Takes a signal (indexed by integers, carrying values) and produces a
Surface by binning and windowed aggregation. This is the signal-centric
entry point — no CanonicalRecords or BinnedRecords needed.

Supports multiple aggregations per surface. Prefix-sum aggregations
(mean, sum, count, std, var, geometric_mean) are O(1) per window.
Non-prefix aggregations (median, percentiles, entropy, spectral)
fall back to per-window computation.

Vectorized binning via np.add.at.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..lattice.sampling import SamplingPlan
from ._signal import LatticeSignal
from ._surface import Surface


# Prefix-sum aggregations: computed from cumulative sums, O(1) per window
def _safe_div(a, b):
    out = np.full_like(a, np.nan)
    np.divide(a, b, out=out, where=b > 0)
    return out


_CUMSUM_AGGS = {
    "mean":           lambda s, sq, cnt, lg: _safe_div(s, cnt),
    "sum":            lambda s, sq, cnt, lg: np.where(cnt > 0, s, np.nan),
    "count":          lambda s, sq, cnt, lg: np.where(cnt > 0, cnt, np.nan),
    "std":            lambda s, sq, cnt, lg: np.where(
                          cnt > 1,
                          np.sqrt(np.maximum(_safe_div(sq, cnt) - _safe_div(s, cnt) ** 2, 0.0)),
                          np.nan),
    "var":            lambda s, sq, cnt, lg: np.where(
                          cnt > 1,
                          np.maximum(_safe_div(sq, cnt) - _safe_div(s, cnt) ** 2, 0.0),
                          np.nan),
    "geometric_mean": lambda s, sq, cnt, lg: np.where(
                          cnt > 0, np.exp(_safe_div(lg, cnt)), np.nan),
}

# Non-prefix aggregations: need actual window values, per-window loop
def _window_agg(dense, starts, ends, agg_name):
    """Compute a non-prefix aggregation per window position."""
    from ..pipeline.aggregation import get_aggregation
    fn = get_aggregation(agg_name)
    result = np.full(len(starts), np.nan)
    for i, (s, e) in enumerate(zip(starts, ends)):
        window_vals = dense[int(s):int(e)]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) > 0:
            result[i] = fn(valid)
    return result


def _count_entropy(dense_count, starts, ends):
    """Shannon entropy of event count distribution within each window.

    Measures temporal spread: low = events clustered in a few sub-bins,
    high = events spread evenly across the window. Returns bits (log2).

    This is the information-theoretic primitive. The existing 'entropy'
    aggregation (in pipeline/aggregation.py) histograms values; this one
    measures the count distribution directly on the lattice bins.
    """
    result = np.full(len(starts), np.nan)
    for i, (s, e) in enumerate(zip(starts, ends)):
        counts = dense_count[int(s):int(e)].astype(np.float64)
        total = counts.sum()
        if total <= 0:
            continue
        p = counts[counts > 0] / total
        result[i] = float(-np.sum(p * np.log2(p)))
    return result


def measure_signal(
    signal: LatticeSignal,
    plan: SamplingPlan,
    agg: Union[str, List[str]] = "mean",
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
    agg : str or list of str
        Aggregation(s) per window. Single string or list.
        Prefix-sum (fast): "mean", "sum", "count", "std", "var", "geometric_mean"
        Per-window (slower): "median", "p25", "p75", "p90", "p95", "p99",
            "spectral_energy", "dominant_freq", "entropy", and any registered aggregation.

    Returns
    -------
    Surface
        With one data array per aggregation name.
    """
    # Normalize agg to list
    if isinstance(agg, str):
        agg_names = [agg]
    else:
        agg_names = list(agg)

    idx = signal.index
    vals = signal.values
    cbin = plan.cbin

    is_complex = np.issubdtype(vals.dtype, np.complexfloating)
    dtype = np.complex128 if is_complex else np.float64

    # --- Step 1: Bin (vectorized) ---
    bin_indices = idx // cbin
    min_bin = int(bin_indices.min())
    max_bin = int(bin_indices.max())
    n_time = max_bin - min_bin + 1

    rel_bins = (bin_indices - min_bin).astype(np.intp)

    dense_sum = np.zeros(n_time, dtype=dtype)
    dense_count = np.zeros(n_time, dtype=np.intp)
    np.add.at(dense_sum, rel_bins, vals)
    np.add.at(dense_count, rel_bins, 1)

    # Dense binned values (mean per bin)
    with np.errstate(divide='ignore', invalid='ignore'):
        dense = np.where(dense_count > 0, dense_sum / dense_count, np.nan)

    # --- Step 2: Prefix sums ---
    time_axis = np.arange(min_bin, max_bin + 1, dtype=np.int64)
    n_scales = len(plan.windows)

    valid_mask = (dense_count > 0).astype(np.float64)
    dense_clean = np.where(valid_mask.astype(bool), dense, 0)

    zero = np.array([0 + 0j]) if is_complex else np.array([0.0])
    cs_val = np.concatenate([zero, dense_clean.cumsum()])
    cs_sq = np.concatenate([zero, (dense_clean ** 2).cumsum()])
    cs_cnt = np.concatenate([[0.0], valid_mask.cumsum()])
    cs_ne = np.concatenate([[0], dense_count.cumsum()])

    # Log for geometric mean (only for positive real values)
    if not is_complex:
        pos_mask = valid_mask.astype(bool) & (dense > 0)
        cs_log = np.concatenate([[0.0], np.where(pos_mask, np.log(np.maximum(dense, 1e-300)), 0.0).cumsum()])
    else:
        cs_log = None

    # --- Step 3: Compute aggregations ---
    data: Dict[str, np.ndarray] = {}
    n_events_arr = np.zeros((n_scales, n_time), dtype=np.intp)
    coverage_arr = np.zeros((n_scales, n_time), dtype=np.float64)

    # Determine which aggs are prefix-sum, count-entropy, or per-window
    cumsum_names = [a for a in agg_names if a in _CUMSUM_AGGS]
    has_count_entropy = "count_entropy" in agg_names
    window_names = [a for a in agg_names
                    if a not in _CUMSUM_AGGS and a != "count_entropy"]

    # Initialize arrays
    for name in agg_names:
        if is_complex and name in _CUMSUM_AGGS:
            data[name] = np.full((n_scales, n_time), np.nan + 0j, dtype=np.complex128)
        else:
            data[name] = np.full((n_scales, n_time), np.nan, dtype=np.float64)

    for scale_idx, (window, hop) in enumerate(zip(plan.windows, plan.hops)):
        w = window // cbin
        h = hop // cbin

        starts = np.arange(0, n_time, h)
        ends = np.minimum(starts + w, n_time)

        # Common: count and coverage
        cnt_w = cs_cnt[ends] - cs_cnt[starts]
        coverage_arr[scale_idx, starts] = cnt_w / w
        n_events_arr[scale_idx, starts] = (cs_ne[ends] - cs_ne[starts]).astype(np.intp)

        # Prefix-sum differences
        s_val = cs_val[ends] - cs_val[starts]
        s_sq = cs_sq[ends] - cs_sq[starts]
        s_log = (cs_log[ends] - cs_log[starts]) if cs_log is not None else None

        # Compute cumsum-based aggregations
        for name in cumsum_names:
            fn = _CUMSUM_AGGS[name]
            data[name][scale_idx, starts] = fn(s_val, s_sq, cnt_w, s_log)

        # Count entropy: Shannon entropy of event count distribution per window
        if has_count_entropy:
            data["count_entropy"][scale_idx, starts] = _count_entropy(
                dense_count, starts, ends)

        # Compute per-window aggregations
        for name in window_names:
            data[name][scale_idx, starts] = _window_agg(dense, starts, ends, name)

    scale_axis = tuple(w // cbin for w in plan.windows)
    profile = ",".join(agg_names) if len(agg_names) > 1 else agg_names[0]

    return Surface(
        time_axis=time_axis,
        scale_axis=scale_axis,
        data=data,
        channel=signal.channel,
        plan=plan,
        keys=signal.keys,
        metric=agg_names[0],
        profile=profile,
        coordinates=plan.coordinates,
        n_events=n_events_arr,
        coverage=coverage_arr,
    )
