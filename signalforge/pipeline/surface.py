"""
signalforge.pipeline.surface

Surface: Stage 3 (Measure) output.

Takes BinnedData and a SamplingPlan and produces a structured 2D array —
time on one axis, scale on the other. Each cell holds an aggregated value
derived from the bins within that window at that position.

Because all windows and hops are multiples of cbin and derived from the same
SamplingPlan, the grid is always the same shape for the same plan. Same plan
→ same grid → surfaces from different sources are directly comparable.
This structural invariance is what makes ML on surfaces meaningful.

Aggregation profiles
--------------------
A profile is a named set of aggregation functions suited to a data type.
Built-in profiles: event, continuous, ratio, anomaly, sparse.
Custom profiles: register_profile("name", ["agg1", "agg2", ...])

Surface stores raw aggregated values. Normalization is a rendering concern,
never baked into the artifact.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ..lattice.sampling import SamplingPlan
from .aggregation import AggFunc, get_aggregation
from .binned import BinnedRecord

# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

_PROFILE_REGISTRY: Dict[str, Tuple[str, ...]] = {}


def register_profile(name: str, agg_names: Iterable[str]) -> None:
    """
    Register a named aggregation profile.

    A profile is an ordered list of aggregation function names from the
    aggregation registry. All names must resolve at registration time.

    Parameters
    ----------
    name : str
        Profile name. Overwrites any existing registration under this name.
    agg_names : iterable of str
        Aggregation function names. Each must be registered in the
        aggregation registry.

    Example
    -------
    >>> register_profile("my_profile", ["mean", "max", "p95"])
    """
    resolved = tuple(agg_names)
    for agg in resolved:
        get_aggregation(agg)  # Validate at registration time.
    _PROFILE_REGISTRY[name] = resolved


def get_profile(name: str) -> Tuple[str, ...]:
    """Return the aggregation names registered under a profile name."""
    if name not in _PROFILE_REGISTRY:
        raise KeyError(
            f"Unknown profile {name!r}. "
            f"Registered: {sorted(_PROFILE_REGISTRY)}"
        )
    return _PROFILE_REGISTRY[name]


def profile_names() -> Tuple[str, ...]:
    """Return all registered profile names in sorted order."""
    return tuple(sorted(_PROFILE_REGISTRY))


# Built-in profiles.
register_profile("event",      ["count", "max", "ewma", "std"])
register_profile("continuous", ["mean", "geometric_mean", "median", "std"])
register_profile("ratio",      ["geometric_mean", "median", "p25", "p75", "p90"])
register_profile("anomaly",    ["mean", "max", "p95", "p99"])
register_profile("sparse",     ["count", "max"])
# gap_mean and gap_max require gap_before from BinnedData — forthcoming.


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


class Surface:
    """
    A 2D measurement grid: time × scale.

    Rows are scales (one per window in the SamplingPlan).
    Columns are time positions (window start positions in bin_index units).
    Each cell holds aggregated values from the bins within that window.

    Immutable after construction.

    Attributes
    ----------
    channel : str
    keys : dict
    metric : str
    profile : str
    time_axis : tuple[int, ...]
        Window start positions in bin_index units.
    scale_axis : tuple[int, ...]
        Window sizes in cbin units. Row i covers scale_axis[i] bins.
    values : dict[str, np.ndarray]
        One (n_scales × n_time) float64 array per aggregation in the profile.
        NaN where no bins fell within the window.
    n_events : np.ndarray
        (n_scales × n_time) int array. Total raw events per cell.
    coverage : np.ndarray
        (n_scales × n_time) float array. Fraction of bins occupied. [0, 1].
    coordinates : tuple[dict, ...]
        Prime exponent vector per scale row, from the SamplingPlan.
    sampling_plan_id : str
        Identifies the geometry used to construct this surface.
    """

    __slots__ = (
        "channel",
        "keys",
        "metric",
        "profile",
        "time_axis",
        "scale_axis",
        "values",
        "n_events",
        "coverage",
        "coordinates",
        "sampling_plan_id",
    )

    def __init__(
        self,
        channel: str,
        keys: dict,
        metric: str,
        profile: str,
        time_axis: Tuple[int, ...],
        scale_axis: Tuple[int, ...],
        values: Dict[str, np.ndarray],
        n_events: np.ndarray,
        coverage: np.ndarray,
        coordinates: Tuple[dict, ...],
        sampling_plan_id: str,
    ) -> None:
        n_scales = len(scale_axis)
        n_time = len(time_axis)
        expected = (n_scales, n_time)

        for agg_name, arr in values.items():
            if arr.shape != expected:
                raise ValueError(
                    f"values[{agg_name!r}].shape {arr.shape} != {expected}"
                )
        if n_events.shape != expected:
            raise ValueError(f"n_events.shape {n_events.shape} != {expected}")
        if coverage.shape != expected:
            raise ValueError(f"coverage.shape {coverage.shape} != {expected}")
        if not np.all((coverage >= 0.0) & (coverage <= 1.0)):
            raise ValueError("coverage values must be in [0.0, 1.0]")

        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "time_axis", time_axis)
        object.__setattr__(self, "scale_axis", scale_axis)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "n_events", n_events)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "sampling_plan_id", sampling_plan_id)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Surface is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Surface is immutable")

    @property
    def shape(self) -> Tuple[int, int]:
        """(n_scales, n_time)"""
        return (len(self.scale_axis), len(self.time_axis))

    def __repr__(self) -> str:
        return (
            f"Surface("
            f"channel={self.channel!r}, "
            f"metric={self.metric!r}, "
            f"profile={self.profile!r}, "
            f"shape={self.shape}, "
            f"keys={self.keys})"
        )


# ---------------------------------------------------------------------------
# measure()
# ---------------------------------------------------------------------------


def _canonical_keys(keys: dict) -> tuple:
    result = []
    for k in sorted(keys):
        v = keys[k]
        result.append((k, tuple(sorted(v)) if isinstance(v, list) else v))
    return tuple(result)


def measure(
    binned_records: List[BinnedRecord],
    plan: SamplingPlan,
    profile: Union[str, List[str]] = "continuous",
) -> List[Surface]:
    """
    Measure a BinnedData sequence into Surfaces using a SamplingPlan.

    One Surface is produced per distinct (channel, keys, metric) combination
    present in the binned records.

    Parameters
    ----------
    binned_records : list of BinnedRecord
        Output of Stage 2. Order within the list does not matter.
    plan : SamplingPlan
        Defines windows, hops, and cbin.
    profile : str or list of str
        Aggregation profile name, or explicit list of aggregation function names.
        Built-in profiles: "event", "continuous", "ratio", "anomaly", "sparse".

    Returns
    -------
    list of Surface
        One surface per (channel, keys, metric) group.

    Notes
    -----
    - Cells where no bins exist within the window are NaN in values and 0 in
      n_events / coverage. They are not an error — gaps are signal.
    - When hops differ per window, time positions that don't align to a given
      window's hop are NaN for that scale row. The time_axis is always the
      full bin range at cbin resolution.
    """
    if not binned_records:
        return []

    # Resolve profile.
    if isinstance(profile, str):
        agg_names = get_profile(profile)
        profile_name = profile
    else:
        agg_names = tuple(profile)
        profile_name = "custom"

    agg_fns: List[Tuple[str, AggFunc]] = [
        (name, get_aggregation(name)) for name in agg_names
    ]

    # Group records by (channel, keys_canonical, metric) → {bin_index: BinnedRecord}.
    groups: Dict[tuple, Dict[int, BinnedRecord]] = defaultdict(dict)
    for rec in binned_records:
        keys_canon = _canonical_keys(rec.keys)
        group_key = (rec.channel, keys_canon, rec.metric)
        groups[group_key][rec.bin_index] = rec

    # Determine the data span across all groups.
    all_bins: set[int] = set()
    for bin_map in groups.values():
        all_bins.update(bin_map.keys())

    min_bin = min(all_bins)
    max_bin = max(all_bins)

    # time_axis: all bin positions from min to max at cbin resolution.
    # This is the common column grid shared by all scale rows.
    time_axis = tuple(range(min_bin, max_bin + 1))
    n_time = len(time_axis)
    n_scales = len(plan.windows)

    sampling_plan_id = repr(plan)

    surfaces: List[Surface] = []

    for group_key, bin_map in groups.items():
        channel, keys_canon, metric = group_key
        keys_dict = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in keys_canon
        }

        values_arrays: Dict[str, np.ndarray] = {
            name: np.full((n_scales, n_time), np.nan, dtype=np.float64)
            for name, _ in agg_fns
        }
        n_events_arr = np.zeros((n_scales, n_time), dtype=np.intp)
        coverage_arr = np.zeros((n_scales, n_time), dtype=np.float64)

        # Build dense arrays for vectorized window queries.
        dense = np.full(n_time, np.nan, dtype=np.float64)
        dense_nevents = np.zeros(n_time, dtype=np.intp)
        for b, rec in bin_map.items():
            t = b - min_bin
            if 0 <= t < n_time:
                dense[t] = rec.value
                dense_nevents[t] = rec.n_events

        # Precompute prefix sums (length n_time+1; index 0 is always 0).
        valid_mask = (~np.isnan(dense)).astype(np.float64)
        cs_val   = np.concatenate([[0.0], np.where(valid_mask, dense, 0.0).cumsum()])
        cs_sq    = np.concatenate([[0.0], np.where(valid_mask, dense**2, 0.0).cumsum()])
        cs_cnt   = np.concatenate([[0.0], valid_mask.cumsum()])
        cs_log   = np.concatenate([[0.0], np.where(valid_mask * (dense > 0), np.log(np.maximum(dense, 1e-300)), 0.0).cumsum()])
        cs_ne    = np.concatenate([[0],   dense_nevents.cumsum()])

        # Cumsum-vectorized aggregations: result = f(prefix[end] - prefix[start]).
        # Use np.divide with out= to avoid divide-by-zero RuntimeWarning when cnt=0.
        _nan_like = np.full(n_time, np.nan, dtype=np.float64)

        def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            out = np.full_like(a, np.nan)
            np.divide(a, b, out=out, where=b > 0)
            return out

        _CUMSUM_AGG = {
            "mean":           lambda s, sq, cnt, lg: _safe_div(s, cnt),
            "sum":            lambda s, sq, cnt, lg: np.where(cnt > 0, s, np.nan),
            "count":          lambda s, sq, cnt, lg: np.where(cnt > 0, cnt, np.nan),
            "std":            lambda s, sq, cnt, lg: np.where(
                cnt > 1,
                np.sqrt(np.maximum(_safe_div(sq, cnt) - _safe_div(s, cnt) ** 2, 0.0)),
                np.nan,
            ),
            "var":            lambda s, sq, cnt, lg: np.where(
                cnt > 1,
                np.maximum(_safe_div(sq, cnt) - _safe_div(s, cnt) ** 2, 0.0),
                np.nan,
            ),
            "geometric_mean": lambda s, sq, cnt, lg: np.where(
                cnt > 0, np.exp(_safe_div(lg, cnt)), np.nan
            ),
        }

        for scale_idx, (window, hop) in enumerate(zip(plan.windows, plan.hops)):
            w = window // plan.cbin
            h = hop // plan.cbin

            # Aligned positions relative to min_bin.
            starts = np.arange(0, n_time, h)
            ends   = np.minimum(starts + w, n_time)

            # Window counts and coverage.
            cnt_w = cs_cnt[ends] - cs_cnt[starts]
            coverage_arr[scale_idx, starts] = cnt_w / w
            n_events_arr[scale_idx, starts] = (cs_ne[ends] - cs_ne[starts]).astype(np.intp)

            # Prefix-sum differences for this window.
            s_val = cs_val[ends] - cs_val[starts]
            s_sq  = cs_sq[ends]  - cs_sq[starts]
            s_log = cs_log[ends] - cs_log[starts]

            for name, fn in agg_fns:
                arr = values_arrays[name]
                if name in _CUMSUM_AGG:
                    arr[scale_idx, starts] = _CUMSUM_AGG[name](s_val, s_sq, cnt_w, s_log)
                else:
                    # Fallback: per-position Python loop over dense slice (no dict).
                    for pos in starts:
                        t_end = min(int(pos) + w, n_time)
                        window_vals = dense[int(pos):t_end]
                        valid_vals = window_vals[~np.isnan(window_vals)]
                        if len(valid_vals) > 0:
                            arr[scale_idx, int(pos)] = fn(valid_vals)

        scale_axis = tuple(w // plan.cbin for w in plan.windows)

        surfaces.append(Surface(
            channel=channel,
            keys=keys_dict,
            metric=metric,
            profile=profile_name,
            time_axis=time_axis,
            scale_axis=scale_axis,
            values=values_arrays,
            n_events=n_events_arr,
            coverage=coverage_arr,
            coordinates=plan.coordinates,
            sampling_plan_id=sampling_plan_id,
        ))

    return surfaces
