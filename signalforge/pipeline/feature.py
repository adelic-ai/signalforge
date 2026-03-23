"""
signalforge.pipeline.feature

FeatureTensor: Stage 4 (Engineer) output.

Takes a Surface and adds derived features — baseline estimates, residuals,
deviation scores, deltas, rates, and cross-scale lattice gradients — all at
the same (n_scales, n_time) shape as the source Surface.

Every derived feature is named "{source}_{type}" so the full feature set is
self-describing. The FeatureTensor carries both the raw Surface values and
all derived features in one place, ready for Stage 5 assembly.

Feature types
-------------
raw             — aggregated values from Surface (carried through unchanged)
baseline_ewma   — exponentially weighted moving average along time axis
baseline_rmed   — rolling median along time axis
residual        — value - baseline_ewma
zscore          — residual / (1.4826 * MAD), robust z-score
delta           — value[t] - value[t-1], NaN at t=0
rate            — delta normalised by coverage (populated bins per window)
gradient        — value at scale i minus value at next coarser scale
                  (lattice cover relation; NaN for the coarsest scale)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..lattice.sampling import SamplingPlan
from .surface import Surface


# ---------------------------------------------------------------------------
# FeatureTensor
# ---------------------------------------------------------------------------


class FeatureTensor:
    """
    Engineered feature representation derived from a single Surface.

    Maintains the (n_scales, n_time) grid from the source Surface.
    Adds derived feature arrays alongside the raw aggregations.

    All arrays in `values` have shape (n_scales, n_time) and dtype float64.
    NaN marks positions where a feature is undefined (edge of time axis,
    missing data, zero coverage).

    Attributes
    ----------
    channel : str
    keys : dict
    metric : str
    profile : str
    time_axis : tuple[int, ...]
    scale_axis : tuple[int, ...]
    coordinates : tuple[dict, ...]
    sampling_plan_id : str
    n_events : np.ndarray        shape (n_scales, n_time), int
    coverage : np.ndarray        shape (n_scales, n_time), float
    values : dict[str, np.ndarray]
        All feature arrays. Keys follow "{source}_{type}" convention.
        Raw arrays are carried through under their original names.
    feature_index : dict[str, str]
        Maps feature name → human-readable description.
    """

    __slots__ = (
        "channel",
        "keys",
        "metric",
        "profile",
        "time_axis",
        "scale_axis",
        "coordinates",
        "sampling_plan_id",
        "n_events",
        "coverage",
        "values",
        "feature_index",
    )

    def __init__(
        self,
        channel: str,
        keys: dict,
        metric: str,
        profile: str,
        time_axis: Tuple[int, ...],
        scale_axis: Tuple[int, ...],
        coordinates: Tuple[dict, ...],
        sampling_plan_id: str,
        n_events: np.ndarray,
        coverage: np.ndarray,
        values: Dict[str, np.ndarray],
        feature_index: Dict[str, str],
    ) -> None:
        n_scales = len(scale_axis)
        n_time = len(time_axis)
        expected = (n_scales, n_time)
        for name, arr in values.items():
            if arr.shape != expected:
                raise ValueError(
                    f"values[{name!r}].shape {arr.shape} != {expected}"
                )

        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "time_axis", time_axis)
        object.__setattr__(self, "scale_axis", scale_axis)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "sampling_plan_id", sampling_plan_id)
        object.__setattr__(self, "n_events", n_events)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "feature_index", feature_index)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("FeatureTensor is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("FeatureTensor is immutable")

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.scale_axis), len(self.time_axis))

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self.values))

    def __repr__(self) -> str:
        return (
            f"FeatureTensor("
            f"channel={self.channel!r}, "
            f"metric={self.metric!r}, "
            f"shape={self.shape}, "
            f"features={len(self.values)})"
        )


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------


def _ewma(arr: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Exponentially weighted moving average along the time axis (axis=1).

    NaN values are skipped — the last non-NaN EWMA value is carried forward.
    """
    # Transpose so time is on rows (pandas operates along axis=0).
    # ignore_na=True skips NaN in weight calculation; ffill carries the last
    # computed EWMA forward at NaN positions, matching the original behaviour.
    df = pd.DataFrame(arr.T)
    result = df.ewm(alpha=alpha, adjust=False, ignore_na=True).mean().ffill()
    return result.to_numpy().T


def _rolling_median(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Rolling median along the time axis (axis=1).

    Uses a lookback window of `window` time steps. NaN values within the
    window are ignored. Result is NaN where no values are available.
    """
    df = pd.DataFrame(arr.T)
    result = df.rolling(window, min_periods=1).median()
    return result.to_numpy().T


def _mad_zscore(residual: np.ndarray) -> np.ndarray:
    """
    Robust z-score using median absolute deviation, per scale row.

    zscore = residual / (1.4826 * MAD)

    The 1.4826 factor makes MAD a consistent estimator of the standard
    deviation for normally distributed data. Per-row MAD is computed over
    all non-NaN time positions.
    """
    result = np.full_like(residual, np.nan)
    for i in range(residual.shape[0]):
        row = residual[i]
        valid = row[~np.isnan(row)]
        if len(valid) < 2:
            continue
        mad = float(np.median(np.abs(valid - np.median(valid))))
        if mad == 0.0:
            continue
        result[i] = row / (1.4826 * mad)
    return result


def _delta(arr: np.ndarray) -> np.ndarray:
    """
    First difference along the time axis. NaN at t=0.
    NaN inputs produce NaN outputs.
    """
    result = np.full_like(arr, np.nan)
    result[:, 1:] = arr[:, 1:] - arr[:, :-1]
    return result


def _lattice_gradient(
    arr: np.ndarray,
    windows: Tuple[int, ...],
) -> np.ndarray:
    """
    For each scale row, value[i] - value[cover(i)].

    cover(i) is the index of the next coarser window that directly covers
    window[i] with no intermediary — the immediate parent in the Hasse diagram.
    The coarsest scale row has no parent: NaN throughout.

    Fine - coarse: positive means fine scale has more signal than coarse.
    """
    result = np.full_like(arr, np.nan)
    n_scales = len(windows)

    # For each scale row find its immediate cover (smallest window > window[i]
    # that divides by window[i] with no intermediary).
    for i in range(n_scales):
        w = windows[i]
        cover_idx = None
        for j in range(i + 1, n_scales):
            if windows[j] % w == 0:
                # Check no intermediary between i and j.
                mediated = any(
                    windows[j] % windows[k] == 0
                    and windows[k] % w == 0
                    and k != i and k != j
                    for k in range(i + 1, j)
                )
                if not mediated:
                    cover_idx = j
                    break
        if cover_idx is not None:
            result[i] = arr[i] - arr[cover_idx]

    return result


# ---------------------------------------------------------------------------
# engineer()
# ---------------------------------------------------------------------------


def engineer(
    surface: Surface,
    plan: SamplingPlan,
    ewma_alpha: float = 0.3,
    rolling_window: int = 5,
) -> FeatureTensor:
    """
    Engineer features from a Surface into a FeatureTensor.

    Computes the full feature set for every aggregation in the source Surface:
    EWMA baseline, rolling median baseline, residuals, robust z-scores,
    deltas, rates, and cross-scale lattice gradients.

    Parameters
    ----------
    surface : Surface
        Source surface from Stage 3.
    plan : SamplingPlan
        The SamplingPlan used to produce the surface.
    ewma_alpha : float
        Smoothing factor for EWMA baseline. Default 0.3.
    rolling_window : int
        Lookback window (in time steps) for rolling median baseline. Default 5.

    Returns
    -------
    FeatureTensor
    """
    values: Dict[str, np.ndarray] = {}
    feature_index: Dict[str, str] = {}

    windows = plan.windows

    for agg_name, raw in surface.values.items():
        # Raw — carried through.
        values[agg_name] = raw.copy()
        feature_index[agg_name] = f"raw {agg_name} from surface"

        # EWMA baseline.
        key = f"{agg_name}_baseline_ewma"
        values[key] = _ewma(raw, alpha=ewma_alpha)
        feature_index[key] = f"EWMA baseline of {agg_name} (alpha={ewma_alpha})"

        # Rolling median baseline.
        key = f"{agg_name}_baseline_rmed"
        values[key] = _rolling_median(raw, window=rolling_window)
        feature_index[key] = (
            f"rolling median baseline of {agg_name} (window={rolling_window})"
        )

        # Residual (value - EWMA baseline).
        baseline = values[f"{agg_name}_baseline_ewma"]
        residual = raw - baseline
        key = f"{agg_name}_residual"
        values[key] = residual
        feature_index[key] = f"residual of {agg_name} vs EWMA baseline"

        # Robust z-score (MAD-based).
        key = f"{agg_name}_zscore"
        values[key] = _mad_zscore(residual)
        feature_index[key] = f"robust z-score of {agg_name} (MAD-based)"

        # Delta (first difference over time).
        key = f"{agg_name}_delta"
        values[key] = _delta(raw)
        feature_index[key] = f"first difference of {agg_name} over time"

        # Rate (delta adjusted by coverage — where coverage > 0).
        delta = values[f"{agg_name}_delta"]
        rate = np.where(surface.coverage > 0, delta / np.maximum(surface.coverage, 1e-9), np.nan)
        key = f"{agg_name}_rate"
        values[key] = rate.astype(np.float64)
        feature_index[key] = f"coverage-adjusted rate of change of {agg_name}"

        # Lattice gradient (fine - coarse along scale axis).
        key = f"{agg_name}_gradient"
        values[key] = _lattice_gradient(raw, windows)
        feature_index[key] = (
            f"lattice gradient of {agg_name}: scale[i] - scale[cover(i)]"
        )

    return FeatureTensor(
        channel=surface.channel,
        keys=surface.keys,
        metric=surface.metric,
        profile=surface.profile,
        time_axis=surface.time_axis,
        scale_axis=surface.scale_axis,
        coordinates=surface.coordinates,
        sampling_plan_id=surface.sampling_plan_id,
        n_events=surface.n_events.copy(),
        coverage=surface.coverage.copy(),
        values=values,
        feature_index=feature_index,
    )
