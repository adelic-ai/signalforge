"""
signalforge.graph._multi_ops

Baseline, Residual, Hilbert, and Stack operators.

These operate on Surface artifacts — they take surfaces in and produce
surfaces out with the same shape but different values.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ._core import Artifact, Op
from ..signal._base import ArtifactType


# ---------------------------------------------------------------------------
# Baseline methods — operate on a single (n_scales x n_time) array
# ---------------------------------------------------------------------------

def _ewma(arr: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Exponentially weighted moving average, per row (per scale).

    alpha: smoothing factor. Higher = more responsive.
    Formula: s_t = alpha * x_t + (1 - alpha) * s_{t-1}
    """
    result = np.full_like(arr, np.nan)
    for i in range(arr.shape[0]):
        row = arr[i]
        out = np.full_like(row, np.nan)
        s = np.nan
        for j in range(len(row)):
            if np.isfinite(row[j]):
                if np.isnan(s):
                    s = row[j]
                else:
                    s = alpha * row[j] + (1 - alpha) * s
                out[j] = s
        result[i] = out
    return result


def _median_filter(arr: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Rolling median baseline, per row (per scale).

    Robust to spikes. Window is centered.
    """
    from scipy.ndimage import median_filter as _mf
    result = np.full_like(arr, np.nan)
    for i in range(arr.shape[0]):
        row = arr[i]
        finite_mask = np.isfinite(row)
        if finite_mask.sum() < window:
            result[i] = row
            continue
        # Fill NaN temporarily for the filter
        filled = np.where(finite_mask, row, np.nanmedian(row))
        filtered = _mf(filled, size=window, mode='reflect')
        result[i] = np.where(finite_mask, filtered, np.nan)
    return result


def _rolling_mean(arr: np.ndarray, window: int = 10) -> np.ndarray:
    """Rolling mean baseline, per row (per scale)."""
    result = np.full_like(arr, np.nan)
    for i in range(arr.shape[0]):
        row = arr[i]
        n = len(row)
        out = np.full(n, np.nan)
        for j in range(n):
            start = max(0, j - window + 1)
            chunk = row[start:j + 1]
            finite = chunk[np.isfinite(chunk)]
            if len(finite) > 0:
                out[j] = np.mean(finite)
        result[i] = out
    return result


_BASELINE_METHODS = {
    "ewma": _ewma,
    "median": _median_filter,
    "rolling_mean": _rolling_mean,
}


def _derive_surface(source, *, data, plan, profile=None, **overrides):
    """Create a new Surface from an existing one, replacing data and optionally other fields."""
    from ..signal import Surface
    return Surface(
        time_axis=overrides.get("time_axis", source.time_axis),
        scale_axis=overrides.get("scale_axis", source.scale_axis),
        data=data,
        channel=overrides.get("channel", source.channel),
        plan=plan,
        keys=overrides.get("keys", source.keys),
        metric=overrides.get("metric", source.metric),
        profile=profile if profile is not None else source.profile,
        coordinates=overrides.get("coordinates", source.coordinates),
        n_events=overrides.get("n_events", source.n_events),
        coverage=overrides.get("coverage", source.coverage),
    )


# ---------------------------------------------------------------------------
# HilbertOp
# ---------------------------------------------------------------------------

class HilbertOp(Op):
    """
    Compute the analytic signal via Hilbert transform.

    Adds amplitude (envelope), phase, and instantaneous frequency
    to each Surface. Operates per-scale row.

    Input: SURFACES. Output: SURFACES (same shape, additional value arrays).
    """

    input_types = (ArtifactType.SURFACES,)
    output_type = ArtifactType.SURFACES

    def execute(self, *inputs: Artifact, plan: Any = None) -> Artifact:
        from scipy.signal import hilbert

        surfaces = inputs[0].value
        result_surfaces = []

        for s in surfaces:
            new_data = dict(s.data)  # keep originals

            # Use "mean" if available, else first value array
            source_key = "mean" if "mean" in s.data else next(iter(s.data))
            arr = s.data[source_key]

            amplitude = np.full_like(arr, np.nan)
            phase = np.full_like(arr, np.nan)
            inst_freq = np.full_like(arr, np.nan)

            for i in range(arr.shape[0]):
                row = arr[i]
                finite_mask = np.isfinite(row)
                if finite_mask.sum() < 4:
                    continue

                # Fill NaN for Hilbert (needs contiguous data)
                filled = np.where(finite_mask, row, np.nanmean(row))
                analytic = hilbert(filled)

                amp = np.abs(analytic)
                ph = np.angle(analytic)
                # Instantaneous frequency: d(phase)/dt, unwrapped
                unwrapped = np.unwrap(ph)
                ifreq = np.gradient(unwrapped)

                amplitude[i] = np.where(finite_mask, amp, np.nan)
                phase[i] = np.where(finite_mask, ph, np.nan)
                inst_freq[i] = np.where(finite_mask, ifreq, np.nan)

            new_data["amplitude"] = amplitude
            new_data["phase"] = phase
            new_data["inst_freq"] = inst_freq

            result_surfaces.append(_derive_surface(s, data=new_data, plan=plan))

        return Artifact(
            type=ArtifactType.SURFACES,
            value=result_surfaces,
            producing_op=self,
            plan=plan,
            metadata={},
        )


# ---------------------------------------------------------------------------
# BaselineOp
# ---------------------------------------------------------------------------

class BaselineOp(Op):
    """
    Compute a baseline from a Surface.

    Produces a new Surface with the same shape where each value
    is the baseline estimate at that (scale, time) position.

    Parameters
    ----------
    method : str
        "ewma", "median", or "rolling_mean"
    alpha : float
        EWMA smoothing factor (only for method="ewma"). Default: 0.1.
    window : int
        Window size for median/rolling_mean. Default: 10.
    """

    input_types = (ArtifactType.SURFACES,)
    output_type = ArtifactType.SURFACES

    def execute(self, *inputs: Artifact, plan: Any = None) -> Artifact:
        surfaces = inputs[0].value
        method_name = self.params.get("method", "ewma")
        method_fn = _BASELINE_METHODS.get(method_name)
        if method_fn is None:
            raise ValueError(
                f"Unknown baseline method {method_name!r}. "
                f"Available: {sorted(_BASELINE_METHODS)}"
            )

        # Build kwargs for the method
        method_kwargs = {}
        if method_name == "ewma":
            method_kwargs["alpha"] = self.params.get("alpha", 0.1)
        elif method_name in ("median", "rolling_mean"):
            method_kwargs["window"] = self.params.get("window", 10)

        result_surfaces = []
        for s in surfaces:
            new_data = {}
            for agg_name, arr in s.data.items():
                new_data[agg_name] = method_fn(arr, **method_kwargs)
            result_surfaces.append(_derive_surface(
                s, data=new_data, plan=plan,
                profile=f"baseline_{method_name}",
            ))

        return Artifact(
            type=ArtifactType.SURFACES,
            value=result_surfaces,
            producing_op=self,
            plan=plan,
            metadata={"method": method_name},
        )


# ---------------------------------------------------------------------------
# ResidualOp
# ---------------------------------------------------------------------------

class ResidualOp(Op):
    """
    Compute the residual between a measured surface and a baseline.

    Two inputs: (measured, baseline). Both must be SURFACES.

    Parameters
    ----------
    mode : str
        "difference" — measured - baseline
        "ratio"      — measured / baseline
        "z"          — (measured - baseline) / std(baseline)
    """

    input_types = (ArtifactType.SURFACES, ArtifactType.SURFACES)
    output_type = ArtifactType.SURFACES

    def execute(self, *inputs: Artifact, plan: Any = None) -> Artifact:
        measured_surfaces = inputs[0].value
        baseline_surfaces = inputs[1].value
        mode = self.params.get("mode", "difference")

        result_surfaces = []
        for ms, bs in zip(measured_surfaces, baseline_surfaces):
            new_data = {}
            for agg_name in ms.data:
                m_arr = ms.data[agg_name]
                b_arr = bs.data[agg_name]
                if mode == "difference":
                    new_data[agg_name] = m_arr - b_arr
                elif mode == "ratio":
                    with np.errstate(divide='ignore', invalid='ignore'):
                        new_data[agg_name] = np.where(
                            b_arr != 0, m_arr / b_arr, np.nan
                        )
                elif mode == "z":
                    # z-score: need std of baseline per scale
                    z = np.full_like(m_arr, np.nan)
                    for i in range(m_arr.shape[0]):
                        b_row = b_arr[i]
                        finite = b_row[np.isfinite(b_row)]
                        if len(finite) > 0 and np.std(finite) > 0:
                            sigma = np.std(finite)
                            z[i] = (m_arr[i] - b_arr[i]) / sigma
                    new_data[agg_name] = z
                else:
                    raise ValueError(
                        f"Unknown residual mode {mode!r}. Use 'difference', 'ratio', or 'z'."
                    )

            result_surfaces.append(_derive_surface(
                ms, data=new_data, plan=plan,
                profile=f"residual_{mode}",
            ))

        return Artifact(
            type=ArtifactType.SURFACES,
            value=result_surfaces,
            producing_op=self,
            plan=plan,
            metadata={"mode": mode},
        )


# ---------------------------------------------------------------------------
# StackOp
# ---------------------------------------------------------------------------

class StackOp(Op):
    """
    Stack multiple Surface artifacts along the feature axis.

    All inputs must be SURFACES with matching shapes.

    Parameters
    ----------
    axis : str
        "feature" — concatenate value dicts, prefixing keys with source index.
    """

    input_types = ()  # variadic
    output_type = ArtifactType.SURFACES
    variadic_inputs = True

    def execute(self, *inputs: Artifact, plan: Any = None) -> Artifact:
        all_surface_lists = [inp.value for inp in inputs]

        # All inputs should have same number of surfaces (one per channel)
        n = len(all_surface_lists[0])
        for sl in all_surface_lists:
            if len(sl) != n:
                raise ValueError("All Stack inputs must have the same number of surfaces.")

        result_surfaces = []
        for ch_idx in range(n):
            base = all_surface_lists[0][ch_idx]
            merged_data = {}
            for src_idx, sl in enumerate(all_surface_lists):
                s = sl[ch_idx]
                prefix = s.profile if s.profile else f"src{src_idx}"
                for agg_name, arr in s.data.items():
                    merged_data[f"{prefix}_{agg_name}"] = arr

            result_surfaces.append(_derive_surface(
                base, data=merged_data, plan=plan,
                profile="stacked",
            ))

        return Artifact(
            type=ArtifactType.SURFACES,
            value=result_surfaces,
            producing_op=self,
            plan=plan,
            metadata={"n_sources": len(inputs)},
        )
