"""
signalforge.graph._multi_ops

Baseline, Residual, and Stack operators.

These operate on Surface artifacts — they take surfaces in and produce
surfaces out with the same shape but different values.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ._core import Artifact, Op
from ._types import ArtifactType


# ---------------------------------------------------------------------------
# Baseline methods — operate on a single (n_scales × n_time) array
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


# ---------------------------------------------------------------------------
# BaselineOp
# ---------------------------------------------------------------------------

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
        from ..pipeline.surface import Surface

        surfaces = inputs[0].value
        result_surfaces = []

        for s in surfaces:
            new_values = dict(s.values)  # keep originals

            # Use "mean" if available, else first value array
            source_key = "mean" if "mean" in s.values else next(iter(s.values))
            arr = s.values[source_key]

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

            new_values["amplitude"] = amplitude
            new_values["phase"] = phase
            new_values["inst_freq"] = inst_freq

            result_surfaces.append(Surface(
                channel=s.channel,
                keys=s.keys,
                metric=s.metric,
                profile=s.profile,
                time_axis=s.time_axis,
                scale_axis=s.scale_axis,
                values=new_values,
                n_events=s.n_events,
                coverage=s.coverage,
                coordinates=s.coordinates,
                sampling_plan_id=s.sampling_plan_id,
            ))

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

        from ..pipeline.surface import Surface

        result_surfaces = []
        for s in surfaces:
            new_values = {}
            for agg_name, arr in s.values.items():
                new_values[agg_name] = method_fn(arr, **method_kwargs)
            result_surfaces.append(Surface(
                channel=s.channel,
                keys=s.keys,
                metric=s.metric,
                profile=f"baseline_{method_name}",
                time_axis=s.time_axis,
                scale_axis=s.scale_axis,
                values=new_values,
                n_events=s.n_events,
                coverage=s.coverage,
                coordinates=s.coordinates,
                sampling_plan_id=s.sampling_plan_id,
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

        from ..pipeline.surface import Surface

        result_surfaces = []
        for ms, bs in zip(measured_surfaces, baseline_surfaces):
            new_values = {}
            for agg_name in ms.values:
                m_arr = ms.values[agg_name]
                b_arr = bs.values[agg_name]
                if mode == "difference":
                    new_values[agg_name] = m_arr - b_arr
                elif mode == "ratio":
                    with np.errstate(divide='ignore', invalid='ignore'):
                        new_values[agg_name] = np.where(
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
                    new_values[agg_name] = z
                else:
                    raise ValueError(
                        f"Unknown residual mode {mode!r}. Use 'difference', 'ratio', or 'z'."
                    )

            result_surfaces.append(Surface(
                channel=ms.channel,
                keys=ms.keys,
                metric=ms.metric,
                profile=f"residual_{mode}",
                time_axis=ms.time_axis,
                scale_axis=ms.scale_axis,
                values=new_values,
                n_events=ms.n_events,
                coverage=ms.coverage,
                coordinates=ms.coordinates,
                sampling_plan_id=ms.sampling_plan_id,
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

        from ..pipeline.surface import Surface

        result_surfaces = []
        for ch_idx in range(n):
            base = all_surface_lists[0][ch_idx]
            merged_values = {}
            for src_idx, sl in enumerate(all_surface_lists):
                s = sl[ch_idx]
                prefix = s.profile if s.profile else f"src{src_idx}"
                for agg_name, arr in s.values.items():
                    merged_values[f"{prefix}_{agg_name}"] = arr

            result_surfaces.append(Surface(
                channel=base.channel,
                keys=base.keys,
                metric=base.metric,
                profile="stacked",
                time_axis=base.time_axis,
                scale_axis=base.scale_axis,
                values=merged_values,
                n_events=base.n_events,
                coverage=base.coverage,
                coordinates=base.coordinates,
                sampling_plan_id=base.sampling_plan_id,
            ))

        return Artifact(
            type=ArtifactType.SURFACES,
            value=result_surfaces,
            producing_op=self,
            plan=plan,
            metadata={"n_sources": len(inputs)},
        )
