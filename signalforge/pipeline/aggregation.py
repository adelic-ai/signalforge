"""
signalforge.pipeline.aggregation

Aggregation function registry for Stage 2 (Materialize).

Standard functions are registered at import time. Custom functions can be
added with @register_aggregation("name") or register_aggregation("name", fn).

A registered function receives a non-empty numpy array of float values for
one bin and returns a single float.

Usage
-----
    from signalforge.pipeline.aggregation import get_aggregation, register_aggregation

    fn = get_aggregation("mean")
    result = fn(values_array)

    @register_aggregation("trimmed_mean")
    def trimmed_mean(values: np.ndarray) -> float:
        ...
"""

from __future__ import annotations

import statistics
from typing import Callable, Dict

import numpy as np

AggFunc = Callable[[np.ndarray], float]

_REGISTRY: Dict[str, AggFunc] = {}


def register_aggregation(name: str, fn: AggFunc | None = None):
    """
    Register an aggregation function by name.

    Can be used as a decorator or called directly:

        @register_aggregation("my_agg")
        def my_agg(values: np.ndarray) -> float: ...

        register_aggregation("my_agg", my_agg)
    """
    if fn is None:
        # Decorator form.
        def decorator(f: AggFunc) -> AggFunc:
            _REGISTRY[name] = f
            return f
        return decorator
    _REGISTRY[name] = fn
    return fn


def get_aggregation(name: str) -> AggFunc:
    """Return the aggregation function registered under name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown aggregation function {name!r}. "
            f"Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def aggregation_names() -> tuple[str, ...]:
    """Return all registered aggregation names in sorted order."""
    return tuple(sorted(_REGISTRY))


# ---------------------------------------------------------------------------
# Standard aggregation functions
# ---------------------------------------------------------------------------

@register_aggregation("count")
def _count(v: np.ndarray) -> float:
    return float(len(v))


@register_aggregation("sum")
def _sum(v: np.ndarray) -> float:
    return float(np.sum(v))


@register_aggregation("mean")
def _mean(v: np.ndarray) -> float:
    return float(np.mean(v))


@register_aggregation("min")
def _min(v: np.ndarray) -> float:
    return float(np.min(v))


@register_aggregation("max")
def _max(v: np.ndarray) -> float:
    return float(np.max(v))


@register_aggregation("median")
def _median(v: np.ndarray) -> float:
    return float(np.median(v))


@register_aggregation("std")
def _std(v: np.ndarray) -> float:
    return float(np.std(v))


@register_aggregation("var")
def _var(v: np.ndarray) -> float:
    return float(np.var(v))


@register_aggregation("range")
def _range(v: np.ndarray) -> float:
    return float(np.max(v) - np.min(v))


@register_aggregation("first")
def _first(v: np.ndarray) -> float:
    return float(v[0])


@register_aggregation("last")
def _last(v: np.ndarray) -> float:
    return float(v[-1])


@register_aggregation("mode")
def _mode(v: np.ndarray) -> float:
    # Most frequent value; ties broken by smallest value.
    values, counts = np.unique(v, return_counts=True)
    return float(values[np.argmax(counts)])


@register_aggregation("p25")
def _p25(v: np.ndarray) -> float:
    return float(np.percentile(v, 25))


@register_aggregation("p75")
def _p75(v: np.ndarray) -> float:
    return float(np.percentile(v, 75))


@register_aggregation("p90")
def _p90(v: np.ndarray) -> float:
    return float(np.percentile(v, 90))


@register_aggregation("p95")
def _p95(v: np.ndarray) -> float:
    return float(np.percentile(v, 95))


@register_aggregation("p99")
def _p99(v: np.ndarray) -> float:
    return float(np.percentile(v, 99))


@register_aggregation("geometric_mean")
def _geometric_mean(v: np.ndarray) -> float:
    # Defined over positive values only; non-positive entries are ignored.
    pos = v[v > 0]
    if len(pos) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(pos))))


@register_aggregation("ewma")
def _ewma(v: np.ndarray) -> float:
    """
    Final value of the exponentially weighted moving average (alpha=0.3).

    Applied to an ordered sequence of bin values within a window — each
    successive bin is a newer observation. The final value reflects the
    smoothed state at the trailing edge of the window.
    """
    result = float(v[0])
    for x in v[1:]:
        result = 0.3 * float(x) + 0.7 * result
    return result


@register_aggregation("spectral_energy")
def _spectral_energy(v: np.ndarray) -> float:
    """
    Total spectral energy: sum(|FFT(v)|²) / len(v).

    Measures the total oscillatory content in the window. Hann window
    applied to suppress spectral leakage from boundary discontinuities.
    DC component removed before FFT. Higher values = more energy.
    """
    centered = v - np.mean(v)
    windowed = centered * np.hanning(len(v))
    spectrum = np.fft.rfft(windowed)
    return float(np.sum(np.abs(spectrum) ** 2) / len(v))


@register_aggregation("dominant_freq")
def _dominant_freq(v: np.ndarray) -> float:
    """
    Dominant frequency: argmax(|FFT(v)|) / len(v).

    The frequency with the most energy in the window, normalized to [0, 0.5]
    (fraction of the sampling rate). Hann window applied. Higher = faster
    oscillation. Returns 0 for constant or very short windows.
    """
    if len(v) < 4:
        return 0.0
    centered = v - np.mean(v)
    windowed = centered * np.hanning(len(v))
    spectrum = np.abs(np.fft.rfft(windowed))
    if spectrum.max() == 0:
        return 0.0
    peak_bin = int(np.argmax(spectrum[1:])) + 1  # skip DC
    return float(peak_bin / len(v))
