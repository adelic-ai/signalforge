"""
signalforge.signal._surface

Surface: a 2D measurement of a LatticeSignal through a SamplingPlan.

A Surface is itself a LatticeSignal — it's indexed by integers (bin
positions) and carries values at each scale. This means you can measure
a Surface to produce another Surface (recursive decomposition), compute
baselines of surfaces, take residuals, etc. Signal in, signal out.

Dimensions:
    scale_axis — window sizes (rows), from the SamplingPlan
    time_axis  — bin positions (columns), integer-indexed

Values are complex128 natively. Real surfaces have imag=0.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..lattice.sampling import SamplingPlan
from ._signal import LatticeSignal


class Surface(LatticeSignal):
    """2D measurement grid: scale x time.

    Each row is a scale (window size). Each column is a time position.
    Values are aggregated observations at that scale and position.

    A Surface satisfies the LatticeSignal contract — its index is the
    time_axis and its values are the 2D grid. This makes surfaces
    composable: you can measure a surface, baseline a surface, take
    residuals of surfaces, feed surfaces into further analysis.

    Parameters
    ----------
    time_axis : array-like of int
        Bin positions along the time dimension.
    scale_axis : tuple of int
        Window sizes, one per row. From the SamplingPlan.
    data : dict of str -> np.ndarray
        Named value arrays, each (n_scales, n_time). complex128 or float64.
        E.g. {"mean": ..., "std": ...} for a continuous profile.
    channel : str
        What signal this surface was measured from.
    plan : SamplingPlan
        The geometry used to construct this surface.
    keys : dict, optional
        Entity dimensions inherited from the source signal.
    metric : str, optional
        Name of the measured quantity.
    profile : str, optional
        Name of the aggregation profile used.
    coordinates : tuple of dict, optional
        Prime exponent vector per scale row.
    n_events : np.ndarray, optional
        (n_scales, n_time) int array of raw event counts per cell.
    coverage : np.ndarray, optional
        (n_scales, n_time) float array, fraction of bins occupied.
    metadata : dict, optional
        Arbitrary metadata.
    """

    def __init__(
        self,
        time_axis: np.ndarray,
        scale_axis: Tuple[int, ...],
        data: Dict[str, np.ndarray],
        channel: str,
        plan: SamplingPlan,
        keys: Optional[Dict[str, Any]] = None,
        metric: Optional[str] = None,
        profile: Optional[str] = None,
        coordinates: Optional[Tuple[Dict[int, int], ...]] = None,
        n_events: Optional[np.ndarray] = None,
        coverage: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._time_axis = np.asarray(time_axis, dtype=np.int64)
        self._scale_axis = tuple(scale_axis)
        self._data = data
        self._channel = channel
        self._plan = plan
        self._keys = keys or {}
        self._metric = metric or ""
        self._profile = profile or ""
        self._coordinates = coordinates or ()
        self._n_events = n_events
        self._coverage = coverage
        self._metadata = metadata or {}

        n_scales = len(self._scale_axis)
        n_time = len(self._time_axis)
        expected = (n_scales, n_time)
        for name, arr in self._data.items():
            if arr.shape != expected:
                raise ValueError(
                    f"data[{name!r}].shape {arr.shape} != {expected}"
                )

    # --- LatticeSignal contract ---

    @property
    def index(self) -> np.ndarray:
        """Time axis — bin positions."""
        return self._time_axis

    @property
    def values(self) -> np.ndarray:
        """Primary value array (first in data dict).

        For the full multi-aggregation view, use surface.data directly.
        """
        return next(iter(self._data.values()))

    @property
    def channel(self) -> str:
        return self._channel

    @property
    def keys(self) -> Dict[str, Any]:
        return self._keys

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def metric(self) -> str:
        return self._metric

    # --- Surface-specific ---

    @property
    def time_axis(self) -> np.ndarray:
        return self._time_axis

    @property
    def scale_axis(self) -> Tuple[int, ...]:
        return self._scale_axis

    @property
    def data(self) -> Dict[str, np.ndarray]:
        """All named value arrays. E.g. {"mean": ..., "std": ...}."""
        return self._data

    @property
    def plan(self) -> SamplingPlan:
        return self._plan

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def coordinates(self) -> Tuple[Dict[int, int], ...]:
        return self._coordinates

    @property
    def n_events(self) -> Optional[np.ndarray]:
        return self._n_events

    @property
    def coverage(self) -> Optional[np.ndarray]:
        return self._coverage

    @property
    def shape(self) -> Tuple[int, int]:
        """(n_scales, n_time)"""
        return (len(self._scale_axis), len(self._time_axis))

    def __repr__(self) -> str:
        n_s, n_t = self.shape
        dtype = "complex" if not self.is_real else "real"
        aggs = list(self._data.keys())
        keys = f", keys={self._keys}" if self._keys else ""
        return (
            f"Surface({self._channel!r}, {n_s}x{n_t}, "
            f"{dtype}, aggs={aggs}{keys})"
        )
