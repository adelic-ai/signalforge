"""
signalforge.signal._complex

Concrete signal types built on LatticeSignal.

ComplexSignal — the atom. Two components on the lattice, combined as
    real + j*imag into a single complex128 array. What the components
    mean (time vs event order, amplitude vs phase, channel A vs channel B)
    is not prescribed here — that's a domain/user choice.

RealSignal — the degenerate case. One real-valued component, imag=0.
    This is what you get from a generic CSV, VIX data, EEG, magnetometer,
    or any single-valued timeseries. It IS a ComplexSignal — just one
    where the imaginary part happens to be zero.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ._signal import LatticeSignal


class ComplexSignal(LatticeSignal):
    """Two components on the lattice, combined as complex128.

    The atom of SignalForge. Everything more complex (Clifford, tensor)
    is built from these. Everything simpler (real) is a special case.

    Parameters
    ----------
    index : array-like of int
        Integer positions on the lattice.
    real_part : array-like of float
        First component.
    imag_part : array-like of float
        Second component.
    channel : str
        What this signal represents.
    keys : dict, optional
        Entity dimensions.
    metadata : dict, optional
        Arbitrary metadata (component labels, source info, etc.).
    """

    def __init__(
        self,
        index: np.ndarray,
        real_part: np.ndarray,
        imag_part: np.ndarray,
        channel: str,
        keys: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        idx = np.asarray(index, dtype=np.int64)
        r = np.asarray(real_part, dtype=np.float64)
        i = np.asarray(imag_part, dtype=np.float64)

        if idx.shape != r.shape or idx.shape != i.shape:
            raise ValueError(
                f"Shape mismatch: index={idx.shape}, "
                f"real={r.shape}, imag={i.shape}"
            )

        self._index = idx
        self._values = r + 1j * i
        self._channel = channel
        self._keys = keys or {}
        self._metadata = metadata or {}

    @property
    def index(self) -> np.ndarray:
        return self._index

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def channel(self) -> str:
        return self._channel

    @property
    def keys(self) -> Dict[str, Any]:
        return self._keys

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata


class RealSignal(LatticeSignal):
    """One real-valued component on the lattice. imag=0.

    The most common case: a timeseries, a count stream, a measurement.
    Stored as float64, promoted to complex128 only when needed.

    Parameters
    ----------
    index : array-like of int
        Integer positions on the lattice.
    values : array-like of float
        Observations.
    channel : str
        What this signal represents.
    keys : dict, optional
        Entity dimensions.
    metadata : dict, optional
        Arbitrary metadata.
    """

    def __init__(
        self,
        index: np.ndarray,
        values: np.ndarray,
        channel: str,
        keys: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._index = np.asarray(index, dtype=np.int64)
        self._values = np.asarray(values, dtype=np.float64)
        self._channel = channel
        self._keys = keys or {}
        self._metadata = metadata or {}

        if self._index.shape != self._values.shape:
            raise ValueError(
                f"Shape mismatch: index={self._index.shape}, "
                f"values={self._values.shape}"
            )

    @property
    def index(self) -> np.ndarray:
        return self._index

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def channel(self) -> str:
        return self._channel

    @property
    def keys(self) -> Dict[str, Any]:
        return self._keys

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata
