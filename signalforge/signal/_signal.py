"""
signalforge.signal._signal

LatticeSignal: the abstract base unit of SignalForge.

Everything in SignalForge is a LatticeSignal — raw data, surfaces,
baselines, residuals. The pipeline takes signals in and produces
signals out. This is the contract.

Values are complex128 natively. Real-valued signals are the degenerate
case (imag=0). This is not a feature flag — complex is the foundation,
real is a projection of it.

The lattice (prime decomposition, scale structure) applies uniformly
to all signals because all signals are indexed by integers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..lattice.sampling import SamplingPlan


class LatticeSignal(ABC):
    """Anything indexed by integers on the prime lattice.

    This is the universal contract. If your data is discrete and
    ordered, it is a LatticeSignal and gets prime decomposition,
    scale structure, and surfaces for free.

    Subclasses must implement:

    Attributes
    ----------
    index : np.ndarray
        Integer positions, monotonically non-decreasing.
    values : np.ndarray
        Complex128 array of observations, same length as index.
    channel : str
        What this signal represents.
    keys : dict
        Entity dimensions (default: empty).
    metadata : dict
        Arbitrary metadata (default: empty).

    Derived properties (available on all signals):

    - ``dtype`` — numpy dtype of values
    - ``is_real`` — True if imaginary component is all zeros
    - ``amplitude()`` — |values|
    - ``phase()`` — angle(values)
    - ``real()`` — real component
    - ``imag()`` — imaginary component
    """

    @property
    @abstractmethod
    def index(self) -> np.ndarray:
        """Integer index array. Monotonically non-decreasing.

        These are the positions on the lattice — primary_order values,
        bin indices, time steps, event counts. The prime decomposition
        of these integers defines the scale structure.
        """
        ...

    @property
    @abstractmethod
    def values(self) -> np.ndarray:
        """Complex128 array of observations, same length as index.

        Real-valued signals: imag component is zero.
        Complex signals: both components carry meaning defined by
        the subclass (time vs event order, amplitude vs phase, etc.).
        """
        ...

    @property
    @abstractmethod
    def channel(self) -> str:
        """What this signal represents."""
        ...

    @property
    def keys(self) -> Dict[str, Any]:
        """Entity dimensions this signal is sliced on.

        Default: empty (unkeyed / aggregate signal).
        Override to carry entity context (user, host, NIC, etc.).
        """
        return {}

    @property
    def metadata(self) -> Dict[str, Any]:
        """Arbitrary metadata. Subclasses may override."""
        return {}

    # --- Derived properties (free from the contract) ---

    @property
    def dtype(self) -> np.dtype:
        return self.values.dtype

    @property
    def is_real(self) -> bool:
        """True if imaginary component is all zeros."""
        v = self.values
        if np.issubdtype(v.dtype, np.complexfloating):
            return np.all(v.imag == 0)
        return True

    def amplitude(self) -> np.ndarray:
        """abs(values). For real signals, this is abs(real part)."""
        return np.abs(self.values)

    def phase(self) -> np.ndarray:
        """angle(values). For real signals, this is 0 or pi."""
        return np.angle(self.values)

    def real(self) -> np.ndarray:
        """Real component."""
        return np.real(self.values)

    def imag(self) -> np.ndarray:
        """Imaginary component."""
        return np.imag(self.values)

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        n = len(self)
        dtype = "complex" if not self.is_real else "real"
        keys = f", keys={self.keys}" if self.keys else ""
        return f"{self.__class__.__name__}({self.channel!r}, n={n}, {dtype}{keys})"
