"""
signalforge.signal

The type foundation of SignalForge.

Everything is a LatticeSignal. The pipeline takes signals in and
produces signals out. Complex-valued natively; real is the special case.

    LatticeSignal       — the ABC. Integer-indexed, complex128 values.
      ComplexSignal     — the atom. Two components on the lattice.
      RealSignal        — imag=0. The common case.
      Surface           — 2D measurement (scale x time). Also a signal.

    Artifact            — wraps any pipeline product with provenance.
    CanonicalRecord     — normalized ingest unit.
"""

from ._base import Artifact, ArtifactType, CanonicalRecord, KeyValue, Keys, OrderType
from ._signal import LatticeSignal
from ._complex import ComplexSignal, RealSignal
from ._surface import Surface

__all__ = [
    "LatticeSignal",
    "ComplexSignal",
    "RealSignal",
    "Surface",
    "Artifact",
    "ArtifactType",
    "CanonicalRecord",
    "KeyValue",
    "Keys",
    "OrderType",
]
