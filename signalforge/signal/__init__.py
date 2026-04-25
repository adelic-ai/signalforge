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
from ._convert import records_to_signals
from ._measure import measure_signal
from ._schema import Schema, Axis, AxisType
from ._record import Record, records_from_csv
from ._segment import Segment, discover_segments, segments_to_signals, print_stats as print_segment_stats
from ._features import (
    entity_signals, entity_channel_matrix,
    label_segments, segment_summary, print_segment_summary,
    segment_features, segments_to_matrix, join_segments,
)
from ._information import (
    entropy, joint_entropy, mutual_information, kl_divergence,
    information_gain, best_split,
    discover_scales, discover_plan,
    entropy_surface, mutual_information_surface,
    divergence_surface, information_gain_surface,
)

__all__ = [
    # Core signal types
    "LatticeSignal",
    "ComplexSignal",
    "RealSignal",
    "Surface",
    # Pipeline plumbing
    "Artifact",
    "ArtifactType",
    "CanonicalRecord",
    "KeyValue",
    "Keys",
    "OrderType",
    # Schema and records (ingest)
    "Schema",
    "Axis",
    "AxisType",
    "Record",
    "records_from_csv",
    # Conversion / measurement
    "records_to_signals",
    "measure_signal",
    # Segments
    "Segment",
    "discover_segments",
    "segments_to_signals",
    "print_segment_stats",
    # Features
    "entity_signals",
    "entity_channel_matrix",
    "label_segments",
    "segment_summary",
    "print_segment_summary",
    "segment_features",
    "segments_to_matrix",
    "join_segments",
    # Information layer
    "entropy",
    "joint_entropy",
    "mutual_information",
    "kl_divergence",
    "information_gain",
    "best_split",
    "discover_scales",
    "discover_plan",
    "entropy_surface",
    "mutual_information_surface",
    "divergence_surface",
    "information_gain_surface",
]
