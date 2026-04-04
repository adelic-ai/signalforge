"""
signalforge.signal._base

Core types that the rest of SignalForge depends on.

Artifact: the universal container — wraps any pipeline product with
    provenance (what produced it, what plan was used, what metadata).

ArtifactType: tags what kind of thing an Artifact carries.

CanonicalRecord: the normalized, domain-agnostic unit of ingest.
    Raw events become these; everything downstream sees only abstract
    channels, ordered numeric observations, and entity keys.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# ArtifactType
# ---------------------------------------------------------------------------

class ArtifactType(enum.Enum):
    """Tags what a graph node produces."""
    RECORDS  = "records"
    BINNED   = "binned"
    SURFACES = "surfaces"
    TENSORS  = "tensors"
    BUNDLE   = "bundle"


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

@dataclass
class Artifact:
    """Wraps a pipeline product with metadata about how it was produced.

    This is the currency of the graph — every node consumes and produces
    Artifacts. The type tag enables validation; the metadata enables
    inspection and lineage tracking.
    """
    type: ArtifactType
    value: Any
    producing_op: Optional[Any] = None  # Op, but avoids circular import
    plan: Optional[Any] = None          # SamplingPlan
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Artifact({self.type.value})"


# ---------------------------------------------------------------------------
# OrderType
# ---------------------------------------------------------------------------

class OrderType(enum.Enum):
    """Declares which ordering dimensions a CanonicalRecord carries.

    TIME      — primary_order is a time epoch integer
    SEQUENCE  — primary_order is a machine sequence number
    BOTH      — both dimensions present; order_delta is meaningful
    """
    TIME = "TIME"
    SEQUENCE = "SEQUENCE"
    BOTH = "BOTH"


# ---------------------------------------------------------------------------
# CanonicalRecord
# ---------------------------------------------------------------------------

KeyValue = Union[str, List[str]]
Keys = Dict[str, KeyValue]


class CanonicalRecord:
    """One normalized observation from a raw source event.

    A single raw event with k metrics produces k CanonicalRecords, one per
    (metric, value) pair. All records from the same event share ordering,
    channel, and keys.

    Invariants:
        - primary_order >= 0
        - value is float
        - time_order present when order_type is TIME or BOTH
        - seq_order present when order_type is SEQUENCE or BOTH
        - order_delta present when order_type is BOTH
    """

    __slots__ = (
        "primary_order",
        "order_type",
        "channel",
        "metric",
        "value",
        "keys",
        "time_order",
        "seq_order",
        "order_delta",
        "orders",
    )

    def __init__(
        self,
        primary_order: int,
        order_type: OrderType,
        channel: str,
        metric: str,
        value: float,
        keys: Optional[Keys] = None,
        time_order: Optional[int] = None,
        seq_order: Optional[int] = None,
        order_delta: Optional[int] = None,
        orders: Optional[Dict[str, int]] = None,
    ) -> None:
        if primary_order < 0:
            raise ValueError(
                f"primary_order must be non-negative, got {primary_order}"
            )
        if order_type in (OrderType.TIME, OrderType.BOTH) and time_order is None:
            raise ValueError(
                f"time_order required when order_type is {order_type.value}"
            )
        if order_type in (OrderType.SEQUENCE, OrderType.BOTH) and seq_order is None:
            raise ValueError(
                f"seq_order required when order_type is {order_type.value}"
            )
        if order_type is OrderType.BOTH and order_delta is None:
            raise ValueError("order_delta required when order_type is BOTH")

        self.primary_order: int = primary_order
        self.order_type: OrderType = order_type
        self.channel: str = channel
        self.metric: str = metric
        self.value: float = float(value)
        self.keys: Keys = keys if keys is not None else {}
        self.time_order: Optional[int] = time_order
        self.seq_order: Optional[int] = seq_order
        self.order_delta: Optional[int] = order_delta
        self.orders: Dict[str, int] = orders if orders is not None else {}

    def __repr__(self) -> str:
        return (
            f"CanonicalRecord("
            f"primary_order={self.primary_order}, "
            f"channel={self.channel!r}, "
            f"metric={self.metric!r}, "
            f"value={self.value}, "
            f"keys={self.keys})"
        )
