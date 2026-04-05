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
    SIGNALS  = "signals"
    BINNED   = "binned"
    SURFACES = "surfaces"
    TENSORS  = "tensors"
    BUNDLE   = "bundle"


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

def _artifact_id(
    artifact_type: ArtifactType,
    plan: Any,
    producing_op: Any,
    parent_ids: tuple = (),
) -> str:
    """Compute a deterministic identity hash for an artifact.

    The hash captures: artifact type, sampling plan, producing op class
    and parameters, and parent artifact IDs (lineage). This means two
    artifacts with the same inputs, same plan, and same transform chain
    produce the same ID — enabling caching, reproducibility checks, and
    ML lineage tracking.
    """
    import hashlib
    import json

    parts = [artifact_type.value]

    # Plan identity
    if plan is not None:
        parts.append(f"H={getattr(plan, 'horizon', '?')}")
        parts.append(f"g={getattr(plan, 'grain', '?')}")
        parts.append(f"cb={getattr(plan, 'cbin', '?')}")
        w = getattr(plan, 'windows', ())
        parts.append(f"W={w}")

    # Op identity
    if producing_op is not None:
        parts.append(producing_op.__class__.__name__)
        try:
            params = json.dumps(producing_op.params, sort_keys=True, default=str)
            parts.append(params)
        except (AttributeError, TypeError):
            pass

    # Lineage
    for pid in parent_ids:
        parts.append(pid)

    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class Artifact:
    """Wraps a pipeline product with metadata about how it was produced.

    This is the currency of the graph — every node consumes and produces
    Artifacts. The type tag enables validation; the metadata enables
    inspection and lineage tracking.

    The `id` field is a deterministic hash of (type, plan, op, parents).
    Two artifacts with the same inputs, plan, and transform chain have
    the same ID. Use for caching, reproducibility, and ML lineage.
    """
    type: ArtifactType
    value: Any
    producing_op: Optional[Any] = None
    plan: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_ids: tuple = ()
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = _artifact_id(
                self.type, self.plan, self.producing_op, self.parent_ids
            )

    def __repr__(self) -> str:
        return f"Artifact({self.type.value}, id={self.id[:8]})"


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
