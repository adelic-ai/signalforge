"""
signalforge.pipeline.canonical

CanonicalRecord: the normalized, domain-agnostic unit of the pipeline.

Stage 0 (Ingest) transforms raw source data into a sequence of these records.
Domain semantics stop here. Every subsequent stage sees only abstract channels,
named entity dimensions, ordered numeric observations, and ordering integers.

One raw event produces one or more CanonicalRecords — one per metric carried
by that event. Aggregation is Stage 2's responsibility, not Stage 0's.

OrderType
---------
TIME      — primary_order is a time epoch integer; gaps are real elapsed time
SEQUENCE  — primary_order is a machine sequence number; gaps indicate missing events
BOTH      — both dimensions are present; order_delta is meaningful

The choice of primary_order (time_order or seq_order) is declared at ingest
configuration time and is fixed for the lifetime of the CanonicalSequence.
"""

from __future__ import annotations

import enum
from typing import Dict, List, Optional, Union


class OrderType(enum.Enum):
    TIME = "TIME"
    SEQUENCE = "SEQUENCE"
    BOTH = "BOTH"


# Keys map dimension names to one or more string values.
# Multiple values per dimension are supported — a single event may carry
# both an IPv4 and IPv6 form of the same address, or multiple related
# entity identifiers.
KeyValue = Union[str, List[str]]
Keys = Dict[str, KeyValue]


class CanonicalRecord:
    """
    One normalized observation from a raw source event.

    A single raw event with k metrics produces k CanonicalRecords, one per
    (metric, value) pair. All records from the same event share the same
    primary_order, order_type, time_order, seq_order, order_delta, orders,
    channel, and keys.

    Parameters
    ----------
    primary_order : int
        Normalized integer that drives pipeline binning.
        Monotonically non-decreasing across the sequence.
        Must be non-negative.

    order_type : OrderType
        Declares which ordering dimensions are present.

    channel : str
        Abstract event category. No domain semantics enforced.

    metric : str
        Names the numeric measurement in value.

    value : float
        The numeric observation. Always float.

    keys : dict[str, str | list[str]], optional
        Named entity dimensions. Order-independent. Defaults to empty.

    time_order : int or None
        Epoch integer. Required when order_type is TIME or BOTH.

    seq_order : int or None
        Machine sequence number. Required when order_type is SEQUENCE or BOTH.

    order_delta : int or None
        seq_order - time_order. Populated when order_type is BOTH.

    orders : dict[str, int], optional
        Additional ordering dimensions by name (e.g. ingest_time).

    Invariants
    ----------
    - primary_order >= 0
    - value is float
    - time_order present when order_type is TIME or BOTH
    - seq_order present when order_type is SEQUENCE or BOTH
    - order_delta present when order_type is BOTH
    - keys contains no numeric measurements
    - no datetime objects anywhere

    Examples
    --------
    >>> rec = CanonicalRecord(
    ...     primary_order=1712000060,
    ...     order_type=OrderType.TIME,
    ...     channel="dns_query",
    ...     metric="count",
    ...     value=1.0,
    ...     keys={"host": "ws-042"},
    ...     time_order=1712000060,
    ... )
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
