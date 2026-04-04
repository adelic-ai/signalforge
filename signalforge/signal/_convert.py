"""
signalforge.signal._convert

Conversion between record-based and signal-based representations.

records_to_signals: list[CanonicalRecord] → list[RealSignal]
    Groups records by (channel, keys) and produces one RealSignal per group.
    Each signal has integer index (primary_order) and float values.

This is the bridge between ingest (which produces records) and the
signal-centric pipeline (which operates on LatticeSignals).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._base import CanonicalRecord
from ._complex import RealSignal


def _canonical_keys(keys: dict) -> tuple:
    """Deterministic, order-independent key for grouping."""
    result = []
    for k in sorted(keys):
        v = keys[k]
        result.append((k, tuple(sorted(v)) if isinstance(v, list) else v))
    return tuple(result)


def records_to_signals(
    records: List[CanonicalRecord],
    agg: str = "mean",
) -> List[RealSignal]:
    """Convert CanonicalRecords into RealSignals, one per (channel, keys) group.

    Records with the same primary_order within a group are aggregated
    using the specified function. The result is a dense signal indexed
    by primary_order.

    Parameters
    ----------
    records : list of CanonicalRecord
        Input records. Need not be sorted.
    agg : str
        Aggregation for multiple values at the same index.
        "mean", "sum", "max", "min", "count", "last".

    Returns
    -------
    list of RealSignal
        One signal per distinct (channel, keys) combination.
    """
    if not records:
        return []

    _AGG_FUNCS = {
        "mean": np.mean,
        "sum": np.sum,
        "max": np.max,
        "min": np.min,
        "count": lambda x: float(len(x)),
        "last": lambda x: x[-1],
    }
    agg_fn = _AGG_FUNCS.get(agg)
    if agg_fn is None:
        raise ValueError(f"Unknown agg {agg!r}. Use: {sorted(_AGG_FUNCS)}")

    # Group by (channel, canonical_keys)
    groups: Dict[Tuple, List[CanonicalRecord]] = defaultdict(list)
    for r in records:
        key = (r.channel, _canonical_keys(r.keys))
        groups[key].append(r)

    signals = []
    for (channel, keys_canon), recs in groups.items():
        # Aggregate by primary_order
        by_order: Dict[int, List[float]] = defaultdict(list)
        for r in recs:
            by_order[r.primary_order].append(r.value)

        sorted_orders = sorted(by_order.keys())
        index = np.array(sorted_orders, dtype=np.int64)
        values = np.array(
            [agg_fn(np.array(by_order[o])) for o in sorted_orders],
            dtype=np.float64,
        )

        keys_dict = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in keys_canon
        }

        signals.append(RealSignal(
            index=index,
            values=values,
            channel=channel,
            keys=keys_dict if keys_dict else None,
            metadata={"n_records": len(recs), "agg": agg},
        ))

    return signals
