"""
signalforge.pipeline.binned

BinnedData: Stage 2 (Materialize) output.

Transforms a CanonicalSequence into an integer-indexed, geometrically
coherent structure aligned with the SamplingPlan. This is the first point
of information reduction — events are aggregated into bins.

    bin_index = floor(primary_order / cbin)

Every bin is aligned to the SamplingPlan's cbin grid. Empty bins are
represented implicitly through gap_before on the next populated bin;
the absence of a record means no events fell in that bin.

BinnedData is always re-derivable from CanonicalSequence. It is a
materialized view, not a source of truth.

Materialization scope
---------------------
Pass a scope dict to restrict which (channel, metric) combinations are
materialized. Omit it (or pass None) for full scope.

    scope = {
        "dns_query": ["count"],
        "http":      ["count", "bytes"],
    }
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..lattice.sampling import SamplingPlan
from .aggregation import AggFunc, get_aggregation
from .canonical import CanonicalRecord, OrderType

# Scope maps channel → set of metrics to materialize.
# None means materialize everything.
Scope = Optional[Dict[str, List[str]]]

# Default aggregation function applied when none is specified per channel/metric.
_DEFAULT_AGG = "mean"


class BinnedRecord:
    """
    One aggregated bin for a specific (channel, keys, metric) combination.

    Fields
    ------
    bin_index  : int              floor(primary_order / cbin)
    channel    : str
    keys       : dict             canonical key dict from CanonicalSequence
    metric     : str
    agg_func   : str              name of the aggregation function applied
    value      : float            aggregated value
    n_events   : int              number of raw events in this bin (>= 1)
    gap_before : int or None      bins since the last populated bin; None for first
    seq_sum    : int or None      sum of seq_order values; only for CONTINUOUS SEQUENCE
    """

    __slots__ = (
        "bin_index",
        "channel",
        "keys",
        "metric",
        "agg_func",
        "value",
        "n_events",
        "gap_before",
        "seq_sum",
    )

    def __init__(
        self,
        bin_index: int,
        channel: str,
        keys: dict,
        metric: str,
        agg_func: str,
        value: float,
        n_events: int,
        gap_before: Optional[int],
        seq_sum: Optional[int],
    ) -> None:
        self.bin_index = bin_index
        self.channel = channel
        self.keys = keys
        self.metric = metric
        self.agg_func = agg_func
        self.value = value
        self.n_events = n_events
        self.gap_before = gap_before
        self.seq_sum = seq_sum

    def __repr__(self) -> str:
        return (
            f"BinnedRecord("
            f"bin={self.bin_index}, "
            f"channel={self.channel!r}, "
            f"metric={self.metric!r}, "
            f"agg={self.agg_func!r}, "
            f"value={self.value}, "
            f"n={self.n_events}, "
            f"gap={self.gap_before})"
        )


def _canonical_keys(keys: dict) -> Tuple:
    """
    Deterministic, order-independent key for grouping and cache identity.
    Returns a sorted tuple of (dimension, value) pairs.
    List values are sorted before hashing.
    """
    result = []
    for k in sorted(keys):
        v = keys[k]
        result.append((k, tuple(sorted(v)) if isinstance(v, list) else v))
    return tuple(result)


def materialize(
    records: Iterable[CanonicalRecord],
    plan: SamplingPlan,
    agg_funcs: Optional[Dict[str, Dict[str, str]]] = None,
    scope: Scope = None,
) -> List[BinnedRecord]:
    """
    Materialize a CanonicalSequence into BinnedData aligned with a SamplingPlan.

    Parameters
    ----------
    records : iterable of CanonicalRecord
        The canonical sequence. Records must be in non-decreasing primary_order.
    plan : SamplingPlan
        Provides cbin — the bin grid alignment.
    agg_funcs : dict[channel, dict[metric, agg_name]], optional
        Aggregation function per (channel, metric). Defaults to "count" for
        any combination not explicitly specified.
    scope : dict[channel, list[metric]], optional
        Restrict materialization to specific (channel, metric) combinations.
        None materializes everything.

    Returns
    -------
    list of BinnedRecord
        One record per populated (bin_index, channel, keys_canonical, metric),
        in ascending bin_index order within each group.

    Notes
    -----
    - Empty bins are not emitted. gap_before on the next record captures the gap.
    - seq_sum is populated only when order_type is SEQUENCE or BOTH and
      seq_order is present on all contributing records.
    """
    cbin = plan.cbin
    agg_funcs = agg_funcs or {}
    scope_channels = set(scope.keys()) if scope else None

    # Accumulator: (channel, keys_canonical, metric) →
    #   { bin_index: {"values": [...], "seq_orders": [...] or None} }
    Accumulator = Dict[int, Dict]
    groups: Dict[Tuple, Accumulator] = defaultdict(lambda: defaultdict(
        lambda: {"values": [], "seq_orders": []}
    ))

    # Track agg_func and order_type per group key.
    group_agg: Dict[Tuple, str] = {}
    group_order_type: Dict[Tuple, OrderType] = {}

    for rec in records:
        # Scope filter.
        if scope_channels is not None:
            if rec.channel not in scope_channels:
                continue
            allowed_metrics = scope[rec.channel]
            if rec.metric not in allowed_metrics:
                continue

        bin_index = math.floor(rec.primary_order / cbin)
        keys_canon = _canonical_keys(rec.keys)
        group_key = (rec.channel, keys_canon, rec.metric)

        # Resolve aggregation function for this group (once, on first encounter).
        if group_key not in group_agg:
            agg_name = (
                agg_funcs
                .get(rec.channel, {})
                .get(rec.metric, _DEFAULT_AGG)
            )
            group_agg[group_key] = agg_name
            group_order_type[group_key] = rec.order_type

        bucket = groups[group_key][bin_index]
        bucket["values"].append(rec.value)
        if rec.seq_order is not None:
            bucket["seq_orders"].append(rec.seq_order)
        else:
            bucket["seq_orders"] = None  # Mark as unavailable for this group.

    # Emit BinnedRecords in bin_index order per group.
    output: List[BinnedRecord] = []

    for group_key, bins in groups.items():
        channel, keys_canon, metric = group_key
        agg_name = group_agg[group_key]
        agg_fn: AggFunc = get_aggregation(agg_name)
        order_type = group_order_type[group_key]

        # Reconstruct keys dict from canonical tuple.
        keys_dict = {k: (list(v) if isinstance(v, tuple) else v) for k, v in keys_canon}

        sorted_bins = sorted(bins.items())  # ascending bin_index
        prev_bin: Optional[int] = None

        for bin_index, bucket in sorted_bins:
            values = np.array(bucket["values"], dtype=float)
            seq_orders = bucket["seq_orders"]

            n_events = len(values)
            value = agg_fn(values)

            gap_before = (bin_index - prev_bin - 1) if prev_bin is not None else None

            # seq_sum: only when sequence ordering is present and complete.
            seq_sum: Optional[int] = None
            if (
                order_type in (OrderType.SEQUENCE, OrderType.BOTH)
                and seq_orders is not None
                and len(seq_orders) == n_events
            ):
                seq_sum = int(sum(seq_orders))

            output.append(BinnedRecord(
                bin_index=bin_index,
                channel=channel,
                keys=keys_dict,
                metric=metric,
                agg_func=agg_name,
                value=value,
                n_events=n_events,
                gap_before=gap_before,
                seq_sum=seq_sum,
            ))

            prev_bin = bin_index

    return output
