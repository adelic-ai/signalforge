"""
signalforge.pipeline.operators

Built-in transform operators for use with Pipeline.transform().

Each operator is a factory function that returns a callable suitable
for passing to .transform(). Operators work on list[BinnedRecord] unless
noted — insert them after .materialize() and before .measure().

Usage
-----
    import signalforge as sf
    import signalforge.ops as ops

    bundle = (
        sf.acquire("eeg", path)
          .materialize()
          .transform(ops.winsorize(0.01, 0.99))
          .transform(ops.drop_sparse(min_coverage=0.5))
          .measure(profile="continuous")
          .engineer()
          .assemble()
          .run()
    )
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_values(binned: list) -> dict:
    """Group BinnedRecord values by (channel, metric)."""
    groups: dict = {}
    for rec in binned:
        key = (rec.channel, rec.metric)
        if key not in groups:
            groups[key] = []
        groups[key].append(rec.value)
    return {k: np.array(v) for k, v in groups.items()}


def _replace_value(rec, new_value: float):
    """Return a copy of rec with value replaced."""
    from .binned import BinnedRecord
    return BinnedRecord(
        bin_index=rec.bin_index,
        channel=rec.channel,
        keys=rec.keys,
        metric=rec.metric,
        agg_func=rec.agg_func,
        value=new_value,
        n_events=rec.n_events,
        gap_before=rec.gap_before,
        seq_sum=rec.seq_sum,
    )


# ---------------------------------------------------------------------------
# Operators on list[BinnedRecord]
# ---------------------------------------------------------------------------

def clip(
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> Callable[[list], list]:
    """
    Clamp bin values to [low, high].

    Parameters
    ----------
    low : float, optional
        Values below this are set to low.
    high : float, optional
        Values above this are set to high.
    """
    def _clip(binned: list) -> list:
        return [
            _replace_value(r, float(np.clip(r.value, low, high)))
            for r in binned
        ]
    return _clip


def winsorize(
    lower: float = 0.01,
    upper: float = 0.99,
) -> Callable[[list], list]:
    """
    Replace extreme values with their percentile bounds, per (channel, metric).

    Parameters
    ----------
    lower : float
        Lower percentile bound (e.g. 0.01 = 1st percentile). Default: 0.01.
    upper : float
        Upper percentile bound (e.g. 0.99 = 99th percentile). Default: 0.99.
    """
    def _winsorize(binned: list) -> list:
        groups = _group_values(binned)
        bounds: dict = {}
        for key, values in groups.items():
            bounds[key] = (
                float(np.nanpercentile(values, lower * 100)),
                float(np.nanpercentile(values, upper * 100)),
            )
        return [
            _replace_value(r, float(np.clip(r.value, *bounds[(r.channel, r.metric)])))
            for r in binned
        ]
    return _winsorize


def drop_sparse(
    min_coverage: float = 0.1,
) -> Callable[[list], list]:
    """
    Drop channels with insufficient bin coverage.

    Coverage is the fraction of bins that are populated relative to the
    total span (first bin to last bin) for that (channel, metric) group.

    Parameters
    ----------
    min_coverage : float
        Minimum coverage fraction to keep. Default: 0.1 (10%).
    """
    def _drop_sparse(binned: list) -> list:
        from collections import defaultdict
        spans: dict = defaultdict(list)
        for rec in binned:
            spans[(rec.channel, rec.metric)].append(rec.bin_index)

        keep: set = set()
        for key, indices in spans.items():
            if len(indices) < 2:
                continue
            span = max(indices) - min(indices) + 1
            coverage = len(indices) / span
            if coverage >= min_coverage:
                keep.add(key)

        return [r for r in binned if (r.channel, r.metric) in keep]
    return _drop_sparse


def drop_channels(*channels: str) -> Callable[[list], list]:
    """
    Drop specific channels by name.

    Parameters
    ----------
    *channels : str
        Channel names to remove.
    """
    drop = set(channels)

    def _drop(binned: list) -> list:
        return [r for r in binned if r.channel not in drop]
    return _drop


def keep_channels(*channels: str) -> Callable[[list], list]:
    """
    Keep only the specified channels, dropping all others.

    Parameters
    ----------
    *channels : str
        Channel names to keep.
    """
    keep = set(channels)

    def _keep(binned: list) -> list:
        return [r for r in binned if r.channel in keep]
    return _keep


def fill_gaps(value: float = 0.0) -> Callable[[list], list]:
    """
    Insert zero-value bins for any gap in the sequence.

    Useful for sparse data where downstream surface computation
    should treat missing bins as a known value rather than absent.

    Parameters
    ----------
    value : float
        Value to insert for gap bins. Default: 0.0.
    """
    def _fill(binned: list) -> list:
        from collections import defaultdict
        from .binned import BinnedRecord

        groups: dict = defaultdict(list)
        for rec in binned:
            key = (rec.channel, tuple(sorted(rec.keys.items())), rec.metric)
            groups[key].append(rec)

        result = []
        for key, recs in groups.items():
            channel, keys_tuple, metric = key
            keys_dict = dict(keys_tuple)
            recs_sorted = sorted(recs, key=lambda r: r.bin_index)
            result.append(recs_sorted[0])
            for prev, curr in zip(recs_sorted, recs_sorted[1:]):
                for gap_bin in range(prev.bin_index + 1, curr.bin_index):
                    result.append(BinnedRecord(
                        bin_index=gap_bin,
                        channel=channel,
                        keys=keys_dict,
                        metric=metric,
                        agg_func=prev.agg_func,
                        value=value,
                        n_events=0,
                        gap_before=None,
                        seq_sum=None,
                    ))
                result.append(curr)

        return sorted(result, key=lambda r: r.bin_index)
    return _fill
