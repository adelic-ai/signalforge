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

Derived channel operators
-------------------------
ops.derive(name, fn)
    Per-bin derivation. fn receives {channel: value} for one bin, returns float.

        ops.derive("ratio", lambda ch: ch["flux_x"] / max(ch["flux_z"], 1.0))
        ops.derive("delta", lambda ch: ch["flux_x"] - ch["flux_z"])

ops.derive_temporal(name, fn)
    Full time-axis derivation. fn receives {channel: np.ndarray}, returns np.ndarray.

        ops.derive_temporal("flux_lag5", lambda ch: ch["flux"] - np.roll(ch["flux"], 5))
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Derived channel operators
# ---------------------------------------------------------------------------

def derive(
    name: str,
    fn: Callable[[Dict[str, float]], float],
    metric: str = "derived",
) -> Callable[[list], list]:
    """
    Create a new channel from a per-bin function of existing channels.

    fn receives a dict of {channel_name: value} for a single bin and returns
    a float. Bins where any required channel is missing are skipped.
    The derived channel is appended alongside the source channels.

    Parameters
    ----------
    name : str
        Name for the derived channel.
    fn : callable
        Function(channel_values: dict[str, float]) -> float.
    metric : str
        Metric label for the derived records. Default: "derived".

    Examples
    --------
        ops.derive("flux_ratio", lambda ch: ch["flux_x"] / max(ch["flux_z"], 1.0))
        ops.derive("component_delta", lambda ch: ch["flux_x"] - ch["flux_z"])
        ops.derive("fused",           lambda ch: 0.6 * ch["A"] + 0.4 * ch["B"])
    """
    def _derive(binned: list) -> list:
        from collections import defaultdict
        from .binned import BinnedRecord

        # Group by bin_index → {channel: record}
        bins: dict = defaultdict(dict)
        for r in binned:
            bins[r.bin_index][r.channel] = r

        derived = []
        for bin_idx in sorted(bins):
            channel_map = bins[bin_idx]
            channel_values = {ch: r.value for ch, r in channel_map.items()}
            try:
                value = float(fn(channel_values))
            except (KeyError, ZeroDivisionError, ValueError):
                continue

            # Inherit metadata from the first available source record
            ref = next(iter(channel_map.values()))
            derived.append(BinnedRecord(
                bin_index=bin_idx,
                channel=name,
                keys=ref.keys,
                metric=metric,
                agg_func=ref.agg_func,
                value=value,
                n_events=ref.n_events,
                gap_before=None,
                seq_sum=None,
            ))

        return sorted(binned + derived, key=lambda r: r.bin_index)
    return _derive


def derive_temporal(
    name: str,
    fn: Callable[[Dict[str, "np.ndarray"]], "np.ndarray"],
    metric: str = "derived",
) -> Callable[[list], list]:
    """
    Create a new channel from a function of full time-axis arrays.

    fn receives a dict of {channel_name: np.ndarray} where each array contains
    the values for that channel across all bins (NaN for missing bins), and
    returns a np.ndarray of the same length. This enables lagged comparisons,
    rolling residuals, and any operation that needs the full time axis.

    Parameters
    ----------
    name : str
        Name for the derived channel.
    fn : callable
        Function(channel_arrays: dict[str, np.ndarray]) -> np.ndarray.
        Input arrays are aligned to a common bin_index axis (NaN-padded).
        Output must have the same length.
    metric : str
        Metric label for the derived records. Default: "derived".

    Examples
    --------
        import numpy as np

        ops.derive_temporal("flux_lag5",
            lambda ch: ch["flux"] - np.roll(ch["flux"], 5))

        ops.derive_temporal("flux_residual",
            lambda ch: ch["flux"] - np.convolve(
                ch["flux"], np.ones(10)/10, mode="same"))
    """
    def _derive_temporal(binned: list) -> list:
        from collections import defaultdict
        from .binned import BinnedRecord

        # Build per-channel index → record maps
        channel_records: dict = defaultdict(dict)
        for r in binned:
            channel_records[r.channel][r.bin_index] = r

        if not channel_records:
            return binned

        # Common bin axis across all channels
        all_bins = sorted({r.bin_index for r in binned})
        bin_to_pos = {b: i for i, b in enumerate(all_bins)}
        n = len(all_bins)

        # Build aligned arrays (NaN where channel has no record for that bin)
        arrays: Dict[str, np.ndarray] = {}
        ref_records: Dict[str, dict] = {}
        for ch, rec_map in channel_records.items():
            arr = np.full(n, np.nan)
            for bin_idx, rec in rec_map.items():
                arr[bin_to_pos[bin_idx]] = rec.value
            arrays[ch] = arr
            ref_records[ch] = rec_map

        try:
            result_arr = fn(arrays)
        except (KeyError, ValueError) as e:
            raise ValueError(f"derive_temporal fn raised: {e}") from e

        if len(result_arr) != n:
            raise ValueError(
                f"derive_temporal fn returned array of length {len(result_arr)}, "
                f"expected {n}"
            )

        # Emit one BinnedRecord per non-NaN output position
        ref = next(iter(next(iter(ref_records.values())).values()))
        derived = []
        for pos, bin_idx in enumerate(all_bins):
            value = float(result_arr[pos])
            if np.isnan(value):
                continue
            derived.append(BinnedRecord(
                bin_index=bin_idx,
                channel=name,
                keys=ref.keys,
                metric=metric,
                agg_func=ref.agg_func,
                value=value,
                n_events=0,
                gap_before=None,
                seq_sum=None,
            ))

        return sorted(binned + derived, key=lambda r: r.bin_index)
    return _derive_temporal
