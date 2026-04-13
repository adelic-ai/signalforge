"""
signalforge.signal._segment

Segment discovery: find natural units of activity in event streams.

Any event stream has bursts of activity separated by gaps. A segment
is one burst — a cluster of events from one entity, bounded by silence.
Domains call them different things (sessions, epochs, cycles) but the
structure is universal.

    from signalforge.signal import discover_segments

    segments, stats = discover_segments(records)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class Segment:
    """One burst of activity from one entity."""
    entity: Dict[str, str]
    start: int
    end: int
    duration: int
    event_count: int
    channels: Dict[str, int]
    events: list = field(default_factory=list)

    @property
    def channel_ratio(self) -> Dict[str, float]:
        """Channel counts as fractions of total."""
        total = max(self.event_count, 1)
        return {ch: cnt / total for ch, cnt in self.channels.items()}

    def __repr__(self):
        chs = ", ".join(f"{k}:{v}" for k, v in self.channels.items())
        return (f"Segment({self.entity}, dur={self.duration}, "
                f"events={self.event_count}, [{chs}])")


def discover_segments(
    records: list,
    gap_threshold: Optional[int] = None,
    gap_method: str = "freedman_diaconis",
    method: str = "gap",
    min_gain: float = 0.1,
    min_segment_events: int = 2,
) -> Tuple[List[Segment], dict]:
    """Discover segments from records.

    Groups events by entity (keys) and splits into Segments using
    one of two methods:

    - "gap" (default): split where inter-event gaps exceed a threshold.
      Simple, fast, works well when sessions have clear pauses.
    - "information_gain": recursively split where Shannon entropy
      reduction is highest. No threshold needed — the data tells you
      where the natural boundaries are. Better for complex patterns
      where gap sizes vary.

    Parameters
    ----------
    records : list of Record or CanonicalRecord
        Events with .primary_order, .channel, .keys
    gap_threshold : int, optional
        Gap size for "gap" method. If None, estimated from distribution.
    gap_method : str
        Estimation method for gap threshold. Default: freedman_diaconis.
    method : str
        "gap" or "information_gain". Default: "gap".
    min_gain : float
        Minimum information gain to justify a split (for "information_gain").
        Default: 0.1 bits.
    min_segment_events : int
        Minimum events per segment (for "information_gain"). Default: 2.

    Returns
    -------
    segments : list of Segment
    stats : dict
        Discovery statistics.
    """
    # Group events by entity
    entities: Dict[tuple, list] = defaultdict(list)
    for r in records:
        key = tuple(sorted(r.keys.items())) if r.keys else (("_all", "_all"),)
        entities[key].append(r)

    for key in entities:
        entities[key].sort(key=lambda r: r.primary_order)

    if method == "gap":
        segments, method_stats = _discover_gap(
            entities, gap_threshold, gap_method)
    elif method == "information_gain":
        segments, method_stats = _discover_information_gain(
            entities, min_gain, min_segment_events)
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'gap' or 'information_gain'.")

    segments.sort(key=lambda s: s.start)

    # Stats
    durations = [s.duration for s in segments]
    stats = {
        "method": method,
        "n_entities": len(entities),
        "n_segments": len(segments),
        "mean_duration": float(np.mean(durations)) if durations else 0,
        "median_duration": float(np.median(durations)) if durations else 0,
        "std_duration": float(np.std(durations)) if durations else 0,
        "min_duration": int(np.min(durations)) if durations else 0,
        "max_duration": int(np.max(durations)) if durations else 0,
        "mean_events": float(np.mean([s.event_count for s in segments])) if segments else 0,
    }
    stats.update(method_stats)

    return segments, stats


def _discover_gap(
    entities: Dict[tuple, list],
    gap_threshold: Optional[int],
    gap_method: str,
) -> Tuple[List[Segment], dict]:
    """Gap-based segment discovery."""
    # Collect all inter-event gaps
    all_gaps = []
    for key, events in entities.items():
        orders = [r.primary_order for r in events]
        gaps = np.diff(orders)
        gaps = gaps[gaps > 0]
        all_gaps.extend(gaps.tolist())

    all_gaps = np.array(all_gaps)

    # Estimate gap threshold
    if gap_threshold is None and len(all_gaps) > 10:
        import binjamin as bj
        estimator = getattr(bj, gap_method)
        est = estimator(all_gaps)
        gap_threshold = int(max(est * 3, np.median(all_gaps) * 5))
    elif gap_threshold is None:
        gap_threshold = int(np.median(all_gaps) * 10) if len(all_gaps) > 0 else 100

    segments = []
    for key, events in entities.items():
        entity_dict = dict(key)
        current = [events[0]]

        for i in range(1, len(events)):
            gap = events[i].primary_order - events[i - 1].primary_order
            if gap > gap_threshold:
                segments.append(_make_segment(entity_dict, current))
                current = [events[i]]
            else:
                current.append(events[i])

        if current:
            segments.append(_make_segment(entity_dict, current))

    return segments, {"gap_threshold": gap_threshold}


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a count array. Bits."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _information_gain(counts: np.ndarray, split: int) -> float:
    """Information gain from splitting a count array at position."""
    if split <= 0 or split >= len(counts):
        return 0.0
    left = counts[:split]
    right = counts[split:]
    total = counts.sum()
    if total <= 0:
        return 0.0
    h_parent = _entropy(counts)
    w_left = left.sum() / total
    w_right = right.sum() / total
    return float(h_parent - w_left * _entropy(left) - w_right * _entropy(right))


def _discover_information_gain(
    entities: Dict[tuple, list],
    min_gain: float,
    min_segment_events: int,
) -> Tuple[List[Segment], dict]:
    """Information-gain-based segment discovery.

    For each entity, bin events at a data-derived resolution, then
    recursively split at the point of maximum information gain.
    Stop when gain drops below min_gain or segments are too small.

    The bin resolution is estimated from inter-event gaps (FD estimator)
    — the same statistic the gap method uses, but here it becomes the
    analysis grain rather than a threshold.
    """
    # Estimate bin resolution from all gaps
    all_gaps = []
    for key, events in entities.items():
        orders = [r.primary_order for r in events]
        gaps = np.diff(sorted(orders))
        gaps = gaps[gaps > 0]
        all_gaps.extend(gaps.tolist())

    if len(all_gaps) > 10:
        import binjamin as bj
        bin_size = int(max(bj.freedman_diaconis(np.array(all_gaps)), 1))
    else:
        bin_size = int(np.median(all_gaps)) if all_gaps else 1

    segments = []
    total_splits = 0

    for key, events in entities.items():
        entity_dict = dict(key)

        if len(events) < min_segment_events:
            segments.append(_make_segment(entity_dict, events))
            continue

        # Bin events at estimated resolution
        orders = np.array([r.primary_order for r in events])
        min_o, max_o = orders.min(), orders.max()

        if min_o == max_o:
            segments.append(_make_segment(entity_dict, events))
            continue

        n_bins = (max_o - min_o) // bin_size + 1
        counts = np.zeros(n_bins, dtype=np.float64)
        for o in orders:
            counts[(o - min_o) // bin_size] += 1

        # Find split points recursively
        split_positions = []
        _recursive_split(counts, 0, n_bins, min_gain, min_segment_events,
                         split_positions)
        split_positions.sort()

        # Convert split positions to event boundaries
        # Split positions are in bin-space; map back to event indices
        if not split_positions:
            segments.append(_make_segment(entity_dict, events))
        else:
            total_splits += len(split_positions)
            # Convert bin positions to primary_order thresholds
            thresholds = [min_o + pos * bin_size for pos in split_positions]

            current = []
            thresh_idx = 0
            for ev in events:
                if (thresh_idx < len(thresholds) and
                        ev.primary_order >= thresholds[thresh_idx]):
                    if current:
                        segments.append(_make_segment(entity_dict, current))
                    current = [ev]
                    thresh_idx += 1
                else:
                    current.append(ev)
            if current:
                segments.append(_make_segment(entity_dict, current))

    return segments, {"min_gain": min_gain, "total_splits": total_splits,
                      "bin_size": bin_size}


def _recursive_split(
    counts: np.ndarray,
    offset: int,
    length: int,
    min_gain: float,
    min_events: int,
    result: list,
    max_depth: int = 20,
) -> None:
    """Recursively find split points by maximum information gain."""
    if length < 2 * min_events or max_depth <= 0:
        return

    window = counts[offset:offset + length]
    if window.sum() < 2 * min_events:
        return

    # Find best split
    best_pos = 0
    best_gain = 0.0
    for i in range(min_events, length - min_events + 1):
        left_sum = window[:i].sum()
        right_sum = window[i:].sum()
        if left_sum < min_events or right_sum < min_events:
            continue
        g = _information_gain(window, i)
        if g > best_gain:
            best_gain = g
            best_pos = i

    if best_gain < min_gain:
        return

    # Record the split point (in absolute position)
    result.append(offset + best_pos)

    # Recurse on both halves
    _recursive_split(counts, offset, best_pos,
                     min_gain, min_events, result, max_depth - 1)
    _recursive_split(counts, offset + best_pos, length - best_pos,
                     min_gain, min_events, result, max_depth - 1)


def _make_segment(entity: dict, events: list) -> Segment:
    channels: Dict[str, int] = defaultdict(int)
    for e in events:
        channels[e.channel] += 1
    return Segment(
        entity=entity,
        start=events[0].primary_order,
        end=events[-1].primary_order,
        duration=events[-1].primary_order - events[0].primary_order,
        event_count=len(events),
        channels=dict(channels),
        events=events,
    )


def segments_to_signals(segments: list, cbin: int = 1) -> list:
    """Convert segments to RealSignals for surfacing.

    Parameters
    ----------
    segments : list of Segment
    cbin : int
        Bin size. Default: 1 (event-order).

    Returns
    -------
    list of RealSignal
    """
    from ._complex import RealSignal

    signals = []
    for i, seg in enumerate(segments):
        if not seg.events:
            continue

        orders = np.array([e.primary_order for e in seg.events])
        min_o = orders.min()
        max_o = orders.max()
        n_bins = (max_o - min_o) // cbin + 1

        counts = np.zeros(n_bins)
        for o in orders:
            counts[(o - min_o) // cbin] += 1

        signals.append(RealSignal(
            index=np.arange(n_bins),
            values=counts,
            channel=f"segment_{i}",
            keys=seg.entity,
            metadata={
                "start": seg.start, "end": seg.end,
                "duration": seg.duration, "channels": seg.channels,
            },
        ))

    return signals


def print_stats(stats: dict) -> None:
    """Print segment discovery statistics."""
    print()
    print(f"  Segment Discovery")
    print(f"  {'─' * 40}")
    print(f"  entities        {stats['n_entities']}")
    print(f"  segments        {stats['n_segments']}")
    print(f"  gap threshold   {stats['gap_threshold']}")
    print(f"  mean duration   {stats['mean_duration']:.1f}")
    print(f"  median duration {stats['median_duration']:.1f}")
    print(f"  events/segment  {stats['mean_events']:.1f}")
    print(f"  {'─' * 40}")
    print()
