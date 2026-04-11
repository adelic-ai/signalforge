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
) -> Tuple[List[Segment], dict]:
    """Discover segments from records.

    Groups events by entity (keys), finds gaps between consecutive events,
    determines a gap threshold, and splits into Segments.

    Parameters
    ----------
    records : list of Record or CanonicalRecord
        Events with .primary_order, .channel, .keys
    gap_threshold : int, optional
        Gap size that defines a segment boundary. If None, estimated
        from the gap distribution.
    gap_method : str
        Estimation method for gap threshold. Default: freedman_diaconis.

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

    # Segment
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

    segments.sort(key=lambda s: s.start)

    # Stats
    durations = [s.duration for s in segments]
    stats = {
        "n_entities": len(entities),
        "n_segments": len(segments),
        "gap_threshold": gap_threshold,
        "mean_duration": float(np.mean(durations)) if durations else 0,
        "median_duration": float(np.median(durations)) if durations else 0,
        "std_duration": float(np.std(durations)) if durations else 0,
        "min_duration": int(np.min(durations)) if durations else 0,
        "max_duration": int(np.max(durations)) if durations else 0,
        "mean_events": float(np.mean([s.event_count for s in segments])) if segments else 0,
    }

    return segments, stats


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
