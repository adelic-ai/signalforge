"""
signalforge.signal._features

Feature extraction from segments and entity signals.

Works with segments from _segment.py. Provides:
- Per-entity interleaved signal extraction
- Segment labeling and comparison
- Segment-to-feature vector conversion for ML

    from signalforge.signal import segment_features, segments_to_matrix
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ._complex import RealSignal
from ._measure import measure_signal
from ..lattice.sampling import SamplingPlan


# ---------------------------------------------------------------------------
# Per-entity signals
# ---------------------------------------------------------------------------

def entity_signals(records: list, entity_keys: list = None) -> Dict[tuple, list]:
    """Group records by entity, preserving all channels interleaved.

    Unlike records_to_signals (which splits by channel), this keeps
    all events from one entity together in time order.

    Parameters
    ----------
    records : list of Record
    entity_keys : list of str, optional
        Which keys define an entity. Default: all keys from records.

    Returns
    -------
    dict
        {entity_tuple: [records sorted by primary_order]}
    """
    entities: Dict[tuple, list] = defaultdict(list)

    for r in records:
        if entity_keys:
            key = tuple((k, str(r.keys.get(k, ""))) for k in entity_keys)
        else:
            key = tuple(sorted(r.keys.items())) if r.keys else (("_all", "_all"),)
        entities[key].append(r)

    for key in entities:
        entities[key].sort(key=lambda r: r.primary_order)

    return entities


def entity_channel_matrix(records: list, entity_keys: list = None,
                          channels: list = None, cbin: int = 1) -> Dict[tuple, dict]:
    """Build per-entity, per-channel count time series.

    Parameters
    ----------
    records : list of Record
    entity_keys : list of str
    channels : list of str, optional
        Default: all channels found.
    cbin : int

    Returns
    -------
    dict
        {entity_tuple: {channel_name: RealSignal, "_total": RealSignal}}
    """
    entities = entity_signals(records, entity_keys)

    if channels is None:
        channels = sorted({r.channel for r in records})

    result = {}
    for entity_key, events in entities.items():
        if not events:
            continue

        orders = [r.primary_order for r in events]
        min_o, max_o = min(orders), max(orders)
        n_bins = (max_o - min_o) // cbin + 1

        ch_signals = {}
        total = np.zeros(n_bins)

        for ch in channels:
            counts = np.zeros(n_bins)
            for r in events:
                if r.channel == ch:
                    b = (r.primary_order - min_o) // cbin
                    counts[b] += 1
            total += counts
            ch_signals[ch] = RealSignal(
                np.arange(n_bins), counts, channel=ch,
                keys=dict(entity_key),
            )

        ch_signals["_total"] = RealSignal(
            np.arange(n_bins), total, channel="_total",
            keys=dict(entity_key),
        )

        result[entity_key] = ch_signals

    return result


# ---------------------------------------------------------------------------
# Segment labeling
# ---------------------------------------------------------------------------

def label_segments(segments: list, rules: dict = None) -> list:
    """Label segments by applying rules.

    Parameters
    ----------
    segments : list of Segment
    rules : dict, optional
        {label: function(segment) -> bool}. First match wins.
        Default: basic heuristics (single event, short burst, etc.)

    Returns
    -------
    list of (Segment, str)
    """
    if rules is None:
        rules = {
            "single_event": lambda s: s.event_count == 1,
            "short_burst": lambda s: s.duration < 5 and s.event_count > 1,
            "normal": lambda s: True,
        }

    labeled = []
    for seg in segments:
        assigned = "unknown"
        for label, rule in rules.items():
            if rule(seg):
                assigned = label
                break
        labeled.append((seg, assigned))

    return labeled


def segment_summary(labeled: list) -> dict:
    """Summarize labeled segments.

    Returns
    -------
    dict
        {label: {count, mean_duration, mean_events, mean_channels}}
    """
    groups: Dict[str, list] = defaultdict(list)
    for seg, label in labeled:
        groups[label].append(seg)

    summary = {}
    for label, segs in groups.items():
        durations = [s.duration for s in segs]
        all_channels: Dict[str, list] = defaultdict(list)
        for s in segs:
            for ch, cnt in s.channels.items():
                all_channels[ch].append(cnt)

        summary[label] = {
            "count": len(segs),
            "mean_duration": float(np.mean(durations)),
            "std_duration": float(np.std(durations)),
            "mean_events": float(np.mean([s.event_count for s in segs])),
            "mean_channels": {ch: float(np.mean(v)) for ch, v in all_channels.items()},
        }

    return summary


def print_segment_summary(summary: dict) -> None:
    """Print segment summary."""
    print()
    print(f"  Segment Summary")
    print(f"  {'─' * 60}")
    print(f"  {'label':<20} {'count':>6} {'mean_dur':>10} {'mean_evt':>10}")
    print(f"  {'─' * 60}")
    for label, s in sorted(summary.items(), key=lambda x: -x[1]["count"]):
        chs = ", ".join(f"{ch}:{v:.1f}" for ch, v in s["mean_channels"].items())
        print(f"  {label:<20} {s['count']:>6} {s['mean_duration']:>10.1f} {s['mean_events']:>10.1f}  [{chs}]")
    print(f"  {'─' * 60}")
    print()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def segment_features(segment, channels: list = None,
                     plan: SamplingPlan = None, cbin: int = 1) -> dict:
    """Convert one segment into a feature dictionary.

    Parameters
    ----------
    segment : Segment
    channels : list of str, optional
    plan : SamplingPlan, optional
        For surface-derived features.
    cbin : int

    Returns
    -------
    dict
        Feature name → float.
    """
    features = {}

    features["duration"] = float(segment.duration)
    features["event_count"] = float(segment.event_count)
    features["event_rate"] = float(segment.event_count / max(segment.duration, 1))

    if channels is None:
        channels = sorted(segment.channels.keys())

    total = max(segment.event_count, 1)
    for ch in channels:
        cnt = segment.channels.get(ch, 0)
        features[f"count_{ch}"] = float(cnt)
        features[f"ratio_{ch}"] = float(cnt / total)

    active = sum(1 for ch in channels if segment.channels.get(ch, 0) > 0)
    features["channel_diversity"] = float(active / max(len(channels), 1))

    # Surface features
    if plan is not None and segment.events:
        orders = np.array([e.primary_order for e in segment.events])
        min_o = orders.min()
        n_bins = (orders.max() - min_o) // cbin + 1

        if n_bins > 2:
            counts = np.zeros(n_bins)
            for o in orders:
                counts[(o - min_o) // cbin] += 1

            sig = RealSignal(np.arange(n_bins), counts, channel="_segment")
            try:
                surface = measure_signal(sig, plan, agg=["mean", "std"])
                mean_arr = surface.data["mean"]
                for i, w in enumerate(plan.windows[:min(len(plan.windows), mean_arr.shape[0])]):
                    row = mean_arr[i]
                    valid = row[np.isfinite(row)]
                    if len(valid) > 1 and np.std(valid) > 0:
                        z = (valid - np.mean(valid)) / np.std(valid)
                        features[f"peak_z_scale_{w}"] = float(np.max(np.abs(z)))
            except Exception:
                pass

    return features


def segments_to_matrix(segments: list, channels: list = None,
                       plan: SamplingPlan = None, cbin: int = 1) -> Tuple[np.ndarray, list, list]:
    """Convert all segments into a feature matrix for ML.

    Parameters
    ----------
    segments : list of Segment
    channels : list of str
    plan : SamplingPlan, optional
    cbin : int

    Returns
    -------
    matrix : np.ndarray (n_segments, n_features)
    feature_names : list of str
    segment_info : list of dict
    """
    all_features = []
    info = []

    for seg in segments:
        feat = segment_features(seg, channels=channels, plan=plan, cbin=cbin)
        all_features.append(feat)
        info.append({"entity": seg.entity, "start": seg.start, "duration": seg.duration})

    if not all_features:
        return np.array([]), [], []

    all_names = sorted(set().union(*(f.keys() for f in all_features)))

    matrix = np.zeros((len(all_features), len(all_names)))
    for i, feat in enumerate(all_features):
        for j, name in enumerate(all_names):
            matrix[i, j] = feat.get(name, 0.0)

    return matrix, all_names, info
