"""
signalforge.analysis

Domain-agnostic analysis engine.

Takes Records from any domain, runs the IL backbone, returns results.
This is the substrate — all analysis logic lives here. Domain modules
only handle ingest and provide an AnalysisSchema that steers the IL.

Contract:
    analyze(records, schema=None) → dict
    summary(records) → dict

AnalysisSchema is optional. Without it, the engine groups signals
by the schema's categorical axes generically. With it, the engine
uses domain-specific families, groupings, and join keys.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .signal import (
    records_to_signals, discover_segments,
    entropy as shannon_entropy, mutual_information as mi_func,
)
from .signal._segment import Segment
from .signal._information import discover_scales, discover_plan
from .signal._complex import RealSignal
from .lattice.sampling import SamplingPlan

import binjamin as bj


# ---------------------------------------------------------------------------
# AnalysisSchema — domain-provided steering for the IL
# ---------------------------------------------------------------------------

@dataclass
class AnalysisSchema:
    """Domain-specific steering for the analysis engine.

    Provided by domain modules (e.g., cloudtrail.ANALYSIS_SCHEMA).
    The analysis engine is generic — this tells it how to group,
    join, and label things for a specific domain.

    Attributes
    ----------
    signal_group_key : str
        Axis name to group signals by for within-group MI
        (e.g., "eventSource" for CloudTrail).
    entity_group_key : str
        Axis name for entity grouping (e.g., "userIdentity").
    entity_classifier : callable, optional
        Function(entity_value) → group_label. Classifies entities
        into types (e.g., "Root", "IAMUser", "AWSService").
        If None, no entity type grouping.
    families : dict, optional
        Event code families: {name: {"events": [...], "join_keys": [...]}}.
        If provided, records are classified into families for per-family analysis.
    family_classifier : callable, optional
        Function(record) → family_name. If None, uses families dict to classify.
    summary_aliases : dict, optional
        Maps axis names to human-readable names for summary output.
        e.g., {"userIdentity": "users", "eventName": "actions"}
    """
    signal_group_key: str = ""
    entity_group_key: str = ""
    entity_classifier: Optional[Callable[[str], str]] = None
    families: Dict[str, Dict] = field(default_factory=dict)
    family_classifier: Optional[Callable] = None
    summary_aliases: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Summary — quick stats from records (domain-agnostic)
# ---------------------------------------------------------------------------

def summary(records: List, schema: Optional[AnalysisSchema] = None) -> Dict[str, Any]:
    """Quick summary statistics from any set of Records.

    Parameters
    ----------
    records : list of Record
    schema : AnalysisSchema, optional
        If provided, adds aliased keys to the result.
    """
    if not records:
        return {"total_events": 0}

    rec_schema = records[0].schema
    epochs = [r.primary_order for r in records]
    span_hours = (max(epochs) - min(epochs)) / 3600

    cat_counters = {}
    for axis in rec_schema.categorical_axes:
        cat_counters[axis.name] = Counter()

    error_axis = None
    for axis in rec_schema.categorical_axes:
        if "error" in axis.name.lower():
            error_axis = axis.name
            break

    errors = 0
    for r in records:
        v = r.values
        for axis in rec_schema.categorical_axes:
            val = v.get(axis.name, "")
            if val:
                cat_counters[axis.name][val] += 1
        if error_axis and v.get(error_axis):
            errors += 1

    result = {
        "total_events": len(records),
        "time_span_hours": round(span_hours, 1),
        "error_rate": round(errors / len(records), 4) if records else 0,
    }

    # Per-axis stats
    for name, counter in cat_counters.items():
        result[f"unique_{name}"] = len(counter)
        result[f"top_{name}"] = counter.most_common(10)

    # Apply aliases from AnalysisSchema
    aliases = schema.summary_aliases if schema else {}
    for axis_name, alias in aliases.items():
        if axis_name in cat_counters:
            result[f"unique_{alias}"] = len(cat_counters[axis_name])
            result[f"top_{alias}"] = cat_counters[axis_name].most_common(10)

    return result


# ---------------------------------------------------------------------------
# Sampling plan (fallback when IL discovery isn't used)
# ---------------------------------------------------------------------------

def default_sampling_plan(
    records: Optional[List] = None,
    horizon: int = 86400,
    grain: int = 60,
) -> SamplingPlan:
    """Build a SamplingPlan from data span or defaults."""
    if records:
        epochs = [r.primary_order for r in records]
        span = max(epochs) - min(epochs)
        if span > 0:
            n_days = max(1, (span // 86400) + 1)
            horizon = n_days * 86400

    cbin = bj.smallest_divisor_gte(horizon, grain)
    valid = set(bj.lattice_members(horizon, cbin))

    anchors = {60, 300, 900, 1800, 3600, 7200, 14400, 43200, 86400}
    if horizon > 86400:
        anchors.add(horizon)
        for d in [86400, 86400 * 7]:
            if d <= horizon and d in valid:
                anchors.add(d)

    fine_cutoff = 3600
    selected = sorted(
        w for w in valid
        if w in anchors or w <= fine_cutoff or w == horizon
    )
    if not selected:
        selected = sorted(valid)

    return SamplingPlan(horizon, grain, windows=selected)


# ---------------------------------------------------------------------------
# Metadata grouping (for MI pruning) — domain-agnostic
# ---------------------------------------------------------------------------

def _build_metadata_groups(
    records: List,
    signals: List,
    schema: Optional[AnalysisSchema] = None,
) -> Dict[str, Any]:
    """Build metadata groups from records for MI pruning.

    Uses AnalysisSchema if provided, otherwise falls back to generic
    grouping by the first non-channel categorical axis.
    """
    rec_schema = records[0].schema

    # Determine grouping keys
    if schema and schema.signal_group_key:
        signal_key = schema.signal_group_key
    else:
        # Fall back: first categorical axis that isn't the channel
        signal_key = None
        for axis in rec_schema.categorical_axes:
            if axis.name != rec_schema.channel_axis:
                signal_key = axis.name
                break

    entity_key = None
    if schema and schema.entity_group_key:
        entity_key = schema.entity_group_key
    elif rec_schema.group_by:
        entity_key = rec_schema.group_by[0]

    # Build channel → signal_group mapping from records
    channel_to_group = {}
    entity_to_type = {}

    for r in records:
        v = r.values
        ch = r.channel
        if signal_key and ch not in channel_to_group:
            channel_to_group[ch] = v.get(signal_key, "unknown")

        if entity_key:
            eid = v.get(entity_key, "")
            if eid and eid not in entity_to_type:
                if schema and schema.entity_classifier:
                    entity_to_type[eid] = schema.entity_classifier(eid)
                else:
                    entity_to_type[eid] = "default"

    # Group signals by signal_group_key
    signal_groups = defaultdict(list)
    for sig in signals:
        group = channel_to_group.get(sig.channel, "unknown")
        signal_groups[group].append(sig)

    # Group signals by entity type
    entity_groups = defaultdict(list)
    if entity_key:
        for sig in signals:
            eid = sig.keys.get(entity_key, "") if sig.keys else ""
            etype = entity_to_type.get(eid, "unknown")
            entity_groups[etype].append(sig)

    return {
        "signal_groups": dict(signal_groups),
        "entity_groups": dict(entity_groups),
    }


def _join_sessions(
    records: List,
    schema: AnalysisSchema,
) -> Tuple[List[Segment], Dict[str, Any]]:
    """Build sessions from records using key-based joins.

    Sessions are defined by the AnalysisSchema's families and their
    session_key. Within each family, records are grouped by the session
    key, then split into sessions by time proximity (using the family's
    natural gap — median inter-event interval × 3).

    No FD estimation. No binjamin. The keys define the session,
    time proximity just splits multiple sessions from the same entity.

    Parameters
    ----------
    records : list of Record
    schema : AnalysisSchema
        Must have families with session_key defined.

    Returns
    -------
    (sessions, stats) where sessions is list of Segment and stats is dict.
    """
    if not schema.families or not schema.family_classifier:
        return [], {"method": "key_join", "n_entities": 0, "n_segments": 0}

    # Classify records into families
    family_records: Dict[str, List] = defaultdict(list)
    for r in records:
        fam = schema.family_classifier(r)
        family_records[fam].append(r)

    all_sessions = []

    for fam_name, fam_info in schema.families.items():
        fam_recs = family_records.get(fam_name, [])
        if len(fam_recs) < 2:
            continue

        session_key = fam_info.get("session_key", "")
        if not session_key:
            continue

        # Group by session key (e.g., userIdentity)
        by_entity: Dict[str, List] = defaultdict(list)
        for r in fam_recs:
            entity_val = r.values.get(session_key, "unknown")
            by_entity[entity_val].append(r)

        for entity_val, entity_recs in by_entity.items():
            entity_recs.sort(key=lambda r: r.primary_order)

            if len(entity_recs) < 2:
                # Single event = single session
                r = entity_recs[0]
                all_sessions.append(Segment(
                    entity={session_key: entity_val},
                    start=r.primary_order,
                    end=r.primary_order,
                    duration=0,
                    event_count=1,
                    channels={r.channel: 1},
                    events=entity_recs,
                ))
                continue

            # Compute gap threshold from this entity's data
            epochs = np.array([r.primary_order for r in entity_recs])
            diffs = np.diff(epochs)
            nonzero = diffs[diffs > 0]
            if len(nonzero) > 0:
                gap_threshold = int(np.median(nonzero) * 3)
                gap_threshold = max(gap_threshold, 1)
            else:
                gap_threshold = 1

            # Split into sessions by gap
            session_start = 0
            for i in range(1, len(entity_recs)):
                gap = entity_recs[i].primary_order - entity_recs[i - 1].primary_order
                if gap > gap_threshold:
                    # Close current session
                    session_recs = entity_recs[session_start:i]
                    channels: Dict[str, int] = defaultdict(int)
                    for r in session_recs:
                        channels[r.channel] += 1
                    all_sessions.append(Segment(
                        entity={session_key: entity_val},
                        start=session_recs[0].primary_order,
                        end=session_recs[-1].primary_order,
                        duration=session_recs[-1].primary_order - session_recs[0].primary_order,
                        event_count=len(session_recs),
                        channels=dict(channels),
                        events=session_recs,
                    ))
                    session_start = i

            # Close last session
            session_recs = entity_recs[session_start:]
            if session_recs:
                channels = defaultdict(int)
                for r in session_recs:
                    channels[r.channel] += 1
                all_sessions.append(Segment(
                    entity={session_key: entity_val},
                    start=session_recs[0].primary_order,
                    end=session_recs[-1].primary_order,
                    duration=session_recs[-1].primary_order - session_recs[0].primary_order,
                    event_count=len(session_recs),
                    channels=dict(channels),
                    events=session_recs,
                ))

    all_sessions.sort(key=lambda s: s.start)

    # Unclassified records get generic gap-based segmentation
    unclassified = family_records.get("unclassified", [])

    # Stats
    durations = [s.duration for s in all_sessions]
    entities = set()
    for s in all_sessions:
        entities.update(s.entity.values())

    stats = {
        "method": "key_join",
        "n_entities": len(entities),
        "n_segments": len(all_sessions),
        "mean_duration": float(np.mean(durations)) if durations else 0,
        "median_duration": float(np.median(durations)) if durations else 0,
        "std_duration": float(np.std(durations)) if durations else 0,
        "min_duration": int(np.min(durations)) if durations else 0,
        "max_duration": int(np.max(durations)) if durations else 0,
        "mean_events": float(np.mean([s.event_count for s in all_sessions])) if all_sessions else 0,
        "gap_threshold": "per-entity (median×3)",
        "n_unclassified": len(unclassified),
        "families_with_sessions": len([f for f in schema.families if family_records.get(f)]),
    }

    return all_sessions, stats


def _compute_mi_pairs(
    signals: List,
    seen: set,
    max_per_group: int = 10,
    group_label: str = "within",
) -> List[Dict]:
    """Compute MI for top signals within a group."""
    top = sorted(signals, key=lambda s: len(s.index), reverse=True)[:max_per_group]
    pairs = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            sa, sb = top[i], top[j]
            pair_key = (id(sa), id(sb))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            mi_val = mi_func(sa.values, sb.values)
            if mi_val > 0:
                pairs.append({
                    "channel_a": sa.channel,
                    "keys_a": sa.keys,
                    "channel_b": sb.channel,
                    "keys_b": sb.keys,
                    "mi": round(mi_val, 4),
                    "group": group_label,
                })
    return pairs


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def _estimate_grain(epochs: np.ndarray, min_grain: int = 60) -> int:
    """Estimate a natural grain from inter-event intervals.

    Uses the median of non-zero gaps to find a representative
    inter-event spacing. Enforces a minimum grain to keep the
    lattice tractable (default: 60 seconds).

    For session-level analysis, the grain doesn't need to be
    finer than the typical gap between events. Going finer just
    adds empty bins.

    Parameters
    ----------
    epochs : np.ndarray
        Sorted epoch timestamps.
    min_grain : int
        Floor for estimated grain. Default: 60 (1 minute).
        Prevents grain=1 on data with second-resolution timestamps.

    Returns
    -------
    int
        Estimated grain. At least min_grain.
    """
    if len(epochs) < 2:
        return min_grain
    diffs = np.diff(epochs)
    nonzero = diffs[diffs > 0]
    if len(nonzero) == 0:
        return min_grain
    median_gap = int(np.median(nonzero))
    return max(min_grain, median_gap)


def _discover_family_scales(
    family_records: List,
    grain: int,
) -> Dict[str, Any]:
    """Discover scales for a single family's records.

    Returns dict with grain, scales, and plan for this family.
    """
    if len(family_records) < 5:
        return {"grain": grain, "scales": [], "plan": None}

    epochs = np.array(
        [r.primary_order for r in family_records], dtype=np.int64
    )
    epochs.sort()

    # Estimate grain from this family's data
    family_grain = _estimate_grain(epochs)
    # Use the coarser of estimated and requested grain
    effective_grain = max(family_grain, grain) if grain > 0 else family_grain

    epoch_min = int(epochs.min())
    il_horizon = 86400  # 1 day

    day_bins = il_horizon // effective_grain
    if day_bins < 2:
        return {"grain": effective_grain, "scales": [], "plan": None}

    day_dense = np.zeros(day_bins, dtype=np.float64)
    for e in epochs:
        sec_of_day = (int(e) - epoch_min) % il_horizon
        b = sec_of_day // effective_grain
        if b < day_bins:
            day_dense[b] += 1.0

    nz = np.nonzero(day_dense)[0]
    if len(nz) < 2:
        return {"grain": effective_grain, "scales": [], "plan": None}

    day_signal = RealSignal(
        index=nz.astype(np.int64),
        values=day_dense[nz],
        channel="_family",
    )

    scales = discover_scales(
        day_signal,
        horizon=day_bins,
        grain=1,
        min_gain=0.01,
        min_events=5,
    )

    # Convert bins back to seconds
    for sc in scales:
        sc["window_bins"] = sc["window"]
        sc["window"] = sc["window"] * effective_grain

    return {
        "grain": effective_grain,
        "scales": scales,
        "plan": None,  # built later if needed
    }


def analyze(
    records: List,
    grain: int = 0,
    top_n: int = 20,
    discover: bool = True,
    schema: Optional[AnalysisSchema] = None,
) -> Dict[str, Any]:
    """Run the IL backbone on any set of Records.

    Domain-agnostic. The optional AnalysisSchema steers grouping
    and classification without the engine knowing domain specifics.

    Parameters
    ----------
    records : list of Record
    grain : int
        Grain hint in seconds. Default: 0 (auto-detect from data).
        If > 0, used as a floor — the engine won't go finer.
        Per-family grains are discovered from inter-event intervals.
    top_n : int
        Number of top channels for MI analysis. Default: 20.
    discover : bool
        If True, use IL to discover natural scales. Default: True.
    schema : AnalysisSchema, optional
        Domain-specific steering. If None, uses generic grouping.
    """
    results: Dict[str, Any] = {}

    # 1. Summary
    results["summary"] = summary(records, schema)

    # 2. Estimate global grain from data if not provided
    all_epochs = np.array(
        [r.primary_order for r in records], dtype=np.int64
    )
    all_epochs.sort()

    if grain <= 0:
        grain = _estimate_grain(all_epochs)
    results["grain"] = grain

    # 3. Build signals
    signals = records_to_signals(records, agg="count")

    # 4. Per-family analysis with per-family grain discovery
    if schema and schema.family_classifier and discover:
        family_records = defaultdict(list)
        for r in records:
            fam = schema.family_classifier(r)
            family_records[fam].append(r)

        family_results = {}
        for name, fam_recs in family_records.items():
            if len(fam_recs) >= 5:
                fam_info = _discover_family_scales(fam_recs, grain)
                fam_info["count"] = len(fam_recs)
                fam_info["pct"] = round(len(fam_recs) / len(records), 4)
                family_results[name] = fam_info
        results["families"] = family_results
    else:
        results["families"] = {}

    # 5. Global scale discovery (aggregate, using estimated grain)
    if discover and signals:
        epoch_min = int(all_epochs.min())
        il_horizon = 86400
        day_bins = il_horizon // grain
        day_dense = np.zeros(day_bins, dtype=np.float64)
        for e in all_epochs:
            sec_of_day = (int(e) - epoch_min) % il_horizon
            b = sec_of_day // grain
            if b < day_bins:
                day_dense[b] += 1.0

        nz_day = np.nonzero(day_dense)[0]
        if len(nz_day) >= 2:
            day_signal = RealSignal(
                index=nz_day.astype(np.int64),
                values=day_dense[nz_day],
                channel="_aggregate_day",
            )
            scales = discover_scales(
                day_signal,
                horizon=day_bins,
                grain=1,
                min_gain=0.01,
                min_events=5,
            )
            for sc in scales:
                sc["window_bins"] = sc["window"]
                sc["window"] = sc["window"] * grain
        else:
            scales = []

        results["discovered_scales"] = scales

        if scales:
            discovered_windows = sorted(set(
                sc["window"] for sc in scales if sc["gain"] > 0.01
            ))
            if not discovered_windows:
                discovered_windows = [86400]
            plan = default_sampling_plan(records, grain=grain)
            plan_windows = [w for w in discovered_windows if w >= grain]
            if plan_windows:
                plan = SamplingPlan(
                    plan.horizon, grain,
                    windows=sorted(set(plan_windows + [plan.horizon]))
                )
        else:
            plan = default_sampling_plan(records, grain=grain)
        results["plan"] = plan
    else:
        results["discovered_scales"] = []
        results["plan"] = default_sampling_plan(records, grain=grain)

    # 6. Per-channel entropy
    channel_entropy = []
    for sig in signals:
        h = shannon_entropy(sig.values)
        channel_entropy.append({
            "channel": sig.channel,
            "keys": sig.keys,
            "entropy": round(h, 4),
            "events": len(sig.index),
        })
    channel_entropy.sort(key=lambda x: x["events"], reverse=True)
    results["channel_entropy"] = channel_entropy

    # 7. Session / segment discovery
    if schema and schema.families and schema.family_classifier:
        # Key-based session joins — no FD, no binjamin
        segments, seg_stats = _join_sessions(records, schema)
    else:
        # Fallback: gap-based discovery for data without known key structure
        segments, seg_stats = discover_segments(records)
    results["segments"] = segments
    results["segment_stats"] = seg_stats

    # 8. Metadata-informed pairwise MI
    meta_groups = _build_metadata_groups(records, signals, schema)
    signal_groups = meta_groups["signal_groups"]
    entity_groups = meta_groups["entity_groups"]

    pairwise_mi = []
    seen_pairs = set()

    for group, sigs in signal_groups.items():
        if len(sigs) >= 2:
            pairwise_mi.extend(_compute_mi_pairs(sigs, seen_pairs))

    for group, sigs in entity_groups.items():
        if len(sigs) >= 2:
            pairwise_mi.extend(_compute_mi_pairs(sigs, seen_pairs))

    bridge_signals = []
    for group, sigs in signal_groups.items():
        top = sorted(sigs, key=lambda s: len(s.index), reverse=True)
        if top:
            bridge_signals.append(top[0])

    bridge_signals = sorted(
        bridge_signals, key=lambda s: len(s.index), reverse=True
    )[:top_n]
    for i in range(len(bridge_signals)):
        for j in range(i + 1, len(bridge_signals)):
            sa, sb = bridge_signals[i], bridge_signals[j]
            pair_key = (id(sa), id(sb))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            mi_val = mi_func(sa.values, sb.values)
            if mi_val > 0:
                pairwise_mi.append({
                    "channel_a": sa.channel,
                    "keys_a": sa.keys,
                    "channel_b": sb.channel,
                    "keys_b": sb.keys,
                    "mi": round(mi_val, 4),
                    "group": "bridge",
                })

    pairwise_mi.sort(key=lambda x: x["mi"], reverse=True)
    results["pairwise_mi"] = pairwise_mi
    results["metadata"] = {
        "signal_groups": {k: len(v) for k, v in signal_groups.items()},
        "entity_groups": {k: len(v) for k, v in entity_groups.items()},
        "n_within_pairs": sum(1 for p in pairwise_mi if p["group"] == "within"),
        "n_bridge_pairs": sum(1 for p in pairwise_mi if p["group"] == "bridge"),
    }

    # 9. Scalemap — entropy across discovered scales, folded into 1 day
    results["scalemap"] = _compute_scalemap(records, results.get("discovered_scales", []),
                                             results.get("grain", 60))

    # 10. Temporal features — time-of-day distribution per entity
    results["temporal"] = _compute_temporal_features(records)

    # 11. Session-level scoring
    results["session_scores"] = _score_sessions(segments, total_events=results["summary"].get("total_events", 1))

    # 12. Source IP analysis — per-entity IP baseline
    results["source_ip"] = _compute_source_ip_features(records)

    # 13. Per-identity error rate
    results["per_identity_errors"] = _compute_per_identity_errors(records)

    # 14. Scoring layer — compress structure into ranked findings
    results["scores"], results["findings"] = _compute_scores_and_findings(results)

    return results


# ---------------------------------------------------------------------------
# Scalemap — entropy heatmap across scales × time-of-day
# ---------------------------------------------------------------------------

def _compute_scalemap(
    records: List,
    discovered_scales: List[Dict],
    grain: int,
) -> Dict[str, Any]:
    """Compute a scalemap: entropy at each discovered scale across time-of-day.

    Folds all events into a single 24h day, then computes entropy
    at each discovered window size. The result is a 2D array:
    scales × time_bins, showing where structure lives.

    Returns dict with 'matrix', 'scale_labels', 'time_labels'.
    """
    if not records or not discovered_scales or grain <= 0:
        return {}

    active_scales = [sc for sc in discovered_scales if sc.get("gain", 0) > 0.01]
    if not active_scales:
        return {}

    all_epochs = np.array([r.primary_order for r in records], dtype=np.int64)
    epoch_min = int(all_epochs.min())

    day_seconds = 86400
    day_bins = day_seconds // grain

    # Build folded density (all events into one 24h period)
    day_dense = np.zeros(day_bins, dtype=np.float64)
    for e in all_epochs:
        sec_of_day = (int(e) - epoch_min) % day_seconds
        b = sec_of_day // grain
        if b < day_bins:
            day_dense[b] += 1.0

    # Compute entropy at each scale
    n_scales = len(active_scales)
    # Use 24 time slots (hourly) for the scalemap display
    n_time_slots = 24
    slot_size = day_bins // n_time_slots

    matrix = np.zeros((n_scales, n_time_slots), dtype=np.float64)
    scale_labels = []

    for si, sc in enumerate(active_scales):
        w_bins = sc.get("window_bins", sc["window"] // grain)
        w_bins = max(w_bins, 1)

        w = sc["window"]
        if w >= 86400:
            scale_labels.append(f"{w // 86400}d")
        elif w >= 3600:
            scale_labels.append(f"{w // 3600}h")
        elif w >= 60:
            scale_labels.append(f"{w // 60}m")
        else:
            scale_labels.append(f"{w}s")

        for ti in range(n_time_slots):
            start = ti * slot_size
            end = min(start + max(w_bins, slot_size), day_bins)
            chunk = day_dense[start:end]
            if chunk.sum() > 0:
                from .signal._information import entropy as _entropy
                matrix[si, ti] = _entropy(chunk)

    time_labels = [f"{h:02d}:00" for h in range(24)]

    return {
        "matrix": matrix.tolist(),
        "scale_labels": scale_labels,
        "time_labels": time_labels,
        "n_scales": n_scales,
        "n_time_slots": n_time_slots,
    }


# ---------------------------------------------------------------------------
# Temporal features — time-of-day distribution per entity
# ---------------------------------------------------------------------------

def _compute_temporal_features(records: List) -> Dict[str, Any]:
    """Compute time-of-day activity distribution per entity.

    Returns dict with per-entity hourly histograms and temporal
    anomaly scores (activity outside normal hours).
    """
    if not records:
        return {}

    # Group by entity
    entity_hours = defaultdict(lambda: np.zeros(24, dtype=np.float64))
    global_hours = np.zeros(24, dtype=np.float64)

    for r in records:
        epoch = r.primary_order
        hour = (epoch % 86400) // 3600
        user = list(r.keys.values())[0] if r.keys else "unknown"
        entity_hours[user][hour] += 1
        global_hours[hour] += 1

    # Normalize global to get "expected" distribution
    total = global_hours.sum()
    if total > 0:
        global_norm = global_hours / total
    else:
        global_norm = np.ones(24) / 24

    # Per-entity temporal anomaly: KL divergence from global pattern
    from .signal._information import entropy as _entropy
    entity_temporal = {}
    for user, hours in entity_hours.items():
        user_total = hours.sum()
        if user_total < 5:
            continue
        user_norm = hours / user_total
        # Simple anomaly: how different from global distribution
        # Use sum of squared differences (chi-squared-like)
        mask = global_norm > 0
        diff = np.sum((user_norm[mask] - global_norm[mask]) ** 2 / global_norm[mask])
        # Peak hour
        peak_hour = int(np.argmax(hours))
        # Activity spread (entropy of hourly distribution)
        temporal_entropy = _entropy(hours)
        max_entropy = np.log2(24)

        short_user = user.split("/")[-1] if "/" in user else user
        entity_temporal[short_user] = {
            "peak_hour": peak_hour,
            "temporal_entropy": round(temporal_entropy, 3),
            "spread": round(temporal_entropy / max_entropy, 3) if max_entropy > 0 else 0,
            "anomaly": round(diff, 4),
            "hours": hours.tolist(),
        }

    return {
        "global_hours": global_hours.tolist(),
        "entity_temporal": entity_temporal,
    }


# ---------------------------------------------------------------------------
# Session-level scoring
# ---------------------------------------------------------------------------

def _score_sessions(segments: List, total_events: int = 1) -> List[Dict]:
    """Score sessions by size, duration, and channel diversity.

    Returns list of scored sessions, sorted by score descending.
    """
    if not segments:
        return []

    # Compute baselines
    event_counts = np.array([s.event_count for s in segments], dtype=np.float64)
    durations = np.array([s.duration for s in segments], dtype=np.float64)
    channel_counts = np.array([len(s.channels) for s in segments], dtype=np.float64)

    mean_events = np.mean(event_counts) if len(event_counts) else 1
    std_events = np.std(event_counts) if len(event_counts) > 1 else 1
    mean_dur = np.mean(durations) if len(durations) else 1
    std_dur = np.std(durations) if len(durations) > 1 else 1

    scored = []
    for seg in segments:
        # Z-scores
        ev_z = (seg.event_count - mean_events) / max(std_events, 0.01)
        dur_z = (seg.duration - mean_dur) / max(std_dur, 0.01)

        # Channel diversity (more channels = more complex session)
        diversity = len(seg.channels) / max(channel_counts.max(), 1)

        # Share of total activity
        share = seg.event_count / max(total_events, 1)

        # Combined
        raw = (
            0.35 * min(abs(ev_z), 5) / 5.0 +
            0.25 * min(abs(dur_z), 5) / 5.0 +
            0.20 * diversity +
            0.20 * np.log1p(share * 1000) / np.log1p(1000)
        )
        score = 1.0 / (1.0 + np.exp(-8 * (raw - 0.3)))

        entity_str = list(seg.entity.values())[0] if seg.entity else "unknown"
        if "/" in entity_str:
            entity_str = entity_str.split("/")[-1]

        top_ch = max(seg.channels, key=seg.channels.get) if seg.channels else ""

        scored.append({
            "entity": entity_str,
            "score": round(score, 4),
            "events": seg.event_count,
            "duration": seg.duration,
            "channels": len(seg.channels),
            "top_channel": top_ch,
            "event_z": round(ev_z, 2),
            "duration_z": round(dur_z, 2),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Source IP analysis
# ---------------------------------------------------------------------------

def _compute_source_ip_features(records: List) -> Dict[str, Any]:
    """Per-entity source IP baseline and anomaly detection.

    For each identity, tracks which IPs they use and flags:
    - New IPs (never seen for this identity)
    - High IP diversity (using many different IPs)
    - Shared IPs (same IP used by multiple identities)
    """
    if not records:
        return {}

    # Per-entity IP sets
    entity_ips = defaultdict(set)
    ip_entities = defaultdict(set)
    entity_ip_counts = defaultdict(lambda: defaultdict(int))

    for r in records:
        v = r.values
        user = v.get("userIdentity", v.get(r.schema.group_by[0] if r.schema.group_by else "", "unknown"))
        ip = v.get("sourceIPAddress", "")
        if not ip or ip == "unknown":
            continue
        if "/" in user:
            user = user.split("/")[-1]
        entity_ips[user].add(ip)
        ip_entities[ip].add(user)
        entity_ip_counts[user][ip] += 1

    # Compute per-entity features
    entity_features = {}
    for user, ips in entity_ips.items():
        counts = entity_ip_counts[user]
        total = sum(counts.values())
        top_ip = max(counts, key=counts.get)
        top_ip_pct = counts[top_ip] / total if total > 0 else 0

        entity_features[user] = {
            "unique_ips": len(ips),
            "total_events": total,
            "top_ip": top_ip,
            "top_ip_pct": round(top_ip_pct, 3),
            "ip_diversity": round(1.0 - top_ip_pct, 3),
        }

    # Shared IPs (used by 3+ identities)
    shared_ips = {
        ip: sorted(users)
        for ip, users in ip_entities.items()
        if len(users) >= 3
    }

    return {
        "entity_features": entity_features,
        "shared_ips": shared_ips,
        "total_unique_ips": len(ip_entities),
    }


# ---------------------------------------------------------------------------
# Per-identity error rate
# ---------------------------------------------------------------------------

def _compute_per_identity_errors(records: List) -> Dict[str, Any]:
    """Per-identity error rate analysis.

    Flags identities with error rates significantly above baseline.
    """
    if not records:
        return {}

    entity_total = defaultdict(int)
    entity_errors = defaultdict(int)
    entity_error_types = defaultdict(lambda: defaultdict(int))

    error_axis = None
    for axis in records[0].schema.categorical_axes:
        if "error" in axis.name.lower():
            error_axis = axis.name
            break

    if not error_axis:
        return {}

    for r in records:
        v = r.values
        user = v.get("userIdentity", v.get(r.schema.group_by[0] if r.schema.group_by else "", "unknown"))
        if "/" in user:
            user = user.split("/")[-1]
        entity_total[user] += 1
        error_code = v.get(error_axis, "")
        if error_code:
            entity_errors[user] += 1
            entity_error_types[user][error_code] += 1

    # Compute rates
    global_error_rate = sum(entity_errors.values()) / max(sum(entity_total.values()), 1)

    entity_rates = {}
    for user in entity_total:
        total = entity_total[user]
        errors = entity_errors.get(user, 0)
        rate = errors / total if total > 0 else 0

        # Flag if significantly above global rate
        flagged = rate > global_error_rate * 2 and errors >= 5

        entity_rates[user] = {
            "total": total,
            "errors": errors,
            "rate": round(rate, 4),
            "flagged": flagged,
            "top_errors": dict(sorted(
                entity_error_types.get(user, {}).items(),
                key=lambda x: -x[1]
            )[:5]),
        }

    flagged_entities = {
        user: info for user, info in entity_rates.items()
        if info["flagged"]
    }

    return {
        "global_error_rate": round(global_error_rate, 4),
        "entity_rates": entity_rates,
        "flagged_entities": flagged_entities,
    }


# ---------------------------------------------------------------------------
# Scoring layer
# ---------------------------------------------------------------------------

def _compute_scores_and_findings(results: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """Compute per-(user, channel) scores and generate ranked findings.

    Uses volume anomaly, entropy anomaly, rarity, MI relationships,
    and session characteristics to produce explainable risk scores.

    Returns (scores, findings) where:
        scores: list of {user, channel, score, components}
        findings: list of {severity, title, description, evidence}
    """
    ce = results.get("channel_entropy", [])
    pmi = results.get("pairwise_mi", [])
    segments = results.get("segments", [])
    s = results.get("summary", {})
    scales = results.get("discovered_scales", [])

    if not ce:
        return [], []

    total_events = s.get("total_events", 1)

    # --- Per-(user, channel) scoring ---

    # Compute baselines per channel (z-score normalization)
    channel_stats = defaultdict(lambda: {"events": [], "entropy": []})
    for entry in ce:
        ch = entry["channel"]
        channel_stats[ch]["events"].append(entry["events"])
        channel_stats[ch]["entropy"].append(entry["entropy"])

    channel_mean_events = {}
    channel_std_events = {}
    channel_mean_entropy = {}
    channel_std_entropy = {}
    for ch, stats in channel_stats.items():
        ev = np.array(stats["events"], dtype=np.float64)
        en = np.array(stats["entropy"], dtype=np.float64)
        channel_mean_events[ch] = float(np.mean(ev)) if len(ev) else 1.0
        channel_std_events[ch] = float(np.std(ev)) if len(ev) > 1 else 1.0
        channel_mean_entropy[ch] = float(np.mean(en)) if len(en) else 0.0
        channel_std_entropy[ch] = float(np.std(en)) if len(en) > 1 else 1.0

    # Per-identity baselines (z-score within identity)
    identity_stats = defaultdict(lambda: {"events": [], "entropy": []})
    for entry in ce:
        user = list(entry["keys"].values())[0] if entry["keys"] else "unknown"
        identity_stats[user]["events"].append(entry["events"])
        identity_stats[user]["entropy"].append(entry["entropy"])

    identity_mean_events = {}
    identity_std_events = {}
    for user, stats in identity_stats.items():
        ev = np.array(stats["events"], dtype=np.float64)
        identity_mean_events[user] = float(np.mean(ev)) if len(ev) else 1.0
        identity_std_events[user] = float(np.std(ev)) if len(ev) > 1 else 1.0

    # Count users per channel (rarity)
    channel_user_count = defaultdict(int)
    for entry in ce:
        channel_user_count[entry["channel"]] += 1

    total_users = s.get("unique_users", s.get("unique_userIdentity", 1))

    # Global event stats for percentile scoring
    all_events = np.array([e["events"] for e in ce], dtype=np.float64)
    all_entropy = np.array([e["entropy"] for e in ce], dtype=np.float64)

    scores = []
    for entry in ce:
        ch = entry["channel"]
        events = entry["events"]
        ent = entry["entropy"]
        user = list(entry["keys"].values())[0] if entry["keys"] else "unknown"

        # Volume z-score within channel (how unusual is this volume for this action?)
        std_ev = max(channel_std_events.get(ch, 1), 0.01)
        channel_z = (events - channel_mean_events.get(ch, 0)) / std_ev

        # Volume z-score within identity (how unusual is this action for this user?)
        std_id = max(identity_std_events.get(user, 1), 0.01)
        identity_z = (events - identity_mean_events.get(user, 0)) / std_id

        # Entropy anomaly: z-score of entropy within channel
        std_ent = max(channel_std_entropy.get(ch, 1), 0.01)
        entropy_z = abs(ent - channel_mean_entropy.get(ch, 0)) / std_ent

        # Rarity: how uncommon is this channel (sigmoid for smooth scaling)
        n_users = channel_user_count.get(ch, 1)
        rarity_raw = 1.0 - (n_users / max(total_users, 1))

        # Activity share (log-scaled for dynamic range)
        share = events / max(total_events, 1)
        log_share = np.log1p(share * 1000) / np.log1p(1000)  # 0-1 range, log-scaled

        # Combined score: sigmoid of weighted z-scores for good separation
        raw = (
            0.30 * min(abs(channel_z), 5) / 5.0 +    # channel volume anomaly
            0.20 * min(abs(identity_z), 5) / 5.0 +    # identity volume anomaly
            0.15 * min(entropy_z, 5) / 5.0 +           # entropy anomaly
            0.20 * rarity_raw +                         # rarity
            0.15 * log_share                            # activity share
        )
        # Sigmoid for 0-1 with good separation in the middle
        combined = 1.0 / (1.0 + np.exp(-8 * (raw - 0.4)))

        scores.append({
            "user": user,
            "channel": ch,
            "score": round(combined, 4),
            "events": events,
            "share": round(share, 4),
            "channel_z": round(channel_z, 2),
            "identity_z": round(identity_z, 2),
            "entropy_z": round(entropy_z, 2),
            "rarity": round(rarity_raw, 4),
        })

    scores.sort(key=lambda x: x["score"], reverse=True)

    # --- Findings generation ---
    findings = []

    # Finding: dominant identity
    if scores:
        top = scores[0]
        if top["share"] > 0.1:
            findings.append({
                "severity": "attention" if top["share"] > 0.3 else "info",
                "title": "Dominant Identity",
                "description": (
                    f"`{top['user']}` accounts for {top['share']:.0%} of all activity "
                    f"via `{top['channel']}` ({top['events']:,} events). "
                    f"Verify this identity is authorized and operating as expected."
                ),
                "evidence": {"user": top["user"], "channel": top["channel"],
                             "events": top["events"], "share": top["share"]},
            })

    # Finding: behavioral twins (users with very similar scores on same channels)
    user_vectors = defaultdict(dict)
    for sc in scores:
        user_vectors[sc["user"]][sc["channel"]] = sc["events"]

    users = list(user_vectors.keys())
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            u1, u2 = users[i], users[j]
            common_channels = set(user_vectors[u1]) & set(user_vectors[u2])
            if len(common_channels) < 3:
                continue
            # Compare event counts on common channels
            v1 = np.array([user_vectors[u1].get(ch, 0) for ch in common_channels])
            v2 = np.array([user_vectors[u2].get(ch, 0) for ch in common_channels])
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                continue
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            if cos_sim > 0.95:
                findings.append({
                    "severity": "attention",
                    "title": "Behavioral Twin Detected",
                    "description": (
                        f"`{_shorten_user(u1)}` and `{_shorten_user(u2)}` exhibit nearly identical "
                        f"behavior across {len(common_channels)} shared actions "
                        f"(similarity: {cos_sim:.2f}). This may indicate shared automation, "
                        f"shared role, or a shared credential surface."
                    ),
                    "evidence": {"user_a": u1, "user_b": u2,
                                 "similarity": round(cos_sim, 4),
                                 "common_channels": len(common_channels)},
                })

    # Finding: high error rate
    error_rate = s.get("error_rate", 0)
    if error_rate > 0.1:
        findings.append({
            "severity": "concern",
            "title": "Elevated Error Rate",
            "description": (
                f"{error_rate:.1%} of API calls returned errors. "
                f"This may indicate misconfigurations, permission issues, "
                f"or unauthorized access attempts."
            ),
            "evidence": {"error_rate": error_rate},
        })
    elif error_rate > 0.05:
        findings.append({
            "severity": "attention",
            "title": "Notable Error Rate",
            "description": f"{error_rate:.1%} of API calls returned errors.",
            "evidence": {"error_rate": error_rate},
        })

    # Finding: rare API usage
    for sc in scores:
        if sc["rarity"] > 0.9 and sc["events"] > 5:
            findings.append({
                "severity": "info",
                "title": "Rare API Usage",
                "description": (
                    f"`{_shorten_user(sc['user'])}` used `{sc['channel']}` "
                    f"({sc['events']} events) — an action used by very few identities. "
                    f"Uncommon API usage may indicate specialized tooling or reconnaissance."
                ),
                "evidence": {"user": sc["user"], "channel": sc["channel"],
                             "events": sc["events"], "rarity": sc["rarity"]},
            })
            if len([f for f in findings if f["title"] == "Rare API Usage"]) >= 3:
                break

    # Finding: strong cross-service coupling
    bridges = [p for p in pmi if p.get("group") == "bridge"]
    if bridges:
        top_bridge = bridges[0]
        findings.append({
            "severity": "info",
            "title": "Cross-Service Workflow",
            "description": (
                f"`{top_bridge['channel_a']}` and `{top_bridge['channel_b']}` are "
                f"strongly correlated across different services (MI={top_bridge['mi']:.2f}). "
                f"These actions tend to occur together as part of a workflow."
            ),
            "evidence": top_bridge,
        })

    # Finding: multi-scale structure
    active_scales = [sc for sc in scales if sc.get("gain", 0) > 0.01]
    if len(active_scales) > 5:
        scale_labels = []
        for sc in active_scales[:6]:
            w = sc["window"]
            if w >= 86400: scale_labels.append(f"{w//86400}d")
            elif w >= 3600: scale_labels.append(f"{w//3600}h")
            elif w >= 60: scale_labels.append(f"{w//60}m")
            else: scale_labels.append(f"{w}s")
        findings.append({
            "severity": "info",
            "title": "Rich Multi-Scale Structure",
            "description": (
                f"Activity shows meaningful patterns at {len(active_scales)} scales "
                f"({', '.join(scale_labels)}). This indicates layered behavioral "
                f"rhythms — daily cycles, shift patterns, automated schedules."
            ),
            "evidence": {"n_scales": len(active_scales), "scales": scale_labels},
        })

    # Sort findings by severity
    severity_order = {"concern": 0, "attention": 1, "info": 2}
    findings.sort(key=lambda f: severity_order.get(f["severity"], 3))

    return scores, findings


def _shorten_user(u: str, max_len: int = 30) -> str:
    if "/" in u:
        u = u.split("/")[-1]
    if len(u) > max_len:
        u = u[:max_len - 3] + "..."
    return u
