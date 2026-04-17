"""
signalforge.report

Rendering layer for analysis results.

Takes the dict returned by analyze() and produces output:
    print_analysis()  — CLI terminal output
    to_json()         — JSON-serializable dict (for API/web)
    describe()        — Quick summary for interactive use

Knows nothing about the math. Just formatting.
"""

from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_window(w: int) -> str:
    """Format a window size in human-readable time."""
    if w >= 86400:
        return f"{w // 86400}d"
    if w >= 3600:
        return f"{w // 3600}h"
    if w >= 60:
        return f"{w // 60}m"
    return f"{w}s"


def _shorten_arn(s: str, max_len: int = 20) -> str:
    """Shorten an AWS ARN to its last meaningful part."""
    if "::" in s or "/" in s:
        parts = s.replace("::", ":").split("/")
        s = parts[-1] if parts else s
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    return s


# ---------------------------------------------------------------------------
# CLI output
# ---------------------------------------------------------------------------

def print_analysis(results: Dict[str, Any], verbose: bool = False) -> None:
    """Print analysis results to terminal.

    Default fits one screen (~40 lines). Verbose expands all sections.

    Parameters
    ----------
    results : dict
        Output of analysis.analyze().
    verbose : bool
        Show full detail.
    """
    s = results["summary"]
    segments = results["segments"]
    seg_stats = results["segment_stats"]
    ce = results["channel_entropy"]
    pmi = results["pairwise_mi"]
    plan = results["plan"]

    # --- Header ---
    print()
    print(f"  SignalForge — Analysis")
    print(f"  {'═' * 60}")
    print(f"  {s['total_events']:,} events | "
          f"{s['time_span_hours']:.0f}h span | "
          f"{s.get('unique_users', '?')} users | "
          f"{s.get('unique_actions', '?')} actions | "
          f"{s.get('unique_services', '?')} services")
    detected_grain = results.get("grain", plan.grain)
    print(f"  Error rate: {s['error_rate']:.1%} | "
          f"Segments: {len(segments):,} | "
          f"Grain: {detected_grain}s (auto) | "
          f"{len(plan.windows)} scales")
    print(f"  {'─' * 60}")

    # --- Discovered scales ---
    scales = results.get("discovered_scales", [])
    if scales:
        active = [sc for sc in scales if sc["gain"] > 0]
        print()
        print(f"  Discovered Scales (IL lattice walk)")
        print(f"  {'─' * 60}")
        print(f"  {len(scales)} scales examined, {len(active)} with structure")
        n_show = 8 if not verbose else len(scales)
        for sc in scales[:n_show]:
            w = sc["window"]
            marker = " *" if sc["gain"] > 0.01 else ""
            print(f"    {_fmt_window(w):>6s} ({w:>8,}s)  H={sc['entropy']:>7.3f}  "
                  f"gain={sc['gain']:.4f}  bins={sc['bins']}{marker}")

        if verbose:
            pcs = results.get("per_channel_scales", {})
            if pcs:
                print()
                print(f"  Per-channel scale discovery (top channels):")
                for label, ch_scales in pcs.items():
                    active_ch = [s for s in ch_scales if s["gain"] > 0]
                    if active_ch:
                        windows = [s["window"] for s in active_ch]
                        win_str = ", ".join(_fmt_window(w) for w in windows[:6])
                        ch_name = label[:40] if len(label) <= 40 else label[:37] + "..."
                        print(f"    {ch_name:<40s} {len(active_ch)} scales: [{win_str}]")

    # --- Per-family breakdown ---
    families = results.get("families", {})
    if families:
        print()
        print(f"  Families ({len(families)} detected)")
        print(f"  {'─' * 60}")
        for name, info in sorted(families.items(), key=lambda x: -x[1].get("count", 0)):
            count = info.get("count", 0)
            pct = info.get("pct", 0)
            fam_grain = info.get("grain", "?")
            fam_scales = info.get("scales", [])
            active = len([s for s in fam_scales if s.get("gain", 0) > 0.01])
            print(f"  {name:<20s} {count:>7,} ({pct:>5.1%})  "
                  f"grain={fam_grain}s  {active} scales")

        if verbose:
            for name, info in sorted(families.items(), key=lambda x: -x[1].get("count", 0)):
                fam_scales = info.get("scales", [])
                active = [s for s in fam_scales if s.get("gain", 0) > 0.01]
                if active:
                    windows = [_fmt_window(s["window"]) for s in active[:6]]
                    print(f"    {name}: [{', '.join(windows)}]")

    # --- Channel entropy ---
    n_channels = 10 if not verbose else 25
    print()
    print(f"  {'Channel':<28s} {'Events':>7s}  {'H(bits)':>7s}  {'User'}")
    print(f"  {'─' * 60}")
    for entry in ce[:n_channels]:
        keys_str = next(iter(entry["keys"].values()), "") if entry["keys"] else ""
        keys_str = _shorten_arn(keys_str)
        print(f"  {entry['channel']:<28s} {entry['events']:>7,}  "
              f"{entry['entropy']:>7.3f}  {keys_str}")

    # --- Segments ---
    print()
    print(f"  Segments")
    print(f"  {'─' * 60}")
    print(f"  {len(segments):,} sessions across {seg_stats.get('n_entities', '?')} entities | "
          f"gap threshold: {seg_stats.get('gap_threshold', '?')}s")
    print(f"  duration: mean={seg_stats.get('mean_duration', 0):.0f}s  "
          f"median={seg_stats.get('median_duration', 0):.0f}s  "
          f"max={seg_stats.get('max_duration', 0):.0f}s | "
          f"mean events/seg: {seg_stats.get('mean_events', 0):.1f}")

    if verbose and segments:
        top_segs = sorted(segments, key=lambda seg: seg.event_count, reverse=True)[:10]
        print()
        print(f"  Top segments by size:")
        for seg in top_segs:
            entity_str = next(iter(seg.entity.values()), "") if seg.entity else ""
            entity_str = _shorten_arn(entity_str)
            top_ch = max(seg.channels, key=seg.channels.get) if seg.channels else "?"
            print(f"    {entity_str:<20s} {seg.event_count:>5} events  "
                  f"{seg.duration:>6}s  [{top_ch}]")

    # --- Metadata grouping ---
    meta = results.get("metadata", {})
    if meta:
        print()
        print(f"  Metadata Structure")
        print(f"  {'─' * 60}")
        svc_g = meta.get("signal_groups", meta.get("service_groups", {}))
        utype_g = meta.get("entity_groups", meta.get("user_type_groups", {}))
        n_within = meta.get("n_within_pairs", 0)
        n_bridge = meta.get("n_bridge_pairs", 0)
        print(f"  {len(svc_g)} service groups | {len(utype_g)} user types | "
              f"{n_within} within-group pairs + {n_bridge} bridge pairs")
        if verbose:
            print()
            print(f"  Services ({len(svc_g)} groups):")
            for svc, count in sorted(svc_g.items(), key=lambda x: -x[1])[:10]:
                svc_short = svc.replace(".amazonaws.com", "")
                print(f"    {svc_short:<30s} {count:>4} signals")
            print()
            print(f"  User types:")
            for utype, count in sorted(utype_g.items(), key=lambda x: -x[1]):
                print(f"    {utype:<30s} {count:>4} signals")

    # --- Relationships ---
    n_mi = 10 if not verbose else 25
    print()
    print(f"  Relationships (mutual information)")
    print(f"  {'─' * 60}")
    if pmi:
        for entry in pmi[:n_mi]:
            group_tag = f" [{entry.get('group', '')}]" if verbose else ""
            print(f"  {entry['channel_a']:<25s} <-> {entry['channel_b']:<25s}  "
                  f"MI={entry['mi']:.3f}{group_tag}")
    else:
        print(f"  No significant relationships found.")

    if verbose:
        top_services = s.get("top_services", s.get("top_eventSource", []))
        if top_services:
            print()
            print(f"  Services")
            print(f"  {'─' * 60}")
            for name, count in top_services:
                svc = name.replace(".amazonaws.com", "")
                print(f"  {svc:<30s} {count:>8,}")

        if s["error_rate"] > 0:
            print()
            print(f"  Errors")
            print(f"  {'─' * 60}")
            print(f"  {s['error_rate']:.1%} of all events returned an error")

    # --- Footer ---
    print()
    print(f"  Plan: H={plan.horizon:,}s  g={plan.grain}s  cbin={plan.cbin}s  "
          f"windows={len(plan.windows)}")
    print(f"  {'═' * 60}")
    print()


# ---------------------------------------------------------------------------
# JSON output (for API / web)
# ---------------------------------------------------------------------------

def to_json(results: Dict[str, Any], analysis_id: str = "") -> Dict[str, Any]:
    """Convert analysis results to a JSON-serializable dict.

    Parameters
    ----------
    results : dict
        Output of analysis.analyze().
    analysis_id : str
        Optional ID for this analysis.

    Returns
    -------
    dict
        JSON-serializable. No numpy, no custom objects.
    """
    import time

    plan = results["plan"]
    seg_stats = results["segment_stats"]
    segments = results["segments"]

    return {
        "id": analysis_id,
        "timestamp": time.time(),
        "summary": results["summary"],
        "channel_entropy": results["channel_entropy"],
        "segments": {
            "method": seg_stats.get("method", "unknown"),
            "n_entities": seg_stats.get("n_entities", 0),
            "n_segments": seg_stats.get("n_segments", len(segments)),
            "mean_duration": seg_stats.get("mean_duration", 0),
            "median_duration": seg_stats.get("median_duration", 0),
            "max_duration": seg_stats.get("max_duration", 0),
            "mean_events": seg_stats.get("mean_events", 0),
            "gap_threshold": seg_stats.get("gap_threshold", 0),
        },
        "pairwise_mi": results["pairwise_mi"],
        "metadata": results.get("metadata", {}),
        "discovered_scales": results.get("discovered_scales", []),
        "plan": {
            "horizon": plan.horizon,
            "grain": plan.grain,
            "cbin": plan.cbin,
            "n_windows": len(plan.windows),
            "windows": list(plan.windows),
        },
    }


# ---------------------------------------------------------------------------
# Quick describe (interactive use)
# ---------------------------------------------------------------------------

def describe(results: Dict[str, Any]) -> None:
    """Print a minimal one-screen summary."""
    print_analysis(results, verbose=False)
