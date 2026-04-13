"""
signalforge.signal._information

Information-theoretic operations on the lattice.

The lattice provides a principled bin structure. Entropy measured
over arbitrary bins is meaningless. Entropy measured over Div(H)
is structural — the bins are forced by arithmetic, not chosen.

Core operations:
    entropy(counts)          — Shannon entropy of a distribution
    joint_entropy(a, b)      — entropy of joint distribution
    mutual_information(a, b) — how much knowing A tells you about B
    kl_divergence(p, q)      — divergence from Q to P
    information_gain(counts, split) — entropy reduction from splitting

Surface operations:
    entropy_surface(surface)         — entropy at each (scale, time) cell
    mutual_information_surface(a, b) — MI between two surfaces
    information_gain_surface(surface) — where does splitting reduce entropy?
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from ._surface import Surface
from ..lattice.sampling import SamplingPlan


# ---------------------------------------------------------------------------
# Core information functions (pure numpy)
# ---------------------------------------------------------------------------

def entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a count array.

    H(X) = -sum(p * log2(p)) where p = counts / sum(counts)

    Parameters
    ----------
    counts : array-like
        Non-negative counts. Zeros are ignored.

    Returns
    -------
    float
        Entropy in bits. 0 = perfectly concentrated, log2(n) = uniform.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def joint_entropy(a: np.ndarray, b: np.ndarray) -> float:
    """Joint entropy of two count arrays.

    H(A, B) from the joint distribution. Inputs must be
    aligned — same time positions.

    Parameters
    ----------
    a, b : array-like
        Count arrays of the same length.

    Returns
    -------
    float
        Joint entropy in bits.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # Build joint histogram from paired values
    pairs = {}
    for va, vb in zip(a.flat, b.flat):
        key = (int(va), int(vb))
        pairs[key] = pairs.get(key, 0) + 1
    joint_counts = np.array(list(pairs.values()), dtype=np.float64)
    return entropy(joint_counts)


def mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """Mutual information between two count arrays.

    I(A; B) = H(A) + H(B) - H(A, B)

    How much knowing A reduces uncertainty about B (and vice versa).

    Parameters
    ----------
    a, b : array-like
        Count arrays of the same length.

    Returns
    -------
    float
        MI in bits. 0 = independent. min(H(A), H(B)) = one determines the other.
    """
    return entropy(a) + entropy(b) - joint_entropy(a, b)


def kl_divergence(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """KL divergence from Q to P.

    D_KL(P || Q) = sum(p * log2(p / q))

    How much P diverges from Q. Not symmetric.
    Use for: "how different is this window's distribution from baseline?"

    Parameters
    ----------
    p_counts, q_counts : array-like
        Count arrays. Must be the same length. Q must not have
        zeros where P has non-zeros (infinite divergence).

    Returns
    -------
    float
        KL divergence in bits. 0 = identical distributions.
    """
    p = np.asarray(p_counts, dtype=np.float64)
    q = np.asarray(q_counts, dtype=np.float64)

    p_total = p.sum()
    q_total = q.sum()
    if p_total <= 0 or q_total <= 0:
        return 0.0

    p_norm = p / p_total
    q_norm = q / q_total

    # Only where both are positive
    mask = (p_norm > 0) & (q_norm > 0)
    if not mask.any():
        return float('inf')

    return float(np.sum(p_norm[mask] * np.log2(p_norm[mask] / q_norm[mask])))


def information_gain(counts: np.ndarray, split: int) -> float:
    """Information gain from splitting a count array at a position.

    IG = H(parent) - weighted_avg(H(left), H(right))

    Parameters
    ----------
    counts : array-like
        Count array to split.
    split : int
        Position to split at (left = [:split], right = [split:]).

    Returns
    -------
    float
        Information gain in bits. Higher = more informative split.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if split <= 0 or split >= len(counts):
        return 0.0

    left = counts[:split]
    right = counts[split:]

    total = counts.sum()
    if total <= 0:
        return 0.0

    h_parent = entropy(counts)
    h_left = entropy(left)
    h_right = entropy(right)

    w_left = left.sum() / total
    w_right = right.sum() / total

    return float(h_parent - w_left * h_left - w_right * h_right)


def best_split(counts: np.ndarray, min_size: int = 1) -> Tuple[int, float]:
    """Find the split position that maximizes information gain.

    Parameters
    ----------
    counts : array-like
        Count array.
    min_size : int
        Minimum elements on each side of the split.

    Returns
    -------
    (split_position, gain)
    """
    counts = np.asarray(counts, dtype=np.float64)
    best_pos = 0
    best_gain = 0.0

    for i in range(min_size, len(counts) - min_size):
        g = information_gain(counts, i)
        if g > best_gain:
            best_gain = g
            best_pos = i

    return best_pos, best_gain


# ---------------------------------------------------------------------------
# Lattice walk — scale discovery
# ---------------------------------------------------------------------------

def discover_scales(
    signal,
    horizon: int,
    grain: int = 1,
    min_gain: float = 0.05,
    min_events: int = 2,
) -> list:
    """Walk the lattice top-down to find scales with information content.

    Starts at the coarsest scale (horizon) and divides by prime factors.
    At each scale, measures information gain from the split. Keeps
    descending where gain is above threshold, stops where it's not.

    This replaces FD-based grain estimation. The lattice provides the
    valid scales, information gain selects which ones matter.

    Parameters
    ----------
    signal : LatticeSignal
        The signal to analyze.
    horizon : int
        Outer boundary (largest window to consider).
    grain : int
        Finest resolution to descend to. Default: 1.
    min_gain : float
        Minimum information gain to justify descending. Default: 0.05 bits.
    min_events : int
        Minimum total events in a window to evaluate it. Default: 2.

    Returns
    -------
    list of dict
        One entry per discovered scale:
        {"window": int, "entropy": float, "gain": float, "bins": int}
        Sorted coarsest to finest.
    """
    from binjamin import factorize, divisors

    idx = signal.index
    vals = signal.values

    # Bin at grain resolution
    bin_indices = idx // grain
    min_bin = int(bin_indices.min())
    max_bin = int(bin_indices.max())
    n_bins = max_bin - min_bin + 1

    rel_bins = (bin_indices - min_bin).astype(np.intp)
    dense = np.zeros(n_bins, dtype=np.float64)
    np.add.at(dense, rel_bins, vals if len(vals.shape) == 1 else np.ones(len(vals)))

    total_events = dense.sum()
    if total_events < min_events:
        return []

    # Get all divisors of horizon that are multiples of grain
    all_scales = sorted([d for d in divisors(horizon) if d >= grain], reverse=True)

    discovered = []

    def _walk(window):
        if window < grain:
            return

        w_bins = window // grain
        if w_bins > n_bins:
            # Window larger than data — compute single entropy
            h = entropy(dense)
            discovered.append({
                "window": window,
                "entropy": h,
                "gain": 0.0,
                "bins": 1,
            })
            # Still try to descend
        else:
            # Compute entropy at this scale
            n_windows = n_bins // w_bins
            if n_windows < 1:
                return

            # Count events per window-sized bin
            window_counts = np.zeros(n_windows)
            for i in range(n_windows):
                window_counts[i] = dense[i * w_bins:(i + 1) * w_bins].sum()

            if window_counts.sum() < min_events:
                return

            h = entropy(window_counts)

            # Best split of the window counts
            if n_windows >= 2:
                _, gain = best_split(window_counts, min_size=1)
            else:
                gain = 0.0

            discovered.append({
                "window": window,
                "entropy": h,
                "gain": gain,
                "bins": n_windows,
            })

            if gain < min_gain:
                return

        # Descend: find the next valid scales below this one
        # These are divisors of this window that are also divisors of horizon
        factors = factorize(window // grain) if window > grain else {}
        children = set()
        for p in factors:
            child = window // p
            if child >= grain and child in set(all_scales):
                children.add(child)

        for child in sorted(children, reverse=True):
            _walk(child)

    _walk(horizon)

    # Deduplicate and sort
    seen = set()
    unique = []
    for entry in discovered:
        if entry["window"] not in seen:
            seen.add(entry["window"])
            unique.append(entry)

    unique.sort(key=lambda x: -x["window"])
    return unique


def discover_plan(
    signal,
    horizon: int,
    grain: int = 1,
    min_gain: float = 0.05,
) -> SamplingPlan:
    """Build a SamplingPlan from data using the lattice walk.

    No FD, no manual window selection. The lattice provides valid scales,
    information gain selects which ones carry structure.

    Parameters
    ----------
    signal : LatticeSignal
        The signal to analyze.
    horizon : int
        Outer boundary.
    grain : int
        Finest resolution. Default: 1.
    min_gain : float
        Minimum gain to include a scale. Default: 0.05.

    Returns
    -------
    SamplingPlan
        With windows selected by information content.
    """
    scales = discover_scales(signal, horizon, grain, min_gain)

    if not scales:
        # Fallback: use horizon as single window
        return SamplingPlan(horizon, grain)

    # Select windows where gain > 0 (they revealed structure)
    windows = [s["window"] for s in scales if s["gain"] > 0 or s["window"] == horizon]

    if not windows:
        windows = [horizon]

    # Ensure grain is included in lcm for clean horizon
    from math import gcd
    from functools import reduce
    all_for_lcm = windows + [grain]
    plan_horizon = reduce(lambda a, b: a * b // gcd(a, b), all_for_lcm)

    # Build plan with selected windows
    return SamplingPlan(plan_horizon, grain, windows=sorted(set(windows)))


# ---------------------------------------------------------------------------
# Surface operations
# ---------------------------------------------------------------------------

def _bin_signal(signal, cbin):
    """Bin a signal's events at cbin resolution. Returns dense_count and time info."""
    idx = signal.index
    bin_indices = idx // cbin
    min_bin = int(bin_indices.min())
    max_bin = int(bin_indices.max())
    n_time = max_bin - min_bin + 1
    rel_bins = (bin_indices - min_bin).astype(np.intp)
    dense_count = np.zeros(n_time, dtype=np.intp)
    np.add.at(dense_count, rel_bins, 1)
    time_axis = np.arange(min_bin, max_bin + 1, dtype=np.int64)
    return dense_count, time_axis, n_time


def entropy_surface(
    signal,
    plan: SamplingPlan,
) -> Surface:
    """Entropy surface: Shannon entropy of event distribution within each window.

    At each (scale, time) cell, measures how evenly events are spread
    across the sub-bins within that window. High entropy = spread,
    low entropy = concentrated.

    Parameters
    ----------
    signal : LatticeSignal
        The signal to measure.
    plan : SamplingPlan
        Lattice geometry.

    Returns
    -------
    Surface
        With data keys: "entropy", "count".
    """
    dense_count, time_axis, n_time = _bin_signal(signal, plan.cbin)
    cbin = plan.cbin
    n_scales = len(plan.windows)

    ent_arr = np.full((n_scales, n_time), np.nan)
    cnt_arr = np.full((n_scales, n_time), np.nan)

    for scale_idx, (window, hop) in enumerate(zip(plan.windows, plan.hops)):
        w = window // cbin
        h = hop // cbin
        starts = np.arange(0, n_time, h)

        for start in starts:
            end = min(start + w, n_time)
            window_counts = dense_count[start:end].astype(np.float64)
            total = window_counts.sum()
            if total <= 0:
                continue
            cnt_arr[scale_idx, start] = total
            ent_arr[scale_idx, start] = entropy(window_counts)

    scale_axis = tuple(w // cbin for w in plan.windows)
    return Surface(
        time_axis=time_axis,
        scale_axis=scale_axis,
        data={"entropy": ent_arr, "count": cnt_arr},
        channel=signal.channel,
        plan=plan,
        keys=signal.keys,
        metric="entropy",
        profile="information",
        coordinates=plan.coordinates,
    )


def mutual_information_surface(
    signal_a,
    signal_b,
    plan: SamplingPlan,
) -> Surface:
    """Mutual information between two signals at each (scale, time) cell.

    At each window position, computes MI between the sub-bin count
    distributions of the two signals. High MI = the signals' temporal
    patterns are correlated at that scale. Low MI = independent.

    Parameters
    ----------
    signal_a, signal_b : LatticeSignal
        Two signals to compare (e.g., channel 4768 and 4769).
    plan : SamplingPlan
        Lattice geometry. Both signals are binned with this plan.

    Returns
    -------
    Surface
        With data keys: "mi", "entropy_a", "entropy_b".
    """
    cbin = plan.cbin
    count_a, time_a, n_time_a = _bin_signal(signal_a, cbin)
    count_b, time_b, n_time_b = _bin_signal(signal_b, cbin)

    # Align to common time range
    min_bin = min(time_a[0], time_b[0])
    max_bin = max(time_a[-1], time_b[-1])
    n_time = int(max_bin - min_bin + 1)

    aligned_a = np.zeros(n_time, dtype=np.intp)
    aligned_b = np.zeros(n_time, dtype=np.intp)
    offset_a = int(time_a[0] - min_bin)
    offset_b = int(time_b[0] - min_bin)
    aligned_a[offset_a:offset_a + len(count_a)] = count_a
    aligned_b[offset_b:offset_b + len(count_b)] = count_b

    n_scales = len(plan.windows)
    mi_arr = np.full((n_scales, n_time), np.nan)
    ent_a_arr = np.full((n_scales, n_time), np.nan)
    ent_b_arr = np.full((n_scales, n_time), np.nan)

    for scale_idx, (window, hop) in enumerate(zip(plan.windows, plan.hops)):
        w = window // cbin
        h = hop // cbin
        starts = np.arange(0, n_time, h)

        for start in starts:
            end = min(start + w, n_time)
            wa = aligned_a[start:end].astype(np.float64)
            wb = aligned_b[start:end].astype(np.float64)

            if wa.sum() <= 0 and wb.sum() <= 0:
                continue

            ha = entropy(wa)
            hb = entropy(wb)
            ent_a_arr[scale_idx, start] = ha
            ent_b_arr[scale_idx, start] = hb
            mi_arr[scale_idx, start] = ha + hb - joint_entropy(wa, wb)

    time_axis = np.arange(min_bin, max_bin + 1, dtype=np.int64)
    scale_axis = tuple(w // cbin for w in plan.windows)

    ch_label = f"MI({signal_a.channel},{signal_b.channel})"
    keys = signal_a.keys if signal_a.keys == signal_b.keys else {}

    return Surface(
        time_axis=time_axis,
        scale_axis=scale_axis,
        data={"mi": mi_arr, "entropy_a": ent_a_arr, "entropy_b": ent_b_arr},
        channel=ch_label,
        plan=plan,
        keys=keys,
        metric="mutual_information",
        profile="information",
        coordinates=plan.coordinates,
    )


def divergence_surface(
    signal,
    signal_baseline,
    plan: SamplingPlan,
) -> Surface:
    """KL divergence surface: how different is the signal from a baseline?

    At each (scale, time) cell, computes KL divergence between the
    sub-bin count distribution of the signal and the baseline.
    High divergence = the distribution shape changed.

    Parameters
    ----------
    signal : LatticeSignal
        Current signal.
    signal_baseline : LatticeSignal
        Reference signal (e.g., historical baseline).
    plan : SamplingPlan
        Lattice geometry.

    Returns
    -------
    Surface
        With data keys: "kl_divergence".
    """
    cbin = plan.cbin
    count_s, time_s, n_time_s = _bin_signal(signal, cbin)
    count_b, time_b, n_time_b = _bin_signal(signal_baseline, cbin)

    min_bin = min(time_s[0], time_b[0])
    max_bin = max(time_s[-1], time_b[-1])
    n_time = int(max_bin - min_bin + 1)

    aligned_s = np.zeros(n_time, dtype=np.intp)
    aligned_b = np.zeros(n_time, dtype=np.intp)
    offset_s = int(time_s[0] - min_bin)
    offset_b = int(time_b[0] - min_bin)
    aligned_s[offset_s:offset_s + len(count_s)] = count_s
    aligned_b[offset_b:offset_b + len(count_b)] = count_b

    n_scales = len(plan.windows)
    kl_arr = np.full((n_scales, n_time), np.nan)

    for scale_idx, (window, hop) in enumerate(zip(plan.windows, plan.hops)):
        w = window // cbin
        h = hop // cbin
        starts = np.arange(0, n_time, h)

        for start in starts:
            end = min(start + w, n_time)
            ws = aligned_s[start:end].astype(np.float64)
            wb = aligned_b[start:end].astype(np.float64)

            if ws.sum() <= 0 or wb.sum() <= 0:
                continue

            kl_arr[scale_idx, start] = kl_divergence(ws, wb)

    time_axis = np.arange(min_bin, max_bin + 1, dtype=np.int64)
    scale_axis = tuple(w // cbin for w in plan.windows)

    return Surface(
        time_axis=time_axis,
        scale_axis=scale_axis,
        data={"kl_divergence": kl_arr},
        channel=signal.channel,
        plan=plan,
        keys=signal.keys,
        metric="kl_divergence",
        profile="information",
        coordinates=plan.coordinates,
    )


def information_gain_surface(
    signal,
    plan: SamplingPlan,
) -> Surface:
    """Information gain surface: where splitting reduces entropy most.

    At each (scale, time) cell, computes the information gain from
    splitting the window at its midpoint. High gain = the window
    contains a natural boundary (session edge, regime change).

    Parameters
    ----------
    signal : LatticeSignal
        The signal to analyze.
    plan : SamplingPlan
        Lattice geometry.

    Returns
    -------
    Surface
        With data keys: "information_gain", "entropy".
    """
    dense_count, time_axis, n_time = _bin_signal(signal, plan.cbin)
    cbin = plan.cbin
    n_scales = len(plan.windows)

    ig_arr = np.full((n_scales, n_time), np.nan)
    ent_arr = np.full((n_scales, n_time), np.nan)

    for scale_idx, (window, hop) in enumerate(zip(plan.windows, plan.hops)):
        w = window // cbin
        h = hop // cbin
        starts = np.arange(0, n_time, h)

        for start in starts:
            end = min(start + w, n_time)
            window_counts = dense_count[start:end].astype(np.float64)

            if window_counts.sum() <= 0:
                continue

            ent_arr[scale_idx, start] = entropy(window_counts)

            # Best split within this window
            best_gain = 0.0
            for split in range(1, len(window_counts)):
                g = information_gain(window_counts, split)
                if g > best_gain:
                    best_gain = g
            ig_arr[scale_idx, start] = best_gain

    scale_axis = tuple(w // cbin for w in plan.windows)
    return Surface(
        time_axis=time_axis,
        scale_axis=scale_axis,
        data={"information_gain": ig_arr, "entropy": ent_arr},
        channel=signal.channel,
        plan=plan,
        keys=signal.keys,
        metric="information_gain",
        profile="information",
        coordinates=plan.coordinates,
    )
