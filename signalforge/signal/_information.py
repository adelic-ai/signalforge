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
# Surface operations
# ---------------------------------------------------------------------------

def entropy_surface(surface: Surface) -> np.ndarray:
    """Compute entropy at each (scale, time) cell of a surface.

    Uses the count data from the surface. At each cell, the "distribution"
    is the event count — entropy measures how concentrated vs spread
    the events are within each window.

    For richer entropy (distribution of values, not just counts),
    use entropy_from_signal which builds per-window histograms.

    Parameters
    ----------
    surface : Surface
        Must have 'count' in surface.data.

    Returns
    -------
    np.ndarray
        2D array (n_scales, n_time) of entropy values in bits.
    """
    count_data = surface.data.get("count")
    if count_data is None:
        raise ValueError("Surface must have 'count' data for entropy computation")

    n_scales, n_time = count_data.shape
    result = np.full((n_scales, n_time), np.nan)

    for s in range(n_scales):
        row = count_data[s]
        valid = ~np.isnan(row)
        if not valid.any():
            continue

        # Sliding entropy: for each position, compute entropy of the
        # window centered/starting at that position
        # Since count_data already has the windowed counts, we compute
        # entropy of the local distribution of counts across neighboring bins
        for t in range(n_time):
            if not valid[t]:
                continue
            # Local window: the count at this cell represents the
            # total events in one window. The entropy of a single count
            # is 0. We need the distribution WITHIN the window.
            # This requires the sub-bin counts, which we compute from
            # the dense binned data.
            result[s, t] = 0.0  # placeholder — needs sub-bin distribution

    return result


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
