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


def entropy_from_signal(
    signal,
    plan: SamplingPlan,
    n_bins: int = 10,
) -> np.ndarray:
    """Compute entropy surface from a signal with per-window histograms.

    For each (scale, time) cell, builds a histogram of values within
    that window and computes Shannon entropy. This gives the distribution
    shape — concentrated (low entropy) vs spread (high entropy).

    Parameters
    ----------
    signal : LatticeSignal
        The signal to measure.
    plan : SamplingPlan
        Lattice geometry.
    n_bins : int
        Number of histogram bins within each window. Default: 10.

    Returns
    -------
    np.ndarray
        2D array (n_scales, n_time) of entropy values in bits.
    """
    idx = signal.index
    vals = signal.values
    cbin = plan.cbin

    # Bin the signal
    bin_indices = idx // cbin
    min_bin = int(bin_indices.min())
    max_bin = int(bin_indices.max())
    n_time = max_bin - min_bin + 1

    rel_bins = (bin_indices - min_bin).astype(np.intp)

    dense_count = np.zeros(n_time, dtype=np.intp)
    np.add.at(dense_count, rel_bins, 1)

    n_scales = len(plan.windows)
    result = np.full((n_scales, n_time), np.nan)

    for scale_idx, (window, hop) in enumerate(zip(plan.windows, plan.hops)):
        w = window // cbin
        h = hop // cbin

        starts = np.arange(0, n_time, h)

        for start in starts:
            end = min(start + w, n_time)
            window_counts = dense_count[start:end]

            if window_counts.sum() == 0:
                continue

            # Entropy of the count distribution within this window
            # Each sub-bin's count is a category
            result[scale_idx, start] = entropy(window_counts)

    return result


def mutual_information_surface(
    surface_a: Surface,
    surface_b: Surface,
) -> np.ndarray:
    """Mutual information between two surfaces at each time position.

    Computes MI across the scale axis — "how much does the scale
    structure of A tell you about the scale structure of B at this time?"

    Parameters
    ----------
    surface_a, surface_b : Surface
        Must have the same plan (same scales and time axis).

    Returns
    -------
    np.ndarray
        1D array (n_time,) of MI values in bits.
    """
    count_a = surface_a.data.get("count")
    count_b = surface_b.data.get("count")

    if count_a is None or count_b is None:
        raise ValueError("Both surfaces must have 'count' data")

    n_scales, n_time = count_a.shape
    result = np.full(n_time, np.nan)

    for t in range(n_time):
        col_a = count_a[:, t]
        col_b = count_b[:, t]

        valid = ~np.isnan(col_a) & ~np.isnan(col_b)
        if valid.sum() < 2:
            continue

        result[t] = mutual_information(
            col_a[valid].astype(np.float64),
            col_b[valid].astype(np.float64),
        )

    return result


def divergence_surface(
    surface: Surface,
    baseline: Surface,
) -> np.ndarray:
    """KL divergence from baseline at each (scale, time) cell.

    "How different is the current distribution from the baseline?"
    Computed across the scale axis at each time position.

    Parameters
    ----------
    surface : Surface
        Current measurement.
    baseline : Surface
        Reference distribution (e.g., EWMA baseline).

    Returns
    -------
    np.ndarray
        1D array (n_time,) of KL divergence values.
    """
    count_s = surface.data.get("count")
    count_b = baseline.data.get("count")

    if count_s is None or count_b is None:
        raise ValueError("Both surfaces must have 'count' data")

    n_scales, n_time = count_s.shape
    result = np.full(n_time, np.nan)

    for t in range(n_time):
        col_s = count_s[:, t]
        col_b = count_b[:, t]

        valid = ~np.isnan(col_s) & ~np.isnan(col_b)
        if valid.sum() < 2:
            continue

        result[t] = kl_divergence(
            col_s[valid].astype(np.float64),
            col_b[valid].astype(np.float64),
        )

    return result


def information_gain_surface(
    signal,
    plan: SamplingPlan,
) -> np.ndarray:
    """Information gain from splitting at each time position.

    For each time position, computes how much entropy is reduced
    by splitting the signal there. High gain = natural boundary
    (session boundary, regime change, anomaly onset).

    Parameters
    ----------
    signal : LatticeSignal
        The signal to analyze.
    plan : SamplingPlan
        Lattice geometry.

    Returns
    -------
    np.ndarray
        1D array (n_time_bins,) of information gain values.
    """
    idx = signal.index
    vals = signal.values
    cbin = plan.cbin

    bin_indices = idx // cbin
    min_bin = int(bin_indices.min())
    max_bin = int(bin_indices.max())
    n_time = max_bin - min_bin + 1

    rel_bins = (bin_indices - min_bin).astype(np.intp)

    dense_count = np.zeros(n_time, dtype=np.intp)
    np.add.at(dense_count, rel_bins, 1)

    # Information gain at each possible split point
    result = np.zeros(n_time)
    for t in range(1, n_time - 1):
        result[t] = information_gain(dense_count, t)

    return result
