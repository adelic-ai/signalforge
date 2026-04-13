"""
Tests for signalforge.signal._information — information-theoretic operations.
"""

import numpy as np
import pytest

from signalforge.signal import (
    entropy, joint_entropy, mutual_information, kl_divergence,
    information_gain, best_split,
    discover_scales, discover_plan,
    entropy_surface, mutual_information_surface,
    divergence_surface, information_gain_surface,
    RealSignal, Surface,
)
from signalforge.lattice.sampling import SamplingPlan


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_uniform(self):
        # Uniform over 4 bins: log2(4) = 2.0
        assert abs(entropy([1, 1, 1, 1]) - 2.0) < 1e-10

    def test_concentrated(self):
        # All in one bin: 0.0
        assert entropy([10, 0, 0, 0]) == 0.0

    def test_binary(self):
        # Two equal bins: log2(2) = 1.0
        assert abs(entropy([5, 5]) - 1.0) < 1e-10

    def test_empty(self):
        assert entropy([0, 0, 0]) == 0.0

    def test_single(self):
        assert entropy([7]) == 0.0

    def test_unequal(self):
        # [3, 1] -> p=[0.75, 0.25] -> H = -0.75*log2(0.75) - 0.25*log2(0.25)
        h = entropy([3, 1])
        expected = -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))
        assert abs(h - expected) < 1e-10


class TestJointEntropy:
    def test_joint_leq_sum(self):
        # H(A,B) <= H(A) + H(B) always
        a = np.array([3, 1, 2, 0])
        b = np.array([1, 2, 0, 3])
        h_joint = joint_entropy(a, b)
        assert h_joint <= entropy(a) + entropy(b) + 1e-10

    def test_independent(self):
        # Independent: H(A,B) = H(A) + H(B)
        # Use deterministic arrays that approximate independence
        a = np.array([1, 1, 0, 0])
        b = np.array([1, 0, 1, 0])
        h_joint = joint_entropy(a, b)
        # 4 distinct pairs, each count 1: entropy = log2(4) = 2.0
        assert abs(h_joint - 2.0) < 1e-10


class TestMutualInformation:
    def test_self_mi_positive(self):
        # MI of signal with itself is positive
        a = np.array([3, 1, 2, 4])
        mi = mutual_information(a, a)
        assert mi > 0

    def test_independent(self):
        # Independent: MI = 0
        a = np.array([1, 1, 0, 0])
        b = np.array([1, 0, 1, 0])
        mi = mutual_information(a, b)
        assert abs(mi) < 1e-10

    def test_non_negative(self):
        a = np.array([5, 2, 0, 3, 1])
        b = np.array([1, 4, 2, 0, 3])
        mi = mutual_information(a, b)
        assert mi >= -1e-10  # MI is non-negative


class TestKLDivergence:
    def test_identical(self):
        p = np.array([3, 2, 5])
        assert kl_divergence(p, p) == 0.0

    def test_non_negative(self):
        p = np.array([5, 2, 3])
        q = np.array([3, 3, 4])
        assert kl_divergence(p, q) >= -1e-10

    def test_asymmetric(self):
        p = np.array([5, 1])
        q = np.array([1, 5])
        # KL(P||Q) != KL(Q||P) in general
        d_pq = kl_divergence(p, q)
        d_qp = kl_divergence(q, p)
        assert d_pq > 0
        assert d_qp > 0
        # They're equal in this symmetric case
        assert abs(d_pq - d_qp) < 1e-10

    def test_empty(self):
        assert kl_divergence([0, 0], [1, 1]) == 0.0


class TestInformationGain:
    def test_perfect_split(self):
        # [10, 0, 0, 10] split at 2: left=[10,0], right=[0,10]
        # H(parent) = 1.0, H(left) = 0, H(right) = 0, gain = 1.0
        g = information_gain(np.array([10, 0, 0, 10]), 2)
        assert abs(g - 1.0) < 1e-10

    def test_uniform_has_gain(self):
        # Uniform [5,5,5,5] split at 2: parent H=2.0, children H=1.0 each
        # gain = 2.0 - 1.0 = 1.0 (splitting reduces entropy by halving the space)
        g = information_gain(np.array([5, 5, 5, 5]), 2)
        assert abs(g - 1.0) < 1e-10

    def test_edge_split(self):
        assert information_gain(np.array([1, 2, 3]), 0) == 0.0
        assert information_gain(np.array([1, 2, 3]), 3) == 0.0


class TestBestSplit:
    def test_obvious_split(self):
        # Clear gap in the middle
        counts = np.array([10, 10, 0, 0, 10, 10])
        pos, gain = best_split(counts, min_size=1)
        assert 2 <= pos <= 4  # Should split at the gap
        assert gain > 0

    def test_concentrated_no_gain(self):
        # All events in one bin — no split helps
        counts = np.array([100, 0, 0, 0])
        pos, gain = best_split(counts, min_size=1)
        assert gain < 1e-10


# ---------------------------------------------------------------------------
# Surface operations
# ---------------------------------------------------------------------------

def _make_signal(n=500, channel="test"):
    """Create a simple test signal."""
    idx = np.arange(n)
    vals = np.zeros(n)
    # Bursty pattern: events at regular intervals with some clustering
    for i in range(0, n, 10):
        vals[i] = np.random.randint(1, 5)
    # Add a burst
    vals[200:210] = 10
    return RealSignal(idx, vals, channel=channel)


class TestEntropySurface:
    def test_returns_surface(self):
        sig = _make_signal()
        plan = SamplingPlan(360, 1)
        result = entropy_surface(sig, plan)
        assert isinstance(result, Surface)

    def test_has_entropy_data(self):
        sig = _make_signal()
        plan = SamplingPlan(360, 1)
        result = entropy_surface(sig, plan)
        assert "entropy" in result.data
        assert "count" in result.data

    def test_entropy_non_negative(self):
        sig = _make_signal()
        plan = SamplingPlan(360, 1)
        result = entropy_surface(sig, plan)
        ent = result.data["entropy"]
        valid = ent[~np.isnan(ent)]
        assert (valid >= -1e-10).all()

    def test_preserves_channel(self):
        sig = _make_signal(channel="4768")
        plan = SamplingPlan(360, 1)
        result = entropy_surface(sig, plan)
        assert result.channel == "4768"


class TestMISurface:
    def test_returns_surface(self):
        sig_a = _make_signal(channel="A")
        sig_b = _make_signal(channel="B")
        plan = SamplingPlan(360, 1)
        result = mutual_information_surface(sig_a, sig_b, plan)
        assert isinstance(result, Surface)

    def test_has_mi_data(self):
        sig_a = _make_signal(channel="A")
        sig_b = _make_signal(channel="B")
        plan = SamplingPlan(360, 1)
        result = mutual_information_surface(sig_a, sig_b, plan)
        assert "mi" in result.data
        assert "entropy_a" in result.data
        assert "entropy_b" in result.data

    def test_channel_label(self):
        sig_a = _make_signal(channel="4768")
        sig_b = _make_signal(channel="4769")
        plan = SamplingPlan(360, 1)
        result = mutual_information_surface(sig_a, sig_b, plan)
        assert "4768" in result.channel
        assert "4769" in result.channel


class TestDivergenceSurface:
    def test_returns_surface(self):
        sig = _make_signal(channel="current")
        baseline = _make_signal(channel="baseline")
        plan = SamplingPlan(360, 1)
        result = divergence_surface(sig, baseline, plan)
        assert isinstance(result, Surface)

    def test_has_kl_data(self):
        sig = _make_signal()
        baseline = _make_signal()
        plan = SamplingPlan(360, 1)
        result = divergence_surface(sig, baseline, plan)
        assert "kl_divergence" in result.data


class TestInformationGainSurface:
    def test_returns_surface(self):
        sig = _make_signal()
        plan = SamplingPlan(360, 1)
        result = information_gain_surface(sig, plan)
        assert isinstance(result, Surface)

    def test_has_ig_data(self):
        sig = _make_signal()
        plan = SamplingPlan(360, 1)
        result = information_gain_surface(sig, plan)
        assert "information_gain" in result.data
        assert "entropy" in result.data

    def test_gain_non_negative(self):
        sig = _make_signal()
        plan = SamplingPlan(360, 1)
        result = information_gain_surface(sig, plan)
        ig = result.data["information_gain"]
        valid = ig[~np.isnan(ig)]
        assert (valid >= -1e-10).all()


# ---------------------------------------------------------------------------
# Lattice walk
# ---------------------------------------------------------------------------

class TestDiscoverScales:
    def test_returns_list(self):
        sig = _make_signal(n=1000)
        scales = discover_scales(sig, horizon=360, grain=1)
        assert isinstance(scales, list)

    def test_each_entry_has_fields(self):
        sig = _make_signal(n=1000)
        scales = discover_scales(sig, horizon=360, grain=1)
        assert len(scales) > 0
        for s in scales:
            assert "window" in s
            assert "entropy" in s
            assert "gain" in s
            assert "bins" in s

    def test_sorted_coarsest_first(self):
        sig = _make_signal(n=1000)
        scales = discover_scales(sig, horizon=360, grain=1)
        windows = [s["window"] for s in scales]
        assert windows == sorted(windows, reverse=True)

    def test_respects_grain(self):
        sig = _make_signal(n=1000)
        scales = discover_scales(sig, horizon=360, grain=10)
        for s in scales:
            assert s["window"] >= 10

    def test_finds_structure(self):
        # Dense signal with obvious structure
        idx = np.arange(360)
        vals = np.zeros(360)
        # Busy first half, quiet second half
        vals[:180] = np.random.randint(1, 10, 180)
        sig = RealSignal(idx, vals, channel="structured")
        scales = discover_scales(sig, horizon=360, grain=1, min_gain=0.01)
        # Should find multiple scales
        assert len(scales) > 1
        # At least some should have gain
        gains = [s["gain"] for s in scales if s["bins"] >= 2]
        assert len(gains) > 0 and max(gains) > 0


class TestDiscoverPlan:
    def test_returns_sampling_plan(self):
        sig = _make_signal(n=1000)
        plan = discover_plan(sig, horizon=360, grain=1)
        assert isinstance(plan, SamplingPlan)

    def test_plan_has_windows(self):
        sig = _make_signal(n=1000)
        plan = discover_plan(sig, horizon=360, grain=1)
        assert len(plan.windows) > 0

    def test_windows_divide_horizon(self):
        sig = _make_signal(n=1000)
        plan = discover_plan(sig, horizon=360, grain=1)
        for w in plan.windows:
            assert plan.horizon % w == 0
