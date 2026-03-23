"""
Tests for signalforge.lattice — coordinates, FlipFlop, SamplingPlan, Neighborhood.
"""

import pytest

from signalforge.lattice import factorize, vec_add, to_int, FlipFlop, SamplingPlan
from signalforge.lattice.neighborhood import neighborhood


# ---------------------------------------------------------------------------
# factorize
# ---------------------------------------------------------------------------


def test_factorize_12():
    assert factorize(12) == {2: 2, 3: 1}


def test_factorize_1():
    assert factorize(1) == {}


def test_factorize_prime():
    assert factorize(7) == {7: 1}


def test_factorize_prime_power():
    assert factorize(8) == {2: 3}


def test_factorize_invalid():
    with pytest.raises(ValueError):
        factorize(0)


# ---------------------------------------------------------------------------
# vec_add
# ---------------------------------------------------------------------------


def test_vec_add_basic():
    result = vec_add({2: 1}, {2: 1, 3: 1})
    assert result == {2: 2, 3: 1}


def test_vec_add_disjoint():
    result = vec_add({2: 1}, {3: 1})
    assert result == {2: 1, 3: 1}


def test_vec_add_empty():
    result = vec_add({}, {2: 3})
    assert result == {2: 3}


# ---------------------------------------------------------------------------
# to_int
# ---------------------------------------------------------------------------


def test_to_int_12():
    assert to_int({2: 2, 3: 1}) == 12


def test_to_int_empty():
    assert to_int({}) == 1


def test_to_int_roundtrip():
    for n in [1, 2, 6, 12, 30, 360, 1800, 86400]:
        assert to_int(factorize(n)) == n


# ---------------------------------------------------------------------------
# FlipFlop
# ---------------------------------------------------------------------------


def test_flipflop_p2_seed():
    f = FlipFlop(2)
    assert f.sequence == (0, 1)
    assert f.prime == 2
    assert f.depth == 0


def test_flipflop_p2_flip_once():
    f = FlipFlop(2).flip()
    assert f.sequence == (0, 1, 0, 2)


def test_flipflop_p2_flip_twice():
    f = FlipFlop(2).flip().flip()
    assert f.sequence == (0, 1, 0, 2, 0, 1, 0, 3)


def test_flipflop_p3_seed():
    f = FlipFlop(3)
    assert f.sequence == (0, 0, 1)
    assert f.depth == 0


def test_flipflop_p3_flip_once():
    f = FlipFlop(3).flip()
    assert f.sequence == (0, 0, 1, 0, 0, 1, 0, 0, 2)


def test_flipflop_p2_callable_alias():
    f = FlipFlop(2)
    assert f().sequence == f.flip().sequence


def test_flipflop_coverage():
    f = FlipFlop(2)
    assert f.coverage == 2
    f2 = f.flip()
    assert f2.coverage == 4


def test_flipflop_expand_to():
    f = FlipFlop(2).expand_to(8)
    assert f.coverage >= 8


def test_flipflop_valuation_at():
    # v_2(4) = 2
    f = FlipFlop(2)
    assert f.valuation_at(4) == 2
    # v_3(9) = 2
    g = FlipFlop(3)
    assert g.valuation_at(9) == 2


# ---------------------------------------------------------------------------
# SamplingPlan
# ---------------------------------------------------------------------------


def test_sampling_plan_basic():
    plan = SamplingPlan(86400, 60)
    assert plan.cbin == 60
    assert plan.prime_basis == {2: 5, 3: 2, 5: 1}


def test_sampling_plan_horizon():
    plan = SamplingPlan(86400, 60)
    assert plan.horizon == 86400
    assert plan.grain == 60


def test_sampling_plan_all_windows_divide_horizon():
    plan = SamplingPlan(86400, 60)
    for w in plan.windows:
        assert plan.horizon % w == 0, f"window {w} does not divide horizon {plan.horizon}"


def test_sampling_plan_all_windows_multiple_of_cbin():
    plan = SamplingPlan(86400, 60)
    for w in plan.windows:
        assert w % plan.cbin == 0, f"window {w} is not a multiple of cbin {plan.cbin}"


def test_sampling_plan_dense_includes_horizon():
    plan = SamplingPlan(86400, 60)
    assert plan.horizon in plan.windows


def test_sampling_plan_small():
    plan = SamplingPlan(3600, 250)
    assert plan.cbin == 300
    assert plan.prime_basis == {2: 2, 3: 1}
    assert plan.windows == (300, 600, 900, 1200, 1800, 3600)


def test_sampling_plan_explicit_windows():
    plan = SamplingPlan(3600, 250, windows=[600, 1800, 3600])
    assert plan.windows == (600, 1800, 3600)


def test_sampling_plan_invalid_window_raises():
    with pytest.raises(ValueError):
        SamplingPlan(3600, 250, windows=[700])


def test_sampling_plan_immutable():
    plan = SamplingPlan(3600, 250)
    with pytest.raises(AttributeError):
        plan.horizon = 999


def test_sampling_plan_n_values():
    plan = SamplingPlan(3600, 250)
    for w, h, n in zip(plan.windows, plan.hops, plan.n_values):
        assert w == n * h


# ---------------------------------------------------------------------------
# Neighborhood
# ---------------------------------------------------------------------------


def test_neighborhood_anchor_and_shape():
    nb = neighborhood(12, 3)
    assert nb.anchor == 12
    assert nb.radius == 3
    assert 2 in nb.prime_basis
    assert 3 in nb.prime_basis


def test_neighborhood_column_36():
    nb = neighborhood(36, 0)
    col = nb.column(36)
    # 36 = 2^2 * 3^2
    i2 = nb.prime_basis.index(2)
    i3 = nb.prime_basis.index(3)
    assert int(col[i2]) == 2
    assert int(col[i3]) == 2


def test_neighborhood_prime_column():
    nb = neighborhood(7, 2)
    assert nb.is_prime_column(7)


def test_neighborhood_immutable():
    nb = neighborhood(12, 3)
    with pytest.raises(AttributeError):
        nb.anchor = 99


def test_neighborhood_radius_zero():
    nb = neighborhood(12, 0)
    assert nb.integers == (12,)
    assert nb.shape[1] == 1


def test_neighborhood_out_of_range_raises():
    nb = neighborhood(12, 3)
    with pytest.raises(ValueError):
        nb.column(1)  # 1 is below lo = max(1, 12-3) = 9
