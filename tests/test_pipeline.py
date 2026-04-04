"""
Tests for signalforge.pipeline — CanonicalRecord, materialize, measure,
engineer, assemble.

All tests use small synthetic data. No file I/O required.
"""

import math

import numpy as np
import pytest

from signalforge.signal import CanonicalRecord, OrderType, Surface
from signalforge.pipeline import (
    materialize,
    measure,
    engineer,
    assemble,
    FeatureTensor,
    FeatureBundle,
)
from signalforge.lattice import SamplingPlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_record(primary_order, value=1.0, channel="ch", metric="val", time_order=None):
    """Minimal CanonicalRecord with TIME order_type."""
    t = time_order if time_order is not None else primary_order
    return CanonicalRecord(
        primary_order=primary_order,
        order_type=OrderType.TIME,
        channel=channel,
        metric=metric,
        value=value,
        time_order=t,
    )


def make_plan(horizon=3600, grain=250):
    return SamplingPlan(horizon, grain)


# ---------------------------------------------------------------------------
# CanonicalRecord — construction and invariants
# ---------------------------------------------------------------------------


def test_canonical_record_basic():
    rec = make_record(100, value=3.14)
    assert rec.primary_order == 100
    assert rec.value == 3.14
    assert rec.channel == "ch"
    assert rec.metric == "val"
    assert rec.order_type == OrderType.TIME


def test_canonical_record_value_is_float():
    rec = make_record(0, value=7)
    assert isinstance(rec.value, float)


def test_canonical_record_negative_primary_order_raises():
    with pytest.raises(ValueError):
        CanonicalRecord(
            primary_order=-1,
            order_type=OrderType.TIME,
            channel="x",
            metric="y",
            value=0.0,
            time_order=0,
        )


def test_canonical_record_time_required_for_time_order():
    with pytest.raises(ValueError):
        CanonicalRecord(
            primary_order=0,
            order_type=OrderType.TIME,
            channel="x",
            metric="y",
            value=0.0,
            # time_order omitted intentionally
        )


def test_canonical_record_seq_required_for_sequence_order():
    with pytest.raises(ValueError):
        CanonicalRecord(
            primary_order=0,
            order_type=OrderType.SEQUENCE,
            channel="x",
            metric="y",
            value=0.0,
            # seq_order omitted
        )


def test_canonical_record_both_requires_delta():
    with pytest.raises(ValueError):
        CanonicalRecord(
            primary_order=0,
            order_type=OrderType.BOTH,
            channel="x",
            metric="y",
            value=0.0,
            time_order=0,
            seq_order=0,
            # order_delta omitted
        )


def test_canonical_record_keys_default_empty():
    rec = make_record(0)
    assert rec.keys == {}


def test_canonical_record_zero_primary_order_ok():
    rec = make_record(0)
    assert rec.primary_order == 0


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------


def test_materialize_basic():
    plan = make_plan()
    # cbin = 300 for horizon=3600, grain=250
    records = [
        make_record(0,   value=1.0),
        make_record(300, value=2.0),
        make_record(600, value=3.0),
    ]
    binned = materialize(records, plan)
    assert len(binned) == 3


def test_materialize_bin_index_formula():
    plan = make_plan()
    cbin = plan.cbin  # 300
    records = [make_record(cbin * 5, value=9.0)]
    binned = materialize(records, plan)
    assert len(binned) == 1
    assert binned[0].bin_index == 5


def test_materialize_same_bin_aggregated():
    plan = make_plan()
    cbin = plan.cbin
    # Two records in the same bin — default agg is "count"
    records = [
        make_record(cbin * 2, value=1.0),
        make_record(cbin * 2 + 1, value=1.0),
    ]
    binned = materialize(records, plan)
    # Both events land in the same bin
    bin_indices = [b.bin_index for b in binned]
    assert bin_indices.count(2) == 1
    b = next(b for b in binned if b.bin_index == 2)
    assert b.n_events == 2


def test_materialize_empty_returns_empty():
    plan = make_plan()
    assert materialize([], plan) == []


def test_materialize_preserves_channel_and_metric():
    plan = make_plan()
    records = [make_record(0, channel="sensor", metric="temp")]
    binned = materialize(records, plan)
    assert binned[0].channel == "sensor"
    assert binned[0].metric == "temp"


# ---------------------------------------------------------------------------
# measure — Surface
# ---------------------------------------------------------------------------


def _make_binned_records(plan, n_bins=30, value=1.0):
    """Produce synthetic BinnedRecords across n_bins."""
    from signalforge.pipeline.binned import BinnedRecord
    records = []
    for i in range(n_bins):
        records.append(BinnedRecord(
            bin_index=i,
            channel="ch",
            keys={},
            metric="val",
            agg_func="count",
            value=float(value),
            n_events=1,
            gap_before=0 if i > 0 else None,
            seq_sum=None,
        ))
    return records


def test_measure_returns_surfaces():
    plan = make_plan()
    binned = _make_binned_records(plan, n_bins=50)
    surfaces = measure(binned, plan, profile="continuous")
    assert len(surfaces) > 0
    assert all(isinstance(s, Surface) for s in surfaces)


def test_measure_surface_shape():
    plan = make_plan()
    n_bins = 50
    binned = _make_binned_records(plan, n_bins=n_bins)
    surfaces = measure(binned, plan, profile="continuous")
    s = surfaces[0]
    n_scales = len(plan.windows)
    n_time = n_bins  # time_axis spans from 0 to n_bins-1
    assert s.shape == (n_scales, n_time)


def test_measure_surface_has_profile_keys():
    plan = make_plan()
    binned = _make_binned_records(plan, n_bins=20)
    surfaces = measure(binned, plan, profile="continuous")
    s = surfaces[0]
    # "continuous" profile: mean, geometric_mean, median, std
    assert "mean" in s.data
    assert s.profile == "continuous"


def test_measure_groups_by_channel_metric():
    plan = make_plan()
    from signalforge.pipeline.binned import BinnedRecord

    def make_br(bin_index, channel, metric):
        return BinnedRecord(
            bin_index=bin_index,
            channel=channel,
            keys={},
            metric=metric,
            agg_func="count",
            value=1.0,
            n_events=1,
            gap_before=None,
            seq_sum=None,
        )

    binned = [make_br(i, "chA", "m1") for i in range(10)]
    binned += [make_br(i, "chB", "m2") for i in range(10)]
    surfaces = measure(binned, plan, profile="sparse")
    channels = {s.channel for s in surfaces}
    assert "chA" in channels
    assert "chB" in channels


def test_measure_empty_returns_empty():
    plan = make_plan()
    assert measure([], plan) == []


# ---------------------------------------------------------------------------
# engineer — FeatureTensor
# ---------------------------------------------------------------------------


def test_engineer_returns_feature_tensor():
    plan = make_plan()
    binned = _make_binned_records(plan, n_bins=50)
    surfaces = measure(binned, plan, profile="continuous")
    ft = engineer(surfaces[0], plan)
    assert isinstance(ft, FeatureTensor)


def test_engineer_feature_tensor_shape_matches_surface():
    plan = make_plan()
    binned = _make_binned_records(plan, n_bins=50)
    s = measure(binned, plan, profile="continuous")[0]
    ft = engineer(s, plan)
    assert ft.shape == s.shape


def test_engineer_derived_feature_names():
    plan = make_plan()
    binned = _make_binned_records(plan, n_bins=50)
    s = measure(binned, plan, profile="continuous")[0]
    ft = engineer(s, plan)
    names = ft.feature_names
    # For each raw agg, we expect baseline_ewma, baseline_rmed, residual,
    # zscore, delta, rate, gradient derivatives.
    assert "mean" in names
    assert "mean_baseline_ewma" in names
    assert "mean_residual" in names
    assert "mean_zscore" in names
    assert "mean_delta" in names
    assert "mean_gradient" in names


def test_engineer_feature_arrays_correct_dtype():
    plan = make_plan()
    binned = _make_binned_records(plan, n_bins=50)
    s = measure(binned, plan, profile="continuous")[0]
    ft = engineer(s, plan)
    for name, arr in ft.values.items():
        assert arr.dtype == np.float64, f"Feature {name!r} has dtype {arr.dtype}"


def test_engineer_feature_index_populated():
    plan = make_plan()
    binned = _make_binned_records(plan, n_bins=50)
    s = measure(binned, plan, profile="continuous")[0]
    ft = engineer(s, plan)
    assert len(ft.feature_index) == len(ft.values)


# ---------------------------------------------------------------------------
# assemble — FeatureBundle
# ---------------------------------------------------------------------------


def test_assemble_returns_bundle():
    plan = make_plan()
    binned = _make_binned_records(plan, n_bins=50)
    s = measure(binned, plan, profile="continuous")[0]
    ft = engineer(s, plan)
    bundle = assemble([ft])
    assert isinstance(bundle, FeatureBundle)


def test_assemble_bundle_shape():
    plan = make_plan()
    binned = _make_binned_records(plan, n_bins=50)
    s = measure(binned, plan, profile="continuous")[0]
    ft = engineer(s, plan)
    bundle = assemble([ft])
    n_channels, n_scales, n_time = bundle.shape
    assert n_channels == len(ft.values)
    assert n_scales == ft.shape[0]
    assert n_time == ft.shape[1]


def test_assemble_channel_index_covers_all_features():
    plan = make_plan()
    binned = _make_binned_records(plan, n_bins=50)
    s = measure(binned, plan, profile="continuous")[0]
    ft = engineer(s, plan)
    bundle = assemble([ft])
    assert len(bundle.channel_index) == len(ft.values)


def test_assemble_entity_count():
    plan = make_plan()
    binned = _make_binned_records(plan, n_bins=50)
    s = measure(binned, plan, profile="continuous")[0]
    ft = engineer(s, plan)
    bundle = assemble([ft])
    # One entity — keys is empty {}
    assert len(bundle) == 1


def test_assemble_empty_raises():
    with pytest.raises(ValueError):
        assemble([])


def test_assemble_mismatched_plan_raises():
    plan_a = SamplingPlan(3600, 250)
    plan_b = SamplingPlan(7200, 250)

    binned_a = _make_binned_records(plan_a, n_bins=50)
    s_a = measure(binned_a, plan_a, profile="continuous")[0]
    ft_a = engineer(s_a, plan_a)

    binned_b = _make_binned_records(plan_b, n_bins=50)
    s_b = measure(binned_b, plan_b, profile="continuous")[0]
    ft_b = engineer(s_b, plan_b)

    with pytest.raises(ValueError):
        assemble([ft_a, ft_b])
