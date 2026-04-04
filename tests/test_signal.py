"""
Tests for signalforge.signal — LatticeSignal, ComplexSignal, RealSignal,
Surface, records_to_signals, measure_signal.
"""

import numpy as np
import pytest

from signalforge.signal import (
    LatticeSignal,
    ComplexSignal,
    RealSignal,
    Surface,
    Artifact,
    ArtifactType,
    CanonicalRecord,
    OrderType,
    records_to_signals,
    measure_signal,
)
from signalforge.lattice.sampling import SamplingPlan


# ---------------------------------------------------------------------------
# RealSignal
# ---------------------------------------------------------------------------


def test_real_signal_creation():
    idx = np.arange(100)
    vals = np.random.randn(100)
    s = RealSignal(idx, vals, channel="test")
    assert len(s) == 100
    assert s.channel == "test"
    assert s.is_real
    assert s.dtype == np.float64


def test_real_signal_keys_default_empty():
    s = RealSignal(np.arange(5), np.ones(5), channel="x")
    assert s.keys == {}


def test_real_signal_with_keys():
    s = RealSignal(np.arange(5), np.ones(5), channel="x", keys={"host": "dc01"})
    assert s.keys == {"host": "dc01"}


def test_real_signal_metadata():
    s = RealSignal(np.arange(5), np.ones(5), channel="x", metadata={"source": "test"})
    assert s.metadata == {"source": "test"}


def test_real_signal_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        RealSignal(np.arange(5), np.ones(10), channel="x")


def test_real_signal_amplitude():
    vals = np.array([-3.0, 0.0, 4.0])
    s = RealSignal(np.arange(3), vals, channel="x")
    np.testing.assert_array_equal(s.amplitude(), np.array([3.0, 0.0, 4.0]))


def test_real_signal_phase():
    vals = np.array([-1.0, 0.0, 1.0])
    s = RealSignal(np.arange(3), vals, channel="x")
    phase = s.phase()
    assert phase[0] == pytest.approx(np.pi)  # negative real → phase = pi
    assert phase[2] == pytest.approx(0.0)    # positive real → phase = 0


def test_real_signal_repr():
    s = RealSignal(np.arange(50), np.ones(50), channel="vix")
    r = repr(s)
    assert "RealSignal" in r
    assert "vix" in r
    assert "n=50" in r
    assert "real" in r


# ---------------------------------------------------------------------------
# ComplexSignal
# ---------------------------------------------------------------------------


def test_complex_signal_creation():
    idx = np.arange(100)
    r = np.random.randn(100)
    i = np.random.randn(100)
    s = ComplexSignal(idx, r, i, channel="eeg")
    assert len(s) == 100
    assert s.channel == "eeg"
    assert not s.is_real
    assert s.dtype == np.complex128


def test_complex_signal_values():
    idx = np.arange(3)
    r = np.array([1.0, 0.0, -1.0])
    i = np.array([0.0, 1.0, 0.0])
    s = ComplexSignal(idx, r, i, channel="x")
    expected = np.array([1+0j, 0+1j, -1+0j])
    np.testing.assert_array_equal(s.values, expected)


def test_complex_signal_amplitude():
    idx = np.arange(2)
    r = np.array([3.0, 0.0])
    i = np.array([4.0, 1.0])
    s = ComplexSignal(idx, r, i, channel="x")
    np.testing.assert_array_almost_equal(s.amplitude(), np.array([5.0, 1.0]))


def test_complex_signal_phase():
    idx = np.arange(2)
    r = np.array([1.0, 0.0])
    i = np.array([0.0, 1.0])
    s = ComplexSignal(idx, r, i, channel="x")
    np.testing.assert_array_almost_equal(s.phase(), np.array([0.0, np.pi / 2]))


def test_complex_signal_real_imag():
    idx = np.arange(3)
    r = np.array([1.0, 2.0, 3.0])
    i = np.array([4.0, 5.0, 6.0])
    s = ComplexSignal(idx, r, i, channel="x")
    np.testing.assert_array_equal(s.real(), r)
    np.testing.assert_array_equal(s.imag(), i)


def test_complex_signal_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        ComplexSignal(np.arange(5), np.ones(5), np.ones(10), channel="x")


def test_complex_signal_repr():
    s = ComplexSignal(np.arange(20), np.ones(20), np.ones(20), channel="eeg")
    r = repr(s)
    assert "ComplexSignal" in r
    assert "complex" in r


# ---------------------------------------------------------------------------
# LatticeSignal ABC
# ---------------------------------------------------------------------------


def test_lattice_signal_is_abstract():
    with pytest.raises(TypeError):
        LatticeSignal()


def test_real_signal_is_lattice_signal():
    s = RealSignal(np.arange(5), np.ones(5), channel="x")
    assert isinstance(s, LatticeSignal)


def test_complex_signal_is_lattice_signal():
    s = ComplexSignal(np.arange(5), np.ones(5), np.ones(5), channel="x")
    assert isinstance(s, LatticeSignal)


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def _make_surface(n_scales=6, n_time=50, complex_valued=False):
    plan = SamplingPlan(360, 1)
    # Use first n_scales windows
    windows = plan.windows[:n_scales]
    time_axis = np.arange(n_time)
    scale_axis = tuple(w for w in windows)
    if complex_valued:
        data = {"mean": np.random.randn(n_scales, n_time) + 1j * np.random.randn(n_scales, n_time)}
    else:
        data = {"mean": np.random.randn(n_scales, n_time)}
    return Surface(
        time_axis=time_axis,
        scale_axis=scale_axis,
        data=data,
        channel="test",
        plan=plan,
        coordinates=plan.coordinates[:n_scales],
    )


def test_surface_creation():
    s = _make_surface()
    assert s.shape == (6, 50)
    assert s.channel == "test"
    assert "mean" in s.data


def test_surface_is_lattice_signal():
    s = _make_surface()
    assert isinstance(s, LatticeSignal)


def test_surface_values_returns_first_data_array():
    s = _make_surface()
    np.testing.assert_array_equal(s.values, s.data["mean"])


def test_surface_index_is_time_axis():
    s = _make_surface()
    np.testing.assert_array_equal(s.index, s.time_axis)


def test_surface_real():
    s = _make_surface(complex_valued=False)
    assert s.is_real


def test_surface_complex():
    s = _make_surface(complex_valued=True)
    assert not s.is_real
    assert s.dtype == np.complex128


def test_surface_amplitude_phase():
    s = _make_surface(complex_valued=True)
    amp = s.amplitude()
    phase = s.phase()
    assert amp.shape == s.shape
    assert phase.shape == s.shape
    # Amplitude should be non-negative
    assert np.all(amp >= 0)


def test_surface_shape_validation():
    plan = SamplingPlan(360, 1)
    with pytest.raises(ValueError, match="shape"):
        Surface(
            time_axis=np.arange(50),
            scale_axis=(1, 2, 3),
            data={"mean": np.ones((4, 50))},  # 4 scales != 3
            channel="bad",
            plan=plan,
        )


def test_surface_repr():
    s = _make_surface()
    r = repr(s)
    assert "Surface" in r
    assert "test" in r


# ---------------------------------------------------------------------------
# Artifact and ArtifactType
# ---------------------------------------------------------------------------


def test_artifact_creation():
    a = Artifact(type=ArtifactType.RECORDS, value=[1, 2, 3])
    assert a.type == ArtifactType.RECORDS
    assert a.value == [1, 2, 3]


def test_artifact_type_signals_exists():
    assert ArtifactType.SIGNALS.value == "signals"


def test_artifact_repr():
    a = Artifact(type=ArtifactType.SURFACES, value=None)
    assert "surfaces" in repr(a)


# ---------------------------------------------------------------------------
# CanonicalRecord
# ---------------------------------------------------------------------------


def test_canonical_record_creation():
    r = CanonicalRecord(
        primary_order=100,
        order_type=OrderType.SEQUENCE,
        channel="test",
        metric="value",
        value=3.14,
        seq_order=100,
    )
    assert r.primary_order == 100
    assert r.value == pytest.approx(3.14)
    assert r.channel == "test"


def test_canonical_record_negative_order_raises():
    with pytest.raises(ValueError, match="non-negative"):
        CanonicalRecord(
            primary_order=-1,
            order_type=OrderType.SEQUENCE,
            channel="x",
            metric="v",
            value=1.0,
            seq_order=-1,
        )


def test_canonical_record_time_requires_time_order():
    with pytest.raises(ValueError, match="time_order required"):
        CanonicalRecord(
            primary_order=0,
            order_type=OrderType.TIME,
            channel="x",
            metric="v",
            value=1.0,
        )


# ---------------------------------------------------------------------------
# records_to_signals
# ---------------------------------------------------------------------------


def _make_records(n=100, channel="test", keys=None):
    return [
        CanonicalRecord(
            primary_order=i,
            order_type=OrderType.SEQUENCE,
            channel=channel,
            metric="value",
            value=float(np.sin(i / 10)),
            seq_order=i,
            keys=keys,
        )
        for i in range(n)
    ]


def test_records_to_signals_basic():
    records = _make_records(100)
    signals = records_to_signals(records)
    assert len(signals) == 1
    s = signals[0]
    assert isinstance(s, RealSignal)
    assert s.channel == "test"
    assert len(s) == 100


def test_records_to_signals_multiple_channels():
    records = _make_records(50, channel="a") + _make_records(50, channel="b")
    signals = records_to_signals(records)
    assert len(signals) == 2
    channels = {s.channel for s in signals}
    assert channels == {"a", "b"}


def test_records_to_signals_with_keys():
    records = (
        _make_records(30, channel="x", keys={"host": "dc01"}) +
        _make_records(30, channel="x", keys={"host": "dc02"})
    )
    signals = records_to_signals(records)
    assert len(signals) == 2


def test_records_to_signals_empty():
    signals = records_to_signals([])
    assert signals == []


def test_records_to_signals_aggregation():
    # Two records at the same primary_order
    records = [
        CanonicalRecord(primary_order=0, order_type=OrderType.SEQUENCE,
                        channel="x", metric="v", value=2.0, seq_order=0),
        CanonicalRecord(primary_order=0, order_type=OrderType.SEQUENCE,
                        channel="x", metric="v", value=4.0, seq_order=0),
    ]
    signals = records_to_signals(records, agg="mean")
    assert len(signals) == 1
    assert signals[0].values[0] == pytest.approx(3.0)


def test_records_to_signals_count_agg():
    records = _make_records(10)
    signals = records_to_signals(records, agg="count")
    assert len(signals) == 1
    # Each primary_order has exactly 1 record
    assert all(v == 1.0 for v in signals[0].values)


# ---------------------------------------------------------------------------
# measure_signal
# ---------------------------------------------------------------------------


def test_measure_signal_basic():
    idx = np.arange(200)
    vals = np.sin(idx / 20.0)
    sig = RealSignal(idx, vals, channel="sine")
    plan = SamplingPlan(360, 1)
    surface = measure_signal(sig, plan)

    assert isinstance(surface, Surface)
    assert isinstance(surface, LatticeSignal)
    assert surface.channel == "sine"
    assert surface.shape[0] == len(plan.windows)  # n_scales
    assert surface.shape[1] > 0  # n_time > 0


def test_measure_signal_complex():
    idx = np.arange(200)
    r = np.sin(idx / 20.0)
    i = np.cos(idx / 20.0)
    sig = ComplexSignal(idx, r, i, channel="analytic")
    plan = SamplingPlan(360, 1)
    surface = measure_signal(sig, plan)

    assert not surface.is_real
    assert surface.dtype == np.complex128
    assert surface.channel == "analytic"


def test_measure_signal_preserves_keys():
    sig = RealSignal(
        np.arange(100), np.ones(100), channel="x",
        keys={"host": "dc01"},
    )
    plan = SamplingPlan(360, 1)
    surface = measure_signal(sig, plan)
    assert surface.keys == {"host": "dc01"}


def test_measure_signal_has_coverage():
    sig = RealSignal(np.arange(100), np.ones(100), channel="x")
    plan = SamplingPlan(360, 1)
    surface = measure_signal(sig, plan)
    assert surface.coverage is not None
    assert surface.coverage.shape == surface.shape


def test_measure_signal_has_n_events():
    sig = RealSignal(np.arange(100), np.ones(100), channel="x")
    plan = SamplingPlan(360, 1)
    surface = measure_signal(sig, plan)
    assert surface.n_events is not None
    assert surface.n_events.shape == surface.shape


def test_measure_signal_plan_attached():
    sig = RealSignal(np.arange(100), np.ones(100), channel="x")
    plan = SamplingPlan(360, 1)
    surface = measure_signal(sig, plan)
    assert surface.plan is plan


# ---------------------------------------------------------------------------
# Graph integration — signal path
# ---------------------------------------------------------------------------


def test_graph_signal_path():
    """Input → Measure via signal path produces surfaces."""
    from signalforge.graph import Input, Measure, Pipeline

    records = _make_records(200)
    x = Input()
    m = Measure()(x)
    pipe = Pipeline(x, m)
    result = pipe.run(records, windows=[10, 20, 50])
    surfaces = result.value

    assert len(surfaces) == 1
    assert isinstance(surfaces[0], Surface)
    assert surfaces[0].channel == "test"


def test_graph_legacy_path():
    """Input(mode='records') → Bin → Measure still works."""
    from signalforge.graph import Input, Bin, Measure, Pipeline

    records = _make_records(200)
    x = Input(mode="records")
    b = Bin(agg="mean")(x)
    m = Measure(profile="continuous")(b)
    pipe = Pipeline(x, m)
    result = pipe.run(records, windows=[10, 20, 50])
    surfaces = result.value

    assert len(surfaces) == 1
    assert isinstance(surfaces[0], Surface)


def test_graph_baseline_on_signals():
    """Signal path with baseline and residual."""
    from signalforge.graph import Input, Measure, Baseline, Residual, Pipeline

    records = _make_records(200)
    x = Input()
    m = Measure()(x)
    bl = Baseline(method="ewma")(m)
    r = Residual(mode="difference")(m, bl)
    pipe = Pipeline(x, r)
    result = pipe.run(records, windows=[10, 20, 50])
    surfaces = result.value

    assert len(surfaces) == 1
    assert isinstance(surfaces[0], Surface)
