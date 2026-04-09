"""
Tests for signalforge.signal Schema, Record, and typed-axis pipeline.
"""

import numpy as np
import pytest
import json
from pathlib import Path

from signalforge.signal import (
    Schema, Axis, AxisType, Record, records_from_csv,
    records_to_signals, measure_signal, RealSignal, Surface,
)
from signalforge.lattice.sampling import SamplingPlan


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_creation():
    schema = Schema(
        name="test",
        axes=[Axis("time", AxisType.ORDERED), Axis("value", AxisType.NUMERIC)],
        primary_order="time",
    )
    assert schema.name == "test"
    assert len(schema.axes) == 2
    assert schema.primary_order == "time"
    assert schema.value_axis == "value"  # auto-detected


def test_schema_invalid_primary():
    with pytest.raises(ValueError, match="not in axes"):
        Schema(
            name="bad",
            axes=[Axis("time", AxisType.ORDERED)],
            primary_order="missing",
        )


def test_schema_primary_must_be_ordered():
    with pytest.raises(ValueError, match="must be ORDERED"):
        Schema(
            name="bad",
            axes=[Axis("name", AxisType.CATEGORICAL)],
            primary_order="name",
        )


def test_schema_group_by_must_be_categorical():
    with pytest.raises(ValueError, match="must be CATEGORICAL"):
        Schema(
            name="bad",
            axes=[
                Axis("time", AxisType.ORDERED),
                Axis("value", AxisType.NUMERIC),
            ],
            primary_order="time",
            group_by=["value"],
        )


def test_schema_axis_properties():
    schema = Schema(
        name="test",
        axes=[
            Axis("time", AxisType.ORDERED),
            Axis("code", AxisType.CATEGORICAL),
            Axis("value", AxisType.NUMERIC),
            Axis("hash", AxisType.RELATIONAL),
        ],
        primary_order="time",
    )
    assert len(schema.ordered_axes) == 1
    assert len(schema.categorical_axes) == 1
    assert len(schema.numeric_axes) == 1
    assert len(schema.relational_axes) == 1


def test_schema_get_axis():
    schema = Schema(
        name="test",
        axes=[Axis("time", AxisType.ORDERED), Axis("val", AxisType.NUMERIC)],
        primary_order="time",
    )
    assert schema.get_axis("time").type == AxisType.ORDERED
    assert schema.get_axis("missing") is None


def test_schema_save_load(tmp_path):
    schema = Schema(
        name="test",
        axes=[
            Axis("time", AxisType.ORDERED),
            Axis("channel", AxisType.CATEGORICAL),
            Axis("value", AxisType.NUMERIC),
        ],
        primary_order="time",
        group_by=[],
        channel_axis="channel",
    )
    path = tmp_path / "test.schema.json"
    schema.save(path)

    loaded = Schema.load(path)
    assert loaded.name == "test"
    assert len(loaded.axes) == 3
    assert loaded.primary_order == "time"
    assert loaded.channel_axis == "channel"


def test_schema_from_csv(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("date,value\n2020-01-01,1.0\n2020-01-02,2.0\n2020-01-03,3.0\n")

    schema = Schema.from_csv(csv)
    assert schema.primary_order == "date"
    assert any(a.type == AxisType.NUMERIC for a in schema.axes)


def test_schema_from_csv_numeric_index(tmp_path):
    csv = tmp_path / "test.csv"
    lines = ["idx,val"] + [f"{i},{float(i)*0.5}" for i in range(100)]
    csv.write_text("\n".join(lines))

    schema = Schema.from_csv(csv)
    assert schema.primary_order == "idx"
    assert schema.get_axis("idx").type == AxisType.ORDERED


def test_schema_repr():
    schema = Schema(
        name="test",
        axes=[Axis("t", AxisType.ORDERED)],
        primary_order="t",
    )
    assert "test" in repr(schema)


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


def _make_schema():
    return Schema(
        name="test",
        axes=[
            Axis("time", AxisType.ORDERED),
            Axis("channel", AxisType.CATEGORICAL),
            Axis("machine", AxisType.CATEGORICAL),
            Axis("value", AxisType.NUMERIC),
        ],
        primary_order="time",
        channel_axis="channel",
        group_by=["machine"],
        value_axis="value",
    )


def test_record_creation():
    schema = _make_schema()
    r = Record(schema, {"time": 100, "channel": "a", "machine": "m1", "value": 3.14})
    assert r.primary_order == 100
    assert r.channel == "a"
    assert r.value == pytest.approx(3.14)
    assert r.keys == {"machine": "m1"}


def test_record_missing_primary():
    schema = _make_schema()
    with pytest.raises(ValueError, match="missing primary_order"):
        Record(schema, {"channel": "a", "value": 1.0})


def test_record_default_value():
    schema = Schema(
        name="events",
        axes=[Axis("time", AxisType.ORDERED)],
        primary_order="time",
    )
    r = Record(schema, {"time": 1})
    assert r.value == 1.0  # default: count


def test_record_get():
    schema = _make_schema()
    r = Record(schema, {"time": 1, "channel": "a", "machine": "m1", "value": 2.0})
    assert r.get("machine") == "m1"
    assert r.get("missing", "default") == "default"


def test_record_repr():
    schema = _make_schema()
    r = Record(schema, {"time": 1, "channel": "a", "machine": "m1", "value": 2.0})
    s = repr(r)
    assert "order=1" in s
    assert "ch='a'" in s


# ---------------------------------------------------------------------------
# records_from_csv
# ---------------------------------------------------------------------------


def test_records_from_csv(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("idx,val\n0,1.0\n1,2.0\n2,3.0\n")

    schema = Schema(
        name="test",
        axes=[Axis("idx", AxisType.ORDERED), Axis("val", AxisType.NUMERIC)],
        primary_order="idx",
    )
    records = records_from_csv(str(csv), schema)
    assert len(records) == 3
    assert records[0].primary_order == 0
    assert records[2].value == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Full pipeline: Schema → Record → Signal → Surface
# ---------------------------------------------------------------------------


def test_schema_to_surface(tmp_path):
    csv = tmp_path / "test.csv"
    lines = ["idx,val"] + [f"{i},{np.sin(i/10.0):.4f}" for i in range(200)]
    csv.write_text("\n".join(lines))

    schema = Schema.from_csv(csv)
    records = records_from_csv(str(csv), schema)
    signals = records_to_signals(records)

    assert len(signals) == 1
    assert isinstance(signals[0], RealSignal)

    plan = SamplingPlan(180, 1)
    surface = measure_signal(signals[0], plan)
    assert isinstance(surface, Surface)
    assert surface.shape[0] > 0
    assert surface.shape[1] > 0


def test_multichannel_schema_to_surface():
    schema = Schema(
        name="multi",
        axes=[
            Axis("time", AxisType.ORDERED),
            Axis("channel", AxisType.CATEGORICAL),
            Axis("value", AxisType.NUMERIC),
        ],
        primary_order="time",
        channel_axis="channel",
    )

    records = []
    for i in range(100):
        for ch in ["a", "b", "c"]:
            records.append(Record(schema, {
                "time": i, "channel": ch, "value": float(i),
            }))

    signals = records_to_signals(records)
    assert len(signals) == 3
    channels = {s.channel for s in signals}
    assert channels == {"a", "b", "c"}


def test_keyed_schema_to_surface():
    schema = Schema(
        name="keyed",
        axes=[
            Axis("time", AxisType.ORDERED),
            Axis("entity", AxisType.CATEGORICAL),
            Axis("value", AxisType.NUMERIC),
        ],
        primary_order="time",
        group_by=["entity"],
    )

    records = []
    for i in range(50):
        for e in ["m1", "m2"]:
            records.append(Record(schema, {
                "time": i, "entity": e, "value": float(i),
            }))

    signals = records_to_signals(records)
    assert len(signals) == 2


def test_schema_round_trip(tmp_path):
    """Schema save → load → use → same result."""
    schema = Schema(
        name="rt",
        axes=[Axis("t", AxisType.ORDERED), Axis("v", AxisType.NUMERIC)],
        primary_order="t",
    )
    path = tmp_path / "rt.schema.json"
    schema.save(path)
    loaded = Schema.load(path)

    records = [Record(loaded, {"t": i, "v": float(i)}) for i in range(50)]
    signals = records_to_signals(records)
    assert len(signals) == 1
    assert len(signals[0]) == 50
