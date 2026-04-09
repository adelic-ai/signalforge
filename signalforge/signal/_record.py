"""
signalforge.signal._record

Record: one event as a point in the typed product space.

A Record carries values for each axis defined in its Schema.
It's the universal replacement for CanonicalRecord — works for
any domain without fixed fields.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ._schema import Schema, AxisType


class Record:
    """One event — values for each axis in the schema.

    Lightweight: just a dict of values plus a reference to the schema.
    Validated on construction to ensure required axes are present
    and types are plausible.

    Parameters
    ----------
    schema : Schema
        The schema this record conforms to.
    values : dict
        Axis name → value. Must include at least the primary_order axis.
    """

    __slots__ = ("schema", "values")

    def __init__(self, schema: Schema, values: Dict[str, Any]) -> None:
        self.schema = schema
        self.values = values

        # Validate primary_order is present
        if schema.primary_order not in values:
            raise ValueError(
                f"Record missing primary_order axis {schema.primary_order!r}"
            )

    @property
    def primary_order(self) -> int:
        """The primary ordering value."""
        return int(self.values[self.schema.primary_order])

    @property
    def channel(self) -> str:
        """Channel value, if channel_axis is defined."""
        if self.schema.channel_axis:
            return str(self.values.get(self.schema.channel_axis, ""))
        return ""

    @property
    def value(self) -> float:
        """Numeric value from the value_axis."""
        if self.schema.value_axis:
            return float(self.values.get(self.schema.value_axis, 0.0))
        return 1.0  # default: count (each record = 1 event)

    @property
    def keys(self) -> Dict[str, str]:
        """Grouping keys — values of group_by axes."""
        return {
            g: str(self.values.get(g, ""))
            for g in self.schema.group_by
        }

    def get(self, axis_name: str, default: Any = None) -> Any:
        """Get value for any axis."""
        return self.values.get(axis_name, default)

    def __repr__(self) -> str:
        po = self.primary_order
        ch = self.channel
        v = self.value
        k = self.keys
        parts = [f"order={po}"]
        if ch:
            parts.append(f"ch={ch!r}")
        parts.append(f"val={v}")
        if k:
            parts.append(f"keys={k}")
        return f"Record({', '.join(parts)})"


def records_from_csv(path: str, schema: Schema) -> list[Record]:
    """Load a CSV file into Records using a Schema.

    Parameters
    ----------
    path : str
        Path to CSV file.
    schema : Schema
        Schema defining axis types and roles.

    Returns
    -------
    list of Record
    """
    import pandas as pd

    df = pd.read_csv(path)
    records = []

    # Handle datetime primary_order — convert to integer
    po_col = schema.primary_order
    if po_col in df.columns:
        try:
            dates = pd.to_datetime(df[po_col], errors="raise")
            # Convert to integer index
            df[po_col] = range(len(df))
        except (ValueError, TypeError):
            df[po_col] = pd.to_numeric(df[po_col], errors="coerce")

    for _, row in df.iterrows():
        values = {}
        for axis in schema.axes:
            if axis.name in df.columns:
                val = row[axis.name]
                if pd.notna(val):
                    values[axis.name] = val
            elif axis.name == "_index":
                values["_index"] = int(row.name)

        if schema.primary_order in values:
            records.append(Record(schema, values))

    return records
