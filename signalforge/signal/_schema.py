"""
signalforge.signal._schema

Schema: declares the typed axes of a data source.

An event is a point in a product of typed axes. The schema declares
which axes exist and what type each one is. Records are values
conforming to the schema.

Three fundamental axis types:
    ORDERED     — has sequence, lattice operates here (time, position, index)
    CATEGORICAL — discrete labels, partitions the space (event code, sensor type)
    NUMERIC     — measurable quantity (count, amplitude, temperature)
    RELATIONAL  — points to another event (ticket hash, parent process ID)

Axis types are extensible — add new types to AxisType as needed.
SF handles built-in types; unknown types are carried as metadata.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class AxisType(str, enum.Enum):
    """Type of an axis in the schema.

    Inherits from str so it serializes cleanly to JSON.
    """
    ORDERED = "ordered"          #: Has sequence — lattice operates here.
    CATEGORICAL = "categorical"  #: Discrete labels — partitions the space.
    NUMERIC = "numeric"          #: Measurable quantity — what you surface.
    RELATIONAL = "relational"    #: Points to another event — cross-event links.


@dataclass
class Axis:
    """One axis in a schema.

    Parameters
    ----------
    name : str
        Column or field name.
    type : AxisType
        One of ORDERED, CATEGORICAL, NUMERIC, RELATIONAL.
    description : str, optional
        Human-readable description.
    """
    name: str
    type: AxisType
    description: str = ""

    def to_dict(self) -> dict:
        d = {"name": self.name, "type": self.type.value}
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Axis":
        return cls(
            name=d["name"],
            type=AxisType(d["type"]),
            description=d.get("description", ""),
        )


@dataclass
class Schema:
    """Declares the typed axes of a data source.

    A schema is the domain adapter in its simplest form — a declaration
    of what axes exist and what type each one is. No code needed.

    Attributes
    ----------
    name : str
        Name of this schema (e.g., "kerberos", "eeg", "vix").
    axes : list of Axis
        The typed axes, in declaration order.
    primary_order : str
        Name of the ordered axis used as the primary index.
    group_by : list of str
        Categorical axes to group by (produce per-entity surfaces).
    value_axis : str
        Name of the numeric axis to surface. Default: first numeric axis.
    channel_axis : str, optional
        Categorical axis that defines channels (one surface per value).
    """
    name: str
    axes: List[Axis]
    primary_order: str
    group_by: List[str] = field(default_factory=list)
    value_axis: str = ""
    channel_axis: str = ""

    def __post_init__(self):
        # Validate primary_order exists and is ordered
        ax = self.get_axis(self.primary_order)
        if ax is None:
            raise ValueError(f"primary_order {self.primary_order!r} not in axes")
        if ax.type != AxisType.ORDERED:
            raise ValueError(f"primary_order {self.primary_order!r} must be ORDERED")

        # Validate group_by axes exist and are categorical
        for g in self.group_by:
            ax = self.get_axis(g)
            if ax is None:
                raise ValueError(f"group_by {g!r} not in axes")
            if ax.type != AxisType.CATEGORICAL:
                raise ValueError(f"group_by {g!r} must be CATEGORICAL")

        # Default value_axis: first numeric axis
        if not self.value_axis:
            for a in self.axes:
                if a.type == AxisType.NUMERIC:
                    self.value_axis = a.name
                    break

    def get_axis(self, name: str) -> Optional[Axis]:
        """Get an axis by name."""
        for a in self.axes:
            if a.name == name:
                return a
        return None

    @property
    def axis_names(self) -> List[str]:
        return [a.name for a in self.axes]

    @property
    def ordered_axes(self) -> List[Axis]:
        return [a for a in self.axes if a.type == AxisType.ORDERED]

    @property
    def categorical_axes(self) -> List[Axis]:
        return [a for a in self.axes if a.type == AxisType.CATEGORICAL]

    @property
    def numeric_axes(self) -> List[Axis]:
        return [a for a in self.axes if a.type == AxisType.NUMERIC]

    @property
    def relational_axes(self) -> List[Axis]:
        return [a for a in self.axes if a.type == AxisType.RELATIONAL]

    # --- Serialization ---

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "axes": [a.to_dict() for a in self.axes],
            "primary_order": self.primary_order,
            "group_by": self.group_by,
            "value_axis": self.value_axis,
            "channel_axis": self.channel_axis,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        return cls(
            name=d["name"],
            axes=[Axis.from_dict(a) for a in d["axes"]],
            primary_order=d["primary_order"],
            group_by=d.get("group_by", []),
            value_axis=d.get("value_axis", ""),
            channel_axis=d.get("channel_axis", ""),
        )

    def save(self, path: str | Path) -> None:
        """Save schema to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Schema":
        """Load schema from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # --- Inference ---

    @classmethod
    def infer(cls, path: str | Path, name: str = "") -> "Schema":
        """Infer a schema from a CSV file.

        Reads the file, detects column types, caches the DataFrame.
        Use ``schema.records()`` to get Records from the cached data
        without re-reading the file.

        Parameters
        ----------
        path : str or Path
            Path to CSV file.
        name : str, optional
            Schema name. Defaults to filename stem.

        Returns
        -------
        Schema
            With ``_df`` cached for ``records()`` and ``_path`` stored.
        """
        import pandas as pd

        path = Path(path)
        df = pd.read_csv(path)
        if not name:
            name = path.stem

        axes = []
        primary_order = ""

        # Use first 1000 rows for type inference
        sample = df.head(1000)

        for col in sample.columns:
            series = sample[col].dropna()
            if len(series) == 0:
                continue

            # Try numeric
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().all():
                if numeric.is_monotonic_increasing and len(numeric) > 5:
                    axes.append(Axis(col, AxisType.ORDERED))
                    if not primary_order:
                        primary_order = col
                else:
                    axes.append(Axis(col, AxisType.NUMERIC))
                continue

            # Try datetime
            try:
                dates = pd.to_datetime(series, errors="raise")
                axes.append(Axis(col, AxisType.ORDERED))
                if not primary_order:
                    primary_order = col
                continue
            except (ValueError, TypeError):
                pass

            # String column
            n_unique = series.nunique()
            n_total = len(series)
            if n_total > 0 and n_unique / n_total > 0.8:
                axes.append(Axis(col, AxisType.RELATIONAL))
            else:
                axes.append(Axis(col, AxisType.CATEGORICAL))

        if not primary_order:
            axes.insert(0, Axis("_index", AxisType.ORDERED))
            primary_order = "_index"

        schema = cls(name=name, axes=axes, primary_order=primary_order)
        schema._df = df
        schema._path = str(path)
        return schema

    # Alias for Python convention
    from_csv = infer

    # --- Data loading ---

    def records(self) -> list:
        """Produce Records from cached data (after infer).

        Returns
        -------
        list of Record
            One Record per row in the cached DataFrame.

        Raises
        ------
        ValueError
            If no cached data (schema was loaded from JSON, not inferred).
        """
        df = getattr(self, "_df", None)
        if df is None:
            raise ValueError(
                "No cached data. Use Schema.infer() to cache, "
                "or schema.read(path) to load a file."
            )
        return self._df_to_records(df)

    def read(self, path: str | Path) -> list:
        """Load a CSV file through this schema.

        Parameters
        ----------
        path : str or Path
            Path to CSV file.

        Returns
        -------
        list of Record
        """
        import pandas as pd
        df = pd.read_csv(path)
        return self._df_to_records(df)

    def _df_to_records(self, df) -> list:
        """Convert a DataFrame to Records using this schema."""
        import pandas as pd
        from ._record import Record

        # Handle datetime primary_order
        po_col = self.primary_order
        if po_col in df.columns:
            try:
                dates = pd.to_datetime(df[po_col], errors="raise")
                df = df.copy()
                df[po_col] = range(len(df))
            except (ValueError, TypeError):
                df = df.copy()
                df[po_col] = pd.to_numeric(df[po_col], errors="coerce")

        records = []
        for _, row in df.iterrows():
            values = {}
            for axis in self.axes:
                if axis.name in df.columns:
                    val = row[axis.name]
                    if pd.notna(val):
                        values[axis.name] = val
                elif axis.name == "_index":
                    values["_index"] = int(row.name)

            if self.primary_order in values:
                records.append(Record(self, values))

        return records

    # --- Display ---

    def describe(self) -> None:
        """Print the schema."""
        print()
        print(f"  Schema: {self.name}")
        print(f"  {'─' * 50}")
        for a in self.axes:
            flags = []
            if a.name == self.primary_order:
                flags.append("primary")
            if a.name in self.group_by:
                flags.append("group-by")
            if a.name == self.value_axis:
                flags.append("value")
            if a.name == self.channel_axis:
                flags.append("channel")
            flag_str = f"  ({', '.join(flags)})" if flags else ""
            print(f"    {a.name:<20s} {a.type.value:<12s}{flag_str}")
        print(f"  {'─' * 50}")
        print()

    def __repr__(self) -> str:
        return f"Schema({self.name!r}, {len(self.axes)} axes, primary={self.primary_order!r})"
