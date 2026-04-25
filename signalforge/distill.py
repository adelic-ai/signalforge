"""
signalforge.distill — top-level convenience for segment discovery and featurization.

Wraps the IL-driven segment discovery and feature toolkit into a
user-facing two-step flow:

    distilled = sf.distill(records)        # discover segments
    features  = sf.featurize(distilled)    # extract feature matrix

Domain-agnostic. Works on any ordered event stream — time-ordered or
event-ordered. The records carry their own ordering via primary_order;
distill does not assume a domain-specific schema.

When records arrive without explicit entity grouping (e.g., when
Schema.infer didn't pick a group_by axis), pass ``entity_key=`` to
re-group records by a named field before discovery.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import numpy as np

from .signal import (
    discover_segments,
    segments_to_matrix,
    join_segments,
    Segment,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_field(record: Any, name: str) -> Any:
    """Best-effort field access on a record.

    Tries ``.values[name]``, then ``.get(name)``, then ``getattr``.
    Returns None if not found.
    """
    values = getattr(record, "values", None)
    if isinstance(values, dict):
        v = values.get(name)
        if v is not None:
            return v
    if hasattr(record, "get"):
        try:
            v = record.get(name)
            if v is not None:
                return v
        except (TypeError, KeyError):
            pass
    return getattr(record, name, None)


def _rekey_records(
    records: Sequence[Any],
    entity_key: str,
    default: str = "_unknown",
) -> List[Any]:
    """Return new records keyed by ``entity_key``.

    Two record shapes are supported:

    1. Record (schema-based) — ``.keys`` is derived from
       ``schema.group_by``. We deep-copy the schema, set
       ``group_by=[entity_key]``, and construct new Records
       that reuse the original ``values`` dicts.
    2. CanonicalRecord (slot-based) — ``.keys`` is a mutable
       slot. We shallow-copy each record and overwrite ``.keys``.

    Original records are not mutated.
    """
    if not records:
        return []

    first = records[0]

    # Path A: Record with schema-derived keys
    schema = getattr(first, "schema", None)
    if schema is not None and hasattr(schema, "group_by"):
        from .signal._record import Record

        new_schema = copy.deepcopy(schema)
        new_schema.group_by = [entity_key]

        rekeyed: List[Any] = []
        for r in records:
            values = getattr(r, "values", None)
            if not isinstance(values, dict):
                continue
            if entity_key not in values or values.get(entity_key) is None:
                values = dict(values)
                values[entity_key] = default
            rekeyed.append(Record(new_schema, values))
        return rekeyed

    # Path B: CanonicalRecord with mutable .keys slot
    rekeyed = []
    for r in records:
        val = _get_field(r, entity_key)
        new_keys = {entity_key: str(val) if val is not None else default}
        try:
            new_r = copy.copy(r)
            new_r.keys = new_keys
        except (TypeError, AttributeError):
            r.keys = new_keys
            new_r = r
        rekeyed.append(new_r)
    return rekeyed


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DistillResult:
    """Output of sf.distill() — discovered segments with discovery stats.

    Attributes
    ----------
    segments : list of Segment
        The discovered dense regions of activity per entity.
    stats : dict
        Aggregate statistics from the discovery pass.
    records : list
        The (possibly re-keyed) input records, retained for traceback,
        featurization, and downstream re-analysis.
    """
    segments: List[Segment]
    stats: dict
    records: List[Any] = field(default_factory=list, repr=False)

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)

    def summary(self) -> dict:
        """Discovery statistics."""
        return self.stats

    def channels(self) -> List[str]:
        """Distinct channels across all input records."""
        return sorted({r.channel for r in self.records}) if self.records else []

    def featurize(self, **kwargs) -> "FeatureSet":
        """Compute a feature matrix from these segments.

        Equivalent to ``sf.featurize(result, **kwargs)``. See ``featurize``
        for parameters.
        """
        return featurize(self, **kwargs)


@dataclass
class FeatureSet:
    """Output of sf.featurize() — named feature matrix.

    Attributes
    ----------
    matrix : np.ndarray, shape (n_segments, n_features)
    names : list of str
        Column labels.
    info : dict
        Metadata from the underlying feature extractors.
    """
    matrix: np.ndarray
    names: List[str]
    info: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        return self.matrix.shape

    def __len__(self) -> int:
        return int(self.matrix.shape[0])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def distill(
    records: Sequence[Any],
    *,
    entity_key: Optional[str] = None,
    method: str = "information_gain",
    min_gain: float = 0.1,
    min_segment_events: int = 2,
) -> DistillResult:
    """Discover segments in records via IL-driven analysis.

    A segment is a burst of activity from one entity, bounded by silence
    or a natural information-theoretic boundary. The default method splits
    via Shannon-entropy reduction — no gap threshold needed; the data tells
    you where the boundaries are.

    Parameters
    ----------
    records : sequence
        Records carrying ``.primary_order``, ``.channel``, and ``.keys``
        (CanonicalRecord or Record). Order can be time-based or
        event-based — distill does not assume which.
    entity_key : str, optional
        Field name to group records by before discovery. Use this when
        ``Schema.infer`` didn't pick a group_by axis and your records'
        ``.keys`` collapse everything into one entity. The named field is
        looked up on each record (via ``.values`` dict, ``.get()``, or
        attribute access) and becomes the new sole key.
    method : {"information_gain", "gap"}, default "information_gain"
        Splitting strategy. "information_gain" is the IL-driven default;
        "gap" is the legacy threshold-based fallback.
    min_gain : float, default 0.1
        Minimum information gain (bits) to justify a split.
    min_segment_events : int, default 2
        Minimum events per segment.

    Returns
    -------
    DistillResult

    Examples
    --------
    >>> result = sf.distill(records)
    >>> result_per_user = sf.distill(records, entity_key="Account_Name")
    """
    rec_list = list(records)
    if entity_key is not None:
        rec_list = _rekey_records(rec_list, entity_key)
    segments, stats = discover_segments(
        rec_list,
        method=method,
        min_gain=min_gain,
        min_segment_events=min_segment_events,
    )
    return DistillResult(segments=segments, stats=stats, records=rec_list)


def featurize(
    distilled: DistillResult,
    *,
    joins: Optional[Sequence[str]] = None,
    channels: Optional[Sequence[str]] = None,
    plan: Optional[Any] = None,
) -> FeatureSet:
    """Compute a feature matrix from distilled segments.

    Builds the standard SF segment feature matrix and optionally enriches
    it with cross-segment join features (e.g., per-source-key fan-out
    such as IP, service, or any other field present on records).

    Parameters
    ----------
    distilled : DistillResult
        Output of ``distill()``.
    joins : sequence of str, optional
        Field names to compute join-enrichment over. Each key adds a
        block of cross-segment features.
    channels : sequence of str, optional
        Channels to feature over. Defaults to all distinct channels in
        ``distilled.records``.
    plan : SamplingPlan, optional
        Lattice plan for measurement-based features. If None, a default
        is constructed inside ``segments_to_matrix``.

    Returns
    -------
    FeatureSet
    """
    segments = distilled.segments
    if not segments:
        return FeatureSet(matrix=np.zeros((0, 0)), names=[], info={})

    if channels is None:
        channels = distilled.channels()

    matrix, names, info = segments_to_matrix(
        segments,
        channels=list(channels),
        plan=plan,
    )

    if joins:
        for key in joins:
            join_features = join_segments(segments, key)
            if not join_features:
                continue
            value_field = f"join_{key}_value"
            join_names = sorted({
                k for jf in join_features
                for k, v in jf.items()
                if k != value_field and isinstance(v, (int, float))
            })
            if not join_names:
                continue
            join_matrix = np.zeros((len(segments), len(join_names)), dtype=np.float64)
            for i, jf in enumerate(join_features):
                for j, name in enumerate(join_names):
                    val = jf.get(name)
                    if isinstance(val, (int, float)):
                        join_matrix[i, j] = float(val)
            matrix = np.hstack([matrix, join_matrix])
            names = list(names) + join_names

    return FeatureSet(matrix=matrix, names=list(names), info=info)
