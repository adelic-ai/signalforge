"""
signalforge.pipeline.dag

Fluent DAG interface for chaining SignalForge pipeline stages.

    bundle = (
        sf.acquire("eeg", "chb01_03_eeg_rms.csv")
          .materialize()
          .measure(profile="continuous")
          .engineer()
          .assemble()
          .run()
    )

Custom operators can be inserted at any stage via .transform():

    bundle = (
        sf.acquire("eeg", path)
          .materialize()
          .transform(my_filter)          # operates on list[BinnedRecord]
          .measure(profile="continuous")
          .transform(my_surface_filter)  # operates on list[Surface]
          .engineer()
          .assemble()
          .run()
    )

The transform function receives the current artifact and must return
the same type. The pipeline does not validate the return type —
it is the caller's responsibility.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional


class Pipeline:
    """
    Fluent interface for chaining SignalForge pipeline stages.

    Stages execute eagerly in the order they are called. Each method
    returns self so calls can be chained.

    Parameters
    ----------
    records : list[CanonicalRecord]
        Output of Stage 0 (Ingest). Must be non-decreasing on primary_order.
    plan : SamplingPlan
        Output of Stage 1 (Plan). Defines the lattice geometry.
    """

    def __init__(self, records: List[Any], plan: Any) -> None:
        self._records = records
        self._plan = plan
        self._binned: Optional[List[Any]] = None
        self._surfaces: Optional[List[Any]] = None
        self._tensors: Optional[List[Any]] = None
        self._bundle: Optional[Any] = None
        self._stage: str = "records"

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def materialize(self, agg_funcs=None) -> "Pipeline":
        """Stage 2 — CanonicalRecords → BinnedRecords."""
        from .binned import materialize as _materialize
        self._binned = _materialize(self._records, self._plan, agg_funcs=agg_funcs)
        self._stage = "binned"
        return self

    def measure(self, profile: str = "continuous") -> "Pipeline":
        """Stage 3 — BinnedRecords → Surfaces (time × scale grid)."""
        from .surface import measure as _measure
        self._surfaces = _measure(self._binned, self._plan, profile=profile)
        self._stage = "surfaces"
        return self

    def engineer(self) -> "Pipeline":
        """Stage 4 — Surfaces → FeatureTensors."""
        from .feature import engineer as _engineer
        self._tensors = [_engineer(s, self._plan) for s in self._surfaces]
        self._stage = "tensors"
        return self

    def assemble(self) -> "Pipeline":
        """Stage 5 — FeatureTensors → FeatureBundle."""
        from .bundle import assemble as _assemble
        self._bundle = _assemble(self._tensors)
        self._stage = "bundle"
        return self

    def run(self) -> Any:
        """Return the assembled FeatureBundle."""
        if self._bundle is None:
            raise RuntimeError(
                "Pipeline has not been assembled. Call .assemble() before .run()."
            )
        return self._bundle

    # ------------------------------------------------------------------
    # Custom operators
    # ------------------------------------------------------------------

    def transform(self, fn: Callable[[Any], Any]) -> "Pipeline":
        """
        Apply a custom function to the current artifact.

        The function receives the artifact at the current stage and must
        return the same type. Valid at any stage.

        Parameters
        ----------
        fn : callable
            Receives and returns the current artifact:
              - after materialize : list[BinnedRecord]
              - after measure     : list[Surface]
              - after engineer    : list[FeatureTensor]
        """
        if self._stage == "records":
            self._records = fn(self._records)
        elif self._stage == "binned":
            self._binned = fn(self._binned)
        elif self._stage == "surfaces":
            self._surfaces = fn(self._surfaces)
        elif self._stage == "tensors":
            self._tensors = fn(self._tensors)
        else:
            raise RuntimeError(
                f"Cannot transform at stage {self._stage!r}."
            )
        return self

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Pipeline(stage={self._stage!r}, plan={self._plan})"
