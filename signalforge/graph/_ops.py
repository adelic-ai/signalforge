"""
signalforge.graph._ops

Concrete Op subclasses wrapping existing pipeline stages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._core import Artifact, Op
from ..signal._base import ArtifactType
from ._types import parse_duration


class InputOp(Op):
    """Wraps raw records into an Artifact. No computation."""

    input_types = ()
    output_type = ArtifactType.RECORDS

    def execute(self, *, records: Any = None, plan: Any = None) -> Artifact:
        if records is None:
            raise ValueError("InputOp.execute requires records.")
        return Artifact(
            type=ArtifactType.RECORDS,
            value=records,
            producing_op=self,
            plan=plan,
            metadata={"n_records": len(records)},
        )

    def contribute_constraints(self) -> Dict[str, Any]:
        c: dict = {}
        if "grain" in self.params:
            c["grain"] = parse_duration(self.params["grain"])
        if "horizon" in self.params:
            c["horizon"] = parse_duration(self.params["horizon"])
        return c


class BinOp(Op):
    """Stage 2 — CanonicalRecords → BinnedRecords via materialize()."""

    input_types = (ArtifactType.RECORDS,)
    output_type = ArtifactType.BINNED

    def execute(self, *inputs: Artifact, plan: Any = None) -> Artifact:
        from ..pipeline.binned import materialize
        records = inputs[0].value
        agg_funcs = self.params.get("agg_funcs")

        # If agg is a simple string like "mean", derive full agg_funcs from channels
        agg = self.params.get("agg")
        if agg and not agg_funcs:
            channels = sorted({r.channel for r in records})
            agg_funcs = {ch: {"value": agg} for ch in channels}

        binned = materialize(records, plan, agg_funcs=agg_funcs)
        return Artifact(
            type=ArtifactType.BINNED,
            value=binned,
            producing_op=self,
            plan=plan,
            metadata={"n_binned": len(binned)},
        )

    def contribute_constraints(self) -> Dict[str, Any]:
        c: dict = {}
        if "grain" in self.params:
            c["grain"] = parse_duration(self.params["grain"])
        return c


class MeasureOp(Op):
    """Stage 3 — BinnedRecords → Surfaces via measure()."""

    input_types = (ArtifactType.BINNED,)
    output_type = ArtifactType.SURFACES

    def execute(self, *inputs: Artifact, plan: Any = None) -> Artifact:
        from ..pipeline.surface import measure
        binned = inputs[0].value
        profile = self.params.get("profile", "continuous")
        surfaces = measure(binned, plan, profile=profile)
        return Artifact(
            type=ArtifactType.SURFACES,
            value=surfaces,
            producing_op=self,
            plan=plan,
            metadata={"n_surfaces": len(surfaces)},
        )

    def contribute_constraints(self) -> Dict[str, Any]:
        c: dict = {}
        if "windows" in self.params:
            c["windows"] = [parse_duration(w) for w in self.params["windows"]]
        return c


class EngineerOp(Op):
    """Stage 4 — Surfaces → FeatureTensors via engineer()."""

    input_types = (ArtifactType.SURFACES,)
    output_type = ArtifactType.TENSORS

    def execute(self, *inputs: Artifact, plan: Any = None) -> Artifact:
        from ..pipeline.feature import engineer
        surfaces = inputs[0].value
        tensors = [engineer(s, plan) for s in surfaces]
        return Artifact(
            type=ArtifactType.TENSORS,
            value=tensors,
            producing_op=self,
            plan=plan,
            metadata={"n_tensors": len(tensors)},
        )


class AssembleOp(Op):
    """Stage 5 — FeatureTensors → FeatureBundle via assemble()."""

    input_types = (ArtifactType.TENSORS,)
    output_type = ArtifactType.BUNDLE

    def execute(self, *inputs: Artifact, plan: Any = None) -> Artifact:
        from ..pipeline.bundle import assemble
        tensors = inputs[0].value
        bundle = assemble(tensors)
        return Artifact(
            type=ArtifactType.BUNDLE,
            value=bundle,
            producing_op=self,
            plan=plan,
            metadata={},
        )
