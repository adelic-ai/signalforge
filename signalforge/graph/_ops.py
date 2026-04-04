"""
signalforge.graph._ops

Concrete Op subclasses for the computation graph.

Two paths through the pipeline:

Signal path (new):
    InputOp(records) → SIGNALS (list[RealSignal])
    MeasureOp(signals) → SURFACES (list[Surface])

Record path (legacy, still supported):
    InputOp(records) → RECORDS (list[CanonicalRecord])
    BinOp(records) → BINNED (list[BinnedRecord])
    MeasureOp(binned) → SURFACES (list[Surface])
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._core import Artifact, Op
from ..signal._base import ArtifactType
from ._types import parse_duration


class InputOp(Op):
    """Ingest records into the graph.

    By default, converts records to RealSignals (one per channel/key group).
    Set mode="records" to pass raw CanonicalRecords through (legacy path).
    """

    input_types = ()
    output_type = ArtifactType.SIGNALS

    def __init__(self, **params):
        super().__init__(**params)
        mode = params.get("mode", "signals")
        if mode == "records":
            self.output_type = ArtifactType.RECORDS

    def execute(self, *, records: Any = None, plan: Any = None) -> Artifact:
        if records is None:
            raise ValueError("InputOp.execute requires records.")

        mode = self.params.get("mode", "signals")

        # Detect if records are already LatticeSignals
        from ..signal._signal import LatticeSignal
        if isinstance(records, list) and records and isinstance(records[0], LatticeSignal):
            return Artifact(
                type=ArtifactType.SIGNALS,
                value=records,
                producing_op=self,
                plan=plan,
                metadata={"n_signals": len(records)},
            )

        if mode == "signals":
            from ..signal import records_to_signals
            agg = self.params.get("agg", "mean")
            signals = records_to_signals(records, agg=agg)
            return Artifact(
                type=ArtifactType.SIGNALS,
                value=signals,
                producing_op=self,
                plan=plan,
                metadata={"n_signals": len(signals), "n_records": len(records)},
            )
        else:
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
    """Stage 2 — CanonicalRecords → BinnedRecords via materialize().

    Legacy path. For the signal path, MeasureOp handles binning internally.
    """

    input_types = (ArtifactType.RECORDS,)
    output_type = ArtifactType.BINNED

    def execute(self, *inputs: Artifact, plan: Any = None) -> Artifact:
        from ..pipeline.binned import materialize
        records = inputs[0].value
        agg_funcs = self.params.get("agg_funcs")

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
    """Measure signals or binned records into Surfaces.

    Accepts either:
    - SIGNALS (list[LatticeSignal]) → measures each signal directly
    - BINNED (list[BinnedRecord]) → uses pipeline.surface.measure() (legacy)
    """

    input_types = (ArtifactType.SIGNALS,)
    output_type = ArtifactType.SURFACES

    def __call__(self, *inputs, **kwargs):
        # Accept either SIGNALS or BINNED input
        if inputs and hasattr(inputs[0], 'output_type'):
            if inputs[0].output_type == ArtifactType.BINNED:
                self.input_types = (ArtifactType.BINNED,)
            elif inputs[0].output_type == ArtifactType.SIGNALS:
                self.input_types = (ArtifactType.SIGNALS,)
        return super().__call__(*inputs, **kwargs)

    def execute(self, *inputs: Artifact, plan: Any = None) -> Artifact:
        input_artifact = inputs[0]

        if input_artifact.type == ArtifactType.SIGNALS:
            # Signal path: measure each LatticeSignal directly
            from ..signal import measure_signal
            signals = input_artifact.value
            agg = self.params.get("agg", "mean")
            surfaces = [measure_signal(s, plan, agg=agg) for s in signals]
        else:
            # Legacy path: binned records
            from ..pipeline.surface import measure
            binned = input_artifact.value
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
