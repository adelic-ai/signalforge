"""
signalforge.chain

Fluent chaining API for quick exploration.

    import signalforge as sf

    result = (
        sf.load("data.csv")
        .measure(windows=[10, 60, 360])
        .baseline("ewma", alpha=0.1)
        .residual("z")
        .run()
    )

This builds a graph pipeline under the hood. For DAG composition
(branching, merging, multiple outputs), use signalforge.graph directly.

The chain always follows: load → measure → [ops...] → run/heatmap.
Each step returns a new Chain, so the original is never mutated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


class Chain:
    """Fluent pipeline builder.

    Each method returns a new Chain with the step appended.
    Nothing executes until .run() or .heatmap() is called.
    """

    def __init__(self, records=None, *, _steps=None, _csv_path=None):
        self._records = records
        self._steps = list(_steps) if _steps else []
        self._csv_path = _csv_path

    def _copy(self, **overrides) -> "Chain":
        c = Chain(
            records=overrides.get("records", self._records),
            _steps=overrides.get("_steps", self._steps),
            _csv_path=overrides.get("_csv_path", self._csv_path),
        )
        return c

    # --- Data entry ---

    @staticmethod
    def load(source: Union[str, Path, list]) -> "Chain":
        """Load data from a CSV path or a list of CanonicalRecords.

        This is the entry point for chaining.

            sf.load("data.csv").measure().run()
            sf.load(records).measure().run()
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            # Defer import to avoid circular
            from .cli import _auto_ingest
            records = _auto_ingest(path)
            if not records:
                raise ValueError(f"Could not ingest {path}")
            return Chain(records=records, _csv_path=str(path))
        else:
            # Assume list of records
            return Chain(records=source)

    @staticmethod
    def from_signal(signal) -> "Chain":
        """Start a chain from a LatticeSignal directly.

            sig = RealSignal(index, values, channel="x")
            sf.from_signal(sig).measure().run()
        """
        # Wrap single signal in a list for consistency
        if not isinstance(signal, (list, tuple)):
            signal = [signal]
        return Chain(records=signal, _steps=[("signal", {})])

    # --- Pipeline steps ---

    def measure(self, **kwargs) -> "Chain":
        """Measure signals into surfaces.

        kwargs: windows, agg, profile (for legacy path).
        """
        return self._copy(_steps=self._steps + [("measure", kwargs)])

    def baseline(self, method: str = "ewma", **kwargs) -> "Chain":
        """Apply a baseline method to the surfaces.

        method: "ewma", "median", "rolling_mean"
        kwargs: alpha (for ewma), window (for median/rolling_mean)
        """
        kwargs["method"] = method
        return self._copy(_steps=self._steps + [("baseline", kwargs)])

    def residual(self, mode: str = "z", **kwargs) -> "Chain":
        """Compute residual between measured and baseline.

        mode: "difference", "ratio", "z"
        Requires .baseline() before this.
        """
        kwargs["mode"] = mode
        return self._copy(_steps=self._steps + [("residual", kwargs)])

    def hilbert(self, **kwargs) -> "Chain":
        """Apply Hilbert transform — adds amplitude, phase, inst_freq."""
        return self._copy(_steps=self._steps + [("hilbert", kwargs)])

    def gradient(self, **kwargs) -> "Chain":
        """Compute discrete gradient on the lattice — grad_t, grad_p2, grad_p3, ..."""
        return self._copy(_steps=self._steps + [("gradient", kwargs)])

    def stack(self, *others: "Chain") -> "Chain":
        """Stack surfaces from multiple chains along the feature axis.

        Use when you want to combine different processing paths:

            m = sf.load("data.csv").measure()
            bl = m.baseline("ewma")
            r = m.residual("z")  # needs baseline first
            combined = m.stack(bl, r)
        """
        return self._copy(_steps=self._steps + [("stack", {"others": others})])

    # --- Execution ---

    def run(self, **resolve_kwargs) -> Any:
        """Build and execute the pipeline. Returns the final Artifact.

        resolve_kwargs: windows, grain, horizon — override plan derivation.
        """
        from .graph import (
            Input, Measure, Baseline, Residual,
            Hilbert, Gradient, Stack, Pipeline,
        )

        records = self._records
        if records is None:
            raise ValueError("No data loaded. Start with sf.load(path) or sf.load(records).")

        # Build graph
        x = Input()
        measured = None
        current = None

        is_signal_input = False
        for step_name, kwargs in self._steps:
            if step_name == "signal":
                # from_signal path — records is already signals, skip conversion
                is_signal_input = True
                continue
            elif step_name == "measure":
                m_kwargs = {k: v for k, v in kwargs.items() if k != "windows"}
                current = Measure(**m_kwargs)(x)
                measured = current
            elif step_name == "baseline":
                if current is None:
                    raise ValueError(".baseline() requires .measure() first")
                current = Baseline(**kwargs)(measured or current)
            elif step_name == "residual":
                if measured is None:
                    raise ValueError(".residual() requires .measure() first")
                # Find the most recent baseline
                bl = current
                current = Residual(**kwargs)(measured, bl)
            elif step_name == "hilbert":
                if current is None:
                    raise ValueError(".hilbert() requires .measure() first")
                current = Hilbert(**kwargs)(current)
            elif step_name == "gradient":
                if current is None:
                    raise ValueError(".gradient() requires .measure() first")
                current = Gradient(**kwargs)(current)
            elif step_name == "stack":
                # Execute other chains to get their surfaces
                raise NotImplementedError("stack() in chaining not yet implemented — use graph API")

        if current is None:
            # No steps — just measure with defaults
            current = Measure()(x)
            measured = current

        pipe = Pipeline(x, current)

        # Extract windows from steps for resolve
        for step_name, kwargs in self._steps:
            if step_name == "measure" and "windows" in kwargs:
                resolve_kwargs.setdefault("windows", kwargs["windows"])

        return pipe.run(records, **resolve_kwargs)

    def surfaces(self, **resolve_kwargs) -> list:
        """Shortcut: run and return the list of surfaces."""
        result = self.run(**resolve_kwargs)
        return result.value

    def heatmap(self, **resolve_kwargs) -> None:
        """Run the pipeline and display a heatmap."""
        surfaces = self.surfaces(**resolve_kwargs)
        result = self.run(**resolve_kwargs)
        plan = result.plan

        from .cli import _render_heatmap
        _render_heatmap(surfaces, plan, self._csv_path or "")

    def __repr__(self) -> str:
        steps = " → ".join(s[0] for s in self._steps) or "(empty)"
        src = self._csv_path or f"{len(self._records)} records" if self._records else "no data"
        return f"Chain({src} → {steps})"
