# Architecture

Internal structure for contributors and anyone reading the source.

## Table of Contents

- [Module Layout](#module-layout)
- [Dependency Direction](#dependency-direction)
- [Signal Module](#signal-module)
- [Graph Module](#graph-module)
- [Pipeline Module](#pipeline-module)
- [Lattice Module](#lattice-module)

---

## Module Layout

```
binjamin (external)  # Lattice geometry — factorize, divisors, LatticeGeometry
signalforge/
  lattice/           # SF wrappers — SamplingPlan, FlipFlop, Neighborhood
  signal/            # Type foundation — LatticeSignal, ComplexSignal, Surface, Artifact
  graph/             # Computation graph — Op, Node, Pipeline, resolve
  pipeline/          # Stage implementations — binned, surface, feature, bundle
  domains/           # Domain adapters — ingest functions per data type
  chain.py           # Fluent chaining API
  cli.py             # CLI commands
```

## Dependency Direction

```
binjamin    <- depends on nothing (standalone)
lattice/    <- depends on binjamin
signal/     <- depends on lattice/
graph/      <- depends on signal/, lattice/
pipeline/   <- depends on signal/, lattice/
chain.py    <- depends on graph/, signal/
cli.py      <- depends on everything
```

No circular dependencies. Each layer only looks down. New modules (workspace, export, domain adapters) slot in at the `graph/` level or beside `cli.py` without touching the core.

---

## Signal Module

`signalforge/signal/` is the type foundation. Everything in SignalForge is a `LatticeSignal`.

```
signal/
  _base.py      # Artifact, ArtifactType, CanonicalRecord, OrderType
  _signal.py    # LatticeSignal ABC
  _complex.py   # ComplexSignal, RealSignal
  _surface.py   # Surface (also a LatticeSignal)
  _convert.py   # records_to_signals()
  _measure.py   # measure_signal()
  __init__.py   # Public API
```

**Key design decisions:**

- Values are `complex128` natively. Real signals (`float64`) are the common case.
- `Surface` implements `LatticeSignal` — signal in, signal out. Surfaces can be measured recursively.
- `Artifact` lives here, not in `graph/` — artifacts are the currency of the whole system.
- `CanonicalRecord` lives here — it's a type, not a pipeline stage.

---

## Graph Module

`signalforge/graph/` implements the Keras-style functional computation graph.

```
graph/
  _core.py        # Op ABC, Node, GraphPipeline
  _ops.py         # InputOp, BinOp, MeasureOp, EngineerOp, AssembleOp
  _multi_ops.py   # BaselineOp, ResidualOp, HilbertOp, StackOp
  _resolve.py     # Constraint collection, SamplingPlan derivation
  _types.py       # parse_duration utility
  __init__.py     # User-facing factories (Input, Measure, Baseline, etc.)
```

**Two pipeline paths:**

1. **Signal path** (default): `Input` converts records to `RealSignal`s, `Measure` produces surfaces directly. No explicit `Bin` step.
2. **Legacy path**: `Input(mode="records")` → `Bin` → `Measure`. Still works for backward compatibility.

`InputOp` auto-detects if input is already `LatticeSignal`s (from `sf.from_signal()`) and passes them through without conversion.

---

## Pipeline Module

`signalforge/pipeline/` contains the stage implementations that graph ops call internally.

```
pipeline/
  canonical.py    # Re-exports from signal (backward compat)
  binned.py       # materialize() — records to binned data
  surface.py      # measure() — binned data to surfaces (legacy path)
  aggregation.py  # Aggregation function registry
  feature.py      # engineer() — surfaces to feature tensors
  bundle.py       # assemble() — feature tensors to ML-ready bundle
  operators.py    # Transform operators (clip, winsorize, etc.)
```

The pipeline module is called by graph ops — not used directly by end users. The public API is the graph module and the chaining API.

---

## Lattice Module

`signalforge/lattice/` provides SF-specific wrappers around [binjamin](https://github.com/adelic-ai/binjamin), which owns the lattice geometry math.

```
lattice/
  sampling.py     # SamplingPlan — wraps binjamin.LatticeGeometry + adds hops
  neighborhood.py # Neighborhood — the p-adic viewing box (SF-specific)
  flipflop.py     # FlipFlop — seed-based lattice navigator (SF-specific)
  __init__.py     # Public API
```

**Boundary:** binjamin owns all lattice geometry derivation (factorization, divisors, lattice members, coordinate vectors, grain/cbin/horizon computation). SignalForge consumes that geometry and adds signal-specific layers (hops, profiles, validation). No geometry is derived in SF — all paths go through `binjamin.lattice()`.
