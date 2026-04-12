# Architecture

Internal structure for contributors and anyone reading the source.

## Module Layout

```
binjamin (external)  # Lattice geometry — factorize, divisors, LatticeGeometry
signalforge/
  lattice/           # SF wrappers — SamplingPlan, FlipFlop, Neighborhood
  signal/            # Type foundation — LatticeSignal, Schema, Record, Surface
  graph/             # Computation graph — Op, Node, Pipeline, resolve
  pipeline/          # Legacy stage implementations (being phased out)
  domains/           # Domain adapters — ingest via Schema + Record
  chain.py           # Fluent chaining API
  cli.py             # CLI commands
```

## Dependency Direction

```
binjamin    <- depends on nothing (standalone)
lattice/    <- depends on binjamin
signal/     <- depends on lattice/
graph/      <- depends on signal/, lattice/
chain.py    <- depends on graph/, signal/
cli.py      <- depends on everything
```

No circular dependencies. Each layer only looks down.

---

## Signal Module

`signalforge/signal/` is the type foundation. Everything is a `LatticeSignal`.

```
signal/
  _base.py      # Artifact, ArtifactType, (legacy: CanonicalRecord)
  _signal.py    # LatticeSignal ABC
  _complex.py   # ComplexSignal, RealSignal
  _surface.py   # Surface (also a LatticeSignal)
  _schema.py    # Schema, Axis, AxisType — typed axes
  _record.py    # Record — universal event type
  _convert.py   # records_to_signals()
  _measure.py   # measure_signal() — multi-aggregation, prefix sums
  _segment.py   # Segment discovery — gap-based activity boundaries
  _features.py  # Feature extraction, labeling, join enrichment
  __init__.py   # Public API
```

**Key design decisions:**

- Values are `complex128` natively. Real signals (`float64`) are the common case.
- `Surface` implements `LatticeSignal` — signal in, signal out.
- `Schema` + `Record` replace `CanonicalRecord` for data ingest. An event is a point in a product of typed axes (ORDERED, CATEGORICAL, NUMERIC, RELATIONAL).
- `measure_signal()` supports multiple aggregations per surface via prefix sums (fast) and per-window fallback (any registered aggregation).
- `discover_segments()` finds natural activity boundaries from inter-event gaps. Domain-agnostic.
- `join_segments()` enriches segments with cross-segment context — group by a different key, compute fanout/fanin/co-occurrence. The caller picks the join key; SF computes the aggregates.
- Each `Artifact` has a deterministic ID (hash of plan + op + parent IDs) for caching and lineage.

---

## Graph Module

`signalforge/graph/` implements the Keras-style functional computation graph.

```
graph/
  _core.py        # Op ABC, Node, GraphPipeline, transform validation
  _ops.py         # InputOp, BinOp, MeasureOp, EngineerOp, AssembleOp
  _multi_ops.py   # BaselineOp, ResidualOp, HilbertOp, GradientOp, StackOp
  _resolve.py     # Constraint collection, SamplingPlan derivation via binjamin
  _types.py       # parse_duration utility
  __init__.py     # User-facing factories (Input, Measure, Baseline, Gradient, etc.)
```

**Two pipeline paths:**

1. **Signal path** (default): `Input` converts records to `RealSignal`s, `Measure` produces surfaces directly.
2. **Legacy path**: `Input(mode="records")` → `Bin` → `Measure`. Being phased out.

`InputOp` auto-detects if input is already `LatticeSignal`s and passes them through.

Every op output is validated at runtime — type checks, shape consistency, non-empty values.

---

## Pipeline Module

`signalforge/pipeline/` contains legacy stage implementations. Being replaced by `signal/_measure.py` (multi-aggregation, vectorized).

Remaining uses: `aggregation.py` (aggregation function registry — still active), `feature.py` and `bundle.py` (feature engineering — used by EngineerOp and AssembleOp).

---

## Lattice Module

`signalforge/lattice/` provides SF-specific wrappers around [binjamin](https://github.com/adelic-ai/binjamin).

```
lattice/
  sampling.py     # SamplingPlan — wraps binjamin.LatticeGeometry + adds hops
  neighborhood.py # Neighborhood — the p-adic viewing box
  flipflop.py     # FlipFlop — seed-based lattice navigator
  __init__.py     # Public API
```

**Boundary:** binjamin owns all lattice geometry derivation. SignalForge consumes it. No geometry is derived in SF — all paths go through `binjamin.lattice()`.

---

## Domain Adapters

`signalforge/domains/` provides ingest functions for specific data formats. Each adapter creates a `Schema` and returns `Record`s.

```
domains/
  timeseries.py   # Two-column CSV (date + value)
  eeg.py          # CHB-MIT EEG (t_sec + eeg_rms)
  equities.py     # Yahoo Finance (timestamp + ticker + metric + value)
  intermagnet.py  # IAGA-2002 geomagnetic (timestamp + station + component + value)
```

For CSV data, `sf schema` replaces domain adapters — no code needed. Domain adapters are for native binary formats or complex parsing logic.
