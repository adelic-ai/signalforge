"""
signalforge — multiscale signal analysis on a normalized scale space.

SignalForge transforms any ordered sequence into structurally invariant
multiscale surfaces. The measurement space is derived from the arithmetic
of the signal's grain and window declarations — not designed, not chosen.

The graph API is the primary interface. Compose operators into a DAG,
resolve the geometry, run.

    from signalforge.graph import Input, Bin, Measure, Baseline, Residual, Pipeline
    from signalforge.domains import timeseries

    records = timeseries.ingest("data.csv")

    x = Input()
    b = Bin()(x)
    m = Measure()(b)
    bl = Baseline("ewma", alpha=0.1)(m)
    r = Residual("ratio")(m, bl)

    pipe = Pipeline(x, r)
    result = pipe.run(records)

Key objects
-----------
    SamplingPlan    — the complete geometric specification of the measurement space
    CanonicalRecord — normalized, domain-agnostic unit of ingest
    Surface         — 2D measurement grid: time × scale
    Artifact        — typed container with data, lineage, and plan context

Mathematical primitives
-----------------------
    factorize       — prime exponent coordinate of an integer
    FlipFlop        — seed-based navigator for the p-adic valuation sequence
    Neighborhood    — arithmetic viewing box: the coordinate space made visible
"""

# Graph API (primary interface)
from .graph import (
    Input,
    Bin,
    Measure,
    Engineer,
    Assemble,
    Baseline,
    Residual,
    Hilbert,
    Stack,
    Pipeline,
    Artifact,
    ArtifactType,
    Node,
    Op,
)

# Lattice
from .lattice import SamplingPlan, factorize, FlipFlop, horizon_for
from .lattice.neighborhood import neighborhood, neighborhood_from_vector, Neighborhood

# Types (still needed for domain ingest functions)
from .pipeline.canonical import CanonicalRecord, OrderType
from .pipeline.surface import Surface

# Domains
from . import domains

__version__ = "0.2.1"

__all__ = [
    # Graph API
    "Input",
    "Bin",
    "Measure",
    "Engineer",
    "Assemble",
    "Baseline",
    "Residual",
    "Hilbert",
    "Stack",
    "Pipeline",
    "Artifact",
    "ArtifactType",
    "Node",
    "Op",
    # Lattice
    "SamplingPlan",
    "factorize",
    "FlipFlop",
    "horizon_for",
    # Neighborhood
    "neighborhood",
    "neighborhood_from_vector",
    "Neighborhood",
    # Types
    "CanonicalRecord",
    "OrderType",
    "Surface",
    # Subpackages
    "domains",
    # Version
    "__version__",
]
