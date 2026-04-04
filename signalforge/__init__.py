"""
signalforge — multiscale signal analysis on a normalized scale space.

SignalForge transforms any ordered sequence into structurally invariant
multiscale surfaces. The measurement space is derived from the arithmetic
of the signal's grain and window declarations — not designed, not chosen.

Quick exploration (chaining API):

    import signalforge as sf

    surfaces = (
        sf.load("data.csv")
        .measure(windows=[10, 60, 360])
        .baseline("ewma", alpha=0.1)
        .residual("z")
        .surfaces()
    )

Full composition (graph API):

    from signalforge.graph import Input, Measure, Baseline, Residual, Pipeline

    x = Input()
    m = Measure()(x)
    bl = Baseline("ewma", alpha=0.1)(m)
    r = Residual("z")(m, bl)
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

# Types — canonical source is signalforge.signal
from .signal import CanonicalRecord, OrderType, Surface

# Chaining API
from .chain import Chain
load = Chain.load
from_signal = Chain.from_signal

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
    # Chaining API
    "Chain",
    "load",
    "from_signal",
    # Subpackages
    "domains",
    # Version
    "__version__",
]
