"""
signalforge — multiscale signal processing on the p-adic divisibility lattice.

SignalForge transforms raw sequential data into structurally invariant feature
tensors for machine learning and analysis. The mathematical foundation is the
p-adic divisibility lattice: every positive integer has a unique prime
factorization, and this module treats that factorization as a coordinate
system — each prime is an axis, each exponent is a coordinate.

The pipeline is domain-agnostic after Stage 0 (Ingest). Domain knowledge lives
exclusively in signalforge.domains. The lattice ensures that surfaces produced
from different data sources with the same SamplingPlan have identical shape and
are directly comparable.

Pipeline stages
---------------
    Stage 0  Ingest        raw source data → CanonicalRecord
    Stage 1  Plan          declarations → SamplingPlan (geometry, not a transform)
    Stage 2  Materialize   CanonicalRecords → BinnedRecord
    Stage 3  Measure       BinnedRecords → Surface (time × scale grid)
    Stage 4  Engineer      Surface → FeatureTensor (derived features)
    Stage 5  Assemble      FeatureTensors → FeatureBundle (ML-ready tensor)

Key objects
-----------
    SamplingPlan    — the complete geometric specification of the measurement space
    CanonicalRecord — normalized, domain-agnostic unit of the pipeline
    Surface         — 2D measurement grid: time × scale
    FeatureTensor   — engineered feature representation derived from a Surface
    FeatureBundle   — multi-channel tensor dataset ready for machine learning

Mathematical primitives
-----------------------
    factorize       — prime exponent coordinate of an integer
    FlipFlop        — seed-based navigator for the p-adic valuation sequence
    Neighborhood    — arithmetic viewing box: the coordinate space made visible
"""

from .pipeline import (
    CanonicalRecord,
    OrderType,
    materialize,
    measure,
    engineer,
    assemble,
    register_profile,
    get_profile,
    Surface,
    FeatureTensor,
    FeatureBundle,
)
from .pipeline.dag import Pipeline
from .lattice import SamplingPlan, factorize, FlipFlop
from .lattice.neighborhood import neighborhood, neighborhood_from_vector, Neighborhood
from . import domains
from .pipeline import operators as ops


def acquire(domain, path: str, **plan_kwargs) -> Pipeline:
    """
    Load data from a domain and return a Pipeline ready to run.

    Parameters
    ----------
    domain : str or module
        Domain name (e.g. "eeg", "intermagnet") or a domain module directly.
    path : str
        Path to the preprocessed CSV file.
    **plan_kwargs
        Passed to domain.sampling_plan().

    Returns
    -------
    Pipeline

    Examples
    --------
    >>> import signalforge as sf
    >>> bundle = (
    ...     sf.acquire("eeg", "chb01_03_eeg_rms.csv")
    ...       .materialize()
    ...       .measure(profile="continuous")
    ...       .engineer()
    ...       .assemble()
    ...       .run()
    ... )
    """
    from importlib import import_module
    if isinstance(domain, str):
        mod = import_module(f"signalforge.domains.{domain}")
    else:
        mod = domain
    records = mod.ingest(path)
    plan = mod.sampling_plan(**plan_kwargs)
    return Pipeline(records, plan)

__version__ = "0.1.0"

__all__ = [
    # Pipeline: Stage 0
    "CanonicalRecord",
    "OrderType",
    # Pipeline: Stage 2
    "materialize",
    # Pipeline: Stage 3
    "measure",
    "Surface",
    "register_profile",
    "get_profile",
    # Pipeline: Stage 4
    "engineer",
    "FeatureTensor",
    # Pipeline: Stage 5
    "assemble",
    "FeatureBundle",
    # Lattice
    "SamplingPlan",
    "factorize",
    "FlipFlop",
    # Neighborhood
    "neighborhood",
    "neighborhood_from_vector",
    "Neighborhood",
    # DAG / fluent API
    "acquire",
    "Pipeline",
    "ops",
    # Subpackages
    "domains",
    # Version
    "__version__",
]
