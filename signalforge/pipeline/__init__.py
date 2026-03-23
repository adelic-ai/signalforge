"""
signalforge.pipeline

The six-stage data pipeline.

    Stage 0  canonical.py   — normalize raw source data → CanonicalSequence
    Stage 2  binned.py      — materialize bins → BinnedData
    Stage 3  surface.py     — measure windows → Surface          (forthcoming)
    Stage 4  feature.py     — engineer features → FeatureTensor  (forthcoming)
    Stage 5  bundle.py      — assemble for ML → FeatureBundle     (forthcoming)

Support:
    aggregation.py          — aggregation function registry

Stage 1 (Plan / SamplingPlan) lives in signalforge.lattice.sampling —
it is a geometric artifact, not a pipeline transform.
"""

from .canonical import CanonicalRecord, OrderType
from .binned import BinnedRecord, materialize
from .surface import Surface, measure, register_profile, get_profile
from .feature import FeatureTensor, engineer
from .bundle import FeatureBundle, SurfaceDataset, assemble
