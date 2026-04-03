"""
signalforge.pipeline

Internal stage implementations. These are called by graph Ops, not
used directly. The public API is signalforge.graph.

    canonical.py    — CanonicalRecord (ingest output)
    binned.py       — materialize()
    surface.py      — measure()
    feature.py      — engineer()
    bundle.py       — assemble()
    aggregation.py  — aggregation function registry
    operators.py    — transform operators (clip, winsorize, derive, etc.)
"""

from .canonical import CanonicalRecord, OrderType
from .binned import BinnedRecord, materialize
from .surface import Surface, measure, register_profile, get_profile
from .feature import FeatureTensor, engineer
from .bundle import FeatureBundle, SurfaceDataset, assemble
