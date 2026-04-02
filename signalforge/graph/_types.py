"""
signalforge.graph._types

Artifact types and utility functions for the computation graph.
"""

from __future__ import annotations

import enum
import re


class ArtifactType(enum.Enum):
    """Tags what a graph node produces."""
    RECORDS  = "records"    # list[CanonicalRecord]
    BINNED   = "binned"     # list[BinnedRecord]
    SURFACES = "surfaces"   # list[Surface]
    TENSORS  = "tensors"    # list[FeatureTensor]
    BUNDLE   = "bundle"     # FeatureBundle


# Duration string → integer seconds
_DURATION_RE = re.compile(r"^(\d+)\s*(s|m|h|d|w)$", re.IGNORECASE)
_DURATION_MULTIPLIERS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_duration(s: str | int) -> int:
    """
    Parse a duration string like '5m', '1h', '24h' to integer seconds.
    Integers pass through unchanged.
    """
    if isinstance(s, (int, float)):
        return int(s)
    m = _DURATION_RE.match(str(s).strip())
    if not m:
        raise ValueError(f"Cannot parse duration: {s!r}. Use e.g. '5m', '1h', '24h'.")
    return int(m.group(1)) * _DURATION_MULTIPLIERS[m.group(2).lower()]
