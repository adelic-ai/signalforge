"""
signalforge.pipeline.canonical

Re-exports from signalforge.signal for backward compatibility.
Canonical source is signalforge.signal._base.
"""

from ..signal._base import CanonicalRecord, OrderType, Keys, KeyValue

__all__ = ["CanonicalRecord", "OrderType", "Keys", "KeyValue"]
