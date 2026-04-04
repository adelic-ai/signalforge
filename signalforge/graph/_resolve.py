"""
signalforge.graph._resolve

Constraint collection and SamplingPlan derivation from a graph.

Geometry derivation is delegated to binjamin.lattice() — the single
canonical path. This module collects constraints from graph nodes
and passes them to binjamin.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def collect_constraints(nodes: list) -> Dict[str, Any]:
    """
    Walk nodes in topological order, merge constraints from each Op.

    Merging rules:
      - grain: take the maximum (coarsest grain that satisfies all ops)
      - windows: take the union
      - horizon: take the maximum if multiple are specified
    """
    grains: list = []
    windows: list = []
    horizons: list = []

    for node in nodes:
        c = node.op.contribute_constraints()
        if "grain" in c:
            grains.append(c["grain"])
        if "windows" in c:
            windows.extend(c["windows"])
        if "horizon" in c:
            horizons.append(c["horizon"])

    result: Dict[str, Any] = {}
    if grains:
        result["grain"] = max(grains)
    if windows:
        result["windows"] = sorted(set(windows))
    if horizons:
        result["horizon"] = max(horizons)

    return result


def derive_grain_from_records(records: Any, method: str = "freedman_diaconis") -> int:
    """
    Estimate data_grain from CanonicalRecords or LatticeSignals.

    Uses binjamin.grain_from_orders().
    """
    import binjamin as bj
    from ..signal._signal import LatticeSignal
    import numpy as np

    # Single LatticeSignal
    if isinstance(records, LatticeSignal):
        return bj.grain_from_orders(records.index.tolist(), method=method)

    # List of LatticeSignals
    if records and isinstance(records[0], LatticeSignal):
        all_orders = np.concatenate([s.index for s in records])
        return bj.grain_from_orders(all_orders.tolist(), method=method)

    orders = [r.primary_order for r in records]
    return bj.grain_from_orders(orders, method=method)


def derive_plan(
    constraints: Dict[str, Any],
    records: Any = None,
) -> Any:
    """
    Derive a SamplingPlan from collected constraints and optionally from data.

    Delegates geometry to binjamin.lattice() — single canonical path.

    Priority:
      1. Explicit overrides in constraints (grain, windows, horizon)
      2. Grain estimated from records if not specified
      3. Geometry derived via binjamin.lattice()
      4. Defaults (horizon=360, grain=1) if nothing is specified

    Returns
    -------
    SamplingPlan
    """
    import binjamin as bj
    from ..lattice.sampling import SamplingPlan

    grain = constraints.get("grain")
    windows = constraints.get("windows")
    horizon = constraints.get("horizon")

    # Derive grain from data if not explicitly set
    if grain is None and records is not None:
        grain = derive_grain_from_records(records)
    if grain is None:
        grain = 1

    # Case 1: windows given — use binjamin.lattice()
    if windows:
        geo = bj.lattice(
            windows=windows,
            grain=grain,
            horizon=horizon,
        )
        return SamplingPlan(
            geo.horizon, grain,
            windows=list(geo.windows),
        )

    # Case 2: horizon given, no windows — dense
    if horizon is not None:
        return SamplingPlan(horizon, grain)

    # Case 3: nothing — default
    return SamplingPlan(360, grain)
