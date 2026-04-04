"""
signalforge.graph._resolve

Constraint collection and SamplingPlan derivation from a graph.

resolve() can derive grain from records automatically using grain_from_orders(),
and horizon from windows using lcm. Explicit overrides always take precedence.
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
    Estimate grain from CanonicalRecords or LatticeSignals.

    Uses grain_from_orders() from the lattice module.
    """
    from ..lattice.sampling import grain_from_orders
    from ..signal._signal import LatticeSignal
    import numpy as np

    # Single LatticeSignal
    if isinstance(records, LatticeSignal):
        return grain_from_orders(records.index.tolist(), method=method)

    # List of LatticeSignals
    if records and isinstance(records[0], LatticeSignal):
        all_orders = np.concatenate([s.index for s in records])
        return grain_from_orders(all_orders.tolist(), method=method)

    orders = [r.primary_order for r in records]
    return grain_from_orders(orders, method=method)


def derive_plan(
    constraints: Dict[str, Any],
    records: Any = None,
) -> Any:
    """
    Derive a SamplingPlan from collected constraints and optionally from data.

    Priority:
      1. Explicit overrides in constraints (grain, windows, horizon)
      2. Grain estimated from records if not specified
      3. Horizon derived from windows via lcm (per paper: H = lcm(W ∪ {g}))
      4. Defaults (horizon=360, grain=1) if nothing is specified

    Returns
    -------
    SamplingPlan
    """
    from ..lattice.sampling import SamplingPlan, suggest_cbin
    from ..lattice.coordinates import smallest_divisor_gte, lattice_members

    grain = constraints.get("grain")
    windows = constraints.get("windows")
    horizon = constraints.get("horizon")

    # Derive grain from data if not explicitly set
    if grain is None and records is not None:
        grain = derive_grain_from_records(records)
    if grain is None and windows:
        # Paper default: g = gcd(W), finest grain the window family permits
        from math import gcd
        from functools import reduce
        grain = reduce(gcd, windows)
    if grain is None:
        grain = 1

    # Case 1: explicit horizon
    if horizon is not None:
        cbin = smallest_divisor_gte(horizon, grain)
        if windows:
            valid = set(lattice_members(horizon, cbin))
            requested = set(windows)
            selected = sorted(w for w in valid if w in requested or w == horizon)
            if not selected:
                selected = sorted(valid)
        else:
            valid = set(lattice_members(horizon, cbin))
            selected = sorted(valid)
        return SamplingPlan(horizon, grain, windows=selected)

    # Case 2: windows given, derive horizon as lcm(W ∪ {grain})
    if windows:
        from math import lcm
        from functools import reduce
        all_vals = windows + [grain]
        horizon = reduce(lcm, all_vals)
        cbin = smallest_divisor_gte(horizon, grain)
        valid = set(lattice_members(horizon, cbin))
        requested = set(windows)
        # Active domain: Div(H) ∩ [cbin, max(W)]
        max_w = max(windows)
        selected = sorted(
            w for w in valid
            if (w in requested or w <= max_w) and w >= cbin
        )
        return SamplingPlan(horizon, grain, windows=selected)

    # Case 3: no windows, no horizon — use default
    horizon = 360
    cbin = smallest_divisor_gte(horizon, grain)
    valid = set(lattice_members(horizon, cbin))
    selected = sorted(valid)
    return SamplingPlan(horizon, grain, windows=selected)
