"""
signalforge.graph._resolve

Constraint collection and SamplingPlan derivation from a graph.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._types import parse_duration


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


def derive_plan(constraints: Dict[str, Any]) -> Any:
    """
    Derive a SamplingPlan from collected constraints.

    If windows and grain are both present, constructs a plan from those.
    If only grain is present, uses a default horizon.
    Falls back to a minimal default plan if no constraints are given.
    """
    from ..lattice.sampling import SamplingPlan
    from ..lattice.coordinates import smallest_divisor_gte, lattice_members

    grain = constraints.get("grain", 1)
    windows = constraints.get("windows")
    horizon = constraints.get("horizon")

    if horizon is not None:
        # Explicit horizon — use it, derive windows from lattice if not given
        cbin = smallest_divisor_gte(horizon, grain)
        if windows:
            # Validate that requested windows are valid divisors
            valid = set(lattice_members(horizon, cbin))
            selected = sorted(w for w in valid if w in set(windows) or w == horizon)
            if not selected:
                selected = sorted(valid)
        else:
            # All lattice members as windows
            valid = set(lattice_members(horizon, cbin))
            selected = sorted(valid)
        return SamplingPlan(horizon, grain, windows=selected)

    if windows:
        # Derive horizon from windows as lcm
        from math import lcm
        from functools import reduce
        all_vals = windows + [grain]
        horizon = reduce(lcm, all_vals)
        cbin = smallest_divisor_gte(horizon, grain)
        valid = set(lattice_members(horizon, cbin))
        # Include requested windows plus any lattice members between them
        selected = sorted(w for w in valid if w in set(windows) or w <= max(windows))
        return SamplingPlan(horizon, grain, windows=selected)

    # No constraints — default plan (360, grain=1)
    horizon = 360
    cbin = smallest_divisor_gte(horizon, grain)
    valid = set(lattice_members(horizon, cbin))
    selected = sorted(valid)
    return SamplingPlan(horizon, grain, windows=selected)
