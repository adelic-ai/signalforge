"""
signalforge.lattice.sampling

SamplingPlan: the complete geometric specification of the measurement space.

Two declarations drive everything:

    horizon : int  — outer boundary of the coordinate space; all valid windows
                     divide it; may be larger than the actual data span
    grain   : int  — finest resolution the data supports; used to derive cbin

All other fields are derived. The plan is immutable after construction and
validated at instantiation time: a SamplingPlan that violates any invariant
cannot be created.

Window selection
----------------
Pass an explicit sequence of window values, or the string "dense" to use all
valid lattice members. Domain-specific presets belong in domain modules
(e.g. signalforge.domains.intermagnet), not here.

Helper
------
horizon_for(windows, grain) — compute the smallest horizon that makes all
    windows lattice members, given a grain.
"""

from __future__ import annotations

from functools import reduce
from math import gcd
from typing import Sequence, Tuple, Union

from .coordinates import (
    Coordinate,
    factorize,
    lattice_members,
    smallest_divisor_gte,
)

WindowSpec = Union[str, Sequence[int]]


def horizon_for(windows: Sequence[int], grain: int) -> int:
    """
    Compute the smallest horizon that makes all windows valid lattice members.

    The horizon must be divisible by every window. The smallest such value
    is lcm(windows). The grain is used to verify the result is compatible
    (cbin will be derivable from the returned horizon).

    Parameters
    ----------
    windows : sequence of int
        The windows you need. All must be positive integers.
    grain : int
        The finest resolution the data supports.

    Returns
    -------
    int
        The smallest horizon >= max(windows) such that all windows divide it
        and smallest_divisor_gte(horizon, grain) divides all windows.

    Examples
    --------
    >>> horizon_for([900, 1800, 3600], 60)
    3600
    >>> horizon_for([900, 7200], 60)
    7200
    """
    if not windows:
        raise ValueError("windows must be non-empty")
    lcm = reduce(lambda a, b: a * b // gcd(a, b), windows)
    return lcm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_windows(
    spec: WindowSpec,
    valid: Tuple[int, ...],
) -> Tuple[int, ...]:
    """
    Resolve a window specification to a sorted tuple of valid lattice members.

    "dense"    — all valid lattice members
    [int, ...] — explicit list; every value must be a valid lattice member
    """
    if isinstance(spec, str):
        if spec.lower() != "dense":
            raise ValueError(
                f"Unknown window preset {spec!r}. "
                f"Use 'dense' or pass an explicit sequence of window values. "
                f"For domain-specific presets see signalforge.domains."
            )
        return valid

    explicit = tuple(sorted(set(int(w) for w in spec)))
    if not explicit:
        raise ValueError("Window sequence is empty.")
    valid_set = set(valid)
    bad = [w for w in explicit if w not in valid_set]
    if bad:
        raise ValueError(
            f"Windows {bad} are not valid lattice members. "
            f"Valid set: {valid}"
        )
    return explicit


def _default_hops(windows: Tuple[int, ...], cbin: int) -> Tuple[int, ...]:
    """
    Default hop: cbin for every window.

    cbin divides every window by construction and is a multiple of itself,
    so it is always a valid hop. This gives the finest resolution.
    """
    return tuple(cbin for _ in windows)


def _validate_hops(
    windows: Tuple[int, ...],
    hops: Tuple[int, ...],
    cbin: int,
) -> None:
    if len(hops) != len(windows):
        raise ValueError(
            f"len(hops)={len(hops)} must equal len(windows)={len(windows)}"
        )
    for window, hop in zip(windows, hops):
        if hop % cbin != 0:
            raise ValueError(f"hop {hop} is not a multiple of cbin {cbin}")
        if window % hop != 0:
            raise ValueError(f"hop {hop} does not divide window {window}")


def _validate_invariants(
    horizon: int,
    cbin: int,
    windows: Tuple[int, ...],
    hops: Tuple[int, ...],
    n_values: Tuple[int, ...],
    coordinates: Tuple[Coordinate, ...],
) -> None:
    """
    Full invariant check. Raises ValueError on the first violation.

    Invariants:
        cbin | horizon

        for every (window, hop, n, coord):
            window | horizon
            cbin   | window
            cbin   | hop
            hop    | window
            window == n * hop
            window == product(p**e for p, e in coord.items()) * cbin
    """
    if horizon % cbin != 0:
        raise ValueError(f"cbin {cbin} does not divide horizon {horizon}")

    for window, hop, n, coord in zip(windows, hops, n_values, coordinates):
        if horizon % window != 0:
            raise ValueError(
                f"window {window} does not divide horizon {horizon}"
            )
        if window % cbin != 0:
            raise ValueError(f"cbin {cbin} does not divide window {window}")
        if hop % cbin != 0:
            raise ValueError(f"cbin {cbin} does not divide hop {hop}")
        if window % hop != 0:
            raise ValueError(f"hop {hop} does not divide window {window}")
        if window != n * hop:
            raise ValueError(f"window {window} != n={n} * hop={hop}")
        reconstructed = cbin
        for prime, exp in coord.items():
            reconstructed *= prime ** exp
        if reconstructed != window:
            raise ValueError(
                f"Coordinate check failed for window {window}: "
                f"coord={dict(coord)} * cbin={cbin} = {reconstructed}"
            )


# ---------------------------------------------------------------------------
# SamplingPlan
# ---------------------------------------------------------------------------


class SamplingPlan:
    """
    The complete geometric specification of the measurement space.

    Constructed from two declarations; everything else is derived and
    validated. The object is immutable after construction.

    Parameters
    ----------
    horizon : int
        Outer boundary of the coordinate space. All valid windows divide it.
        May be larger than the actual data span — that is fine and useful.
        Must be a positive integer.
    grain : int
        Finest resolution the data supports. Used to derive cbin.
        Must satisfy 1 <= grain <= horizon.
    windows : str or sequence of int, optional
        Window selection:
        - "dense"    : all valid lattice members (default)
        - [int, ...] : explicit list; each must divide horizon and be a
                       multiple of cbin
        Domain-specific presets live in signalforge.domains.
    hops : sequence of int, optional
        One hop per selected window. Each must be a multiple of cbin that
        divides its window. Defaults to cbin for every window.

    Fields (read-only)
    ------------------
    horizon      : int
    grain        : int                 construction input (read-only view)
    cbin         : int                 smallest divisor of horizon >= grain
    prime_basis  : dict[int, int]      factorize(horizon // cbin)
    windows      : tuple[int, ...]     selected window sizes, ascending
    hops         : tuple[int, ...]     one hop per window
    n_values     : tuple[int, ...]     window // hop per pair
    coordinates  : tuple[dict, ...]    prime exponent vector per window

    Examples
    --------
    >>> plan = SamplingPlan(3600, 250)
    >>> plan.cbin
    300
    >>> plan.prime_basis
    {2: 2, 3: 1}
    >>> plan.windows
    (300, 600, 900, 1200, 1800, 3600)
    >>> plan.coordinates
    ({}, {2: 1}, {3: 1}, {2: 2}, {2: 1, 3: 1}, {2: 2, 3: 1})
    """

    __slots__ = (
        "_horizon",
        "_grain",
        "_cbin",
        "_prime_basis",
        "_windows",
        "_hops",
        "_n_values",
        "_coordinates",
    )

    def __init__(
        self,
        horizon: int,
        grain: int,
        windows: WindowSpec = "dense",
        hops: Sequence[int] | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError(
                f"horizon must be a positive integer, got {horizon}"
            )
        if grain < 1:
            raise ValueError(
                f"grain must be a positive integer, got {grain}"
            )
        if grain > horizon:
            raise ValueError(
                f"grain {grain} > horizon {horizon}"
            )

        cbin = smallest_divisor_gte(horizon, grain)
        ratio = horizon // cbin
        prime_basis: Coordinate = factorize(ratio) if ratio > 1 else {}
        valid = lattice_members(horizon, cbin)
        resolved_windows = _resolve_windows(windows, valid)

        if hops is None:
            resolved_hops = _default_hops(resolved_windows, cbin)
        else:
            resolved_hops = tuple(int(h) for h in hops)
        _validate_hops(resolved_windows, resolved_hops, cbin)

        n_values = tuple(w // h for w, h in zip(resolved_windows, resolved_hops))

        coordinates: Tuple[Coordinate, ...] = tuple(
            factorize(w // cbin) if w > cbin else {}
            for w in resolved_windows
        )

        _validate_invariants(
            horizon, cbin, resolved_windows, resolved_hops, n_values, coordinates
        )

        object.__setattr__(self, "_horizon", horizon)
        object.__setattr__(self, "_grain", grain)
        object.__setattr__(self, "_cbin", cbin)
        object.__setattr__(self, "_prime_basis", prime_basis)
        object.__setattr__(self, "_windows", resolved_windows)
        object.__setattr__(self, "_hops", resolved_hops)
        object.__setattr__(self, "_n_values", n_values)
        object.__setattr__(self, "_coordinates", coordinates)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("SamplingPlan is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("SamplingPlan is immutable")

    # --- Fields ---

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def grain(self) -> int:
        """Construction input. Read-only view."""
        return self._grain

    @property
    def cbin(self) -> int:
        return self._cbin

    @property
    def prime_basis(self) -> Coordinate:
        return dict(self._prime_basis)

    @property
    def windows(self) -> Tuple[int, ...]:
        return self._windows

    @property
    def hops(self) -> Tuple[int, ...]:
        return self._hops

    @property
    def n_values(self) -> Tuple[int, ...]:
        return self._n_values

    @property
    def coordinates(self) -> Tuple[Coordinate, ...]:
        return tuple(dict(c) for c in self._coordinates)

    # --- Inspection ---

    def valid_windows(self) -> Tuple[int, ...]:
        """Full lattice member set — all divisors of horizon that are multiples of cbin."""
        return lattice_members(self._horizon, self._cbin)

    def describe(self) -> None:
        """Print the full derivation chain from declarations to coordinates."""
        ratio = self._horizon // self._cbin
        valid = self.valid_windows()
        print("SamplingPlan")
        print(f"  horizon     : {self._horizon}")
        print(f"  grain       : {self._grain}")
        print(
            f"  cbin        : {self._cbin}"
            f"  (smallest divisor of {self._horizon} >= {self._grain})"
        )
        print(f"  ratio       : {self._horizon} / {self._cbin} = {ratio}")
        print(f"  prime_basis : {dict(self._prime_basis)}")
        print(f"  valid windows ({len(valid)}): {valid}")
        print(f"  selected ({len(self._windows)} windows):")
        for w, h, n, c in zip(
            self._windows, self._hops, self._n_values, self._coordinates
        ):
            print(f"    window={w:>10}  hop={h:>8}  n={n:>6}  coord={dict(c)}")

    def show_lattice(self) -> None:
        """
        Render a text Hasse diagram of the selected windows.

        An edge a → b means a divides b and no c in the selected set
        satisfies a | c | b (cover relation).
        """
        windows = self._windows
        edges: list[tuple[int, int]] = []
        for i, a in enumerate(windows):
            for j, b in enumerate(windows):
                if i >= j:
                    continue
                if b % a != 0:
                    continue
                mediated = any(
                    b % c == 0 and c % a == 0 and c != a and c != b
                    for c in windows
                )
                if not mediated:
                    edges.append((a, b))

        print(f"Hasse diagram  cbin={self._cbin}  ({len(windows)} windows)")
        print(f"  Nodes : {windows}")
        print(f"  Covers (a → b  means  a | b, no intermediary):")
        for a, b in edges:
            print(f"    {a} → {b}")

    # --- Dunder ---

    def __repr__(self) -> str:
        return (
            f"SamplingPlan("
            f"horizon={self._horizon}, "
            f"cbin={self._cbin}, "
            f"prime_basis={dict(self._prime_basis)}, "
            f"windows={self._windows})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SamplingPlan):
            return NotImplemented
        return (
            self._horizon == other._horizon
            and self._cbin == other._cbin
            and self._windows == other._windows
            and self._hops == other._hops
        )

    def __hash__(self) -> int:
        return hash((self._horizon, self._cbin, self._windows, self._hops))
