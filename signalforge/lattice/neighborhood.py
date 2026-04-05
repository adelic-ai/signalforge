"""
signalforge.lattice.neighborhood

The arithmetic viewing box: the prime coordinate space made visible.

For an anchor integer n and radius r, constructs a 2D grid:

    rows  : prime basis (ascending)
    cols  : integers n-r, n-r+1, ..., n, ..., n+r
    cells : v_p(integer) — dense, zeros explicit

This is the same shape as a signal Surface, but constructed from
arithmetic rather than signal data. Every column is a dense coordinate
vector in the prime basis. The zeros are data — v_p(n) = 0 is the
statement that p does not divide n, and it belongs in the grid.

What you see:
    prime      — column with exactly one nonzero entry, value 1
    prime power — column with one row elevated above 1, rest zero
    composite  — column with nonzero entries in multiple rows
    1          — all-zero column (origin of the lattice)

The prime basis extends on demand. Auto-detection collects every prime
that appears as a factor of any integer in the window. Passing an
explicit basis pins the axes, including primes with all-zero columns.

Original motivation: the gnuplot visualization from upfpy (~2013).
See: docs/discovery_flipflop_prime_structure.md
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from binjamin import Coordinate, factorize, to_int


class Neighborhood:
    """
    The arithmetic viewing box centered on an anchor integer.

    Rows are primes (ascending). Columns are consecutive integers.
    Every cell holds v_p(integer) — dense, zeros explicit.

    This IS the coordinate space. The signal pipeline uses the lattice
    as scaffolding; this makes the lattice itself the subject.

    Immutable after construction.

    Attributes
    ----------
    anchor : int
        The center integer.
    radius : int
        Half-width. Total columns = min(anchor, radius+1) + radius + 1
        (clamped at 1 on the left).
    prime_basis : tuple[int, ...]
        Prime row axes, ascending.
    integers : tuple[int, ...]
        Column positions: max(1, anchor-radius) .. anchor+radius.
    valuations : np.ndarray
        Shape (n_primes, n_integers), dtype int32.
        valuations[i, j] = v_{prime_basis[i]}(integers[j]).
    """

    __slots__ = ("anchor", "radius", "prime_basis", "integers", "valuations")

    def __init__(
        self,
        anchor: int,
        radius: int,
        prime_basis: Tuple[int, ...],
        integers: Tuple[int, ...],
        valuations: np.ndarray,
    ) -> None:
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "prime_basis", prime_basis)
        object.__setattr__(self, "integers", integers)
        object.__setattr__(self, "valuations", valuations)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Neighborhood is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Neighborhood is immutable")

    @property
    def shape(self) -> Tuple[int, int]:
        """(n_primes, n_integers)"""
        return self.valuations.shape  # type: ignore[return-value]

    def column(self, n: int) -> np.ndarray:
        """
        Dense coordinate vector for integer n.

        Returns a 1D array of length n_primes where entry i is
        v_{prime_basis[i]}(n). Zeros are explicit.
        """
        lo = self.integers[0]
        j = n - lo
        if j < 0 or j >= len(self.integers):
            raise ValueError(
                f"{n} is outside the window "
                f"[{self.integers[0]}, {self.integers[-1]}]"
            )
        return self.valuations[:, j].copy()

    def row(self, p: int) -> np.ndarray:
        """
        Valuation sequence for prime p across the full window.

        Returns a 1D array of length n_integers where entry j is
        v_p(integers[j]).
        """
        if p not in self.prime_basis:
            raise ValueError(f"Prime {p} not in basis {self.prime_basis}")
        i = self.prime_basis.index(p)
        return self.valuations[i, :].copy()

    def is_prime_column(self, n: int) -> bool:
        """
        True if n appears prime relative to the current basis.

        A prime column has exactly one nonzero entry with value 1 —
        one prime factor, exponent 1, no others.

        Note: returns False for primes whose only prime factor lies
        outside the current basis (they appear as all-zero columns).
        Use residual() to detect those.
        """
        col = self.column(n)
        return int(col.sum()) == 1 and int(col.max()) == 1

    def residual(self, n: int) -> int:
        """
        The part of n not accounted for by the current prime basis.

            residual(n) = n // prod(p^v_p(n) for p in basis)

        If residual > 1, n has prime factors outside the current basis.
        residual == 1 means n is fully factored by the basis.
        residual == n means none of the basis primes divide n.
        """
        col = self.column(n)
        accounted = 1
        for p, v in zip(self.prime_basis, col.tolist()):
            accounted *= p ** v
        return n // accounted

    def show(self, zeros: str = ".") -> None:
        """
        Print the viewing box grid.

        Parameters
        ----------
        zeros : str
            Character to display for zero entries. Default ".".
        """
        n_primes, n_cols = self.shape
        col_w = max(len(str(max(self.integers, default=0))), 3)
        prime_w = max(len(str(max(self.prime_basis, default=0))), 3)

        # Header row: integer values.
        pad = " " * (prime_w + 3)
        header = pad + "  ".join(str(n).rjust(col_w) for n in self.integers)
        sep = pad + "-" * (col_w * n_cols + 2 * (n_cols - 1))
        print(header)
        print(sep)

        # One row per prime.
        for i, p in enumerate(self.prime_basis):
            cells = "  ".join(
                str(int(v)).rjust(col_w) if v > 0 else zeros.rjust(col_w)
                for v in self.valuations[i]
            )
            print(f"  {str(p).rjust(prime_w)}  {cells}")

    def __repr__(self) -> str:
        return (
            f"Neighborhood(anchor={self.anchor}, radius={self.radius}, "
            f"n_primes={len(self.prime_basis)}, shape={self.shape})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _auto_basis(integers: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Derive the prime basis from the factorizations of integers in the range.
    Collects every prime that appears as a factor of any integer in the window.
    """
    primes: set = set()
    for n in integers:
        if n > 1:
            primes.update(factorize(n).keys())
    return tuple(sorted(primes))


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------


def neighborhood(
    anchor: int,
    radius: int,
    basis: Optional[List[int]] = None,
) -> Neighborhood:
    """
    Build the arithmetic viewing box centered on anchor.

    Parameters
    ----------
    anchor : int
        The center integer. Must be >= 1.
    radius : int
        Half-width. Window covers max(1, anchor-radius) .. anchor+radius.
    basis : list of int, optional
        Prime axes. If None, auto-detected from the factorizations of
        integers in the window — the basis is exactly the primes needed
        to explain the window, no more.

        Pass an explicit basis to:
          - include primes with all-zero columns (context primes)
          - fix the axes for comparison across multiple neighborhoods
          - control which primes are visible

    Returns
    -------
    Neighborhood
        Dense grid. Zeros are explicit.

    Examples
    --------
    >>> nb = neighborhood(36, 6)
    >>> nb.show()

    >>> nb = neighborhood(100, 10, basis=[2, 3, 5, 7, 11, 13])
    >>> nb.show()
    """
    if anchor < 1:
        raise ValueError(f"anchor must be >= 1, got {anchor}")
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")

    lo = max(1, anchor - radius)
    hi = anchor + radius
    integers = tuple(range(lo, hi + 1))

    prime_basis = tuple(sorted(set(basis))) if basis is not None else _auto_basis(integers)

    n_primes = len(prime_basis)
    n_cols = len(integers)
    valuations = np.zeros((n_primes, n_cols), dtype=np.int32)

    for j, n in enumerate(integers):
        if n < 2:
            continue
        coord = factorize(n)
        for i, p in enumerate(prime_basis):
            valuations[i, j] = coord.get(p, 0)

    return Neighborhood(anchor, radius, prime_basis, integers, valuations)


def neighborhood_from_vector(
    vector: Coordinate,
    radius: int,
) -> Neighborhood:
    """
    Build a Neighborhood from a prime exponent coordinate vector.

    The anchor integer is reconstructed via to_int(vector). The basis
    starts with the primes in the vector and is extended by any primes
    discovered in the window.

    Parameters
    ----------
    vector : dict[int, int]
        Prime exponent coordinate. {2: 2, 3: 1} → anchor = 12.
    radius : int
        Half-width of the window.

    Returns
    -------
    Neighborhood

    Notes
    -----
    For vectors with large exponents, to_int() returns an exact but
    astronomically large integer. Factorizing its neighborhood integers
    may be slow. Pass an explicit basis and pre-computed anchor if needed.

    Examples
    --------
    >>> nb = neighborhood_from_vector({2: 2, 3: 1}, 6)   # anchor = 12
    >>> nb.show()
    """
    anchor = to_int(vector)
    lo = max(1, anchor - radius)
    hi = anchor + radius
    integers = tuple(range(lo, hi + 1))
    discovered = _auto_basis(integers)
    combined = tuple(sorted(set(vector.keys()) | set(discovered)))
    return neighborhood(anchor, radius, basis=list(combined))
