"""
signalforge.lattice.coordinates

Prime exponent coordinate system for the integers.

Every positive integer n maps uniquely to a coordinate vector in N^k where
each axis corresponds to a prime and each coordinate is the exponent of that
prime in the factorization of n:

    n = p1^a1 * p2^a2 * ... * pk^ak  <-->  (a1, a2, ..., ak)

Under this embedding:
    multiplication   <-->  coordinate addition
    divisibility     <-->  coordinate-wise <=
    gcd              <-->  coordinate-wise min
    lcm              <-->  coordinate-wise max

This module provides the factorization and vector arithmetic that the
SamplingPlan uses to derive the prime basis, enumerate lattice members,
and assign coordinates to windows.

Original implementation by Shun Richard Honda (upfpy, 2002–2013).

Factorization backend
---------------------
Default: internal wheel trial division (6k±1, up to √n). Fast and
self-contained for the integers this system works with.

Power users can swap to sympy or any callable that satisfies:
    fn(n: int) -> dict[int, int]   # prime → exponent

    from signalforge.lattice.coordinates import set_factorization_backend
    set_factorization_backend("sympy")
    set_factorization_backend(my_fn)

The cache is shared regardless of backend. Clearing it after a backend
switch ensures consistency: clear_factorization_cache().
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional, Tuple

# A Coordinate is a dict mapping prime -> exponent.
# e.g. 12 = 2^2 * 3^1  -->  {2: 2, 3: 1}
Coordinate = Dict[int, int]

# ---------------------------------------------------------------------------
# Factorization backends
# ---------------------------------------------------------------------------

# Timing threshold in seconds — if a single factorization takes longer than
# this, the adaptive switcher will recommend (but not force) a backend change.
_TIMING_THRESHOLD: float = 0.01  # 10ms


def _wheel_factorize(n: int) -> Coordinate:
    """
    Trial division with 6k±1 wheel.

    Every prime > 3 is of the form 6k−1 or 6k+1. We check 2 and 3 first,
    then only 6k±1 candidates up to √n. If no factor is found by √n, n
    is prime.

    Correct and fast for all integers this system encounters (≤ horizon,
    typically ≤ 10^7).
    """
    if n < 1:
        raise ValueError(f"factorize requires a positive integer, got {n}")
    if n == 1:
        return {}

    factors: Coordinate = {}
    remaining = n

    # Extract factors of 2.
    while remaining % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        remaining //= 2

    # Extract factors of 3.
    while remaining % 3 == 0:
        factors[3] = factors.get(3, 0) + 1
        remaining //= 3

    # Trial division with 6k±1 candidates up to √remaining.
    k = 1
    while True:
        candidate = 6 * k - 1
        if candidate * candidate > remaining:
            break
        while remaining % candidate == 0:
            factors[candidate] = factors.get(candidate, 0) + 1
            remaining //= candidate

        candidate = 6 * k + 1
        if candidate * candidate > remaining:
            break
        while remaining % candidate == 0:
            factors[candidate] = factors.get(candidate, 0) + 1
            remaining //= candidate

        k += 1

    # If remaining > 1, it is prime.
    if remaining > 1:
        factors[remaining] = factors.get(remaining, 0) + 1

    return factors


def _sympy_factorize(n: int) -> Coordinate:
    """Sympy backend. Requires sympy to be installed."""
    try:
        from sympy import factorint
    except ImportError:
        raise ImportError(
            "sympy is not installed. "
            "Install it or use the default wheel backend."
        )
    if n < 1:
        raise ValueError(f"factorize requires a positive integer, got {n}")
    if n == 1:
        return {}
    return dict(factorint(n))


# Active backend callable.
_backend: Callable[[int], Coordinate] = _wheel_factorize

# Persistent unbounded factorization cache.
# Grows on demand. Never evicts — these integers are small and the results
# are tiny (a few prime→exponent pairs each). Matches upfpy behavior.
_cache: Dict[int, Coordinate] = {}


def set_factorization_backend(backend) -> None:
    """
    Set the factorization backend.

    Parameters
    ----------
    backend : str or callable
        "wheel"  — internal 6k±1 trial division (default)
        "sympy"  — sympy.factorint (requires sympy installed)
        callable — any fn(n: int) -> dict[int, int]

    The cache is NOT cleared automatically. Call clear_factorization_cache()
    if you need to recompute previously cached values with the new backend.

    Example
    -------
    >>> set_factorization_backend("sympy")
    >>> set_factorization_backend(my_custom_fn)
    >>> set_factorization_backend("wheel")  # back to default
    """
    global _backend
    if backend == "wheel":
        _backend = _wheel_factorize
    elif backend == "sympy":
        _backend = _sympy_factorize
    elif callable(backend):
        _backend = backend
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            f"Use 'wheel', 'sympy', or a callable fn(n: int) -> dict[int, int]."
        )


def clear_factorization_cache() -> None:
    """Clear the factorization cache. Useful after switching backends."""
    _cache.clear()


def factorization_cache_size() -> int:
    """Return the number of integers currently in the cache."""
    return len(_cache)


# ---------------------------------------------------------------------------
# factorize() — the public entry point
# ---------------------------------------------------------------------------


def factorize(n: int) -> Coordinate:
    """
    Return the prime exponent coordinate of integer n.

    factorize(1)   -> {}
    factorize(2)   -> {2: 1}
    factorize(12)  -> {2: 2, 3: 1}
    factorize(360) -> {2: 3, 3: 2, 5: 1}

    The empty dict is the coordinate of 1 — the origin of the lattice.

    Results are cached indefinitely. The cache grows as new integers are
    encountered — it "learns" all integers up to the largest seen so far.

    The backend can be changed with set_factorization_backend(). If a single
    call takes longer than the timing threshold, a warning is printed
    suggesting the sympy backend for large inputs.
    """
    if n in _cache:
        return _cache[n]

    t0 = time.monotonic()
    result = _backend(n)
    elapsed = time.monotonic() - t0

    _cache[n] = result

    if elapsed > _TIMING_THRESHOLD and _backend is _wheel_factorize:
        print(
            f"[signalforge] factorize({n}) took {elapsed*1000:.1f}ms. "
            f"For large integers consider: "
            f"set_factorization_backend('sympy')"
        )

    return result


# ---------------------------------------------------------------------------
# Vector arithmetic
# ---------------------------------------------------------------------------


def to_int(coord: Coordinate) -> int:
    """
    Reconstruct the integer from its prime exponent coordinate.

    Inverse of factorize:
        to_int(factorize(n)) == n  for all positive integers n.

    to_int({})           -> 1
    to_int({2: 1})       -> 2
    to_int({2: 2, 3: 1}) -> 12
    """
    result = 1
    for prime, exp in coord.items():
        result *= prime ** exp
    return result


def vec_add(a: Coordinate, b: Coordinate) -> Coordinate:
    """
    Add two coordinate vectors — equivalent to multiplying the integers.

    vec_add({2: 1}, {2: 1, 3: 1})  ->  {2: 2, 3: 1}   (2 * 6 = 12)
    """
    result = dict(a)
    for prime, exp in b.items():
        result[prime] = result.get(prime, 0) + exp
    return {p: e for p, e in result.items() if e != 0}


def vec_sub(a: Coordinate, b: Coordinate) -> Coordinate:
    """
    Subtract coordinate vectors — equivalent to dividing the integers.
    b must divide a (all exponents in b <= corresponding exponents in a).

    vec_sub({2: 2, 3: 1}, {2: 1})  ->  {2: 1, 3: 1}   (12 / 2 = 6)

    Extends naturally to rationals: v_p(a/b) = v_p(a) - v_p(b).
    """
    result = dict(a)
    for prime, exp in b.items():
        new_exp = result.get(prime, 0) - exp
        if new_exp < 0:
            raise ValueError(
                f"vec_sub: {b} does not divide {a} "
                f"(prime {prime}: {result.get(prime, 0)} < {exp})"
            )
        result[prime] = new_exp
    return {p: e for p, e in result.items() if e != 0}


def vec_le(a: Coordinate, b: Coordinate) -> bool:
    """
    True if a divides b — coordinate-wise <=.

    vec_le({2: 1}, {2: 2, 3: 1})  ->  True   (2 divides 12)
    vec_le({2: 3}, {2: 2, 3: 1})  ->  False  (8 does not divide 12)
    """
    for prime, exp in a.items():
        if b.get(prime, 0) < exp:
            return False
    return True


def vec_min(a: Coordinate, b: Coordinate) -> Coordinate:
    """
    Coordinate-wise minimum — equivalent to gcd of the integers.

    vec_min({2: 3, 3: 1}, {2: 1, 5: 2})  ->  {2: 1}   (gcd(24, 50) = 2)
    """
    primes = set(a) | set(b)
    result = {p: min(a.get(p, 0), b.get(p, 0)) for p in primes}
    return {p: e for p, e in result.items() if e != 0}


def vec_max(a: Coordinate, b: Coordinate) -> Coordinate:
    """
    Coordinate-wise maximum — equivalent to lcm of the integers.

    vec_max({2: 3, 3: 1}, {2: 1, 5: 2})  ->  {2: 3, 3: 1, 5: 2}   (lcm(24, 50) = 600)
    """
    primes = set(a) | set(b)
    result = {p: max(a.get(p, 0), b.get(p, 0)) for p in primes}
    return {p: e for p, e in result.items() if e != 0}


# ---------------------------------------------------------------------------
# Lattice operations
# ---------------------------------------------------------------------------


def divisors(n: int) -> Tuple[int, ...]:
    """
    Return all positive divisors of n in ascending order.

    divisors(12)  ->  (1, 2, 3, 4, 6, 12)

    In lattice terms: all integers whose coordinate is <= factorize(n).
    """
    coord = factorize(n)
    result = [1]
    for prime, exp in coord.items():
        result = [d * prime**e for d in result for e in range(exp + 1)]
    return tuple(sorted(result))


def lattice_members(horizon: int, cbin: int) -> Tuple[int, ...]:
    """
    Return all valid window sizes in the divisibility lattice.

    A valid window is any divisor of horizon that is also a multiple of cbin.
    These are the points on the lattice available for window selection.

    lattice_members(3600, 300)  ->  (300, 600, 900, 1200, 1800, 3600)
    """
    if horizon % cbin != 0:
        raise ValueError(f"cbin {cbin} must divide horizon {horizon}")
    ratio = horizon // cbin
    return tuple(d * cbin for d in divisors(ratio))


def smallest_divisor_gte(n: int, floor: int) -> int:
    """
    Return the smallest divisor of n that is >= floor.

    This is how cbin is derived from horizon and grain:
        cbin = smallest_divisor_gte(horizon, grain)

    smallest_divisor_gte(3600, 250)  ->  300
    smallest_divisor_gte(3600, 300)  ->  300
    smallest_divisor_gte(3600, 301)  ->  400
    """
    for d in divisors(n):
        if d >= floor:
            return d
    raise ValueError(f"No divisor of {n} is >= {floor}")
