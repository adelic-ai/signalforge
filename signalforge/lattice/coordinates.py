"""
signalforge.lattice.coordinates

Re-exports from binjamin. The canonical source for all coordinate
and factorization math is binjamin.coordinates.

Original implementation by Shun Richard Honda (upfpy, 2002-2013).
Migrated to binjamin as the standalone lattice geometry package.
"""

from binjamin.coordinates import (
    Coordinate,
    factorize,
    to_int,
    divisors,
    lattice_members,
    smallest_divisor_gte,
    vec_add,
    vec_sub,
    vec_le,
    vec_min,
    vec_max,
    set_factorization_backend,
    clear_factorization_cache,
    factorization_cache_size,
)

__all__ = [
    "Coordinate",
    "factorize",
    "to_int",
    "divisors",
    "lattice_members",
    "smallest_divisor_gte",
    "vec_add",
    "vec_sub",
    "vec_le",
    "vec_min",
    "vec_max",
    "set_factorization_backend",
    "clear_factorization_cache",
    "factorization_cache_size",
]
