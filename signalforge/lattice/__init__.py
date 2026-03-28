"""
signalforge.lattice

The p-adic coordinate system underlying the SignalForge measurement geometry.

Every positive integer has a unique representation as a product of prime powers
(Fundamental Theorem of Arithmetic). This module treats that representation as
a coordinate system — each prime is an axis, each exponent is a coordinate.

The lattice of divisors of a chosen max_window, expressed in these coordinates,
is the geometric foundation of the SamplingPlan. Windows are points on this
lattice. The scale axis of every Surface is a path through it.

Modules:
    coordinates  — prime exponent vectors, factorization, lattice arithmetic
    flipflop     — seed-based navigation of p-adic valuation structure
    sampling     — window and hop derivation from the divisibility lattice
"""

from .coordinates import (
    factorize, vec_add, vec_sub, vec_le, to_int, Coordinate,
    set_factorization_backend, clear_factorization_cache, factorization_cache_size,
)
from .flipflop import FlipFlop
from .sampling import SamplingPlan, horizon_for, grain_from_orders, suggest_cbin
from .neighborhood import Neighborhood, neighborhood, neighborhood_from_vector
