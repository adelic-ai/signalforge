"""
signalforge.lattice

SF-specific wrappers around binjamin's lattice geometry.

Lattice math (factorize, divisors, coordinates, LatticeGeometry) lives
in binjamin. This module adds SF-specific types: SamplingPlan (with hops),
FlipFlop (valuation navigator), Neighborhood (viewing box).

For lattice math directly: import binjamin
For SF signal analysis: import from here
"""

from .flipflop import FlipFlop
from .sampling import SamplingPlan, horizon_for, grain_from_orders, suggest_cbin
from .neighborhood import Neighborhood, neighborhood, neighborhood_from_vector
