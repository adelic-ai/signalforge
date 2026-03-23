"""
signalforge.domains

Domain-specific SamplingPlan factories.

Each domain module implements:

    sampling_plan(max_window: int, data_minbin: int) -> SamplingPlan

using expert knowledge of that domain's natural periodicities and
meaningful analysis scales. The returned SamplingPlan is a standard
lattice artifact — domain knowledge lives only in the window selection.

Shipped domains
---------------
intermagnet  — geomagnetic observatory data (INTERMAGNET standard)

Adding a domain
---------------
Create a module in this package. Implement sampling_plan(). No
registration or plugin machinery required — import directly.

    from signalforge.domains import intermagnet
    plan = intermagnet.sampling_plan(86400, 60)
"""
