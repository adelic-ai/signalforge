# Design Note: Grain Snapping and the Lattice Constraint

## What snapping is

When `grain_from_orders` estimates a grain from data, it does not use the
raw estimate directly. It snaps it to the nearest divisor of the horizon:

```python
grain = smallest_divisor_gte(horizon, raw_estimate)
```

If the data suggests a natural bin width of 58 seconds and the horizon is
86400, the snapped grain becomes 60 — the smallest divisor of 86400 that is
≥ 58.

This nudge is intentional. It is what makes the grain lattice-compatible.

---

## Why snapping is the default

A grain that divides the horizon gives you three guarantees:

**1. Artifact-free tiling.** Every window in `Div(H)` partitions the sequence
without remainder. No partial bins, no truncation at the boundary.

**2. Free aggregation across scales.** Because windows nest exactly
(if `d1 | d2`, then `d2/d1` consecutive `d1`-windows tile each `d2`-window),
a feature at a coarser scale can be computed by aggregating already-computed
values from a finer scale. The lattice is a DAG; computation flows bottom-up
in a single pass. No re-reading of the raw data at each scale.

**3. Structural invariance.** Two sequences with the same `H = horizon/grain`
share an identical lattice. Features at scale `d` measure the same structural
property in both, with no normalization required between sequences.

Snapping trades a small distortion in the grain estimate for all three of
these. For most data, the distortion is negligible — the nearest divisor is
rarely far from the raw estimate.

---

## When snapping is a problem

Snapping is most disruptive when:

- The horizon is poorly chosen and has few divisors (e.g., a prime number),
  causing large jumps between candidate grains
- The user has a domain reason to use a specific grain that does not divide
  the horizon — for example, a 7-day window that does not divide a 360-day
  horizon cleanly (360/7 is not an integer)
- The bin width method (e.g., `knuth`, `stone`) produces an estimate that is
  meaningful in isolation but loses that meaning when nudged

In these cases, the user may want to opt out of snapping and use an arbitrary
grain. This is currently not supported.

---

## What free-grain mode would look like

Without snapping, the lattice guarantees are lost:

- Windows may not tile the sequence without remainder
- The DAG aggregation structure breaks down
- Surfaces from different sequences are no longer structurally invariant

However, a different efficiency strategy becomes available: **prefix sums**.

Precompute a cumulative sum array at grain resolution (O(H/grain) setup).
Any window `[i, i+w)` is then a single array subtraction — O(1) per position.
Total cost: O(H/grain + H/grain × W), where W is the number of windows
requested. For small W this is comparable to the lattice cost; for large W
the lattice wins because it reuses aggregations across the DAG.

The prefix sum approach handles mean and sum exactly. For variance and
higher-order statistics, the same trick extends to prefix sums of squares
(and higher powers), at the cost of more precomputed arrays.

A natural design would be two separate types:

- `SamplingPlan` — current behavior, grain snapped, all lattice guarantees
- `FreeSamplingPlan` — arbitrary grain, user-specified windows, prefix-sum
  computation, no structural invariance guarantee

The type itself signals what guarantees you have, rather than a flag that is
easy to ignore.

---

## Current status

Free-grain mode is not implemented. `grain_from_orders` always snaps.
`SamplingPlan` requires a lattice-compatible grain.

If you have a use case where snapping causes a real problem, or a better
design for free-grain mode, feedback is welcome:

**[Open an issue on GitHub](https://github.com/adelic-ai/signalforge/issues)**

Specifically useful:
- A concrete domain where snapping produces a wrong grain
- A better criterion for when snapping should be skipped automatically
- A design for `FreeSamplingPlan` that preserves as many guarantees as possible
- Benchmarks comparing lattice aggregation vs prefix-sum for your data size
