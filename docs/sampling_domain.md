# The Sampling Domain

SignalForge does not choose window sizes. It derives them.

Declare a **grain** (the pixel — the smallest meaningful unit: a position, step,
or time interval) and the **windows** you want to compute. The horizon is derived
automatically as `lcm(windows + [grain])`, and the valid window sizes are exactly
the divisors of that horizon. This is not a design choice. It is a consequence of
requiring that windows partition the sequence without remainder and nest into each
other without overlap.

---

## Why divisors

A window of size `w` tiles a signal of length `H` cleanly if and only if `w`
divides `H`. Any other window size leaves a partial bin at the boundary —
either truncating the signal or introducing an artifact. The divisors of `H`
are the complete set of artifact-free window sizes, and no window outside that
set belongs to it.

Nesting follows from the same arithmetic: if `d1` divides `d2`, then every
`d2`-window is partitioned exactly by `d2/d1` consecutive `d1`-windows, with
no remainder. The divisibility relation on windows is the nesting relation on
the signal.

---

## The lattice structure

The divisors of `H` do not just form a set — they form a lattice under
divisibility. The meet of two windows is their gcd (the finest scale both
share); the join is their lcm (the coarsest scale both divide into).

By the Fundamental Theorem of Arithmetic, this lattice decomposes as a product
of independent chains, one per prime dividing `H`:

```
Div(H) ≅ {0, ..., v₂(H)} × {0, ..., v₃(H)} × {0, ..., v₅(H)} × ...
```

where `vₚ(H)` is the exponent of prime `p` in the factorization of `H`. Each
prime is an independent scale axis. Moving one step along the prime-2 axis
doubles the window; moving along the prime-3 axis triples it. The full lattice
exhausts every combination.

---

## Computational efficiency

Because windows nest perfectly, a feature computed at a coarser scale can be
obtained by aggregating already-computed values from a finer scale — no
re-reading of the raw signal. The lattice is a DAG; computation flows
bottom-up from grain-level windows to the horizon in a single pass.

Total cost: one raw-data read at grain resolution, plus lightweight aggregation
up the lattice. This is nearly linear in signal length regardless of how many
scales are computed. No other family of window sizes has this property, because
it requires the family to be closed under gcd — which is exactly the structure
of `Div(H)`.

`Div(H)` is the unique maximal window family that is simultaneously
artifact-free, perfectly nested, and closed under finest-common-refinement.

---

## What this means for a SamplingPlan

The ergonomic entry point is `from_windows` — declare the scales you want and
the grain, and the horizon is derived automatically as `lcm(windows + [grain])`:

```python
plan = SamplingPlan.from_windows([1, 5, 15, 30, 60, 360], grain=1)
```

The derived horizon is `lcm(1, 5, 15, 30, 60, 360, 1) = 360`. The grain divides
the horizon exactly — no snapping. The pipeline computes features at all selected
scales in one pass. Units are whatever grain represents: seconds, minutes, events,
ticks.

For full control, declare the horizon directly:

```python
plan = sf.SamplingPlan(horizon=3600, grain=1)
```

`H = 3600 = 2³ × 3² × 5²` gives 36 divisors across a 3-axis lattice (prime axes
2, 3, 5). The grain is snapped to the nearest divisor of 3600 if it does not
already divide 3600.

See [design_grain_snapping.md](design_grain_snapping.md) for why `from_windows`
is preferred when the grain comes from data.

---

*Full treatment with proofs: [arXiv preprint — forthcoming]*
