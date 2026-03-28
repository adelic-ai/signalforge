# The Sampling Domain

SignalForge does not choose the measurement space. It derives it from the windows you declare.

The construction has two stages. **Unitization** embeds the analyst's
quantities into a common integer domain: specify a **grain** (the smallest
meaningful unit — a position, step, or time interval) and the **windows** you
want to compute. The grain can be declared directly when the cadence is known,
or estimated from the data using [binjamin](binjamin.md) when it is not — any
estimation method produces a valid grain. Once declared, all window sizes
become integer multiples of the grain. This produces commensurability but not
structure.

**Normalization** completes the unitized set to the divisibility lattice: the
horizon is derived automatically as `lcm(windows + [grain])`, and the valid
window sizes are exactly the divisors of that horizon. This is not a design
choice. It is the necessary completion — the minimal lattice containing the
unitized windows that is closed under divisibility and gcd.

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

## The phantom horizon

`H = lcm(windows + [grain])` may be larger than `max(windows)`. This is
intentional. `H` is not the largest window computed — it is the arithmetic
boundary that gives the lattice its structure. Call it the **phantom horizon**:
it shapes the divisibility relationships but is never materialized as a
computation.

The purpose is precision. If `grain` is declared independently of `H`, the
only way to guarantee `grain | H` is to snap the grain to the nearest divisor
of `H` — introducing a rounding error that propagates into every feature
computed at grain resolution. By deriving `H` as `lcm(windows + [grain])`,
the grain is exact by construction. No snapping. No forced error amount. The
grain you specify — whether declared or estimated from data — is the grain the
lattice uses.

Computation is bounded by `[grain, max(windows)]`. The lattice above
`max(windows)` exists arithmetically but is never swept. The phantom horizon
costs nothing.

---

## Computational efficiency

The origin of this approach was practical frustration. Earlier multi-scale
pipelines computed each window size independently: bin the data at the finest
resolution, then for each window size re-read those bins, re-aggregate, and
re-score. Adding a new window added another full pass. Adding a new feature
multiplied the cost again. The pipeline grew cumbersome — and it was doing
something that felt like integration, badly.

The insight is that multi-scale aggregation is discrete integration. In a basic
calculus course, integration is introduced as a sum over a partition: divide the
domain into intervals of width Δx, sum the values, and the result approximates
the area. Make Δx smaller and the approximation improves. Take the limit as Δx
goes to zero and you get the exact area under the curve — the triumphant
conclusion of the whole story. But for a discrete signal there is no limit to
take. The grain is the finest partition. At the grain you are already exact.

The same structure applies here. The grain is Δx. Binning the raw data at grain
resolution is forming the partition. Aggregating up to a coarser window — say,
5-grain windows — is summing five adjacent Riemann rectangles. The 5-grain value
is already determined by the grain-level values; it does not require re-reading
the raw data. The 15-grain value is determined by three 5-grain values. The
lattice is the complete system of these relationships, made exact by divisibility.

This is the DAG: computation flows upward from grain through the lattice in a
single pass. Each coarser scale costs only the aggregation of already-computed
finer-scale values. Reading the raw data happens once.

The resolution — the grain — is a choice matched to the analysis task. A finer
grain resolves structure at shorter scales; patterns that are invisible at
coarser resolution become detectable. A coarser grain is cheaper. The pipeline
scales to whatever compute is available, and the precision is whatever the task
requires — not a fixed constraint imposed by the method.

Total cost: one raw-data read at grain resolution, plus lightweight aggregation
up the lattice — O(max(W) · τ(H)), nearly linear in the maximum window size
regardless of how many window sizes are selected.

`Div(H)` is the unique maximal window family that is simultaneously
artifact-free, perfectly nested, and closed under finest-common-refinement.
The artifact-free guarantee applies within the lattice structure; recording
boundaries are handled the same way as in any windowed analysis (stop at the
last complete window, pad, or flag partial windows).

---

## What this means for a SamplingPlan

The ergonomic entry point is `from_windows` — specify the scales you want and
the grain, and the phantom horizon is derived automatically as
`lcm(windows + [grain])`:

```python
# Grain known from domain (e.g. 1-second cadence)
plan = SamplingPlan.from_windows([1, 5, 15, 30, 60, 360], grain=1)

# Grain estimated from data — any binjamin method works
from signalforge.lattice.sampling import grain_from_orders
g    = grain_from_orders(orders)                    # Freedman-Diaconis default
g    = grain_from_orders(orders, method="knuth")    # or any other method
plan = SamplingPlan.from_windows([g*5, g*15, g*60, g*360], grain=g)
```

In both cases the grain divides the horizon exactly — no snapping, no distortion
of the lattice. The pipeline computes features at all selected scales in one pass.
Units are whatever grain represents: seconds, minutes, events, ticks.

For full control, declare the horizon directly:

```python
plan = sf.SamplingPlan(horizon=3600, grain=1)
```

`H = 3600 = 2³ × 3² × 5²` gives 36 divisors across a 3-axis lattice (prime axes
2, 3, 5). If the grain does not already divide the declared horizon, it is
snapped to the nearest divisor — which is why `from_windows` is preferred when
the grain comes from data: it derives `H` from the grain, guaranteeing exactness
by construction.

See [design_grain_snapping.md](design_grain_snapping.md) for the design rationale.

---

*Full treatment with proofs: [arXiv preprint — forthcoming]*
