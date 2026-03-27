# Design Note: Grain, Horizon, and the Derived Lattice

## The problem snapping was solving

`SamplingPlan(horizon, grain)` derives `cbin = smallest_divisor_gte(horizon, grain)`.
If `grain` does not divide `horizon`, `cbin != grain` — the grain is nudged to
the nearest divisor. This is snapping.

Snapping exists because the lattice requires `cbin | horizon`. Without it, windows
cannot tile the sequence without remainder, nesting breaks, and the aggregation
DAG falls apart.

The cost: the grain the data suggests and the grain actually used are different.
For small nudges (58 → 60) this is usually acceptable. For larger nudges, or when
the grain carries precise meaning, it is not.

---

## The resolution: horizon as a derived value

The root cause of snapping is treating `horizon` as a fixed declaration and forcing
`grain` to fit it. The fix is to flip the relationship: **let the grain be exact and
derive the horizon to fit it**.

Given desired windows `[w₁, w₂, ..., wₙ]` and a grain `g`:

```
horizon = lcm(w₁, w₂, ..., wₙ, g)
```

Because `g` is included in the lcm, `g | horizon` exactly. So `cbin = g` — no
snapping, no distortion. The grain the data suggests is the grain used.

The horizon is now scaffolding. It is larger than `max(windows)` in general, but
only the slice `[g, max(windows)]` is ever computed. The lattice above `max(windows)`
is never materialized — it shapes the arithmetic without contributing computation.

---

## The API: SamplingPlan.from_windows()

```python
plan = SamplingPlan.from_windows(
    windows=[7, 30, 90, 360],
    grain=g,
)
```

`from_windows` computes `horizon = lcm(windows + [grain])` internally. The user
declares three things:

| Parameter | Meaning |
|-----------|---------|
| `windows` | The scales you want to compute — anchor windows and max window |
| `grain` | The pixel — finest resolution unit, exact, no snapping |
| `hops` | (optional) stride per window, defaults to grain |

`horizon` is never declared. It is always derived. `plan.horizon` is readable for
inspection but is not a user-facing input.

---

## Getting the grain from data

`grain_from_orders` without a horizon argument returns a raw estimate with no
snapping — suitable for passing directly to `from_windows`:

```python
from signalforge.lattice.sampling import grain_from_orders

g    = grain_from_orders(orders)                           # raw estimate
plan = SamplingPlan.from_windows([g*7, g*30, g*90], grain=g)
```

With a horizon argument, the old snapping behavior is preserved for backward
compatibility:

```python
g = grain_from_orders(orders, horizon=86400)   # snapped to divisor of 86400
```

---

## Why the ghost horizon costs nothing

The horizon `lcm(windows + [grain])` may be much larger than `max(windows)`.
This is fine because:

1. **Computation is bounded by `[cbin, max_window]`** — the pipeline only
   materializes windows in the selected set. The lattice above `max_window`
   is never swept.

2. **Cost scales with divisor count, not horizon magnitude** — the lattice
   traversal cost is O(H · τ(H)) where τ(H) is the number of divisors of H,
   not H itself. A large horizon with few prime factors (e.g., a power of 2)
   is cheaper than a small horizon with many.

3. **The new prime axes from `g` are shallow** — if `g` introduces a new prime
   factor (e.g., `g = 58 = 2 × 29` adds a 29-axis), it does so at depth 1.
   One extra scale level, not a recursive blowup.

4. **The ratio constraint keeps things bounded** — `cbin` must be a reasonable
   fraction of your smallest window, and your smallest window must be a
   reasonable fraction of your largest. This means `g` and the anchor windows
   are in the same order of magnitude, so `lcm(windows + [g])` stays manageable.

---

## The original SamplingPlan(horizon, grain) constructor

Still available and unchanged. Use it when:

- You know your horizon and grain and they are already compatible
- You are working in a domain with a fixed, well-understood coordinate space
- You want precise control over which lattice is constructed

`from_windows` is the ergonomic entry point for the common case. The underlying
representation is identical — `from_windows` is just a factory that derives the
horizon for you.

---

## FreeSamplingPlan: still open

This design resolves snapping for the case where you can declare your windows
and grain upfront. The case it does not cover: when even the window list cannot
be declared in advance — e.g., fully data-driven window selection where the
windows themselves emerge from the data.

For that case, a `FreeSamplingPlan` with prefix-sum computation (rather than
DAG aggregation) remains a future possibility. Feedback welcome:

**[Open an issue on GitHub](https://github.com/adelic-ai/signalforge/issues)**
