# Grain, Horizon, and the Sampling Lattice

## How the lattice is defined

Two values define the measurement space:

- **grain** `g` — the pixel, the finest resolution unit
- **horizon** `H` — the outer boundary; all valid windows divide it

The valid window sizes are exactly the divisors of `H`. For this to include
`g` itself — so the pixel is a lattice member — `g` must divide `H`. When it
does, `cbin = g` exactly and the lattice is fully determined.

---

## from_windows: deriving the horizon

The natural way to satisfy `g | H` is to derive `H` from your windows and
grain rather than declaring it independently:

```python
plan = SamplingPlan.from_windows(
    windows=[7, 30, 90, 360],
    grain=g,
)
```

Internally: `H = lcm(windows + [g])`. Because `g` is included in the lcm,
`g | H` exactly. `cbin = g`. No adjustment to the grain, regardless of which
estimation method produced it.

The user declares three things:

| Parameter | Meaning |
|-----------|---------|
| `windows` | The scales to compute — anchor windows and largest window |
| `grain`   | The pixel — finest resolution unit, exact |
| `hops`    | (optional) stride per window, defaults to grain |

`H` is never declared. It is derived and readable via `plan.horizon` for
inspection, but it is scaffolding — not a user-facing parameter.

---

## Grain estimation from data

`grain_from_orders` estimates the grain from inter-event intervals using any
binjamin method. Without a horizon argument, it returns the raw estimate
directly — suitable for `from_windows`:

```python
from signalforge.lattice.sampling import grain_from_orders

g    = grain_from_orders(orders)                      # Freedman-Diaconis default
g    = grain_from_orders(orders, method="knuth")      # or any other method
plan = SamplingPlan.from_windows([g*7, g*30, g*90], grain=g)
```

Every binjamin method produces a valid grain for `from_windows`. The method
choice affects the quality of the estimate, not the lattice validity —
`from_windows` guarantees `cbin = g` regardless of how `g` was obtained.

See [binjamin.md](binjamin.md) for the full method list.

---

## Why the derived horizon costs nothing

`lcm(windows + [g])` may be larger than `max(windows)`. This is fine:

- **Computation is bounded by `[cbin, max_window]`** — the pipeline only
  materializes the selected windows. The lattice above `max_window` is never
  swept.

- **Cost scales with divisor count, not horizon magnitude** — lattice traversal
  is O(max(W) · τ(H)) where τ(H) is the number of divisors, not H itself.

- **New prime axes from `g` are shallow** — if `g = 58 = 2 × 29` adds a
  29-axis, it does so at depth 1. One extra scale level.

- **The grain-to-window ratio keeps the lcm bounded** — `g` must be a
  meaningful fraction of your smallest window, and your smallest window a
  meaningful fraction of your largest. This means `g` and the anchor windows
  are in the same order of magnitude, so the lcm stays manageable.

---

## Direct declaration

`SamplingPlan(horizon, grain)` is still available and unchanged. Use it when
the horizon and grain are known and compatible — fixed-cadence domains like
EEG, geomagnetic, or equities where the coordinate space is well-understood.

When the grain is estimated from data or the windows are the natural starting
point, `from_windows` is the right entry point.

---

## Fully data-driven window selection

`from_windows` covers the case where you know your windows and grain upfront.
It does not cover the case where even the window list emerges from the data.
For that, a `FreeSamplingPlan` using prefix-sum computation rather than DAG
aggregation remains a future possibility. Feedback welcome:

**[Open an issue on GitHub](https://github.com/adelic-ai/signalforge/issues)**
