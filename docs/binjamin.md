# Grain Selection and Binjamin

The grain is the most consequential declaration in a SamplingPlan. Too coarse
and you lose the fine structure that makes anomalies detectable. Too fine and
the lattice becomes large with many sparse windows. For familiar data — EEG at
256 Hz, geomagnetic at 1-minute cadence — the right grain is obvious. For
unfamiliar data, it is not.

SignalForge uses [binjamin](https://pypi.org/project/binjamin/) to estimate
the grain automatically when the data can inform that choice.

---

## What binjamin is

`binjamin` is a bin width estimation library. It implements all major
statistical methods for choosing an optimal histogram bin width in one package:

| Method | Description |
|--------|-------------|
| Freedman-Diaconis | `h = 2 × IQR × n⁻¹/³` — robust to outliers, default choice |
| Bayesian blocks | Variable-width bins that adapt to local event density |
| Scott | `h = 3.49 × σ × n⁻¹/³` — assumes normality |
| Knuth | Maximum likelihood bin count via Bayesian model selection |
| Sturges | `k = ⌈log₂(n)⌉ + 1` — fast, less accurate for large n |

Freedman-Diaconis is the default in SignalForge because it makes no assumption
about the distribution shape and is stable under outliers — both important
properties for event-driven or sensor data.

---

## How grain_from_orders works

`grain_from_orders` is available in `signalforge.lattice.sampling`:

```python
from signalforge.lattice.sampling import grain_from_orders

grain = grain_from_orders(orders, horizon=86400)
grain = grain_from_orders(orders, horizon=86400, method="scott")
```

It takes a sequence of `primary_order` values, a candidate horizon, and an
optional method name, and returns a grain that is:

1. **Statistically grounded** — the chosen method estimates the natural bin
   width from the inter-event intervals in the data
2. **Lattice-compatible** — the estimate is snapped to the nearest divisor of
   the horizon via `smallest_divisor_gte(horizon, estimate)`, ensuring the
   grain divides the horizon cleanly

```python
# Uniformly spaced events at 60-second intervals
grain_from_orders(range(0, 3600, 60), horizon=86400)
# → 60
```

**Available methods:**

| Method | Notes |
|--------|-------|
| `freedman_diaconis` | Default. Robust, no distributional assumption |
| `auto` | `max(freedman_diaconis, sturges)` — numpy default |
| `scott` | Assumes normality, sensitive to outliers |
| `sturges` | Simple, tends to underbin for large n |
| `rice` | No assumption, more bins than Sturges |
| `sqrt` | Simplest, exploratory use |
| `doane` | Sturges adjusted for skewness |
| `stone` | Leave-one-out cross-validation; slower, accurate |
| `knuth` | Maximum likelihood; optimal for uniform bins |
| `gcd_interval` | Exact GCD of intervals — lattice-native but brittle |

**Fallbacks**: if fewer than 3 intervals are available, the minimum observed
interval is used directly regardless of method.

---

## When to use it

`grain_from_orders` is useful when:

- Your data arrives at irregular intervals and you need to choose a bin size
- You are writing a new domain and are unsure what grain is appropriate
- You want the grain to adapt to the actual resolution of the data rather
  than a fixed declaration

For data with a known, fixed cadence (EEG at 256 Hz, INTERMAGNET at 1 minute,
equity bars at 1 minute), declare the grain directly — `grain_from_orders`
adds no value when the cadence is already known.

---

## Example: unknown-cadence event log

```python
import signalforge as sf
from signalforge.lattice.sampling import grain_from_orders

# Load raw event timestamps (e.g. from a log file)
orders = [r.primary_order for r in records]
horizon = 86400  # one day

grain = grain_from_orders(orders, horizon=horizon)
plan  = sf.SamplingPlan(horizon, grain)

print(f"Estimated grain: {grain}s")
print(f"Windows: {plan.windows}")
```

The pipeline then runs without modification. The grain estimate is reproducible
— given the same data and horizon, `grain_from_orders` always returns the same
value.

---

---

## The snapping constraint

`grain_from_orders` always snaps the estimate to the nearest divisor of the
horizon. This is what makes the grain lattice-compatible and unlocks the
aggregation and structural invariance guarantees — but it means the raw
estimate is nudged, which may not always be desirable.

Free-grain mode (no snapping, arbitrary windows, prefix-sum computation) is
not yet implemented. For the design tradeoffs and how to provide feedback, see
[design_grain_snapping.md](design_grain_snapping.md).

---

*binjamin source: [github.com/adelic-ai/binjamin](https://github.com/adelic-ai/binjamin)*
*binjamin on PyPI: [pypi.org/project/binjamin](https://pypi.org/project/binjamin)*
