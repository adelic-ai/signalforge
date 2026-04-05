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

`binjamin` is a bin width estimation library implementing all major statistical
methods in one package. Every function takes a 1-D array and returns a scalar
bin width estimate.

| Method | Notes |
|--------|-------|
| `freedman_diaconis` | `h = 2·IQR·n⁻¹/³` — robust, no distributional assumption. Default. |
| `auto` | `max(freedman_diaconis, sturges)` — numpy default rule |
| `scott` | `h = 3.5·σ·n⁻¹/³` — assumes normality, sensitive to outliers |
| `sturges` | `(max-min)/(1+log₂n)` — simple, underbins for large n |
| `rice` | `(max-min)/(2·n¹/³)` — no assumption, more bins than Sturges |
| `sqrt` | `(max-min)/√n` — simplest, exploratory use |
| `doane` | Sturges adjusted for skewness |
| `stone` | Leave-one-out cross-validation — slower, accurate, no assumption |
| `knuth` | Maximum likelihood — optimal for uniform bins |
| `gcd_interval` | GCD of intervals — exact but brittle on irregular data |

Freedman-Diaconis is the default because it makes no distributional assumption
and is stable under outliers — both important for event-driven or sensor data.

---

## How grain_from_orders works

`grain_from_orders` estimates the grain from the inter-event intervals in a
sequence of `primary_order` values:

```python
from signalforge.lattice.sampling import grain_from_orders

g = grain_from_orders(orders)                   # Freedman-Diaconis (default)
g = grain_from_orders(orders, method="knuth")   # any other method
```

Without a `horizon` argument, it returns the raw rounded estimate. This is
the recommended form — pass the result directly to `SamplingPlan.from_windows`:

```python
from signalforge.lattice.sampling import SamplingPlan

plan = SamplingPlan.from_windows([g*5, g*15, g*60, g*360], grain=g)
```

`from_windows` derives `horizon = lcm(windows + [g])`, guaranteeing `g`
divides the horizon exactly. Every binjamin method produces a valid grain —
the lattice is always correct regardless of which estimator was used.

**Fallbacks**: if fewer than 3 intervals are available, the minimum observed
interval is used directly regardless of method.

---

## When to use it

`grain_from_orders` is useful when:

- Your data arrives at irregular intervals and you need to choose a bin size
- You are writing a new domain and are unsure what grain is appropriate
- You want the grain to adapt to the actual resolution of the data

For data with a known, fixed cadence (EEG at 256 Hz, INTERMAGNET at 1 minute,
equity bars at 1 minute), declare the grain directly — `grain_from_orders`
adds no value when the cadence is already known.

---

## Example: unknown-cadence event log

```python
from signalforge.lattice.sampling import grain_from_orders, SamplingPlan

orders = [r.primary_order for r in records]

g    = grain_from_orders(orders)
plan = SamplingPlan.from_windows(
    windows=[g*5, g*20, g*100, g*500],
    grain=g,
)

print(f"Estimated grain : {g}")
print(f"Derived horizon : {plan.horizon}")
print(f"Windows         : {plan.windows}")
```

The grain estimate is reproducible — given the same data and method,
`grain_from_orders` always returns the same value.

---

For the design rationale behind `from_windows` and why the derived horizon
costs nothing, see [design_grain_snapping.md](design_grain_snapping.md).

---

*binjamin source: [github.com/adelic-ai/binjamin](https://github.com/adelic-ai/binjamin)*
*binjamin on PyPI: [pypi.org/project/binjamin](https://pypi.org/project/binjamin)*
