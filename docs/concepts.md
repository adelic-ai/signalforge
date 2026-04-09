# Concepts

You've used SF. Here's what's happening underneath — only if you're curious.

## Table of Contents

- [Surfaces](#surfaces)
- [Scales and the lattice](#scales-and-the-lattice)
- [Resolution: cbin and grain](#resolution-cbin-and-grain)
- [Why surfaces are comparable](#why-surfaces-are-comparable)
- [Signals](#signals)
- [Complex signals](#complex-signals)
- [Anomalies are shapes](#anomalies-are-shapes)

---

## Surfaces

When you run `sf surface`, you get a 2D grid: time on one axis, scale on the other. Each cell is the aggregated signal within a window of that size at that position.

The surface isn't a heatmap — the heatmap is how you look at it. The surface is the data structure underneath: a measurement at every combination of time and scale. Everything else (baselines, residuals, gradients, features) operates on this grid.

A surface is also a signal. You can feed a surface back through SF to measure surfaces of surfaces — recursive decomposition. That's why baselines and residuals work: they're just surfaces derived from other surfaces.

---

## Scales and the lattice

You asked for `--max-window 360`. SF gave you 18 scales. Why 18? Why those specific numbers?

Every scale must divide the horizon cleanly — no leftover partial bins. The valid scales are the **divisors** of the horizon. For horizon 360:

```
360 = 2³ × 3² × 5
```

That factorization gives 24 divisors. The scales between cbin and max-window are the ones SF uses.

Each scale has a coordinate — the exponents in the factorization. For example:

```
60  = 2² × 3 × 5  → coordinate (2, 1, 1)
30  = 2  × 3 × 5  → coordinate (1, 1, 1)
12  = 2² × 3      → coordinate (2, 1, 0)
```

Each prime is an independent axis. Moving along the prime-2 axis doubles the window. Moving along prime-3 triples it. The lattice gives every scale a unique address.

`sf inspect lattice` shows more. `sf neighborhood 360` shows the lattice around any integer.

---

## Resolution: cbin and grain

**cbin** — the resolution your analysis runs at. Derived as the greatest common divisor of your windows. It's the coarsest bin size that divides every window cleanly.

**grain** — the finest resolution the data supports. Estimated from the spacing between data points. When grain < cbin, you can zoom finer — re-analyze a subset with smaller windows.

The relationship: `grain ≤ cbin`. cbin comes from the windows (what you asked for). Grain comes from the data (what it supports).

`sf load` shows the estimated grain. Override with `--grain` if you know your data's cadence.

**Choosing cbin:** SF picks `gcd(windows)` by default — the coarsest valid resolution, cheapest to compute. A finer cbin gives more resolution at more compute cost.

**Estimation methods:** Grain is estimated using Freedman-Diaconis by default — robust, no assumptions about the data distribution. Other methods are available through [binjamin](https://github.com/adelic-ai/binjamin), the math library underneath SF.

---

## Why surfaces are comparable

Two signals analyzed with the same windows produce surfaces on the same grid — same scales, same coordinates, same structure. No alignment needed.

This means: surface A from machine 1 and surface B from machine 2 can be stacked, subtracted, correlated, or fed to ML directly. The lattice guarantees they're measuring the same thing at each scale.

This is why SF is useful for ML. The feature tensor is just stacked surfaces — and every entry in the tensor occupies the same structural position across all entities. No feature engineering required.

---

## Signals

Everything in SF is a **LatticeSignal** — raw data, surfaces, baselines, residuals. The pipeline takes signals in and produces signals out.

```python
from signalforge.signal import RealSignal

sig = RealSignal(index, values, channel="my_signal")
```

Values are complex-native (`complex128`). Real signals are the common case (`imag=0`). This means amplitude and phase are always available — the complex machinery is there when you need it, invisible when you don't.

---

## Complex signals

For data that's naturally oscillatory (EEG, radar, vibration), the Hilbert transform promotes a real signal to its complex (analytic) form:

```python
surfaces = sf.load("eeg.csv").measure().hilbert().surfaces()
```

The analytic signal carries amplitude (envelope) and phase (position in the cycle) at every point. Phase relationships between channels — which channels are synchronized, which are drifting — are visible only through complex-valued analysis.

For non-oscillatory data (counts, rates, market prices), complex signals aren't needed. Real values work fine. The complex foundation is there for domains that need it, not imposed on domains that don't.

---

## Anomalies are shapes

A z-score at one point isn't an anomaly — it's one measurement. The anomaly is the pattern across the surface.

An event that matches a window's size produces a peak at that scale. Too fine a window catches noise. Too coarse a window dilutes the signal. The peak at the matching scale, extended across time, forms a **ridge** — the anomaly's shape in scale-time space.

Every anomaly has a **scale signature**: the coarsest scale where it appears (detection), the scale where it peaks (resolution), and the finest scale where it holds together (support). This signature is determined by the lattice — not by analyst choices — so the same type of anomaly has the same signature across different recordings.
