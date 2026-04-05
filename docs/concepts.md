# Concepts

SignalForge transforms any ordered sequence into structurally invariant multiscale surfaces. This page explains the core ideas: what the measurement space is, how it's derived, and why the results are directly comparable across recordings.

## Table of Contents

- [Signals](#signals)
- [Grain](#grain)
- [Lattice](#lattice)
- [Horizon](#horizon)
- [Surface](#surface)
- [Anomalies Are Shapes](#anomalies-are-shapes)
- [Structural Invariance](#structural-invariance)
- [Complex Signals](#complex-signals)

---

## Signals

Everything in SignalForge is a **LatticeSignal** — an abstract base type for anything indexed by integers on the prime lattice. Raw data, surfaces, baselines, residuals — all are signals. The pipeline takes signals in and produces signals out.

```python
from signalforge.signal import RealSignal, ComplexSignal, Surface

sig = RealSignal(index, values, channel="my_signal")
```

Values are complex-native (`complex128`). Real signals are the common case (`imag=0`). This means amplitude and phase are always available via `sig.amplitude()` and `sig.phase()`, and complex-valued data (EEG analytic signals, radar, MRI) flows through the pipeline without special handling.

A `Surface` is also a `LatticeSignal` — so you can measure surfaces of surfaces, compute baselines of baselines, or feed any intermediate back through the pipeline.

---

## Grain and cbin

Three related concepts control resolution:

**data_grain** — the raw cadence of the data. For event-ordered sequences (log entries, EEG samples), this is trivially 1 — each event is one unit. For time-ordered data with irregular spacing, it's estimated from inter-event intervals using [binjamin](https://github.com/adelic-ai/binjamin).

**grain** — the finest admissible analysis resolution. Must be >= data_grain. This is the floor — you can't analyze finer than the data supports. For event-ordered data, grain = 1 is the obvious default.

**cbin** — the materialized computational bin. This is the resolution the analysis actually runs at. It must divide all windows and the horizon. Derived as `gcd(windows)` by default — the coarsest resolution that divides every window cleanly. A finer cbin can be chosen for higher resolution at more compute cost.

The chain: `data_grain <= grain <= cbin`

When `grain < cbin`, zoom is possible — you can re-analyze a subset at finer resolution with a smaller window family. When `grain = cbin`, you're fully refined.

```python
import binjamin as bj

# cbin derived automatically from windows
geo = bj.lattice([10, 30, 60, 360])
geo.cbin   # 10 — gcd(windows)
geo.grain  # 1 — zoom potential

# EEG: event-ordered, grain trivially 1, cbin from windows
geo = bj.lattice([256, 1024, 4096], grain=1)
geo.cbin   # 256 — analysis at 1-second bins
geo.grain  # 1 — could zoom to sample level
```

`sf load data.csv` shows the estimated grain. Override with `--grain` if you know your data's true cadence.

---

## Lattice

SignalForge does not choose the measurement space. It derives it from the windows you declare.

Every positive integer has a unique prime factorization: `360 = 2^3 x 3^2 x 5`. The divisors of 360 form a **lattice** under divisibility — a structure where every pair of windows has a greatest common divisor (finest shared scale) and least common multiple (coarsest shared scale).

By the Fundamental Theorem of Arithmetic, this lattice decomposes as a product of independent chains, one per prime:

```
Div(360) = {0,1,2,3} x {0,1,2} x {0,1}
             prime 2    prime 3   prime 5
```

Each prime is an independent scale axis. Moving one step along the prime-2 axis doubles the window; moving along the prime-3 axis triples it. The full lattice exhausts every combination.

A window of size `w` tiles a signal of length `H` cleanly if and only if `w` divides `H`. Any other window size leaves a partial bin — an artifact. The divisors of `H` are the complete set of artifact-free window sizes.

Nesting follows from the same arithmetic: if `d1` divides `d2`, then every `d2`-window is partitioned exactly by `d2/d1` consecutive `d1`-windows. The divisibility relation on windows IS the nesting relation on the signal.

The **active domain** is the subset of lattice members between cbin and max(windows): `Div(H) ∩ [cbin, max(W)]`. These are the scales the analysis actually computes. The lattice below cbin exists but isn't materialized (that's where zoom goes).

`sf neighborhood 360` shows the lattice around any integer. `sf inspect lattice` explains the concept.

---

## Horizon

The horizon `H = lcm(windows + [cbin])` is the outer boundary of the coordinate space. All valid windows divide it. It may be larger than the data — that's fine and intentional.

The purpose is precision. By deriving `H` from the grain and windows, the grain is exact by construction — no rounding, no snapping, no forced error. The grain you specify is the grain the lattice uses.

Computation is bounded by `[grain, max(windows)]`. The lattice above `max(windows)` exists arithmetically but is never swept. The horizon is a **phantom** — it shapes the divisibility relationships but costs nothing.

In the CLI, you declare `--max-window` and the horizon is derived automatically. `sf plan equities-daily` shows the full sampling plan.

---

## Surface

A surface is a two-dimensional grid: time on one axis, scale on the other. Each cell holds an aggregated value — typically a windowed mean — computed over a window of that size at that position.

This is not a heatmap for visualization. It is a coordinate space. Every point has an address: a position in time, and a position in the divisibility lattice expressed as a prime exponent vector `(v_2, v_3, v_5, ...)`. Moving one step along the scale axis multiplies the window by a prime. The scale axis is logarithmic by construction.

Same plan = same grid = surfaces from different signals are directly comparable. This structural invariance is what makes ML on surfaces meaningful.

`sf surface data.csv -hm` builds and displays a surface. The `-hm` flag renders it as a heatmap.

---

## Anomalies Are Shapes

The z-score at a single point on the surface is not the anomaly. The anomaly is the pattern of change across the surface.

Consider what happens to a real anomaly — a seizure, a geomagnetic storm, a market crash — as you vary the window size while holding time fixed. At fine scales, the window captures noise as readily as signal. At the scale matching the anomaly's characteristic duration, the signal peaks. At coarser scales, the anomaly dilutes.

The result is a triangle profile along the scale axis, with a peak at the scale that fits. Extend that peak across time and you get a **ridge** — a connected structure tracing the anomaly's duration at its natural scale.

Every anomaly has a **scale signature** — three coordinates that triangulate it:

- **Detection scale**: coarsest scale where the anomaly first emerges
- **Resolution scale**: where the signal peaks — the window best matched to the anomaly's duration
- **Support scale**: finest scale where the signal remains coherent

Because the coordinates are determined by arithmetic, not analyst choice, the scale signature of the same type of anomaly is consistent across different recordings processed with the same plan.

---

## Structural Invariance

A surface computed from one sequence is directly comparable to a surface computed from another — across patients, sessions, instruments, sites — without normalization, alignment, or feature engineering.

When two sequences are processed with the same `SamplingPlan`, they have the same `Div(H)`: the same set of window sizes in the same lattice arrangement. The feature at scale `d` measures the same structural property in both signals. There is nothing to align. The lattice is the shared coordinate system.

Standard multi-scale methods produce features whose meaning depends on choices made by the analyst: which window sizes, which basis functions, which normalization scheme. SignalForge features are indexed by lattice position, not by analyst choice. The measurement space does the work that post-hoc normalization would otherwise have to do.

Surfaces from different recordings with the same plan stack directly into an ML-ready tensor. No preprocessing. No alignment. The lattice guarantees the columns mean the same thing in every row.

---

## Complex Signals

SignalForge values are complex-native. Real signals are the degenerate case (`imag=0`).

For signals that are naturally complex — EEG analytic signals, radar, MRI, seismology — the complex representation flows through the pipeline without mode switches. Amplitude and phase are always available.

The **Hilbert transform** promotes a real signal to its analytic (complex) form:

```python
surfaces = (
    sf.load("eeg.csv")
    .measure(windows=[256, 1024, 4096])
    .hilbert()
    .surfaces()
)
# surfaces[0].data now includes 'amplitude', 'phase', 'inst_freq'
```

The analytic signal is `z(t) = x(t) + j*H[x](t)` — the original signal is the real part, the Hilbert transform is the imaginary part. Amplitude is the envelope, phase is the position in the local cycle, instantaneous frequency is the local clock speed.

For best results, apply Hilbert to derived signals (after measurement and smoothing), not to raw events. See the [Python API guide](python-api.md#hilbert) for Hilbert pipeline examples.

---

*Full treatment with proofs: arXiv preprint — forthcoming.*

See also: [comparison with STFT, wavelets, and EMD](comparison.md)
