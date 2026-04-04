# Python API

Two APIs for different needs: **chaining** for quick exploration, **DAG** for full composition.

## Table of Contents

- [Chaining API](#chaining-api)
- [DAG Composition](#dag-composition)
- [Signals](#signals)
- [Surfaces](#surfaces)
- [Hilbert](#hilbert)
- [Operators](#operators)

---

## Chaining API

For linear exploration — try ideas fast, one line each, compare results.

```python
import signalforge as sf

# Load and measure
surfaces = sf.load("data.csv").measure(windows=[10, 60, 360]).surfaces()

# With baseline and residual
surfaces = (
    sf.load("data.csv")
    .measure(windows=[10, 60, 360])
    .baseline("ewma", alpha=0.1)
    .residual("z")
    .surfaces()
)

# With Hilbert transform
surfaces = (
    sf.load("data.csv")
    .measure(windows=[10, 60, 360])
    .hilbert()
    .surfaces()
)
```

Each method returns a new `Chain` — the original is never mutated. Nothing executes until `.run()` or `.surfaces()`.

**Methods:**

| Method | Purpose |
|--------|---------|
| `sf.load(path_or_records)` | Start from CSV or CanonicalRecords |
| `sf.from_signal(signal)` | Start from a LatticeSignal directly |
| `.measure(**kwargs)` | Build surfaces. `windows=` sets analysis scales |
| `.baseline(method, **kwargs)` | Apply baseline: `"ewma"`, `"median"`, `"rolling_mean"` |
| `.residual(mode)` | Compute residual: `"z"`, `"ratio"`, `"difference"` |
| `.hilbert()` | Hilbert transform: adds amplitude, phase, inst_freq |
| `.run(**resolve_kwargs)` | Execute, return Artifact |
| `.surfaces(**resolve_kwargs)` | Execute, return list of Surface |
| `.heatmap(**resolve_kwargs)` | Execute and display heatmap |

### Comparing approaches

```python
records = sf.load("data.csv")

ewma = records.measure(windows=[10, 60, 360]).baseline("ewma", alpha=0.1).residual("z").surfaces()
median = records.measure(windows=[10, 60, 360]).baseline("median", window=20).residual("z").surfaces()
```

### From a signal

```python
from signalforge.signal import RealSignal
import numpy as np

sig = RealSignal(np.arange(500), np.sin(np.arange(500) / 20.0), channel="sine")
surfaces = sf.from_signal(sig).measure(windows=[10, 30, 90]).surfaces()
```

---

## DAG Composition

When you need branching and merging — multiple baselines, multiple residuals, stacked features for ML.

```python
from signalforge.graph import Input, Measure, Baseline, Residual, Hilbert, Stack, Pipeline

x = Input()
m = Measure()(x)

# Branch 1: EWMA baseline + z-score
bl_ewma = Baseline(method="ewma", alpha=0.1)(m)
resid_ewma = Residual(mode="z")(m, bl_ewma)

# Branch 2: median baseline + ratio
bl_median = Baseline(method="median", window=20)(m)
resid_median = Residual(mode="ratio")(m, bl_median)

# Branch 3: Hilbert
hilbert = Hilbert()(m)

# Merge into one feature-rich surface
features = Stack()([m, resid_ewma, resid_median, hilbert])

pipe = Pipeline(x, features)
result = pipe.run(records, windows=[10, 60, 360])
```

The pattern: `Operator(params)(inputs)`. First call creates the op with configuration, second call wires it into the graph. This is the [Keras functional API](https://keras.io/guides/functional_api/) pattern.

**Operators:**

| Operator | Input | Output | Purpose |
|----------|-------|--------|---------|
| `Input()` | records | signals | Ingest data |
| `Measure()` | signals | surfaces | Windowed measurement |
| `Baseline(method)` | surfaces | surfaces | Compute baseline |
| `Residual(mode)` | surfaces, surfaces | surfaces | Measured vs baseline |
| `Hilbert()` | surfaces | surfaces | Analytic signal |
| `Stack()` | [surfaces, ...] | surfaces | Merge feature branches |

`Pipeline(input, output)` defines the graph. `.run(records)` resolves the plan and builds in one call.

### Legacy path

The record-based pipeline still works for backward compatibility:

```python
x = Input(mode="records")
b = Bin(agg="mean")(x)
m = Measure(profile="continuous")(b)
pipe = Pipeline(x, m)
```

---

## Signals

Everything is a [LatticeSignal](concepts.md#signals):

```python
from signalforge.signal import LatticeSignal, RealSignal, ComplexSignal, Surface
```

| Type | Description |
|------|-------------|
| `LatticeSignal` | ABC — integer-indexed, complex-native values |
| `RealSignal` | The common case: one real-valued component (`imag=0`) |
| `ComplexSignal` | Two components on the lattice (real + imaginary) |
| `Surface` | 2D measurement grid (scale x time) — also a LatticeSignal |

**Properties available on all signals:**

```python
sig.index        # integer index array
sig.values       # value array (float64 or complex128)
sig.channel      # channel name
sig.keys         # entity dimensions (e.g. {"host": "dc01"})
sig.is_real      # True if imag component is all zeros
sig.amplitude()  # |z| — abs(values)
sig.phase()      # angle(z)
sig.real()       # real component
sig.imag()       # imaginary component
```

### Converting records to signals

```python
from signalforge.signal import records_to_signals

signals = records_to_signals(records, agg="mean")
# One RealSignal per (channel, keys) group
```

### Measuring a signal directly

```python
from signalforge.signal import measure_signal
from signalforge.lattice.sampling import SamplingPlan

plan = SamplingPlan(360, 1)
surface = measure_signal(sig, plan)
# Returns a Surface (which is a LatticeSignal)
```

---

## Surfaces

A `Surface` is a `LatticeSignal` with additional structure:

```python
surface.data          # dict of named value arrays: {"mean": array, "std": array, ...}
surface.time_axis     # bin positions
surface.scale_axis    # window sizes per row
surface.plan          # the SamplingPlan used
surface.shape         # (n_scales, n_time)
surface.coordinates   # prime exponent vectors per scale
surface.n_events      # event counts per cell
surface.coverage      # fraction of bins occupied per cell
```

The `.values` property (from the LatticeSignal contract) returns the first data array. Use `.data` to access all named arrays.

---

## Hilbert

The Hilbert transform produces the analytic signal — promoting a real signal to its natural complex form.

```python
# Via chaining
surfaces = sf.load("data.csv").measure().hilbert().surfaces()
# surfaces[0].data keys: ['mean', 'geometric_mean', ..., 'amplitude', 'phase', 'inst_freq']

# Via DAG
h = Hilbert()(m)
```

**Key principle:** Apply Hilbert to derived signals, not raw events. Smooth first for cleaner results.

```python
# Noisy — Hilbert on raw measurement
raw = sf.load("data.csv").measure().hilbert().surfaces()

# Cleaner — smooth first, then Hilbert
clean = sf.load("data.csv").measure().baseline("ewma", alpha=0.05).hilbert().surfaces()
```

### Manual ComplexSignal from Hilbert

```python
from signalforge.signal import ComplexSignal
from scipy.signal import hilbert

# Get a surface row
surface = sf.load("data.csv").measure(windows=[60]).surfaces()[0]
row = surface.data["mean"][0]

# Hilbert transform
filled = np.where(np.isfinite(row), row, np.nanmean(row))
analytic = hilbert(filled)

# Build ComplexSignal — real is the signal, imag is its Hilbert transform
sig = ComplexSignal(
    index=surface.index,
    real_part=np.real(analytic),
    imag_part=np.imag(analytic),
    channel="analytic_scale60",
)

# Feed it back — signal in, signal out
complex_surface = sf.from_signal(sig).measure(windows=[5, 10, 20]).surfaces()
```

---

## Operators

### Baseline methods

| Method | `sf inspect` | Parameters |
|--------|-------------|------------|
| EWMA | `sf inspect ewma` | `alpha`: smoothing factor (0,1] |
| Median filter | `sf inspect median` | `window`: centered window size |
| Rolling mean | `sf inspect rolling_mean` | `window`: trailing window size |

### Residual modes

| Mode | `sf inspect` | Formula |
|------|-------------|---------|
| Difference | `sf inspect difference` | `x - baseline` |
| Ratio | `sf inspect ratio` | `x / baseline` |
| Z-score | `sf inspect z` | `(x - baseline) / std(baseline)` |
