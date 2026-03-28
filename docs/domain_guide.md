# Domain Guide

You already know your signal. You know which timescales matter, which windows are
standard in your field, and what "normal" looks like in your data.

Writing a SignalForge domain encodes that knowledge once. What you get back:

- A **multiscale surface** over your data — every meaningful window computed
  simultaneously, across the full time axis, in a single pass
- **Structurally invariant features** — the same 32 features per window, regardless
  of data length or source, directly comparable across recordings, patients, sessions,
  instruments, or sites
- **Unsupervised anomaly scores** — z-scores per scale, no labels required, no
  training, no threshold tuning
- **ML-ready tensors** — surfaces from different sources sharing the same plan stack
  directly, no alignment or normalization step

The EEG domain took an afternoon to write. The pipeline then detected a clinical
seizure at 13.96σ with no labels and no EEG-specific processing. The geomagnetic
domain runs the same pipeline on a year of Yellowknife observatory data. Same code.
Different geometry declaration.

Your domain is next.

---

## Just want specific windows?

If you know which windows you need, let `horizon_for()` compute the geometry
for you:

```python
import signalforge as sf

# I need 7s, 30s, and 60s windows at 1s resolution
plan = sf.SamplingPlan(sf.horizon_for([7, 30, 60], grain=1), grain=1)
```

`horizon_for()` returns the smallest horizon that contains all your requested
windows as lattice members. The lattice then fills in any intermediate scales
automatically. No manual horizon calculation needed.

For most domain masters writing a full domain module, read on.

---

## What a domain actually is

A single Python file in `signalforge/domains/` that implements one function:

```python
def sampling_plan(...) -> SamplingPlan:
```

The `SamplingPlan` answers two questions:

| | |
|---|---|
| `horizon` | What is the natural outer span of your data? One day, one session, one orbit. |
| `grain` | What is the finest meaningful unit? One second, one sample, one tick. |

From those two integers, the p-adic divisibility lattice enumerates every valid
measurement window. You select the ones your field actually uses. The pipeline
builds a surface — a time × scale grid — and extracts features from it.

Two datasets that share a `SamplingPlan` produce surfaces with identical shape.
They stack directly. They are comparable without alignment, normalization, or
feature engineering.

---

## Three questions to answer before writing a line of code

**1. What is your natural outer span?**

The `horizon` is the integer that your important windows divide into cleanly.
For daily data, one day. For clinical EEG, one hour. For equity trading, one
session. It does not need to equal your data length — it defines the coordinate
space.

**2. What is your grain?**

The smallest unit your data meaningfully supports. Not the raw sample rate —
the smallest unit you would actually analyze. For 256 Hz EEG, one second (256
samples). For 1-minute geomagnetic data, one minute (60 seconds).

If your data is unfamiliar and the right grain is not obvious, estimate it
from the data and let the horizon be derived automatically:

```python
from signalforge.lattice.sampling import grain_from_orders, SamplingPlan

orders = [r.primary_order for r in records]
g      = grain_from_orders(orders)               # raw estimate, no snapping
plan   = SamplingPlan.from_windows(
    windows=[g*5, g*15, g*60, g*360],            # anchor windows as multiples of grain
    grain=g,
)
```

`from_windows` derives `horizon = lcm(windows + [grain])`, guaranteeing the
grain is an exact lattice member — no nudging. The grain the data suggests is
the grain used. A different method can be selected:

```python
g = grain_from_orders(orders, method="knuth")
```

See [binjamin.md](binjamin.md) for the full method list and
[design_grain_snapping.md](design_grain_snapping.md) for the design rationale.

**3. Which windows does your field already use?**

These become your anchors. Standard clinical EEG epochs are 1s, 4s, 16s, 30s,
60s. Standard geomagnetic products are 1min, 1hr, 1day. You know these — you
work in this field. List them. The lattice fills in the rest.

---

## The pattern

```python
from signalforge.lattice.coordinates import lattice_members, smallest_divisor_gte
from signalforge.lattice.sampling import SamplingPlan

def sampling_plan(horizon: int, grain: int) -> SamplingPlan:
    cbin = smallest_divisor_gte(horizon, grain)
    valid = set(lattice_members(horizon, cbin))

    anchors = {standard_window_1, standard_window_2, horizon}
    fine_cutoff = your_field_transition_scale

    selected = sorted(
        w for w in valid
        if w in anchors or w <= fine_cutoff or w == horizon
    )
    return SamplingPlan(horizon, grain, windows=selected)
```

`fine_cutoff` is the scale below which you want every lattice member included.
Above it, only anchors and the horizon. This keeps the surface focused without
losing the multiscale structure where it matters most.

---

## A complete example: intraday equities

One trading session (6.5 hours), 1-second resolution. Standard windows: 1min,
5min, 1hr, full session.

```python
"""signalforge.domains.equities — intraday equity tick data."""

from __future__ import annotations
from ..lattice.coordinates import lattice_members, smallest_divisor_gte
from ..lattice.sampling import SamplingPlan

_ONE_MINUTE  = 60
_FIVE_MINUTES = 300
_ONE_HOUR    = 3_600
_SESSION     = 23_400   # 6.5 hours

def sampling_plan(
    horizon: int = _SESSION,
    grain: int = 1,
) -> SamplingPlan:
    """SamplingPlan for 1-second equity data over one trading session."""
    cbin = smallest_divisor_gte(horizon, grain)
    valid = set(lattice_members(horizon, cbin))

    anchors = {_ONE_MINUTE, _FIVE_MINUTES, _ONE_HOUR, horizon}
    fine_cutoff = _FIVE_MINUTES

    selected = sorted(
        w for w in valid
        if w in anchors or w <= fine_cutoff or w == horizon
    )
    return SamplingPlan(horizon, grain, windows=selected)
```

Drop it in `signalforge/domains/equities.py`. That's the entire domain.

---

## Checklist

- [ ] `horizon` and `grain` are in the same units
- [ ] `horizon` is divisible by `grain` — otherwise `cbin > grain` and you lose resolution
- [ ] `horizon` is always an anchor — ensures the full span is always a window
- [ ] Anchors are standard windows from your field, not invented ones
- [ ] Module docstring names the domain, units, sampling rate, and a reference

---

## Reference implementations

| File | Domain | Horizon | Grain |
|------|--------|---------|-------|
| `signalforge/domains/intermagnet.py` | Geomagnetic observatory | 86400 s (1 day) | 60 s (1 min) |
| `signalforge/domains/eeg.py` | Clinical EEG at 256 Hz | 921600 samples (1 hr) | 256 samples (1 s) |

`intermagnet.py` is the canonical reference. Read it first.

For adding derived channels, custom cleaning, and other pipeline operators, see [operators_guide.md](operators_guide.md).
