# Data Guide

How to get your data into SignalForge. Start with the simplest path that works, refine later.

## Table of Contents

- [Simplest Path](#simplest-path)
- [Multi-Channel Data](#multi-channel-data)
- [Event Data (Logs, Transactions)](#event-data)
- [Keyed Data (Per-Entity Analysis)](#keyed-data)
- [Choosing Grain](#choosing-grain)
- [Large Data](#large-data)
- [Domain Adapter Pattern](#domain-adapter-pattern)

---

## Simplest Path

A two-column CSV: first column is the index (date, timestamp, or sequence number), second is the value.

```csv
date,value
2005-01-03,12.07
2005-01-04,11.37
2005-01-05,11.55
```

Load and go:

```bash
sf load data.csv
sf surface data.csv -hm
```

Or in Python:

```python
import signalforge as sf
surfaces = sf.load("data.csv").measure().surfaces()
```

SF auto-detects: first column as index, second as value. Grain estimated from the data. No configuration needed.

**Requirements:**
- Values must be numeric (floats or integers)
- Index should be ordered (ascending dates or sequence numbers)
- Missing values are dropped automatically

---

## Multi-Channel Data

If your data has multiple measurement types, you have two options.

### Option A: Separate CSV files per channel

```bash
sf surface temperature.csv -hm
sf surface pressure.csv -hm
```

Each file is a two-column CSV. Compare surfaces visually or stack them in Python.

### Option B: Build CanonicalRecords in Python

```python
from signalforge.signal import CanonicalRecord, OrderType
import signalforge as sf

records = []
for row in my_data:
    # One record per channel per time step
    records.append(CanonicalRecord(
        primary_order=row.timestamp,
        order_type=OrderType.TIME,
        channel="temperature",
        metric="value",
        value=row.temp,
        time_order=row.timestamp,
    ))
    records.append(CanonicalRecord(
        primary_order=row.timestamp,
        order_type=OrderType.TIME,
        channel="pressure",
        metric="value",
        value=row.pressure,
        time_order=row.timestamp,
    ))

surfaces = sf.load(records).measure().surfaces()
# Returns one surface per channel
```

Each channel gets its own surface on the same lattice — directly comparable.

---

## Event Data

Logs, transactions, network events — anything where each row is an event, not a periodic measurement.

### The key decision: time-ordered or event-ordered?

**Time-ordered** — use timestamps as the index. Events spaced irregularly in time. Gaps between events are real (the system was idle). Grain is estimated from inter-event intervals.

```python
CanonicalRecord(
    primary_order=unix_timestamp,
    order_type=OrderType.TIME,
    channel="kerberos_4768",
    metric="count",
    value=1.0,
    time_order=unix_timestamp,
)
```

**Event-ordered** — use sequence numbers as the index. Events are numbered 1, 2, 3... Gaps in time disappear. Grain is trivially 1. The pattern of events matters, not when they happened.

```python
CanonicalRecord(
    primary_order=event_number,
    order_type=OrderType.SEQUENCE,
    channel="kerberos_4768",
    metric="count",
    value=1.0,
    seq_order=event_number,
)
```

**Both** — when you have both a timestamp and a sequence number:

```python
CanonicalRecord(
    primary_order=unix_timestamp,  # or seq_order — your choice
    order_type=OrderType.BOTH,
    channel="kerberos_4768",
    metric="count",
    value=1.0,
    time_order=unix_timestamp,
    seq_order=event_number,
    order_delta=seq_order - time_order,
)
```

Use `OrderType.BOTH` to surface the same data on both orderings and compare.

### Converting event counts

Raw logs often have one row per event. To get event rates, count events per time bin:

```python
import pandas as pd

df = pd.read_csv("auth_log.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Count events per minute
counts = df.set_index("timestamp").resample("1min").size()

# Save as two-column CSV
counts.to_csv("auth_rate_1min.csv", header=["count"])
```

Now `sf surface auth_rate_1min.csv -hm` works directly.

---

## Keyed Data

When you have multiple entities (machines, users, sensors) and want per-entity surfaces:

```python
records = []
for row in my_data:
    records.append(CanonicalRecord(
        primary_order=row.timestamp,
        order_type=OrderType.TIME,
        channel="auth",
        metric="count",
        value=1.0,
        time_order=row.timestamp,
        keys={"machine": row.hostname, "nic": row.mac_address},
    ))

surfaces = sf.load(records).measure().surfaces()
# One surface per (channel, keys) combination
```

**Important:** all entities end up on the same lattice (same grain, same windows, same horizon). This is what makes them comparable — structural invariance.

### Choosing the grain for keyed data

Different entities may have different cadences. Machine A authenticates every 5 minutes, Machine B every 30 minutes. The grain must accommodate the sparsest entity:

- Estimate data_grain per entity
- Take the maximum (coarsest) across all entities
- Use that as the grain for the shared lattice

If you don't, sparse entities will have mostly empty bins and meaningless surfaces.

---

## Choosing Grain

SF estimates grain automatically from inter-event intervals. But you can override it when you know better.

### Let SF estimate (default)

```bash
sf load data.csv
# Shows: grain  2  (estimated)
```

Good for: unknown data, first exploration, quick look.

### Declare from domain knowledge

```python
# EEG at 256 Hz: one second = 256 samples
surfaces = sf.load(records).measure().surfaces(grain=256)

# Minute-resolution logs
surfaces = sf.load(records).measure().surfaces(grain=60)
```

Good for: known cadence, sensor data, fixed-rate sampling.

### When the estimate is wrong

The estimator (Freedman-Diaconis on inter-event intervals) can be fooled by:
- **Bimodal spacing** — events at 1s and 60s intervals. Estimate lands between them.
- **Perfectly regular data** — constant intervals give IQR=0, falls back to a small estimate.
- **Very sparse data** — few events, unreliable statistics.

Override with `--grain` on the CLI or `grain=` in Python.

### The relationship between grain and windows

Grain must be ≤ cbin (= gcd of windows). If your grain is 7 and your windows are multiples of 10, they don't divide evenly. Either:
- Choose windows that are multiples of your grain
- Let SF derive windows from the lattice (they'll be multiples of cbin automatically)

---

## Large Data

### NetCDF files

For scientific data in netCDF format (GRACE, climate, ocean):

```python
import netCDF4 as nc
import numpy as np
import pandas as pd

ds = nc.Dataset("data.nc")
time = ds.variables["time"][:]
values = ds.variables["temperature"][:, lat_idx, lon_idx]

# Convert to CSV or build records directly
df = pd.DataFrame({"time": time, "value": values})
df.to_csv("extracted.csv", index=False)
```

### Performance tips

- **100k points**: ~30ms. No issues.
- **1M points**: ~500ms. Fine for interactive use.
- **10M points**: ~8s. Consider subsetting first.
- **100M+ points**: Chunk the data. Process segments, combine surfaces.

### Chunking large time series

```python
chunk_size = 1_000_000
for start in range(0, len(records), chunk_size):
    chunk = records[start:start + chunk_size]
    surface = sf.load(chunk).measure().surfaces()[0]
    # Process or save each chunk's surface
```

Surfaces from different chunks with the same SamplingPlan are structurally comparable.

---

## Domain Adapter Pattern

For repeated use with a specific data format, write a domain adapter — a Python module with `ingest()` and `sampling_plan()` functions. See `signalforge/domains/` for examples.

### Minimal adapter

```python
# mydata.py
import pandas as pd
from signalforge.signal import CanonicalRecord, OrderType

def ingest(path):
    """Load my data format into CanonicalRecords."""
    df = pd.read_csv(path)
    return [
        CanonicalRecord(
            primary_order=int(row.timestamp),
            order_type=OrderType.TIME,
            channel=row.sensor_id,
            metric="reading",
            value=float(row.value),
            time_order=int(row.timestamp),
        )
        for _, row in df.iterrows()
    ]
```

Then:

```python
import signalforge as sf
from mydata import ingest

records = ingest("mydata.csv")
surfaces = sf.load(records).measure().surfaces()
```

### Built-in adapters

| Adapter | Data format | Import |
|---------|-------------|--------|
| `timeseries` | Two-column CSV (date + value) | `from signalforge.domains import timeseries` |
| `eeg` | CHB-MIT EEG (t_sec + eeg_rms) | `from signalforge.domains import eeg` |
| `equities` | Yahoo Finance (timestamp + ticker + metric + value) | `from signalforge.domains import equities` |
| `intermagnet` | IAGA-2002 geomagnetic (timestamp + station + component + value) | `from signalforge.domains import intermagnet` |
