# Distill quickstart

> **Tune in to trends in any ordered data.** Give SignalForge a CSV (or any record stream), get back the segments where activity clusters and the features that describe them.

This is the fast path for any user who wants to find structure in event-like or time-series data without writing pipelines by hand. Domain-agnostic — works on financial ticks, sensor logs, web events, security telemetry, or anything else with an order to it.

---

## Install

The repo is private; install from a clone (or fork it first if you want your own copy to push to):

```bash
git clone https://github.com/adelic-ai/signalforge.git
cd signalforge
git checkout distill-pipeline   # current demo branch
uv pip install -e .
```

If you don't have `uv`, install it from <https://docs.astral.sh/uv/> (recommended), or use any Python 3.12+ environment with `pip install -e .` instead.

Verify:

```bash
sf --help
```

---

## 30-second flow

Three commands. Replace `data.csv` with any multi-column CSV that has a timestamp (or any monotonic ordering column) plus at least one categorical or numeric column.

```bash
sf init mywork --csv data.csv
cd mywork
sf distill data.csv --entity-key user_id --joins source_ip --output features.npz
```

What you get:

- `mywork/` — a workspace directory with `cache/` and `output/`
- A printed summary: how many records, how many segments, the channels found, the feature matrix shape
- `features.npz` — numpy archive with `matrix` (n_segments × n_features) and `names` (column labels) ready for ML

---

## Python API

### Direct

```python
import signalforge as sf

records   = sf.Schema.infer("data.csv").records()
distilled = sf.distill(records, entity_key="user_id")
features  = sf.featurize(distilled, joins=["source_ip"])

print(features.shape)       # (n_segments, n_features)
print(features.names[:5])   # column labels
features.matrix             # numpy array, ready for ML
```

### Chained (one-liner)

```python
features = (
    sf.load("data.csv")
      .distill(entity_key="user_id")
      .featurize(joins=["source_ip"])
)
```

### What the objects carry

- **`DistillResult`** — `.segments`, `.stats`, `.records`, `.channels()`, `.summary()`, `.featurize(...)`, `len()`
- **`FeatureSet`** — `.matrix` (numpy array), `.names` (list[str]), `.shape`, `.info`, `len()`

Both are plain dataclasses; inspect them however you like.

---

## What's actually happening

1. **Records** are parsed from CSV with a Schema that infers types (dates, categorical, numeric) and group-by axes.
2. **Distill** discovers segments — bursts of activity, bounded by silence or by information-theoretic boundaries — per entity. The default `method="information_gain"` walks the lattice from coarse to fine and splits where Shannon-entropy reduction is highest. No gap thresholds, no manual binning. (The classic gap-based approach is still available as `method="gap"`.)
3. **Featurize** computes a numeric feature matrix per segment: duration, event count, channel mix ratios, density, plus optional cross-segment join features.

The math primitives are all accessible directly if you want to bypass the convenience layer:

```python
sf.entropy(arr)
sf.mutual_information(a, b)
sf.kl_divergence(p, q)
sf.discover_scales(signal, horizon, grain)
sf.discover_plan(signal, horizon, grain)
sf.discover_segments(records)
```

---

## Common patterns

### Just explore the data

```python
records = sf.load("data.csv").records()
distilled = sf.distill(records)             # auto everything
print(distilled.summary())                   # stats
print(distilled.channels())                  # channels found
```

### Group by a specific entity field

```python
distilled = sf.distill(records, entity_key="user_id")
# Segments are now per-user
```

### Add cross-segment fan-out features

```python
features = sf.featurize(distilled, joins=["source_ip", "service"])
# Each join key adds a block of features about how segments
# relate to each other through that field — entity diversity,
# co-occurrence, fan-out shape.
```

### Pick the analysis method

```python
distilled = sf.distill(records, method="information_gain")  # default, IL-driven
distilled = sf.distill(records, method="gap")               # classic gap-based
```

### Save to disk for ML

```bash
sf distill data.csv --entity-key user_id --output features.npz
```

```python
import numpy as np
arc = np.load("features.npz", allow_pickle=True)
matrix, names = arc["matrix"], arc["names"]
```

---

## What this is not

- **Not a black box.** Every primitive that distill uses is exposed at the top level: `sf.entropy`, `sf.mutual_information`, `sf.kl_divergence`, `sf.discover_scales`, `sf.discover_plan`, `sf.discover_segments`, `sf.segments_to_matrix`, `sf.join_segments`. Compose your own pipeline if the convenience layer doesn't fit.
- **Not a domain tool.** SignalForge knows nothing about cybersecurity, finance, or any specific schema. The records you supply define the domain; the engine just finds structure.
- **Not opinionated about ML.** `FeatureSet.matrix` is a numpy array. Hand it to whatever you'd hand a numpy array to.

---

## Going deeper

| File | What it covers |
|---|---|
| [README](../README.md) | Heatmap-first surface analysis (the visualization side of SF) |
| [docs/python-api.md](python-api.md) | Chaining API, graph DAG composition |
| [docs/cli.md](cli.md) | All CLI commands |
| [docs/concepts.md](concepts.md) | The lattice, scales, what multiscale means |
| [docs/examples.md](examples.md) | VIX, EEG, GRACE, INTERMAGNET worked examples |

For questions: open an issue or ping the author.
