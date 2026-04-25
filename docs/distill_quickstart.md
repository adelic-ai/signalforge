# SignalForge quickstart

> **Tune in to trends in any ordered data.** This walks you all the way from clone-the-repo to ML-ready feature matrices on a dataset of your choice. Domain-agnostic — works on financial ticks, sensor logs, web events, security telemetry, anything with an order to it.

---

## 1. Setup

### 1.1 Prerequisites

- **Python 3.12+** — `python3 --version` to check
- **git** — `git --version`
- **uv** (recommended; faster) — install from <https://docs.astral.sh/uv/>; pip works too

### 1.2 Clone and install

```bash
git clone https://github.com/adelic-ai/signalforge.git
cd signalforge
git checkout distill-pipeline
uv pip install -e .
```

If you'd rather use plain pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 1.3 Verify

```bash
sf --help
```

You should see commands listed: `init`, `load`, `surface`, `schema`, `distill`, `inspect`, etc. If `sf` isn't on your PATH, try `python -m signalforge --help`.

---

## 2. Set up a workspace and see the structure

A workspace is a folder for one dataset's analysis. SignalForge stores config, caches, and outputs there. Recommended: `git init` it too so your work is versioned.

```bash
sf init mywork --csv path/to/your/data.csv
cd mywork
```

What you'll see:

```
SignalForge  workspace
────────────────────────────────────────
created   mywork/
cache     mywork/cache/
output    mywork/output/
data      data.csv
records   25,971
channels  EventCode
grain     1  (estimated)
────────────────────────────────────────

cd mywork && sf surface -hm
```

Workspace layout:

```
mywork/
├── cache/                 (intermediate results)
├── output/                (heatmaps, feature matrices, etc.)
├── sf-config.json         (workspace settings)
└── data.csv               (your data, if you copied/symlinked it in)
```

### See the structure (heatmap)

The very next thing to do is just look:

```bash
sf surface -hm
```

This builds a multiscale surface and displays it as a heatmap. Y-axis is scale (analysis window size); X-axis is time. Bright bands are where the data deviates most from its local baseline. This is the "see the structure" first impression — works on any dataset, no configuration.

Useful variants:

```bash
sf surface -hm --max-window 360                            # cap the largest scale
sf surface -hm --baseline ewma --residual z                # anomaly view
sf surface -hm --start-date 2008-01-01 --end-date 2009-06-01   # zoom in
sf surface -hm --save out.png                              # save instead of display
```

If you don't have a dataset to try yet, **any CSV with a timestamp column + one or more categorical or numeric columns** works. Two-column timeseries (date + value) work too.

---

## 3. Dig deeper into the data

Everything in this section is read-only — no setup decisions made yet. These commands work on any CSV.

### 3.1 Textual summary

```bash
sf load data.csv
```

Record count, distinct channels, time span, estimated grain, default sampling plan.

### 3.2 Schema inference

```bash
sf schema data.csv
```

Shows the inferred axis types (timestamps, categoricals, numerics) and which fields the engine plans to use as group-by axes.

Override the inference if needed:

```bash
sf schema data.csv --channel EventCode --group-by Account_Name
sf schema data.csv --save my.schema.json   # persist the override
```

### 3.3 Inspect operations and concepts

```bash
sf inspect              # list everything inspectable
sf inspect ewma         # explain the ewma baseline
sf inspect lattice      # explain the analysis lattice
sf inspect z            # explain z-scoring
```

---

## 4. Distill: discover segments

A **segment** is a burst of activity from one entity, bounded by silence or by a natural information-theoretic boundary. SignalForge finds them automatically — no gap thresholds, no manual binning.

### 4.1 CLI

```bash
sf distill data.csv \
    --entity-key Account_Name \
    --joins source_ip,service \
    --output features.npz
```

`--entity-key` tells SignalForge which field identifies the "actor" (user, machine, ticker, sensor, etc.). Without it, all records collapse into one global entity.

`--joins` adds cross-segment fan-out features for the named field(s) — useful for catching anomalies that involve many entities (one IP touching many accounts, one service flooded by requests).

Output:

```
SignalForge  distill  data.csv
────────────────────────────────────────
records   25,971
segments  9,330
channels  EventCode
entity    Account_Name
features  (9330, 16)
joins     source_ip, service
method    information_gain
────────────────────────────────────────
distill   18.78s
featurize 0.07s
saved     features.npz
```

### 4.2 Python — direct

```python
import signalforge as sf

records   = sf.Schema.infer("data.csv").records()
distilled = sf.distill(records, entity_key="Account_Name")
features  = sf.featurize(distilled, joins=["source_ip"])

print(features.shape)        # (n_segments, n_features)
print(features.names[:5])    # column labels
features.matrix              # numpy array, ready for ML
```

### 4.3 Python — chained (one expression)

```python
features = (
    sf.load("data.csv")
      .distill(entity_key="Account_Name")
      .featurize(joins=["source_ip"])
)
```

### 4.4 What the result objects carry

**`DistillResult`** (output of `distill()`):

```python
distilled.segments       # list[Segment] — discovered bursts
distilled.stats          # dict — discovery statistics
distilled.records        # the (possibly re-keyed) input records
distilled.channels()     # list[str] — distinct channels
distilled.summary()      # the stats dict
distilled.featurize(...) # convenience: same as sf.featurize(distilled, ...)
len(distilled)           # n_segments
```

**`FeatureSet`** (output of `featurize()`):

```python
features.matrix          # np.ndarray, shape (n_segments, n_features)
features.names           # list[str], one per column
features.shape           # (n, k)
features.info            # metadata from the underlying extractors
len(features)            # n_segments
```

---

## 5. Custom features

If the built-in features aren't enough, compose with the primitives directly. Every function `featurize` uses is exposed at the top level: `sf.entropy`, `sf.mutual_information`, `sf.kl_divergence`, `sf.discover_scales`, `sf.discover_plan`, `sf.discover_segments`, `sf.segments_to_matrix`, `sf.join_segments`.

Example — add per-segment value variance:

```python
import numpy as np

def value_variance(segment):
    vals = [e.value for e in segment.events]
    return float(np.var(vals)) if vals else 0.0

custom_col = np.array([value_variance(s) for s in distilled.segments])

features.matrix = np.hstack([features.matrix, custom_col.reshape(-1, 1)])
features.names  = features.names + ["value_variance"]
```

You can also use `sf.entropy(arr)`, `sf.mutual_information(a, b)`, etc. inside your custom feature function.

---

## 6. Hand off to ML

`features.matrix` is a plain numpy array. Hand it to anything that takes one.

### 6.1 Load saved features

```python
import numpy as np

arc = np.load("features.npz", allow_pickle=True)
matrix, names = arc["matrix"], arc["names"]
```

### 6.2 sklearn — anomaly detection (no labels needed)

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(matrix)

clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X)

scores = clf.score_samples(X)            # higher = more normal
anomalies = clf.predict(X) == -1         # True = anomaly

# Top anomalies
import numpy as np
idx = np.argsort(scores)[:10]
for i in idx:
    print(f"score={scores[i]:.3f}  segment={distilled.segments[i]}")
```

### 6.3 pandas — exploration

```python
import pandas as pd

df = pd.DataFrame(matrix, columns=list(names))
df.describe()
df.corr()
```

### 6.4 PyTorch / autoencoder

```python
import torch
X = torch.from_numpy(matrix).float()
# ... train an autoencoder, reconstruction error → anomaly score
```

---

## 7. Power user: graph DAG

The fluent chain (`.distill().featurize()`) is one path. For branching pipelines (e.g., compute multiple feature sets in parallel from one distill, or combine surface analysis with segment features), use the graph API:

```python
from signalforge.graph import Input, Measure, Baseline, Residual, Stack, Pipeline

x  = Input()
m  = Measure(windows=[10, 60, 360])(x)
bl = Baseline("ewma", alpha=0.1)(m)
r  = Residual("z")(m, bl)
out = Stack()([m, r])

pipe = Pipeline(x, out)
result = pipe.run(records)
```

For now, distill is fluent-only (no `DistillOp` graph node yet); compose it with the graph API by running both passes and combining results in user code.

---

## 8. What this is not

- **Not a black box.** Every primitive `distill` and `featurize` use is exposed at the top level. Compose your own pipeline if the convenience layer doesn't fit.
- **Not a domain tool.** SignalForge knows nothing about cybersecurity, finance, or any specific schema. The records you supply define the domain; the engine just finds structure.
- **Not opinionated about ML.** `FeatureSet.matrix` is a numpy array. Hand it to whatever you'd hand a numpy array to.

---

## 9. Going deeper

| File | What it covers |
|---|---|
| [README](../README.md) | Heatmap-first surface analysis (the visualization side of SF) |
| [docs/python-api.md](python-api.md) | Chaining API, graph DAG composition |
| [docs/cli.md](cli.md) | All CLI commands |
| [docs/concepts.md](concepts.md) | The lattice, scales, what multiscale means |
| [docs/examples.md](examples.md) | VIX, EEG, GRACE, INTERMAGNET worked examples |

For questions: open an issue or ping the author.

---

## Appendix: troubleshooting

**`sf` command not found.** Activate the venv (`source .venv/bin/activate`) or run `python -m signalforge` instead.

**`ModuleNotFoundError: No module named 'binjamin'`.** Re-run `uv pip install -e .` — `binjamin` is a PyPI dependency that should auto-install.

**`sf distill` produces 1 entity (everything collapses).** Pass `--entity-key <field>` to specify which field identifies the actor. Without it, the schema doesn't know how to group records.

**Distill is slow on large datasets.** The default `--method information_gain` is computationally heavier than the gap-based alternative. For initial exploration: `--method gap` is fast. For final analysis: `information_gain` gives sharper segments.

**Heatmaps don't render in a notebook.** Run `sf surface ... --save out.png` and view the file directly, or use `%matplotlib inline` in your notebook.
