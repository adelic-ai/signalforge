# CLI Reference

All commands and flags.

## Table of Contents

- [Commands](#commands)
- [schema](#schema)
- [load](#load)
- [surface](#surface)
- [Baselines](#baselines)
- [Residuals](#residuals)
- [Zoom](#zoom)
- [inspect](#inspect)
- [Workspace](#workspace)
- [neighborhood](#neighborhood)
- [plan](#plan)
- [demo](#demo)

---

## Commands

| Command | Purpose |
|---------|---------|
| `sf schema <csv>` | Infer and edit a data schema |
| `sf load <csv>` | Inspect a data file |
| `sf surface <csv>` | Build and display a surface |
| `sf inspect [name]` | Learn about methods and concepts |
| `sf init <name>` | Create a workspace |
| `sf status` | Show workspace status |
| `sf neighborhood <n>` | View the p-adic lattice around an integer |
| `sf plan <domain>` | Show a sampling plan |
| `sf demo` | Run the built-in EEG demo |

---

## schema

Infer, inspect, and save a data schema. For multi-column data where auto-detection isn't enough.

```bash
sf schema data.csv
```

Shows the inferred axis types. Correct anything wrong:

```bash
sf schema data.csv --set ticket_hash=relational
sf schema data.csv --group-by machine user
sf schema data.csv --channel event_code
sf schema data.csv --save mydata.schema.json
```

Then use it:

```bash
sf surface data.csv --schema mydata.schema.json -hm
```

The schema file is reusable — same format, different data, same schema.

**Flags:**

| Flag | Purpose |
|------|---------|
| `--set col=type` | Override an axis type (ordered, categorical, numeric, relational) |
| `--group-by col1 col2` | Set grouping keys for per-entity surfaces |
| `--channel col` | Set the channel axis |
| `--save path` | Save schema to JSON |
| `--load path` | Load existing schema instead of inferring |

---

## load

See what's in your data before committing to a surface build.

```bash
sf load data.csv
```

```
  SignalForge  data.csv
  ────────────────────────────────────────
  records   2,013
  channels  VIXCLS
  span      0 .. 2,085  (2,085)
  grain     2  (estimated)
  scales    4  [3 .. 2085]
  ────────────────────────────────────────

  Next:
    sf surface data.csv -hm
```

Shows: record count, channels, data range, estimated resolution, and how many scales are available. Suggests the next command. Add `--schema` to load via a saved schema.

---

## surface

The primary command. Build a [surface](concepts.md#surface) and show anomaly summary.

```bash
sf surface data.csv -hm --max-window 360
```

**Flags:**

| Flag | Purpose |
|------|---------|
| `-hm` | Render heatmap |
| `--max-window N` | Largest analysis window (horizon derived automatically) |
| `--grain N` | Override estimated grain |
| `--baseline METHOD` | Apply a baseline: `ewma`, `median`, `rolling_mean` |
| `--alpha F` | EWMA smoothing factor (default: 0.1) |
| `--window N` | Baseline window size for median/rolling_mean (default: 20) |
| `--residual MODE` | Residual mode: `difference`, `ratio`, `z` (requires `--baseline`) |
| `--start N` / `--end N` | Zoom by bin index |
| `--start-date D` / `--end-date D` | Zoom by date (e.g. `2008-01-01`) |
| `--save PATH` | Save heatmap to file instead of displaying |
| `--name LABEL` | Save this run as a named experiment in the workspace |

**Output includes:**
- Record count, channels, horizon, basis, scales
- Anomaly summary with peak z-score per scale
- Peak anomaly location (date or bin number)
- Suggestions for next steps (add heatmap, try baseline, zoom into peak)

Suggestions are suppressed by setting `"suggestions": false` in the workspace `sf.json`.

---

## Baselines

A baseline estimates what "normal" looks like. The residual is what's left after removing it.

```bash
sf surface data.csv -hm --baseline ewma --alpha 0.1
sf surface data.csv -hm --baseline median --window 20
sf surface data.csv -hm --baseline rolling_mean --window 20
```

| Method | Formula | Best for |
|--------|---------|----------|
| `ewma` | `s_t = alpha * x_t + (1-alpha) * s_{t-1}` | Drifting mean, trends |
| `median` | Centered rolling median | Spike-heavy, outlier-heavy data |
| `rolling_mean` | Trailing window mean | Simple smoothing |

Use `sf inspect ewma` (or `median`, `rolling_mean`) for details.

---

## Residuals

Remove the baseline and score what remains.

```bash
sf surface data.csv -hm --baseline ewma --residual z
sf surface data.csv -hm --baseline ewma --residual ratio
sf surface data.csv -hm --baseline median --residual difference
```

| Mode | Formula | Meaning |
|------|---------|---------|
| `difference` | `x - baseline` | Absolute deviation (preserves units) |
| `ratio` | `x / baseline` | Multiplicative deviation (scale-invariant) |
| `z` | `(x - baseline) / std(baseline)` | Normalized by variability (unitless) |

Use `sf inspect z` (or `ratio`, `difference`) for details.

---

## Zoom

Spotted an anomaly? Zoom in for finer resolution.

```bash
# By date
sf surface data.csv -hm --start-date 2007-06-01 --end-date 2009-06-01

# By bin index
sf surface data.csv -hm --start 400 --end 600
```

When you zoom, the [grain](concepts.md#grain) is re-estimated for the smaller region — typically finer than the full dataset. This means more scales are available in the zoomed view. Coarse-to-fine exploration happens naturally.

The CLI suggests a zoom command centered on the peak anomaly after every `sf surface` run. Copy-paste and go.

---

## inspect

Browse and learn every method, operator, and concept in SignalForge.

```bash
sf inspect           # list everything
sf inspect ewma      # details on EWMA baseline
sf inspect lattice   # what the lattice is
sf inspect surface   # what a surface is
```

Shows formula, parameters, description, when to use it, and an example command. Available entries:

**Baselines:** ewma, median, rolling_mean
**Residuals:** difference, ratio, z
**Concepts:** horizon, grain, surface, lattice

---

## Workspace

For sustained exploration, initialize a workspace. It stores your data path, defaults, and named experiments.

```bash
sf init my_project --csv data.csv --max-window 360
cd my_project
sf surface -hm                    # picks up csv and max-window from sf.json
sf surface -hm --baseline ewma --residual z --name ewma_z   # saves the run
sf status                         # see data, defaults, cached surfaces, runs
```

**Directory structure:**

```
my_project/
  sf.json           # workspace config (csv, defaults, suggestions toggle)
  cache/            # cached surfaces (future)
  output/
    ewma_z/         # named experiment
      meta.json     # plan, parameters, csv path
      value_mean.npy  # surface data
      heatmap.png   # if -hm was used
```

`sf status` shows everything in the workspace. Experiments are just directories — version them with git.

To suppress CLI suggestions, add `"suggestions": false` to `sf.json`.

---

## neighborhood

View the p-adic [lattice](concepts.md#lattice) around any integer.

```bash
sf neighborhood 360
sf neighborhood 360 10              # radius 10
sf neighborhood 360 --basis 2 3 5   # explicit prime basis
```

Shows the divisibility structure, prime factorization, and lattice coordinates of nearby integers.

---

## plan

Show a [sampling plan](concepts.md#horizon) for a built-in domain.

```bash
sf plan equities-daily
sf plan eeg
sf plan intermagnet
```

Shows horizon, grain, cbin, prime basis, and every window with its hop, bin count, and lattice coordinate.

---

## demo

Run the built-in EEG seizure detection demo.

```bash
sf demo
```

Requires the CHB-MIT EEG data to be preprocessed first. See [examples](examples.md) for setup.
