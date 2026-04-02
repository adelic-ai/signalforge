# SignalForge

Multiscale signal analysis on the p-adic divisibility lattice.
Give it any ordered sequence and explore its structure across scales — no labels, no training, no domain-specific code.

## Install

```bash
pip install adelic-signalforge
```

Or from source:

```bash
git clone https://github.com/adelic-ai/signalforge
cd signalforge
uv sync
```

## Explore your data

SignalForge works on any two-column CSV (date/index, value). Download some data and start exploring:

```bash
# Grab VIX volatility data (2005–2012)
curl -o vix.csv "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS&cosd=2005-01-01&coed=2012-12-31"

# What's in it?
sf load vix.csv
```

```
  file      : vix.csv
  records   : 2,013
  channels  : ['value']
  range     : 0 → 2,085  (span: 2,085)
  est grain : 2
```

### Build a surface

A surface is a 2D grid — time on one axis, analysis scale on the other. One command:

```bash
sf surface vix.csv -hm --max-window 360
```

This produces a [multiscale heatmap](docs/scale_space.md) showing z-score deviations across all scales. The 2008 financial crisis appears as a vertical band of red — visible across every scale simultaneously.

`--max-window 360` means "analyze at scales up to 360 bins." The [measurement geometry](docs/sampling_domain.md) — horizon, lattice, coordinates — is derived automatically from that declaration.

### Try different baselines

What's the deviation *relative to*? That's the baseline. Swap it:

```bash
sf surface vix.csv -hm --baseline ewma --alpha 0.1
sf surface vix.csv -hm --baseline median --window 20
```

Don't know what EWMA does? Ask:

```bash
sf inspect ewma
```

```
  Exponentially Weighted Moving Average
  =====================================

  Formula:  s_t = α · x_t + (1 - α) · s_{t-1}
  Params:   α (alpha): smoothing factor in (0, 1]. Higher = more responsive.
  ...
```

Available methods: `ewma`, `median`, `rolling_mean`. See `sf inspect <name>` for any of them.

### Score the residual

Remove the baseline and look at what's left:

```bash
sf surface vix.csv -hm --baseline ewma --residual ratio
sf surface vix.csv -hm --baseline median --residual z
```

Residual modes: `difference` (absolute), `ratio` (multiplicative), `z` (normalized by baseline variability). See `sf inspect ratio`, `sf inspect z`.

### Zoom in

Spotted something interesting? Zoom:

```bash
sf surface vix.csv -hm --start-date 2008-01-01 --end-date 2009-12-31
sf surface vix.csv -hm --start 800 --end 1200
```

The surface is recomputed over the zoomed region — the lattice geometry applies fresh to the subset.

### Set up a workspace

When you're doing serious exploration, initialize a workspace:

```bash
sf init my_analysis --csv vix.csv --max-window 360
cd my_analysis
sf surface -hm
sf surface -hm --baseline ewma --residual ratio --save output/ratio.png
```

The workspace stores your config, so you don't repeat flags. Outputs go to `output/`. You can version your experiments with git.

### View the lattice geometry

```bash
sf plan equities-daily
```

Shows the [sampling plan](docs/sampling_domain.md) — every window, hop, and [p-adic coordinate](docs/scale_space.md) in the lattice. This is the measurement space your surfaces live in.

## Compose pipelines in Python

For more control, compose pipelines programmatically using the [graph API](docs/overview.md):

```python
from signalforge.graph import Input, Bin, Measure, Baseline, Residual, Pipeline
from signalforge.domains import timeseries

records = timeseries.ingest("vix.csv")

x = Input()
b = Bin(agg_funcs={"value": {"value": "mean"}})(x)
m = Measure(profile="continuous")(b)
bl = Baseline(method="ewma", alpha=0.1)(m)
r = Residual(mode="ratio")(m, bl)

pipe = Pipeline(x, r)
pipe.resolve(records=records)
result = pipe.build(records)
```

Pipelines are lazy DAGs — define the graph, resolve the geometry, then build. Branch and merge freely: two baselines, two residuals, [stack](docs/operators_guide.md) them into one surface.

## What makes this different

Standard multiscale analysis (STFT, wavelets) requires the analyst to choose window sizes. SignalForge [derives the measurement space](docs/sampling_domain.md) from your declared windows and grain. The valid scales are the divisors of the horizon — a structure from number theory that guarantees artifact-free tiling, perfect nesting, and [structural invariance](docs/structural_invariance.md) across recordings.

Two signals with the same sampling plan produce surfaces that are [directly comparable](docs/structural_invariance.md) — no post-hoc normalization needed. This is what makes ML on surfaces meaningful: features at each scale occupy the same structural position in the lattice.

For a detailed comparison with wavelets, STFT, and EMD: [docs/comparison.md](docs/comparison.md)

## What counts as a signal

Anything ordered. Time-stamped sensor data, sequential event logs, daily market prices, clinical recordings, genomic positions. If it has an order and a value, SignalForge can surface its multiscale structure.

Time is not required — event-ordered sequences work with `grain=1`, where each event is one bin and scales are measured in events rather than time.

[docs/domain_guide.md](docs/domain_guide.md) — how to bring your own data

## Demonstrated results

SignalForge detected a clinical epileptic seizure at **13.96σ** on CHB-MIT EEG data — no training data, no labels, no EEG-specific code. The same pipeline processes INTERMAGNET geomagnetic observatory data unchanged.

[docs/empirical_results.md](docs/empirical_results.md) — full results and reproducibility

## Documentation

| Doc | What it covers |
|-----|----------------|
| [Sampling domain](docs/sampling_domain.md) | The lattice, horizon, grain, why divisors |
| [Scale space](docs/scale_space.md) | Surfaces, coordinates, the geometry of scale |
| [Structural invariance](docs/structural_invariance.md) | Why surfaces are directly comparable |
| [Comparison](docs/comparison.md) | STFT, wavelets, EMD vs SignalForge |
| [Pipeline overview](docs/overview.md) | Stages, architecture, the graph API |
| [Domain guide](docs/domain_guide.md) | Bring your own data |
| [Data guide](docs/data_guide.md) | Downloading datasets, running examples |
| [Operators](docs/operators_guide.md) | Transform operators and derived channels |
| [Grain estimation](docs/binjamin.md) | Automatic grain selection |
| [Grain & horizon design](docs/design_grain_snapping.md) | Internal design decisions |
| [EEG discovery](docs/discovery_eeg_seizure_detection.md) | The seizure detection case study |
| [Empirical results](docs/empirical_results.md) | Cross-domain validation |

## License

Business Source License 1.1. See [LICENSE](LICENSE).

Free for non-commercial use (research, personal projects, evaluation).
Commercial use requires a license from Adelic — contact [shun.honda@adelic.org](mailto:shun.honda@adelic.org).

Converts to Apache 2.0 on 2029-03-22.

## Citation

If you use SignalForge in published work, a citation entry will be available once the arXiv preprint is posted.
