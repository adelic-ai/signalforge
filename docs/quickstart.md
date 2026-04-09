# Quick Start

Zero to heatmap in 3 minutes.

## Install

```bash
pip install adelic-signalforge
```

## Download sample data

```bash
curl -o vix.csv "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS&cosd=2005-01-01&coed=2012-12-31"
```

This is the CBOE Volatility Index (VIX), daily, 2005-2012. It includes the 2008 financial crisis.

## See what's in it

```bash
sf load vix.csv
```

```
  SignalForge  vix.csv
  ────────────────────────────────────────
  records   2,013
  channels  VIXCLS
  grain     2  (estimated)
  scales    4  [3 .. 2085]
  ────────────────────────────────────────

  Next:
    sf surface vix.csv -hm
```

## Build a surface

```bash
sf surface vix.csv -hm --max-window 360
```

A heatmap appears. Time on the x-axis, analysis scales on the y-axis. Red = above normal, blue = below. The 2008 crisis is a vertical red band — visible across every scale at once.

## What's anomalous?

Add a baseline and score the deviation:

```bash
sf surface vix.csv -hm --max-window 360 --baseline ewma --residual z
```

Now each cell shows how many standard deviations it is from the baseline. The crisis stands out even more.

## Zoom in

The output suggests a zoom command centered on the peak anomaly. Copy and paste it:

```bash
sf surface vix.csv -hm --start-date 2007-06-01 --end-date 2009-06-01
```

The zoomed region automatically gets finer resolution — more scales, smaller bins.

## Learn more

Don't know what EWMA is? Ask:

```bash
sf inspect ewma
```

Want to see all available methods and concepts?

```bash
sf inspect
```

## What's next

- [Use your own data](your-data.md) — `sf schema` infers the structure from any CSV
- [CLI reference](cli.md) — all commands and flags
- [Python API](python-api.md) — chaining and DAG composition
- [Examples](examples.md) — VIX, EEG, GRACE satellite data, and more
