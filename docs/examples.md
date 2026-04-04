# Examples

SignalForge is domain-agnostic. These examples demonstrate the same pipeline on different types of data — from financial markets to clinical EEG to geomagnetic observatories.

## Table of Contents

- [VIX Volatility Index](#vix-volatility-index)
- [EEG Seizure Detection](#eeg-seizure-detection)
- [INTERMAGNET Geomagnetic Data](#intermagnet-geomagnetic-data)
- [Generic Time Series](#generic-time-series)
- [Bringing Your Own Data](#bringing-your-own-data)

---

## VIX Volatility Index

Daily CBOE Volatility Index, 2005-2012. The 2008 financial crisis as a regime change detection problem.

### Get the data

```bash
curl -o vix.csv "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS&cosd=2005-01-01&coed=2012-12-31"
```

### CLI

```bash
sf load vix.csv
sf surface vix.csv -hm --max-window 360
sf surface vix.csv -hm --max-window 360 --baseline ewma --residual z
sf surface vix.csv -hm --start-date 2007-06-01 --end-date 2009-06-01
```

### Python

```python
import signalforge as sf

# Quick exploration
surfaces = (
    sf.load("vix.csv")
    .measure(windows=[10, 30, 60, 90, 180, 360])
    .baseline("ewma", alpha=0.1)
    .residual("z")
    .surfaces()
)

# The 2008 crisis appears as a broad vertical band of high z-scores
# across all scales — a regime change visible simultaneously at every
# analysis resolution.
```

**What it shows:** SignalForge detects regime changes in financial time series without knowing what a "regime" is. The 2008 crisis is visible across every scale — not because VIX spiked (that's one scale), but because the entire multiscale structure of the signal changed.

---

## EEG Seizure Detection

CHB-MIT Scalp EEG Database, patient chb01, recording chb01_03. One hour at 256 Hz, 23 channels collapsed to RMS.

### Get the data

```bash
# Download from PhysioNet
# https://physionet.org/content/chbmit/1.0.0/
# Then preprocess:
python notes/eeg_chbmit/edf_to_helix_signal.py
```

### CLI

```bash
sf surface notes/eeg_chbmit/chb01_03_eeg_rms.csv --max-window 360 -hm
```

### Python

```python
import signalforge as sf
from signalforge.domains import eeg

records = eeg.ingest("notes/eeg_chbmit/chb01_03_eeg_rms.csv")

# With Hilbert — amplitude and phase structure
surfaces = (
    sf.load(records)
    .measure(windows=[256, 1024, 4096, 15360])
    .hilbert()
    .surfaces()
)
# surfaces[0].data includes 'amplitude', 'phase', 'inst_freq'
```

**What it shows:** A clinical epileptic seizure detected at **13.96σ** — nearly 14 standard deviations from the scale baseline. No training data, no labels, no EEG-specific code. The seizure window (2996-3036 seconds) appears as a bright band in the heatmap, visible across all scales simultaneously.

---

## INTERMAGNET Geomagnetic Data

Yellowknife observatory, minute-resolution magnetic field measurements.

### Get the data

```bash
# Download from INTERMAGNET
# https://intermagnet.org/data_download.html
# Convert IAGA2002 format:
python examples/iaga2002_to_csv.py input.iaga output.csv
```

### CLI

```bash
sf load yellowknife.csv
sf surface yellowknife.csv -hm --max-window 1440
```

### Python

```python
from signalforge.domains import intermagnet

records = intermagnet.ingest("yellowknife.csv")
plan = intermagnet.sampling_plan()
```

**What it shows:** Geomagnetic storms produce the same kind of multiscale signature as EEG seizures — a ridge in scale space at the storm's characteristic duration. The same pipeline, the same analysis, different domain.

---

## Generic Time Series

Any two-column CSV (date/index + value) works with no configuration.

```bash
sf load my_data.csv
sf surface my_data.csv -hm
```

SignalForge auto-detects the format: first column as index, second as value. Grain is estimated from the data. No domain adapter needed.

---

## Bringing Your Own Data

SignalForge accepts any ordered sequence. The minimum requirement: a list of `CanonicalRecord` objects.

```python
from signalforge.signal import CanonicalRecord, OrderType

records = [
    CanonicalRecord(
        primary_order=i,
        order_type=OrderType.SEQUENCE,
        channel="my_channel",
        metric="value",
        value=float(measurement),
        seq_order=i,
    )
    for i, measurement in enumerate(my_data)
]

import signalforge as sf
surfaces = sf.load(records).measure().surfaces()
```

Or go directly from arrays:

```python
from signalforge.signal import RealSignal
import numpy as np

sig = RealSignal(
    index=np.arange(len(my_data)),
    values=np.array(my_data, dtype=float),
    channel="my_channel",
)
surfaces = sf.from_signal(sig).measure(windows=[10, 30, 90]).surfaces()
```

For multi-column CSVs with known formats (equities, intermagnet, EEG), see the domain modules in `signalforge/domains/`.
