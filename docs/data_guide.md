# Data Guide — Running the Examples and Getting More Data

---

## Running the demos

### EEG seizure detection

```bash
uv run signalforge demo
```

Runs the full pipeline on CHB-MIT patient chb01, recording chb01_03. Prints
z-scores per scale across the annotated seizure window. Expects the preprocessed
CSV at `examples/eeg_chbmit/chb01_03_eeg_rms.csv` (see download instructions below).

To run the full pipeline script directly with optional CSV export:

```bash
uv run python examples/run_eeg.py
uv run python examples/run_eeg.py --out runs/eeg
```

### Geomagnetic observatory

```bash
uv run python examples/run_intermagnet.py --csv path/to/ykc_minute.csv
uv run python examples/run_intermagnet.py --csv path/to/ykc_year.csv --grain 86400
```

Pass `--grain 86400` for year-scale data to use daily bins instead of minute bins.

---

## Downloading more data

### CHB-MIT Scalp EEG (PhysioNet)

The full database contains 24 patients, 983 recordings, 198 annotated seizures.
The demo result (13.96σ seizure detection on chb01_03) used no labels, no
training, and no EEG-specific processing. Try it on any recording and see what
you find.

**Navigating the PhysioNet page** (it's not obvious):

1. Go to https://physionet.org/content/chbmit/1.0.0/
2. Scroll down to the **Files** section
3. You'll see folders named `chb01` through `chb24` — one per patient
4. Each folder contains `.edf` recordings and a `chbXX-summary.txt` that lists
   which recordings contain seizures and at what timestamps
5. Start with `chb01/chb01_03.edf` — that's the demo recording, seizure at ~2996s

Good recordings to try (all have annotated seizures):

| File | Patient | Seizure window |
|------|---------|----------------|
| `chb01/chb01_03.edf` | chb01 | 2996–3036 s |
| `chb01/chb01_04.edf` | chb01 | 1467–1494 s |
| `chb02/chb02_16.edf` | chb02 | 130–212 s |
| `chb03/chb03_01.edf` | chb03 | 362–414 s |

You do not need to know where the seizure is — that's the point. Run SignalForge
on any recording and look for the anomaly score spike.

Or use the PhysioNet downloader:

```bash
pip install wfdb

# Single recording (~50 MB)
python -c "
import wfdb
wfdb.dl_database('chbmit', './data/chbmit', records=['chb01/chb01_03'])
"

# Full corpus (24 patients, ~50 GB) — omit the records argument
```

Each recording is an EDF file. Convert it to the CSV format SignalForge expects:

```bash
uv add mne
uv run python examples/edf_to_rms_csv.py data/chbmit/chb01/chb01_03.edf
# outputs: chb01_03_eeg_rms.csv
```

Then run the pipeline:

```bash
uv run python examples/run_eeg.py --csv chb01_03_eeg_rms.csv
```

### INTERMAGNET geomagnetic data

Minute-cadence observatory data, freely available for research.

1. Go to https://intermagnet.org/data_download.html
2. Select observatory: **YKC** (Yellowknife) for the reference dataset
3. Select format: **IAGA-2002** (text, minute data)
4. Download any date range

The file arrives as a `.min` text file. Use the preprocessing script below
to convert it to CSV.

---

## Preprocessing scripts

These scripts convert raw source files into the CSV format the pipeline expects.
They live in `examples/`.

### EDF → CSV (EEG)

```bash
uv run python examples/run_eeg.py --help
```

The conversion step (EDF → RMS CSV) requires `pyedflib`:

```bash
uv add pyedflib
uv run python examples/edf_to_rms_csv.py data/chbmit/chb01/chb01_03.edf
# outputs: chb01_03_eeg_rms.csv  (columns: t_sec, eeg_rms)
```

### IAGA-2002 → CSV (INTERMAGNET)

```bash
uv run python examples/iaga2002_to_csv.py ykc20250101vmin.min
# outputs: ykc_minute.csv  (columns: timestamp, H, D, Z, F)
```

---

## Using AI to write a preprocessing script for your data

If you have a new data source, you can prompt any capable LLM to write the
preprocessing script. Give it:

1. **A sample of your raw data** — first 20–50 rows, header included
2. **The CanonicalRecord schema** — copied below
3. **Your domain's ordering** — time-based or sequence-based

### The CanonicalRecord schema

```python
from signalforge.pipeline.canonical import CanonicalRecord, OrderType

# One record per observation. Fields:
#
#   primary_order : int       — monotonically non-decreasing integer that drives binning
#                               use epoch seconds for time-ordered data
#                               use sample index for sequence-ordered data
#   order_type    : OrderType — TIME, SEQUENCE, or BOTH
#   channel       : str       — category of signal (e.g. "eeg", "temperature", "flux")
#   metric        : str       — what is being measured (e.g. "rms", "magnitude", "count")
#   value         : float     — the numeric observation
#   keys          : dict      — optional named dimensions (e.g. {"station": "YKC"})
#   time_order    : int       — epoch seconds (required if order_type is TIME or BOTH)
#   seq_order     : int       — sequence number (required if order_type is SEQUENCE or BOTH)

rec = CanonicalRecord(
    primary_order=1712000060,
    order_type=OrderType.TIME,
    channel="geomagnetic",
    metric="H",
    value=14832.5,
    keys={"station": "YKC"},
    time_order=1712000060,
)
```

### Prompt template

```
I have a CSV file with the following columns and sample rows:

[paste your header and ~20 rows here]

Write a Python function that reads this file with pandas and returns a list of
signalforge CanonicalRecord objects.

Use the following schema:

[paste the CanonicalRecord schema above]

Rules:
- primary_order must be a non-negative integer, monotonically non-decreasing
- For time-series data, use epoch seconds as primary_order and time_order
- One CanonicalRecord per measurement per row (if a row has multiple numeric
  columns, emit one record per column)
- value must be float; skip rows where the value is missing or non-numeric
- channel should reflect the signal category, metric the measurement name
- keys should capture any entity dimensions (station, patient, instrument)

Show the complete function and any imports needed.
```

Paste the output into `examples/ingest_yourdata.py`, run it, and feed the
resulting records into the pipeline.

---

## What the pipeline expects

After preprocessing, data flows through:

```
list[CanonicalRecord]  →  materialize()  →  measure()  →  engineer()  →  assemble()
```

The only requirement on the records list: `primary_order` must be non-negative
and monotonically non-decreasing. Everything else is flexible.
