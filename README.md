# SignalForge

Multiscale signal processing pipeline built on the p-adic divisibility lattice. Transforms raw sequential data into structurally invariant feature tensors.

---

## What counts as a signal

SignalForge works on any ordered sequence. Time is not required.

**No timestamps**: If your data is a sequence of events with no time component — log entries, transactions, genomic positions, ranked items — each event is already one bin. Feed it directly. Set `grain=1`.

**Time-ordered data**: The standard path bins by the smallest meaningful time unit (`grain`). A second view is also available: collapsing the time axis entirely, treating each event as the next position in an ordered sequence regardless of when it occurred. This can reveal structure that time-binning distributes across windows. Both views are valid and often complementary.

Within a time-ordered sequence, a finer ordering axis may exist — a sequence number, monotonic counter, or sub-second timestamp — that places events more precisely. Whether that resolution is meaningful is a domain judgment. `grain` is where that decision is encoded.

---

## Install

```bash
git clone https://github.com/adelic-ai/signalforge
cd signalforge
uv sync
```

---

## Quick start

```bash
uv run signalforge demo                        # EEG seizure detection demo
uv run signalforge neighborhood 36 6           # p-adic arithmetic viewing box
uv run signalforge plan eeg                    # sampling plan for EEG
uv run signalforge run eeg path/to/data.csv    # run on your data
```

---

## Bring your own data

SignalForge ships with two domains: `intermagnet` (geomagnetic observatory) and `eeg` (clinical EEG). To add your own, write a single Python file in `signalforge/domains/`.

- **[docs/overview.md](docs/overview.md)** — what it is, the mathematics, the name
- **[docs/domain_guide.md](docs/domain_guide.md)** — how to write a domain
- **[docs/data_guide.md](docs/data_guide.md)** — downloading data, running examples, AI-assisted preprocessing
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — submitting a domain or pipeline change

The short version: declare a `horizon` and a `grain`, let the lattice enumerate valid windows, select the ones meaningful for your data, return a `SamplingPlan`. Everything downstream runs without modification.

---

## Pipeline

| Stage | Name | Input → Output |
|------:|------|----------------|
| 0 | Ingest | Raw source data → CanonicalRecord |
| 1 | Plan | Declarations → SamplingPlan |
| 2 | Materialize | CanonicalRecords → BinnedRecord |
| 3 | Measure | BinnedRecords → Surface (time × scale grid) |
| 4 | Engineer | Surface → FeatureTensor |
| 5 | Assemble | FeatureTensors → FeatureBundle (ML-ready) |

Domain knowledge lives exclusively in `signalforge/domains/`. Stages 1–5 are identical across all data sources.

---

## Result: EEG seizure detection

**Dataset**: CHB-MIT Scalp EEG, patient chb01, recording chb01\_03.
**Input**: RMS across 23 channels, 1-second bins. No labels, no seizure times, no EEG-specific processing.

Peak z-score: **13.96σ at the 30-second scale**, within the annotated seizure window (seconds 2996–3036).

Full result: [docs/discovery_eeg_seizure_detection.md](docs/discovery_eeg_seizure_detection.md)

---

## License

Business Source License 1.1. See [LICENSE](LICENSE).

Free for non-commercial use (research, personal projects, evaluation). Commercial use requires a license from Adelic — contact [shun.honda@adelic.org](mailto:shun.honda@adelic.org).

Converts to Apache 2.0 on 2029-03-22.

---

## Citation

If you use the CHB-MIT EEG data, please cite:

> Shoeb AH, Guttag JV. Application of machine learning to epileptic seizure detection. In: *Proceedings of the 27th International Conference on Machine Learning (ICML)*. 2010.

Raw data: CHB-MIT Scalp EEG Database v1.0.0
https://physionet.org/content/chbmit/1.0.0/
