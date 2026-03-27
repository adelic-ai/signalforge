# SignalForge

Multiscale signal processing pipeline built on the p-adic divisibility lattice.
Give it any ordered sequence — clinical EEG, geomagnetic field measurements,
network traffic, log events — and it produces a multiscale surface of
structurally invariant features, ready for anomaly detection or ML.

No labels. No domain-specific processing. No normalization between recordings.

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

## What it does

SignalForge detected a clinical epileptic seizure at **13.96σ** on CHB-MIT
EEG data — with no training data, no labels, and no EEG-specific code. The
same pipeline, unchanged, processes a full year of INTERMAGNET geomagnetic
observatory data.

The signal types have nothing in common. The pipeline does not know that.
It operates on the arithmetic structure of the data's sequential organization,
not on its physical content.

Full results and reproducibility instructions: [docs/empirical_results.md](docs/empirical_results.md)

---

## How it works

### The sampling domain

A standard multi-scale analysis requires the analyst to choose window sizes.
SignalForge does not. Declare a `horizon` (total length) and a `grain`
(smallest meaningful unit), and the valid windows are determined by
arithmetic: they are exactly the divisors of `H = horizon / grain`.

Divisors are the only window sizes that partition the signal without remainder
and nest into each other without overlap. They form a lattice — one independent
scale axis per prime dividing `H` — and computation flows up the lattice in a
single pass, without re-reading the raw data at each scale.

[docs/sampling_domain.md](docs/sampling_domain.md) — [arXiv preprint — forthcoming]

### Structural invariance

Because the measurement space is determined by `H`, two recordings that share a
SamplingPlan share an identical lattice. Features at scale `d` measure the same
structural property in both — no alignment, no cross-recording normalization
required. Surfaces from different recordings, patients, instruments, and sites
stack directly into an ML-ready bundle.

[docs/structural_invariance.md](docs/structural_invariance.md)

### The pipeline

| Stage | Name | Input → Output |
|------:|------|----------------|
| 0 | Ingest | Raw source data → CanonicalRecord |
| 1 | Plan | Declarations → SamplingPlan |
| 2 | Materialize | CanonicalRecords → BinnedRecord |
| 3 | Measure | BinnedRecords → Surface (time × scale grid) |
| 4 | Engineer | Surface → FeatureTensor |
| 5 | Assemble | FeatureTensors → FeatureBundle (ML-ready) |

Domain knowledge lives exclusively in `signalforge/domains/`. Stages 1–5 are
identical across all data sources. [docs/overview.md](docs/overview.md)

---

## Bring your own data

SignalForge ships with two domains: `eeg` (clinical EEG) and `intermagnet`
(geomagnetic observatory). To add your own, write a single Python file in
`signalforge/domains/`.

Declare a `horizon` and a `grain`, let the lattice enumerate valid windows,
select the ones meaningful for your data, return a `SamplingPlan`. Everything
downstream runs without modification.

- [docs/domain_guide.md](docs/domain_guide.md) — how to write a domain
- [docs/data_guide.md](docs/data_guide.md) — downloading data, running examples
- [CONTRIBUTING.md](CONTRIBUTING.md) — submitting a domain or pipeline change

---

## How it compares to wavelets and STFT

Wavelet analysis uses a dyadic scale hierarchy — windows at powers of 2. This
is one chain along one prime axis. STFT uses a single fixed window chosen by
the analyst. SignalForge uses all prime axes simultaneously, derived from the
horizon rather than chosen. The result is a strictly larger, more complete
measurement space with guaranteed cross-recording comparability.

[docs/comparison.md](docs/comparison.md)

---

## Contents

- [docs/empirical_results.md](docs/empirical_results.md) — EEG and geomagnetic results, reproducibility
- [docs/sampling_domain.md](docs/sampling_domain.md) — why divisors, the lattice, computational efficiency
- [docs/structural_invariance.md](docs/structural_invariance.md) — why features are directly comparable
- [docs/comparison.md](docs/comparison.md) — STFT, wavelets, EMD, and this
- [docs/overview.md](docs/overview.md) — pipeline detail, the mathematics, the name
- [docs/domain_guide.md](docs/domain_guide.md) — writing a domain
- [docs/data_guide.md](docs/data_guide.md) — data access and preprocessing
- [docs/operators_guide.md](docs/operators_guide.md) — pipeline operators

---

## License

Business Source License 1.1. See [LICENSE](LICENSE).

Free for non-commercial use (research, personal projects, evaluation).
Commercial use requires a license from Adelic — contact [shun.honda@adelic.org](mailto:shun.honda@adelic.org).

Converts to Apache 2.0 on 2029-03-22.

---

## Citation

If you use SignalForge in published work, a citation entry will be available
once the arXiv preprint is posted.

If you use the CHB-MIT EEG data, please cite:

> Shoeb AH, Guttag JV. Application of machine learning to epileptic seizure
> detection. In: *Proceedings of the 27th International Conference on Machine
> Learning (ICML)*. 2010.

Raw data: CHB-MIT Scalp EEG Database v1.0.0
https://physionet.org/content/chbmit/1.0.0/
