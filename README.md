# SignalForge

Multiscale signal processing pipeline built on the p-adic divisibility lattice.
Give it any ordered sequence — clinical EEG, geomagnetic field measurements,
network traffic, log events — and it produces a multiscale surface of
structurally invariant features, ready for anomaly detection or ML.

No labels. No domain-specific processing. No post-hoc normalization between recordings.

---

## Where this came from

Computing a signal at multiple window sizes used to mean running each window
separately — re-read the data, re-aggregate, re-score. Adding a window added
another full pass. Adding a feature multiplied the cost again. The pipeline was
doing something that looked like integration, badly.

The cleaner way is hiding in plain sight in any calculus course. Integration
starts with Riemann sums — partition the domain, sum the rectangles — and the
punchline is: take the limit as the partition gets infinitely fine and you get
the exact area under the curve. That is the triumphant conclusion of the whole
story. But for a discrete signal there is no limit to take. The grain *is* the
finest partition. Read the data once at grain resolution, aggregate upward, and
every coarser window is already determined by the finer values beneath it —
nothing to recompute. You get the same compositional structure that makes
integration work, without needing the limit that makes it exact in the
continuous case, because at the grain you are already exact.

The arithmetic that makes this exact is divisibility. A window tiles a sequence
cleanly if and only if it divides the sequence length. Windows nest without
remainder if and only if one divides the other. The set of valid windows for a
given grain and horizon is the divisor lattice of that horizon — a structure
from number theory that turns out to be exactly the right coordinate system for
multi-scale signal analysis.

That is what SignalForge is: multi-scale analysis built on the arithmetic that
was always implicit in windowed computation, made explicit and used directly.

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

## Demonstrated capabilities

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
SignalForge does not. Declare the windows you want and your grain — the finest
resolution unit — and the measurement space is derived in two stages:
unitization embeds your quantities into a common integer domain; normalization
completes that set to the divisibility lattice via the horizon:

```python
plan = SamplingPlan.from_windows([7, 30, 90, 360], grain=g)
```

The horizon is derived as `lcm(windows + [grain])`, guaranteeing the grain is
an exact lattice member with no snapping. Divisors are the only window sizes
that partition the sequence without remainder and nest into each other exactly
— computation flows up the lattice in a single pass, without re-reading the
raw data at each scale.

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

Specify the windows meaningful for your data and a grain — either declared
directly when the cadence is known, or estimated from the data via
[`grain_from_orders`](docs/binjamin.md). Pass both to `from_windows` and the
lattice is determined. Everything downstream runs without modification.

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
- [docs/scale_space.md](docs/scale_space.md) — the surface, gradients, anomaly geometry, and where this is going
- [docs/comparison.md](docs/comparison.md) — STFT, wavelets, EMD, and this
- [docs/overview.md](docs/overview.md) — pipeline detail, the mathematics, the name
- [docs/domain_guide.md](docs/domain_guide.md) — writing a domain
- [docs/binjamin.md](docs/binjamin.md) — automatic grain selection
- [docs/design_grain_snapping.md](docs/design_grain_snapping.md) — grain, horizon as derived value, from_windows design
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
