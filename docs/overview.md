# Overview

## What SignalForge is

SignalForge transforms raw sequential data into structurally invariant feature
tensors. Give it a time series or any ordered sequence — and it produces a
multiscale surface: a grid where one axis is position and the other is scale,
with every meaningful measurement window computed simultaneously.

The output is always the same shape for a given plan. Two recordings from
different instruments, different patients, different sites, on different days —
if they share a SamplingPlan, their surfaces stack directly. No alignment. No
normalization. No feature engineering.

---

## The mathematical foundation

Every positive integer has a unique prime factorization. SignalForge treats that
factorization as a coordinate system — each prime is an axis, each exponent is a
coordinate. Under this embedding, divisibility becomes geometry: one window
divides another if and only if its coordinate vector is componentwise ≤.

This is the p-adic divisibility lattice. It determines which measurement windows
are valid, how they relate to each other, and why surfaces from different sources
are directly comparable. The geometry is not a metaphor — it is the load-bearing
structure of the pipeline.

A SamplingPlan is declared with two integers: `horizon` (the outer boundary of
the coordinate space) and `grain` (the finest meaningful unit). All valid
measurement windows must divide `horizon`. That constraint is what makes the
surfaces invariant.

The surface is not just a feature matrix. The framework can be viewed as a
finite analogue of Riemann integration: the grain defines the finest admissible
partition and no limiting process is required. The `horizon` is the upper
boundary of the measurement domain. The lattice determines the admissible
scales, not arbitrarily, but by the arithmetic structure of the prime
factorization.

---

## The name

The mathematical object that combines all p-adic perspectives simultaneously —
holding the structure of every prime axis at once — is called an adelic number.
That is precisely what a SignalForge surface computes. The company is Adelic.
The name was not chosen for branding. It is a description.

---

## The pipeline

| Stage | Name | What it does |
|------:|------|--------------|
| 0 | Ingest | Raw source data → CanonicalRecord |
| 1 | Plan | Declare geometry → SamplingPlan |
| 2 | Materialize | CanonicalRecords → BinnedRecord |
| 3 | Measure | BinnedRecords → Surface (time × scale) |
| 4 | Engineer | Surface → FeatureTensor |
| 5 | Assemble | FeatureTensors → FeatureBundle (ML-ready) |

Domain knowledge enters at Stage 0 and nowhere else. Stages 1–5 are identical
across all data sources.

---

## What it has done

On CHB-MIT Scalp EEG patient chb01, recording chb01_03, SignalForge detected
a clinical epileptic seizure at **13.96σ** — with no training data, no labels,
and no EEG-specific processing. The pipeline had never seen a brain signal. It
saw structure at the 30-second scale that could not be noise.

Full result: [discovery_eeg_seizure_detection.md](discovery_eeg_seizure_detection.md)

The same pipeline code, without modification, processes a full year of
geomagnetic observatory data from Yellowknife, Canada.

Data sources:
- [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) — PhysioNet
- [INTERMAGNET](https://intermagnet.org) — International Real-time Magnetic Observatory Network

---

## What it can do for your data

If your data has meaningful timescales — and most sequential data does — a
SamplingPlan can encode them. The domain guide walks through exactly how.

[domain_guide.md](domain_guide.md)
