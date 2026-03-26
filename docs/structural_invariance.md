# Structural Invariance

A SignalForge surface computed from one recording is directly comparable to a
surface computed from another — across patients, sessions, instruments, and
sites — without normalization, alignment, or feature engineering. This property
is called **structural invariance**, and it follows from the lattice.

---

## What invariance means

A feature `f(x, d)` is structurally invariant if it depends only on the local
statistics of signal `x` within windows of size `d` — not on when in the
recording those windows occur, not on the absolute scale of the data, and not
on anything outside `x` itself.

When two signals share the same `horizon` and `grain`, they have the same `H`,
and therefore the same `Div(H)` — the same set of window sizes in the same
lattice arrangement. The feature at scale `d` measures the same structural
property in both signals: the same window size, the same position in the
scale hierarchy, the same relationship to every other scale.

There is nothing to align. The lattice is the shared coordinate system.

---

## Why normalization is not needed

Standard multi-scale methods produce features whose meaning depends on choices
made by the analyst: which window sizes, which basis functions, which
normalization scheme. Comparing features across recordings requires those
choices to be identical and requires additional alignment steps to account for
differences in length, sampling rate, or amplitude range.

SignalForge features are indexed by lattice position, not by analyst choice.
Two surfaces at scale `d` are measuring the same thing because `d` has the same
arithmetic meaning in both lattices — the same divisibility relationships, the
same nesting structure, the same distance from grain and horizon.

Amplitude differences between recordings are handled by the z-score
construction in the Engineer stage, which normalizes each scale row relative
to its own baseline and variability. This is within-surface normalization — it
does not require cross-recording calibration.

---

## Practical consequence

Surfaces from different recordings with the same SamplingPlan stack directly
into a feature bundle:

```python
bundle = sf.assemble([surface_1, surface_2, surface_3, ...])
```

The result is an ML-ready tensor. No preprocessing. No alignment. The lattice
guarantees the columns mean the same thing in every row.

This is what makes unsupervised anomaly detection tractable: you can compare
any window in any recording to any other window in any other recording on the
same scale, and the comparison is geometrically meaningful.

---

## Across different domains

The EEG and geomagnetic pipelines share no domain-specific code after Stage 0.
Their SamplingPlans have different horizons and grains — different `H` values —
so their lattices are not identical. But within each domain, every recording
that shares a plan is directly comparable to every other.

When the cybersecurity domain is added, the same will hold: any two network
traffic captures processed with the same plan will produce directly comparable
surfaces, with no feature engineering required to make them compatible.

---

*Full treatment with proofs: [arXiv preprint — forthcoming]*
