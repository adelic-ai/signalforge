# Empirical Results

The central claim of SignalForge is that one pipeline, grounded in the
arithmetic structure of integer divisibility, processes fundamentally different
kinds of sequential data without modification. These results are the empirical
evidence for that claim.

---

## The controlled demonstration

Two domains. One pipeline. No shared code after Stage 0.

| Domain | Data source | Signal | Grain | Horizon |
|--------|-------------|--------|-------|---------|
| EEG | CHB-MIT Scalp EEG Database | RMS across 23 channels | 1 s (256 samples) | 3600 s |
| Geomagnetic | INTERMAGNET, Yellowknife YKC | Field component intensity | 1 min | 525,600 min (1 year) |

The SamplingPlans differ. The pipeline stages — Materialize, Measure, Engineer,
Assemble — are identical. Domain knowledge enters at Stage 0 and nowhere else.

---

## EEG: unsupervised seizure detection

**Dataset**: CHB-MIT Scalp EEG Database, patient chb01, recording chb01_03.
A one-hour recording from a pediatric epilepsy patient. Annotated seizure at
seconds 2996–3036 (approximately 50 minutes in).

**What the pipeline was given**: raw EEG samples. No seizure times, no labels,
no EEG-specific processing.

**What it found**: a peak z-score of **13.96σ at the 30-second scale**, inside
the annotated seizure window.

At 5σ the probability of a chance fluctuation in a Gaussian signal is roughly
1 in 3.5 million — the threshold used to announce the Higgs boson. At 13.96σ
the probability is of order 10⁻⁴³. The detection is not marginal.

The z-score profile peaks sharply between 6 and 30 seconds — the ictal
onset timescale — and falls off at both extremes. The multiscale surface
resolved the characteristic temporal extent of the seizure without being told
what to look for.

Full result and scale-by-scale z-scores: [discovery_eeg_seizure_detection.md](discovery_eeg_seizure_detection.md)

---

## Geomagnetic: full-year observatory data

**Dataset**: INTERMAGNET, Yellowknife YKC station, one full year of
geomagnetic field intensity at 1-minute resolution.

**What the pipeline was given**: the same pipeline used for EEG, with a
different SamplingPlan reflecting a one-year horizon at 1-minute grain.

**What it found**: structured multiscale surfaces across the full year,
with anomaly scores elevated during geomagnetic storm periods.

No modification to any pipeline stage. No geomagnetic-specific feature
engineering. The same arithmetic that organized EEG windows organized
a year of magnetic field observations.

---

## What these results establish

These two results together are a controlled demonstration of domain
agnosticism. The argument is not "it works on EEG" and separately "it works
on geomagnetic data." The argument is:

> The same code, without modification, produces structurally meaningful
> multi-scale anomaly scores on a clinical brain signal and a planetary
> magnetic field measurement — because the organizing structure is
> arithmetic, not domain-specific.

The methodology is described in [sampling_domain.md](sampling_domain.md)
and [structural_invariance.md](structural_invariance.md).

---

## Reproducibility

```bash
git clone https://github.com/adelic-ai/signalforge
cd signalforge
uv sync
uv run signalforge demo    # runs the EEG seizure detection demo
```

Full data download and run instructions: [data_guide.md](data_guide.md)

**Data sources**:
- CHB-MIT Scalp EEG Database v1.0.0 — https://physionet.org/content/chbmit/1.0.0/
- INTERMAGNET — https://intermagnet.org

*Shoeb AH, Guttag JV. Application of machine learning to epileptic seizure
detection. ICML 2010.*
