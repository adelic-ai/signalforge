# The Surface and What Lives on It

## What the pipeline produces

After the pipeline runs, you have a surface: a two-dimensional grid with time
on one axis and scale on the other. Each cell is a feature — typically a
z-score — computed over a window of that size at that moment in the sequence.

This is not a heatmap for visualization. It is a coordinate space. Every point
on it has an address: a position in time, and a position in the divisibility
lattice expressed as a p-adic valuation vector `(v₂, v₃, v₅, ...)`. Moving
one step along the scale axis does not add a fixed amount — it multiplies the
window by a prime. The scale axis is logarithmic by construction.

---

## Anomalies are shapes, not values

The z-score at a single point on the surface is not the anomaly. It is one
measurement of it. The anomaly is the pattern of change across the surface.

Consider what happens to a real anomaly — a seizure, a geomagnetic storm, a
network intrusion — as you vary the window size while holding time fixed. At
very fine scales, the window is too small: it captures noise as readily as
signal. At the scale that matches the anomaly's characteristic duration, the
signal peaks. At coarser scales, the anomaly gets averaged out over a larger
window and dilutes. The result is a triangle profile along the scale axis,
with a peak at the scale that fits.

Extend that peak across time and you get a ridge — a connected structure that
traces the anomaly's duration at its natural scale. The ridge has a location
in scale space. It has a width in time. It has edges where the signal rises
and falls.

An anomaly is a geometric structure on the surface, not a threshold crossing
at a point.

---

## Every anomaly has a scale signature

An anomaly is triangulated by three coordinates in scale space:

- **Detection scale**: the coarsest scale at which the anomaly first becomes
  visible. Large windows average out noise — anomalies tend to emerge here
  first.
- **Resolution scale**: the scale where the signal peaks — the window size
  best matched to the anomaly's actual duration.
- **Support scale**: the finest scale at which the signal remains coherent.
  Below this the anomaly dissolves into noise.

These three points bracket the anomaly from above, at peak, and from below.
Together they triangulate its extent in scale space, the same way three
measurements triangulate a position in physical space. The result is the
anomaly's **scale signature**.

Because the coordinates are p-adic valuation vectors — determined by
arithmetic, not analyst choice — the scale signature of the same type of
anomaly is consistent across different recordings, patients, instruments,
and sites processed with the same SamplingPlan. The coordinate system is
shared. The triangulation means the same thing everywhere.

---

## Gradients and what they mean

A gradient on the surface describes how the feature changes as you move from
one cell to an adjacent one. There are two directions:

**Along time**: how fast is the anomaly building or decaying at a given scale.

**Along scale**: how sharply the feature changes as the window size changes.
This is the derivative of the triangle profile described above. A steep
gradient approaching the resolution scale from below means the anomaly is
tightly concentrated at that scale. A gentle gradient means it is spread
across scales — a slow drift rather than a sharp event.

Because the lattice has multiple paths between any two scales — you can go
from a 60-unit window to a 15-unit window via 30, or via 20 and 10 — the
accumulated gradient along different paths is a consistency check. Real
structure accumulates the same total change regardless of which path you take.
Noise does not. Path consistency is a built-in noise filter, free from the
geometry.

---

## Where SignalForge is going

The surface and its geometry suggest a natural next step: a traversal
algorithm that follows gradients through scale space to locate, characterize,
and validate anomalies. Rather than flagging threshold crossings at individual
scales, such an algorithm would:

- scan at coarse scales where signal-to-noise is highest
- descend the gradient toward finer scales to find the resolution peak
- validate the structure by checking path consistency
- stop at the support boundary where coherence is lost
- return the full scale signature as the detection result

This is the direction of Wellington — a detector that treats anomalies as
geometric objects and reports them as such, not as a list of z-scores.

---

## How this followed from the lattice

None of this was the original goal. The p-adic divisibility lattice was
introduced to solve an efficiency problem: avoid recomputing features
separately at each window size. The solution required a coordinate system
on the window family. That coordinate system turned out to be the p-adic
valuation — a structure from number theory that gives each scale a precise
address in a product of integer axes.

Once the coordinates existed, gradients existed. Once gradients existed,
geometry existed. The scale signature concept, the ridge structure, the
path consistency test, the Wellington traversal — these are the natural
conceptual neighbors of the arithmetic that was already there. They were
not designed; they followed.

This is not unusual in mathematics. A structure introduced for one reason
often turns out to carry more than intended. The p-adic lattice as a signal
sampling domain appears to be such a case, and we are still mapping the
territory.

---

*Rigorous treatment: [zeta/scale_space_geometry.md](../../zeta/scale_space_geometry.md)*
