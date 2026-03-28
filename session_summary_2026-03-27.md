# Session Summary — 2026-03-27

---

## Docs updated in signalforge/dev branch

### README.md
- "What it does" → "Demonstrated capabilities"
- "Bring your own data" section updated from old horizon/grain API to
  from_windows workflow with grain_from_orders
- Added "Where this came from" section — the origin story: multi-window
  pipelines were doing integration badly; the divisibility lattice is the
  clean solution; calculus analogy with the limit story told correctly

### docs/sampling_domain.md
- Opening: "Declare a grain" → "Specify a grain", binjamin path added
- Added "The phantom horizon" section: H = lcm(windows + [grain]) may exceed
  max(windows); phantom horizon shapes arithmetic but is never computed;
  grain exact by construction, no snapping, no forced error
- Rewrote "Computational efficiency": origin story from helix (loop-per-window,
  full re-read per scale, O(H·k)), the discrete integration insight, DAG
  aggregation, resolution as a choice matched to detection task
- Both grain paths (declared and estimated) shown in SamplingPlan code examples

### docs/comparison.md
- No changes needed — confirmed accurate as-is, citations still pending

### docs/scale_space.md (NEW)
- Lay explanation of the surface, gradients, and anomaly geometry
- Triangle/ridge structure across scale
- Anomaly triangulation: (d_det, d_res, d_sup) as canonical scale signature
- Path consistency as built-in noise filter
- Where SignalForge is going: Wellington
- How the geometry followed uninvited from the efficiency decision
- Links to zeta/scale_space_geometry.md for rigorous treatment

---

## New zeta documents

### zeta/scale_space_geometry.md (NEW)
Rigorous treatment of discrete differential geometry on the p-adic lattice:

1. Background — lattice as coordinate space, not just optimization
2. Signal surface F(t, v₂, v₃, v₅, ...)
3. Discrete gradient — prime-axis finite differences, logarithmic scale axis
4. Path integrals — accumulation along refinement chains
5. Path consistency theorem — path independence from FTA/unique factorization
6. Second differences and curvature
7. Triangle and ridge structure — scale peaks, ridge definition
8. Anomaly as geometric object — scale signature triangulation
9. Dual resolution — latent lattice vs operational lattice, grain as GLB
10. Learned scale geometry / eigenspace:
    - Gradient covariance Σ = E[ggᵀ]
    - Eigendecomposition → principal directions of variation
    - Two coordinate systems: arithmetic prior (prime axes) vs data posterior (eigenvectors)
    - Residual score: ‖g − projection‖ as anomaly signal — catches geometrically
      novel behavior that z-score thresholding misses
    - Detection lighter and faster via low-dimensional eigenspace navigation
11. Correspondence table — full discrete DG mapping including eigenspace
12. Wellington — gradient-following traversal returning full scale signature
13. How this followed from efficiency

### zeta/progress_assessment_2026-03-26.md (UPDATED)
- Point 5 (novelty) substantially revised: FTA application is stronger and
  broader than initially framed — efficiency, invariance, path independence,
  canonical triangulation all follow from the same foundation
- Literature search elevated to priority 2 (was 3)
- Two-paper scope documented: Paper 1 (sampling domain), Paper 2 (scale-space
  geometry and detection theory)

### zeta/roadmap_q2_2026.md (NEW)
Three-month milestone plan toward ICS chapter meeting (late June 2026):
- Month 1: EEG benchmark notebook, surface visualization, SWaT/BATADAL eval
- Month 2: ICS domain module, ICS pipeline results, regime-switching note
- Month 3: Demo script, ICS pitch paragraph, citations in comparison.md
- Parallel track: theory docs at own pace, no meeting deadline

---

## Key ideas developed in conversation

**Phantom horizon**: H = lcm(windows + [grain]) may exceed max(windows).
This is intentional — it ensures grain | H exactly. The phantom horizon
shapes arithmetic, is never computed, and costs nothing.

**Calculus analogy (corrected)**: Riemann sums → take the limit → exact area,
triumphant conclusion of calculus. Our story: discrete signal has no limit to
take. Grain is the finest partition. At the grain you are already exact.

**Path independence from FTA**: Every path between two lattice points must
account for the same total valuation change per prime — there is no other
way to get from one unique factorization to another. Path independence is
not proved about the system; it is unique factorization, already there.

**Triangulation**: The scale signature (d_det, d_res, d_sup) triangulates an
anomaly's extent in scale space — bounded from above, located at peak, bounded
from below — in a coordinate system that is canonical across all recordings
sharing a SamplingPlan.

**Eigenspace as learned geometry**: PCA on the gradient field. Arithmetic
basis (prime axes) is the prior; eigenvectors are the data-driven posterior.
Residual from eigenspace projection is an anomaly score that catches
geometrically novel behavior — a different and complementary signal to z-score.

**Attractor intuition**: normal behavior has preferred directions in scale
space (the eigenvectors). Anomalies deviate from those directions. Loosely
analogous to attractors in dynamical systems.

**Two-signal detection**: z-score catches extreme values; residual score
catches geometrically novel directions of change. Together they cover different
failure modes.

---

## ICS meeting context

- Audience: mixed ICS cyber backgrounds (EE, networking, operations)
- Key objection to address: "boring and sparse then goes crazy" — three-state
  regime switching. Answer: within-surface normalization, each regime
  establishes its own baseline.
- Demo anchor: SWaT (Secure Water Treatment) dataset — labeled attacks against
  regime-changing normal operations
- The visual that lands: surface heatmap with anomaly ridge visible

---

## ChatGPT attribution
ChatGPT (OpenAI) contributed to the theoretical development in
jaco_scale_lattice_summary.md and scale_space_gradient_summary.md.
To be acknowledged in CONTRIBUTORS.md.

---

## Still to discuss (user's note)
- Prime lattice observatory / flipflop — user has been carrying this for
  decades, will return to it when ready
