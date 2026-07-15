# The P-Adic Divisibility Lattice as a Signal Sampling Domain

## Abstract

Multi-scale signal analysis requires a family of window sizes. Standard approaches either fix a single window (STFT), restrict to dyadic scales (wavelet decomposition), or choose window sizes arbitrarily. We show that for any time-ordered discrete signal, the natural, complete, and computationally optimal window family is Div(H), the set of divisors of a horizon H derived from the desired windows and grain, ordered by divisibility. This family is the unique maximal set of window sizes that is simultaneously artifact-free, perfectly nested, and closed under finest-common-refinement. Two contributions are separable. First, the lattice construction and DAG aggregation require only divisibility: Div(H) is the unique artifact-free, consistently nested, gcd-closed window family, and features at all scales can be computed in O(max(W)·τ(H)) operations, where τ(H) is the number of divisors of H and satisfies τ(H) = O(Hε) for every ε > 0. Second, by the Fundamental Theorem of Arithmetic, Div(H) is isomorphic to a product of finite chains, one per prime dividing H; the p-adic valuation vector of each window size gives it a canonical coordinate, transforming Div(H) from a computational structure into a geometric measurement space. Features indexed by these coordinates are structurally invariant across signals sharing the same horizon. The dyadic wavelet decomposition is the special case restricted to the prime-2 axis. The framework is implemented in SignalForge, available at https://github.com/adelic-ai/signalforge.

---

## 1. Setup

Let a **discrete signal** be a finite sequence $x : \{0, 1, \ldots, N-1\} \to \mathbb{R}$.

Two parameters are declared:

- **grain** $g$: the smallest time unit carrying meaningful structure (e.g., 1 second). Determines the resolution floor.
- **horizon** $H$: the total signal length in units of grain. By construction, $H \in \mathbb{Z}_{>0}$.

The signal has $H$ positions. Everything below concerns the integer $H$.

---

## 2. The P-Adic Valuation and the Divisibility Lattice

For a prime $p$, the **p-adic valuation** $v_p(n)$ is the largest $k$ such that $p^k \mid n$:

$$v_p(n) = \max\{k \in \mathbb{N} : p^k \mid n\}$$

By the Fundamental Theorem of Arithmetic, every $n \in \mathbb{Z}_{>0}$ has a unique representation:

$$n = \prod_{p \text{ prime}} p^{v_p(n)}$$

with $v_p(n) = 0$ for all but finitely many primes. This gives $n$ a coordinate vector:

$$\mathbf{e}(n) = (v_2(n),\, v_3(n),\, v_5(n),\, \ldots) \in \mathbb{N}^{(\infty)}$$

(finite support). Multiplication becomes vector addition: $\mathbf{e}(mn) = \mathbf{e}(m) + \mathbf{e}(n)$.

The **divisibility lattice** of $H$ is:

$$\mathrm{Div}(H) = \{d \in \mathbb{Z}_{>0} : d \mid H\}$$

with partial order $d_1 \leq d_2 \iff d_1 \mid d_2$, meet $d_1 \wedge d_2 = \gcd(d_1, d_2)$, join $d_1 \vee d_2 = \mathrm{lcm}(d_1, d_2)$.

**Proposition 2.1.** *The map $d \mapsto \mathbf{e}(d)$ gives a lattice isomorphism:*

$$\mathrm{Div}(H) \cong \prod_{p \mid H} \{0, 1, \ldots, v_p(H)\}$$

*Proof.* The coordinate vector of $d \mid H$ satisfies $0 \leq v_p(d) \leq v_p(H)$ for all primes $p$. Divisibility corresponds coordinatewise to $\leq$. The isomorphism follows from unique factorization. $\square$

So $\mathrm{Div}(H)$ is a product of finite chains, one per prime dividing $H$. Each prime $p$ contributes an independent scale axis of depth $v_p(H)$.

---

## 3. Why Divisors Are the Natural Window Family

**Definition 3.1.** A window of size $w$ **tiles** $H$ if $H / w \in \mathbb{Z}_{>0}$, i.e., $w \mid H$.

**Proposition 3.2** *(Completeness).* The set of window sizes that partition $\{0, \ldots, H-1\}$ into $H/w$ contiguous, equal, non-overlapping bins is exactly $\mathrm{Div}(H)$.

No window size outside $\mathrm{Div}(H)$ partitions the signal without a remainder fragment. Any analysis using $w \nmid H$ either truncates the signal or introduces a boundary artifact. The divisors are not a convenience — they are the complete set of artifact-free window sizes for a signal of length $H$.

**Proposition 3.3** *(Nesting).* If $d_1 \mid d_2 \mid H$, then every $d_2$-window is exactly partitioned by $d_2 / d_1$ contiguous $d_1$-windows.

*Proof.* $d_1 \mid d_2$ implies $d_2 / d_1 \in \mathbb{Z}_{>0}$. The $d_2$-window $[kd_2, (k+1)d_2)$ is partitioned by $\{[kd_2 + jd_1,\, kd_2 + (j+1)d_1)\}_{j=0}^{d_2/d_1 - 1}$. $\square$

The lattice order on $\mathrm{Div}(H)$ directly encodes the nesting structure of windows in the signal. Moving up in the lattice aggregates; moving down refines. There are no partial overlaps anywhere in the hierarchy.

**Corollary 3.4.** For any $d_1, d_2 \in \mathrm{Div}(H)$, the windows at scale $\gcd(d_1, d_2)$ nest into both $d_1$-windows and $d_2$-windows without remainder. The meet in the lattice is the finest common refinement.

---

## 4. The Coordinate Representation as a Measurement Space

By Proposition 2.1, each $d \in \mathrm{Div}(H)$ corresponds to a coordinate $(e_2, e_3, e_5, \ldots)$ where $e_p = v_p(d)$.

- Moving one step along the **prime-2 axis** (incrementing $e_2$ by 1) doubles the window size.
- Moving one step along the **prime-3 axis** triples it.
- Moving along both simultaneously gives a window size $2 \cdot 3 = 6$ times the base.

Each prime is an independent scale dimension. The full lattice $\mathrm{Div}(H)$ exhausts all combinations of scales up to $H$.

A feature vector indexed by $\mathrm{Div}(H)$ lives in a space whose axes are labeled by independent prime-scale combinations. There is no basis ambiguity and no free parameter for how to arrange the scales. The structure of $H$ determines the measurement space uniquely.

---

## 5. Structural Invariance

**Definition 5.1.** A feature $f(x, d)$ computed from signal $x$ at scale $d$ is **structurally invariant** if it depends only on the local statistics of $x$ within windows of size $d$, not on the absolute position of those windows along the time axis or on any quantity external to $x$.

**Proposition 5.2.** If two signals $x_1$ and $x_2$ share horizon $H$ and grain $g$, and $f$ is structurally invariant, then $f(x_1, d)$ and $f(x_2, d)$ are directly comparable for every $d \in \mathrm{Div}(H)$.

The comparison is meaningful — not merely numerically possible. Because $d$ has the same lattice position in both signals (same relationship to $H$, same nesting structure, same scale relative to grain), the feature at scale $d$ measures the same structural property in both. No normalization is required; the shared lattice structure guarantees it.

**Remark.** This extends across different instruments or sessions recording the same process, provided the same grain and horizon are declared. The lattice is determined by $H = \text{horizon} / \text{grain}$, not by physical units. Two EEG sessions with the same sampling rate and recording length have identical lattices and directly comparable feature tensors.

---

## 6. Contrast with Standard Multi-Resolution Methods

**STFT.** Uses a fixed window $W$ slid along the signal. The window size is a free parameter chosen by the analyst. There is no mathematical relationship between measurements at different window sizes and no lattice structure organizing the scales.

**Dyadic wavelet decomposition.** Uses windows at scales $\{W, W/2, W/4, \ldots\}$ — a chain along the prime-2 axis only. This is the sub-lattice of $\mathrm{Div}(H)$ generated by powers of 2. It captures 2-adic structure but ignores all other prime axes. The choice of 2 is a convention, not a mathematical necessity.

**The divisibility lattice.** Uses all primes dividing $H$ simultaneously. The scale axes are determined by the horizon, not chosen by the analyst. The lattice is complete (all artifact-free window sizes), consistent (perfect nesting at all levels), and coordinate-invariant (the measurement space is fixed by $H$).

---

## 7. Optimality and Computational Efficiency

The nesting property is not only a structural nicety — it is the basis for efficient computation across all scales simultaneously.

**The aggregation principle.** Because every $d_1$-window tiles every $d_2$-window exactly when $d_1 \mid d_2$, a statistic at a coarser scale can be computed by aggregating the already-computed values at a finer scale. No re-reading of the raw signal is required. The Hasse diagram of $\mathrm{Div}(H)$ is a directed acyclic graph (DAG); computation flows bottom-up from grain-level windows to the horizon.

Concretely: to compute a feature at every scale in $\mathrm{Div}(H)$:

1. Make one pass over the raw signal at grain resolution: $O(H)$ operations.
2. For each covering relation $d_1 \lessdot d_2$ in the Hasse diagram, aggregate $p = d_2/d_1$ consecutive values: $O(H/d_2)$ operations per edge.

The total work is $O(H \cdot \tau(H))$ where $\tau(H)$ is the number of divisors of $H$. Since $\tau(H) = O(H^\varepsilon)$ for any $\varepsilon > 0$, this is nearly linear in $H$ regardless of how many scales are computed.

**Why no other window family achieves this.** The aggregation DAG exists because $\mathrm{Div}(H)$ is closed under $\gcd$: for any $d_1, d_2 \in \mathrm{Div}(H)$, $\gcd(d_1, d_2) \in \mathrm{Div}(H)$. This means any two windows always share a finest common sub-window that is itself in the family. An arbitrary set of window sizes does not have this property. If $w_1, w_2$ are chosen ad hoc and $\gcd(w_1, w_2) \notin \{w_1, w_2\}$, there is no shared finer scale, and each window size requires an independent $O(H)$ pass over the raw data.

**Optimality statement.** $\mathrm{Div}(H)$ is the unique maximal family $\mathcal{F} \subseteq \mathbb{Z}_{>0}$ satisfying all three of:

1. Every $w \in \mathcal{F}$ tiles $H$ (artifact-free partition).
2. $\mathcal{F}$ is closed under $\gcd$ (finest common refinement always in family).
3. $1 \in \mathcal{F}$ and $H \in \mathcal{F}$ (grain and horizon are included).

*Proof.* Any $w$ satisfying (1) must divide $H$, so $\mathcal{F} \subseteq \mathrm{Div}(H)$. Conditions (2) and (3) are satisfied by $\mathrm{Div}(H)$ and by any subset closed under $\gcd$ containing 1 and $H$. The maximal such family is $\mathrm{Div}(H)$ itself. $\square$

Choosing any strict subset of $\mathrm{Div}(H)$ loses scales without computational benefit. Choosing any window sizes outside $\mathrm{Div}(H)$ introduces boundary artifacts and breaks the aggregation structure. The divisibility lattice is the unique family for which multi-scale computation is both artifact-free and maximally efficient.

---

## 8. The Core Claim, Precisely

Given a signal of length $H$ bins (bin size = grain):

1. The valid window sizes are exactly $\mathrm{Div}(H)$.
2. $\mathrm{Div}(H)$ has the structure of a product lattice, one chain per prime dividing $H$.
3. Features indexed by $\mathrm{Div}(H)$ live in a coordinate space determined entirely by $H$.
4. Any two signals with the same $H$ have the same measurement space, and features at each scale are directly comparable without normalization.

The p-adic valuation lattice is not a method applied to signals. It is the natural structure of the set of artifact-free, consistently nested window sizes for a discrete signal of length $H$. SignalForge makes that structure explicit and uses it as the measurement domain.
