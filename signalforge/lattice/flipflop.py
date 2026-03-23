"""
signalforge.lattice.flipflop

Seed-based navigation of p-adic valuation structure.

For a prime p, the p-adic valuation v_p(n) is the exponent of p in the
factorization of n. The sequence (v_p(1), v_p(2), ...) has a recursive
structure that can be navigated structurally — without division — by
starting from a minimal seed and applying a single prime-invariant
operation repeatedly.

The Seed
--------
For prime p, the seed (nut) is:

    _nut = (0, 0, ..., 0, 1)    # p-1 zeros followed by 1

The seed has length p and equals (v_p(1), v_p(2), ..., v_p(p)):
every integer from 1 to p-1 is not divisible by p (valuation 0),
and p itself has valuation 1.

    p=2: (0, 1)             # v_2(1), v_2(2)
    p=3: (0, 0, 1)          # v_3(1), v_3(2), v_3(3)
    p=5: (0, 0, 0, 0, 1)    # v_5(1..5)

The Flip Operation
------------------
One operation. Prime-invariant. Applied to itself repeatedly.

From a sequence s = (v_p(1), ..., v_p(p^k)), produce
s' = (v_p(1), ..., v_p(p^(k+1))):

    For n in 1..p^(k+1):
        s'[n-1] = s[n/p - 1] + 1   if p divides n
        s'[n-1] = 0                  otherwise

In words: upsample by p — place s[i]+1 at every p-th position, zeros
between. This follows directly from v_p(p*m) = v_p(m) + 1.

After k flips, the sequence has length p^(k+1) and equals
(v_p(1), v_p(2), ..., v_p(p^(k+1))).

    p=2, flip^0: (0, 1)
    p=2, flip^1: (0, 1, 0, 2)
    p=2, flip^2: (0, 1, 0, 2, 0, 1, 0, 3)

    p=3, flip^0: (0, 0, 1)
    p=3, flip^1: (0, 0, 1, 0, 0, 1, 0, 0, 2)
    p=3, flip^2: (0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 3)

Growth Rate
-----------
After k flips from seed p, the sequence covers p^(k+1) integers.
Larger primes grow faster — prime 97 after 5 flips covers 97^6 ≈ 8.3e11
integers. The number of flips to encompass integer n for prime p is
max(0, ceil(log_p(n)) - 1).

This inverts the usual computational intuition: large primes are cheaper
to navigate structurally because their seeds span wider intervals.

Mirror Symmetry (p = 2 only)
-----------------------------
For p=2 specifically, the sequence v_2(1..2^k) is a palindrome around the
central value 2^k / 2 = 2^(k-1). This is why the ruler sequence looks
"self-similar" visually. For p > 2, the sequence has recursive structure
but is not a palindrome.

The structure can be reconstructed from any known point in both directions.
There is no privileged origin. Enter at any integer, expand outward.

Original mathematical observation and implementation by Shun Richard Honda.
See: docs/discovery_flipflop_prime_structure.md
"""

from __future__ import annotations
import math
from typing import Tuple


class FlipFlop:
    """
    Seed-based navigator for the p-adic valuation sequence of a prime.

    Each instance represents one level of the valuation structure.
    Calling flip() returns a new instance one level deeper.
    The object is its own state — no mutation, no counters.

    Usage:
        f = FlipFlop(2)           # seed level: (0, 1)
        f = f.flip()              # level 1:    (0, 1, 0, 2)
        f = f.flip().flip()       # level 3:    (0, 1, 0, 2, 0, 1, 0, 3, ...)

        # Read valuation at position n (1-indexed):
        v = f.valuation_at(n)

        # Chain:
        f2 = FlipFlop(2).flip().flip().flip()
    """

    def __init__(self, prime: int, sequence: Tuple[int, ...] | None = None):
        """
        Parameters
        ----------
        prime    : the prime p whose valuation structure this navigates
        sequence : current sequence state; if None, initializes from seed
        """
        if sequence is None:
            self._nut: Tuple[int, ...] = tuple([0] * (prime - 1) + [1])
            self.sequence: Tuple[int, ...] = self._nut
        else:
            self._nut = tuple([0] * (prime - 1) + [1])
            self.sequence = sequence
        self.prime = prime

    def flip(self) -> FlipFlop:
        """
        Apply the flip operation once.

        From sequence s = (v_p(1), ..., v_p(p^k)), produces
        s' = (v_p(1), ..., v_p(p^(k+1))):

            (p * s)[:-1] + (s[-1] + 1,)

        Repeat s exactly p times, then increment the last element by 1.

        Why this works: v_p(j + m·p^k) = v_p(j) for all j in 1..p^k when
        v_p(j) < k — adding m·p^k does not affect the valuation. The only
        exception is the final position p^(k+1), where v_p = k+1 rather
        than k. The p repetitions are correct everywhere except that one
        last element, which the +1 fixes.

        Returns a new FlipFlop one level deeper. The current instance
        is unchanged.
        """
        s = self.sequence
        return FlipFlop(self.prime, (self.prime * s)[:-1] + (s[-1] + 1,))

    def __call__(self) -> FlipFlop:
        """Alias for flip(). Allows: f = f()"""
        return self.flip()

    @property
    def depth(self) -> int:
        """
        Current depth k — number of flips applied from the seed.
        Seed is depth 0 (covers p^1 integers).
        After k flips: covers p^(k+1) integers.
        """
        n = len(self.sequence)
        return round(math.log(n) / math.log(self.prime)) - 1

    @property
    def coverage(self) -> int:
        """
        Number of integers covered by the current sequence.
        Equals len(sequence) = p^(depth+1).
        """
        return len(self.sequence)

    def flips_needed(self, n: int) -> int:
        """
        Number of additional flips required to encompass integer n.

        From the current state, returns the number of flip() calls needed
        so that coverage >= n.
        """
        if self.coverage >= n:
            return 0
        # Need p^(depth+1+extra) >= n, so extra >= log_p(n) - (depth+1)
        extra = math.ceil(math.log(n, self.prime)) - (self.depth + 1)
        return max(0, extra)

    def expand_to(self, n: int) -> FlipFlop:
        """
        Return a FlipFlop deep enough to encompass integer n.

        Applies flip() as many times as needed from the current level.
        """
        result = self
        while result.coverage < n:
            result = result.flip()
        return result

    def valuation_at(self, n: int) -> int:
        """
        Return v_p(n) by navigating to the position in the sequence.

        Parameters
        ----------
        n : positive integer (1-indexed position)

        The sequence is expanded automatically if needed.
        No division is performed — navigation replaces computation.
        """
        if n < 1:
            raise ValueError(f"n must be a positive integer, got {n}")
        navigator = self.expand_to(n)
        return navigator.sequence[n - 1]

    def __repr__(self) -> str:
        preview = self.sequence[:15]
        suffix = "..." if len(self.sequence) > 15 else ""
        return (
            f"FlipFlop(p={self.prime}, depth={self.depth}, "
            f"coverage={self.coverage}, seq={preview}{suffix})"
        )
