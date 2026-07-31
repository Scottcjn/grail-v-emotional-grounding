#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Recompute the paper's headline statistics from the committed artifacts.
=======================================================================
``data/lpips_results.json`` is the released evidence for the efficiency
claim, but nothing in the repository re-derived the numbers the paper quotes
from it -- so a change in how the metric scripts aggregate (population vs
sample standard deviation, for instance) could silently stop matching the
paper without any test noticing.

This script reads the committed per-pair LPIPS means and reports, for each
published claim, the recomputed value next to the value in the paper.

    python code/verify_paper_stats.py

Exit status is 0 when every claim reproduces within tolerance, 1 otherwise.
No new dependencies: the Student-t tail is evaluated here with the
regularized incomplete beta function rather than pulling in SciPy.
"""

import argparse
import json
import math
import sys

import grail_config

# Arcs whose renders are single-subject portraits.  The paper's primary
# analysis is these three arcs x five seeds (supplementary, "Solo Portrait
# LPIPS: One-Sample t-Test").
SOLO_ARCS = ("sophia_contemplation", "sophia_determination", "sophia_realization")

# The paper's "complex multi-character scenes (n=15)" group.  The tension arc
# is deliberately excluded from it and reported separately as the extended
# n=22 analysis; see supplementary, same section.
COMPLEX_ARCS = ("debate_passion", "elyan_claude_focus", "elyan_sophia_focus")

EQUIVALENCE_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Statistics (no SciPy)
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log1p(-x) + a * math.log(x)
    ) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    """P(T <= t) for Student's t with ``df`` degrees of freedom."""
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return tail if t <= 0 else 1.0 - tail


def mean(values) -> float:
    return math.fsum(values) / len(values)


def sample_std(values) -> float:
    """ddof=1, matching every dispersion figure in the paper."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(math.fsum((v - m) ** 2 for v in values) / (len(values) - 1))


def population_std(values) -> float:
    """ddof=0 -- what numpy's default produced before this was pinned down."""
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(math.fsum((v - m) ** 2 for v in values) / len(values))


def one_sample_t(values, mu0: float):
    """(t, one-sided p for H1: mean < mu0, df)."""
    n = len(values)
    df = n - 1
    s = sample_std(values)
    if s == 0.0:
        # Zero spread: the test is degenerate rather than infinitely
        # significant.  Report it as such instead of dividing by zero.
        t = -math.inf if mean(values) < mu0 else math.inf
        return t, (0.0 if mean(values) < mu0 else 1.0), df
    t = (mean(values) - mu0) / (s / math.sqrt(n))
    return t, t_cdf(t, df), df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def arc_of(pair_key: str) -> str:
    """'sophia_realization_s42424242' -> 'sophia_realization'."""
    return pair_key.rsplit("_", 1)[0]


def load_pair_means(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {k: v["mean"] for k, v in data["per_pair"].items()}


def collect(pair_means: dict, arcs) -> list:
    return [v for k, v in sorted(pair_means.items()) if arc_of(k) in arcs]


def check(label: str, computed, published, tolerance, unit=""):
    ok = abs(computed - published) <= tolerance
    mark = "ok " if ok else "OFF"
    print(f"  [{mark}] {label:38s} computed {computed:>10.4f}{unit}   "
          f"paper {published:>9.4f}{unit}")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=grail_config.data_file("lpips_results.json"),
        help="LPIPS results JSON (default: <data dir>/lpips_results.json)",
    )
    args = parser.parse_args(argv)

    pair_means = load_pair_means(args.results)
    solo = collect(pair_means, SOLO_ARCS)
    complex_ = collect(pair_means, COMPLEX_ARCS)

    print("=" * 72)
    print("  GRAIL-V -- published statistics recomputed from committed data")
    print("=" * 72)
    print(f"\nSource: {args.results}")
    print(f"Pairs : {len(pair_means)} total, {len(solo)} solo, "
          f"{len(complex_)} complex (3 arcs), "
          f"{len(pair_means) - len(solo) - len(complex_)} tension\n")

    ok = True

    print("Solo portraits -- primary claim (paper: n=15, 0.011 +/- 0.005, t=-69.59)")
    ok &= check("n", len(solo), 15, 0)
    ok &= check("mean LPIPS", mean(solo), 0.011, 0.0005)
    ok &= check("s (sample std, ddof=1)", sample_std(solo), 0.005, 0.0005)
    t, p, df = one_sample_t(solo, EQUIVALENCE_THRESHOLD)
    ok &= check(f"one-sample t (df={df})", t, -69.59, 0.05)
    s_solo = sample_std(solo)
    d = abs(mean(solo) - EQUIVALENCE_THRESHOLD) / s_solo if s_solo else math.inf
    # The paper derives d from the rounded summary (|0.011 - 0.1| / 0.005 =
    # 17.8); from the unrounded pair means it is 17.97.  Tolerance covers
    # that rounding, not a disagreement.
    ok &= check("Cohen's d", d, 17.8, 0.25)

    print(f"\n  one-sided p = {p:.4g}  (paper states p < 1e-19)")
    if p < 1e-19:
        print("      -> holds")
    else:
        print(f"      -> NOTE: p = {p:.4g} is above 1e-19; the data support "
              f"p < 1e-18.")

    print("\n  Same t recomputed with population std (numpy's ddof=0 default):")
    s_pop = population_std(solo)
    t_pop = ((mean(solo) - EQUIVALENCE_THRESHOLD) / (s_pop / math.sqrt(len(solo)))
             if s_pop else -math.inf)
    print(f"      s = {s_pop:.6f}  ->  t = {t_pop:.2f}  (paper: -69.59)")
    print("      Aggregates must use ddof=1 to reproduce the paper.")

    print("\nComplex multi-character scenes (paper: n=15, sigma = 0.187)")
    ok &= check("n", len(complex_), 15, 0)
    ok &= check("sigma (sample std, ddof=1)", sample_std(complex_), 0.187, 0.001)
    lo = min(mean([v for k, v in pair_means.items() if arc_of(k) == a])
             for a in COMPLEX_ARCS)
    hi = max(mean([v for k, v in pair_means.items() if arc_of(k) == a])
             for a in COMPLEX_ARCS)
    ok &= check("lowest arc mean (paper: 0.446)", lo, 0.446, 0.001)
    ok &= check("highest arc mean (paper: 0.545)", hi, 0.545, 0.001)

    print("\n" + "=" * 72)
    print("  ALL CLAIMS REPRODUCE" if ok else "  MISMATCH -- see [OFF] rows above")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
