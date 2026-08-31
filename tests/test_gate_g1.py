"""Milestone 2 gate: task validity, read back from stored runs.

Milestone 2's definition of done is that this file passes against stored
gate runs, which makes the gate re-runnable rather than a one-off
observation someone reported in a message.

Checks that need trained models skip cleanly when the runs are absent, so
the suite still passes on a machine that has never touched the cluster.
The gate itself is only satisfied when they actually run, and
`python -m loopvision.analysis.gate_g1` is what reports that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from loopvision.data import groups as G
from loopvision.data import validate as V

RUNS = Path(__file__).resolve().parents[1] / "runs"

GATE_RUNS = {
    "d3_looped": RUNS / "g1_famA_d3_looped_s0",
    "d3_ffwd": RUNS / "g1_famA_d3_ffwd_s0",
    "d1_looped": RUNS / "g1_famA_d1_looped_s0",
    "famC": RUNS / "g1_famC_looped_s0",
}


def needs(run: str):
    return pytest.mark.skipif(
        not (GATE_RUNS[run] / "summary.json").is_file(),
        reason=f"gate run {GATE_RUNS[run].name} has not been trained here",
    )


# ---------------------------------------------------------------------------
# Thresholds are pre-registered and must not drift
# ---------------------------------------------------------------------------


def test_thresholds_are_what_was_pre_registered() -> None:
    """Pin the numbers committed before the first gate job was submitted.

    If a gate fails, the temptation is to move a threshold. This test makes
    that impossible to do quietly: changing any of these fails here, and
    the change has to be argued for in docs/decisions.md.
    """
    assert V.TAU == 0.90
    assert V.G1_1_MAX_ACCURACY == 0.05
    assert V.G1_2_MIN_ACCURACY == 0.90
    assert V.G1_3_CEILING_MARGIN == 0.15
    assert V.G1_4_MIN_ACCURACY == 0.90


def test_g1_1_threshold_sits_above_chance() -> None:
    """A near-chance criterion must be above chance, or nothing can pass."""
    assert V.G1_1_MAX_ACCURACY > 1 / G.GROUP_ORDER


# ---------------------------------------------------------------------------
# G1.3, which needs no trained model
# ---------------------------------------------------------------------------


def test_g1_3_ceiling_clears_tau() -> None:
    """The order-blind ceiling must sit well below the M1 threshold.

    This is the check that makes the loop-count curve interpretable: if an
    order-blind model could reach tau, k_min would be measuring the
    emergence of a counting statistic rather than of composition.
    """
    rng = np.random.default_rng(23)
    for depth in range(2, 7):
        ceiling = G.bag_of_operators_ceiling(rng, depth, trials=2000)
        assert ceiling < V.TAU - V.G1_3_CEILING_MARGIN, (
            f"depth {depth} ceiling {ceiling:.3f} is too close to tau"
        )


def test_bag_of_operators_probe_does_not_beat_its_own_ceiling() -> None:
    """A probe above the ceiling means the probe is using order information.

    The ceiling is the Bayes accuracy of any order-blind predictor, so a
    logistic regression on operator counts cannot legitimately exceed it.
    Exceeding it would mean the feature construction leaked order, which
    is exactly the bug class that produced N-002.
    """
    rng = np.random.default_rng(23)
    for depth in (2, 3):
        ceiling = G.bag_of_operators_ceiling(rng, depth, trials=2000)
        probe = V.bag_of_operators_probe(depth, n_train=6000, n_test=2000, epochs=25)
        assert probe <= ceiling + 0.03, (
            f"depth {depth}: probe {probe:.3f} exceeds ceiling {ceiling:.3f}, "
            f"so the probe features leak operator order"
        )


def test_probe_learns_something_but_falls_far_short_of_the_ceiling() -> None:
    """The linear probe is much weaker than the order-blind bound, by design.

    Measured at depth 2: probe about 0.056, ceiling 0.652, chance 0.021. The
    gap is not a bug and not underfitting. A logistic regression on operator
    counts can only score additively,

        score(g) = sum_op count[op] * W[op, g]

    so for a pair {a, b} it must rank by W[a, g] + W[b, g]. Picking a.b for
    every pair simultaneously is not representable that way in a
    non-abelian group, where a.b and b.a differ for most pairs. The
    ceiling, by contrast, is the Bayes accuracy of a predictor that may
    memorise each multiset outright.

    **This is why gate G1.3 gates on the ceiling rather than on this
    probe** (docs/decisions.md D-016). The spec's suggested probe would
    have reported 0.056 and looked like strong evidence of no shortcut,
    when a shortcut worth 0.652 was sitting there unmeasured. A weak probe
    finding little is weak evidence of nothing to find.
    """
    probe = V.bag_of_operators_probe(2, n_train=8000, n_test=2000, epochs=30)
    chance = 1 / G.GROUP_ORDER
    assert probe > 2 * chance, f"probe at {probe:.3f} learned nothing at all"
    assert probe < 0.30, (
        f"probe at {probe:.3f} is far above what an additive model should "
        f"reach. Either the features leak order or the group has changed."
    )


# ---------------------------------------------------------------------------
# G1.1, G1.2, G1.4: read stored runs
# ---------------------------------------------------------------------------


@needs("d3_looped")
@needs("d3_ffwd")
def test_g1_1_depth3_is_unsolvable_at_k1() -> None:
    """Both a looped model at k=1 and a matched feedforward must be near chance.

    If either solves depth 3 in one pass, there is no loop-count curve to
    analyse and the task must be deepened.
    """
    for result in V.check_g1_1(GATE_RUNS["d3_looped"], GATE_RUNS["d3_ffwd"]):
        assert result.passed, result.line()


@needs("d1_looped")
def test_g1_2_depth1_is_solvable_at_k1() -> None:
    """Otherwise the task is broken rather than deep."""
    result = V.check_g1_2(GATE_RUNS["d1_looped"])
    assert result.passed, result.line()


@needs("famC")
def test_g1_4_family_c_is_solvable_at_every_breadth() -> None:
    """Every breadth cell, not the average.

    Family C is the H1 null arm. If counting needs loops at any breadth,
    the breadth axis carries sequential structure and the depth versus
    breadth contrast is void.
    """
    result = V.check_g1_4(GATE_RUNS["famC"])
    assert result.passed, result.line()


@needs("d3_looped")
@needs("d3_ffwd")
@needs("d1_looped")
@needs("famC")
def test_gate_g1_passes_as_a_whole() -> None:
    from loopvision.analysis import gate_g1

    results, passed = gate_g1.run(RUNS, skip_slow=True)
    assert passed, "\n".join(r.line() for r in results if not r.passed)
