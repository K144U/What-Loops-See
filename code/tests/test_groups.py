"""Milestone 1: the D4 x S3 group algebra and the family A stimulus.

These tests exist because every downstream number depends on the group
being the group we think it is. An associativity bug would not crash
anything, it would just produce a dataset whose labels are wrong in a
structured way, and the model would learn the wrong structure perfectly.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from loopvision.data import groups as G


# ---------------------------------------------------------------------------
# Group axioms
# ---------------------------------------------------------------------------


def test_order_is_48() -> None:
    assert G.GROUP_ORDER == 48
    assert G.MUL.shape == (48, 48)


def test_closure_and_latin_square() -> None:
    """Every row and column of the table is a permutation of the group.

    This is the rearrangement theorem. It catches a whole class of table
    construction bugs at once, including duplicated or missing elements.
    """
    assert G.MUL.min() == 0 and G.MUL.max() == G.GROUP_ORDER - 1
    for i in range(G.GROUP_ORDER):
        assert sorted(G.MUL[i, :].tolist()) == list(range(G.GROUP_ORDER))
        assert sorted(G.MUL[:, i].tolist()) == list(range(G.GROUP_ORDER))


def test_associativity_exhaustive() -> None:
    """All 48^3 = 110592 triples. Cheap enough to do exhaustively, so do it.

    The spec asks for this explicitly, and a sampled version would be a
    weaker claim for no saving worth having.
    """
    mul = G.MUL
    # (a.b).c versus a.(b.c), vectorised over c for speed.
    for a in range(G.GROUP_ORDER):
        for b in range(G.GROUP_ORDER):
            ab = mul[a, b]
            left = mul[ab, :]              # (a.b).c for every c
            right = mul[a, mul[b, :]]      # a.(b.c) for every c
            assert np.array_equal(left, right), f"associativity fails at {a},{b}"


def test_identity() -> None:
    for i in range(G.GROUP_ORDER):
        assert G.multiply(G.IDENTITY, i) == i
        assert G.multiply(i, G.IDENTITY) == i


def test_inverses() -> None:
    for i in range(G.GROUP_ORDER):
        assert G.multiply(i, G.inverse(i)) == G.IDENTITY
        assert G.multiply(G.inverse(i), i) == G.IDENTITY
        assert G.inverse(G.inverse(i)) == i


def test_group_is_non_abelian() -> None:
    """Order must matter, otherwise the whole task is a counting problem.

    Also checks that non-commutativity is common rather than a rare corner,
    since a nearly abelian group would make the task nearly order blind.
    """
    non_commuting = sum(
        1
        for i in range(G.GROUP_ORDER)
        for j in range(G.GROUP_ORDER)
        if G.multiply(i, j) != G.multiply(j, i)
    )
    total = G.GROUP_ORDER**2
    assert non_commuting > 0
    # Both factors are non-abelian, so the majority of pairs should fail to
    # commute. If this ever drops sharply the group has been changed.
    assert non_commuting / total > 0.5, f"only {non_commuting}/{total} pairs fail to commute"


def test_factors_have_the_right_orders() -> None:
    assert len(set(G.D4_ELEMENTS)) == 8
    assert len(set(G.S3_ELEMENTS)) == 6
    # Every S3 element is a genuine permutation of three symbols.
    for p in G.S3_ELEMENTS:
        assert sorted(p) == [0, 1, 2]


# ---------------------------------------------------------------------------
# The algebra agrees with the geometry
# ---------------------------------------------------------------------------


def test_geometric_action_matches_the_group_law() -> None:
    """Acting by (x.y) equals acting by y then by x.

    This is the test that ties the abstract table to the pictures. If it
    fails, the rendered stimulus and the label disagree, which is the worst
    possible failure because everything still runs and trains.
    """
    points = [(1, 0), (0, 1), (2, 3), (-1, 2)]
    for i in range(G.GROUP_ORDER):
        for j in range(G.GROUP_ORDER):
            composite = G.multiply(i, j)
            for x, y in points:
                via_table = G.d4_apply_cell(G.d4_of(composite), x, y)
                stepwise = G.d4_apply_cell(
                    G.d4_of(i), *G.d4_apply_cell(G.d4_of(j), x, y)
                )
                assert via_table == stepwise, f"action mismatch at {i}.{j}"


def test_sequence_composition_is_left_to_right_in_time() -> None:
    """ops[0] is applied first. Guards the convention downstream relies on."""
    rng = np.random.default_rng(0)
    ops = [int(x) for x in rng.integers(0, G.GROUP_ORDER, size=5)]

    expected = G.IDENTITY
    for op in ops:
        expected = G.multiply(op, expected)

    assert G.compose_sequence(ops) == expected
    assert G.compose_sequence([]) == G.IDENTITY
    assert G.prefix_products(ops)[-1] == G.compose_sequence(ops)
    assert len(G.prefix_products(ops)) == len(ops)


# ---------------------------------------------------------------------------
# The stimulus
# ---------------------------------------------------------------------------


def test_glyph_stabiliser_is_trivial() -> None:
    """Every group element must give a visually distinct state.

    If the stabiliser is larger than the identity, distinct labels render
    identically and the 48-way task is secretly smaller, which would make
    the reported chance level wrong.
    """
    stab = G.stabiliser()
    assert stab == [G.IDENTITY], (
        f"glyph stabiliser has order {len(stab)}, elements "
        f"{[G.element_name(i) for i in stab]}. The task is only "
        f"{G.distinct_states()}-way, not 48-way."
    )


def test_glyph_has_48_distinct_states() -> None:
    assert G.distinct_states() == G.GROUP_ORDER


def test_l_tromino_is_rejected() -> None:
    """The shape the spec suggested does not work, and this records why.

    An L-tromino with distinctly coloured cells is fixed by the diagonal
    reflection paired with the transposition swapping its two arm colours.
    Keeping this as a test means nobody reintroduces it later thinking the
    tetromino was an arbitrary choice.
    """
    l_tromino = ((0, 0, 0), (1, 0, 1), (0, 1, 2))
    stab = G.stabiliser(l_tromino)
    assert len(stab) == 2
    assert G.distinct_states(l_tromino) == 24


# ---------------------------------------------------------------------------
# Anti-shortcut sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5, 6])
def test_factorisation_composes_to_the_target(depth: int) -> None:
    rng = np.random.default_rng(depth)
    for _ in range(200):
        target = int(rng.integers(0, G.GROUP_ORDER))
        ops = G.random_factorisation(rng, target, depth)
        assert len(ops) == depth
        assert G.compose_sequence(ops) == target


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_every_proper_prefix_is_uniform(depth: int) -> None:
    """Prefix products must carry no information about the answer.

    This is the property that kills prefix-statistics shortcuts. Tested as
    a chi-squared style spread check rather than an exact count, since the
    distribution is uniform only in expectation.
    """
    rng = np.random.default_rng(100 + depth)
    trials = 48 * 400
    for prefix_len in range(1, depth):
        counts = np.zeros(G.GROUP_ORDER, dtype=np.int64)
        for _ in range(trials):
            ops, _ = G.sample_operator_sequence(rng, depth)
            counts[G.prefix_products(ops)[prefix_len - 1]] += 1
        expected = trials / G.GROUP_ORDER
        chi2 = float(((counts - expected) ** 2 / expected).sum())
        # 47 degrees of freedom, 0.999 critical value is about 92.
        assert chi2 < 92.0, (
            f"prefix of length {prefix_len} at depth {depth} is not uniform, "
            f"chi2={chi2:.1f}"
        )


def test_labels_are_uniform() -> None:
    """Chance accuracy must be exactly 1/48, with no exploitable class prior."""
    rng = np.random.default_rng(7)
    trials = 48 * 400
    counts = np.zeros(G.GROUP_ORDER, dtype=np.int64)
    for _ in range(trials):
        _, target = G.sample_operator_sequence(rng, 3)
        counts[target] += 1
    expected = trials / G.GROUP_ORDER
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 92.0, f"labels are not uniform, chi2={chi2:.1f}"


def test_single_operator_marginals_are_uniform() -> None:
    """No individual position leaks the answer, including the solved-for last one."""
    rng = np.random.default_rng(11)
    depth = 4
    trials = 48 * 300
    for position in range(depth):
        counts = np.zeros(G.GROUP_ORDER, dtype=np.int64)
        for _ in range(trials):
            ops, _ = G.sample_operator_sequence(rng, depth)
            counts[ops[position]] += 1
        expected = trials / G.GROUP_ORDER
        chi2 = float(((counts - expected) ** 2 / expected).sum())
        assert chi2 < 92.0, f"operator position {position} is not uniform, chi2={chi2:.1f}"


def test_order_blind_ceiling_falls_with_depth() -> None:
    """The order-blind upper bound must decay monotonically with depth.

    It is NOT at chance, and it cannot be made to sit at chance. See
    docs/decisions.md D-016. An order-blind predictor at depth 2 sees the
    multiset {a, b} and need only choose between a.b and b.a, so it scores
    around 65 percent against a chance level of 2 percent. This is a
    property of the group and of counting, not a defect in the sampler,
    and no sampler can remove it.

    What the sampler does guarantee is tested above: uniform labels,
    uniform prefixes, uniform single-operator marginals.
    """
    rng = np.random.default_rng(23)
    ceilings = {d: G.bag_of_operators_ceiling(rng, d, trials=3000) for d in (2, 4, 6)}

    assert ceilings[2] > ceilings[4] > ceilings[6], (
        f"order-blind ceiling should fall with depth, got {ceilings}"
    )


def test_order_blind_ceiling_stays_below_the_m1_threshold() -> None:
    """The load-bearing property, and the reason M1 survives the shortcut.

    M1 measures the minimum k at which held-out accuracy reaches
    tau = 0.90. If an order-blind model could reach tau, the loop-count
    curve would be measuring the emergence of a counting statistic rather
    than of sequential composition, and H1 would be uninterpretable.

    It cannot. The ceiling peaks at depth 2 near 0.65 and falls from there,
    so tau sits above every order-blind model at every depth by a wide
    margin. Reaching tau requires order sensitivity, which is exactly what
    the depth axis is supposed to be about.

    If this test ever fails, either tau has been lowered or the group has
    been changed, and gate G1 must be revisited before any M1 run.
    """
    rng = np.random.default_rng(29)
    tau = 0.90
    margin = 0.15

    for depth in range(2, 7):
        ceiling = G.bag_of_operators_ceiling(rng, depth, trials=2500)
        assert ceiling < tau - margin, (
            f"order-blind ceiling at depth {depth} is {ceiling:.3f}, within "
            f"{margin} of the M1 threshold {tau}. The loop-count curve would "
            f"no longer isolate sequential composition."
        )


def test_ceiling_at_depth_two_is_the_documented_worst_case() -> None:
    """Pin the worst case so a task change cannot alter it silently."""
    rng = np.random.default_rng(31)
    ceiling = G.bag_of_operators_ceiling(rng, 2, trials=4000)
    assert 0.60 < ceiling < 0.70, (
        f"depth 2 order-blind ceiling is {ceiling:.3f}, outside the recorded "
        f"band of 0.60 to 0.70. The task design has changed and gate G1.3 "
        f"needs its numbers regenerated. See docs/findings.md."
    )
