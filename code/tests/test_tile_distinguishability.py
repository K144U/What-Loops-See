"""D-070's measure, tested on synthetic tiles only.

The registration says the module is tested on synthetic tiles so that no real
value exists before the entry is signed. These tests keep to that: every
number checked here is worked out by hand from tiles made up for the test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from loopvision.analysis import tile_distinguishability as TD
from loopvision.data import groups as G


def _tile(fill: int = 0) -> np.ndarray:
    return np.full((3, 4, 4), fill, dtype=np.uint8)


def test_one_pixel_one_channel_full_swing_is_distance_one() -> None:
    a, b = _tile(), _tile()
    b[0, 1, 2] = 255
    mean, low = TD.m_pix(np.stack([a, b]))
    assert mean == pytest.approx(1.0) and low == pytest.approx(1.0)


def test_pixels_are_scaled_to_zero_one_before_the_distance() -> None:
    """One pixel black against white in all three channels is sqrt 3, not
    255 sqrt 3."""
    a, b = _tile(0), _tile(0)
    b[:, 0, 0] = 255
    assert TD.m_pix(np.stack([a, b]))[0] == pytest.approx(math.sqrt(3))


def test_mean_and_minimum_over_distinct_pairs_only() -> None:
    """Three tiles at distances 1, 2 and sqrt 5 from each other. Counting a
    tile against itself would add zeros and pull the mean and minimum down."""
    a, b, c = _tile(), _tile(), _tile()
    b[0, 0, 0] = 255
    c[1, 0, 0] = 255
    c[1, 0, 1] = 255
    c[1, 1, 0] = 255
    c[1, 1, 1] = 255
    mean, low = TD.m_pix(np.stack([a, b, c]))
    assert mean == pytest.approx((1 + 2 + math.sqrt(5)) / 3)
    assert low == pytest.approx(1.0)


def test_the_measure_ignores_which_element_draws_which_tile() -> None:
    """E1b draws D4 position's tiles in a different order, and the registration
    says the two tie. That holds only if the measure is order free."""
    rng = np.random.default_rng(0)
    t = rng.integers(0, 256, size=(8, 3, 4, 4), dtype=np.uint8)
    shuffled = t[rng.permutation(8)]
    assert TD.m_pix(t) == pytest.approx(TD.m_pix(shuffled))


def test_the_embedding_distance_goes_through_the_convolution_and_averages_the_seeds() -> None:
    """A fake convolution that multiplies by the seed plus one: the distances
    scale by 1, 2 and 3 over three seeds, so the average is twice the pixel
    distance. Using one seed, or none, gives a different number."""
    a, b = _tile(), _tile()
    b[2, 3, 3] = 255
    stack = np.stack([a, b])
    mean, low = TD.m_emb(stack, n_init=3, make_conv=lambda seed: (lambda x: x * (seed + 1)))
    assert mean == pytest.approx(2.0) and low == pytest.approx(2.0)


def test_the_real_patch_embedding_is_the_model_s_own_and_is_seeded() -> None:
    """Shape, bias and spread of a freshly initialised model's patchify, and
    the same seed giving the same weights. No rendered tile is used."""
    conv_a, conv_b, conv_c = TD.model_patchify(0), TD.model_patchify(0), TD.model_patchify(1)
    assert tuple(conv_a.weight.shape) == (384, 3, 4, 4)
    assert torch.equal(conv_a.weight, conv_b.weight)
    assert not torch.equal(conv_a.weight, conv_c.weight)
    assert float(conv_a.bias.abs().max()) == 0.0
    assert float(conv_a.weight.std()) == pytest.approx(0.02, rel=0.1)


def test_average_ranks_share_a_tie() -> None:
    assert list(TD.average_ranks([3.0, 1.0, 2.0, 2.0])) == [4.0, 1.0, 2.5, 2.5]


def test_a_tie_broken_only_by_rounding_error_is_still_a_tie() -> None:
    """The fault this caught. E1b's M_pix and D4 position's are one mean of
    one set of distances, summed in a different order, and an exact
    comparison ranked them apart. D-076."""
    assert 0.1 + 0.2 != 0.3
    assert list(TD.average_ranks([0.1 + 0.2, 0.3, 1.0])) == [1.5, 1.5, 3.0]


def test_a_real_difference_is_not_rounded_into_a_tie() -> None:
    assert list(TD.average_ranks([1.0, 1.000001])) == [1.0, 2.0]


#: M_pix and the escape rates as stored by the run of 2026-10-22, at full
#: precision, D4 position and E1b differing only in their last digits.
STORED_PIX = [2.245522029421255, 1.410386483116888, 2.394779570317641,
              2.2455220294212546, 1.6936974862908776, 2.278701194420493,
              1.286757781697509]
STORED_RATES = [9 / 10, 12 / 40, 16 / 20, 14 / 20, 13 / 20, 6 / 10, 3 / 10]


def test_the_stored_measures_give_the_registered_rho() -> None:
    """Average ranks for the named tie give 18.5 / 27.5. Breaking it one way
    gave the 0.703 first reported, and the other way gives 0.631."""
    assert TD.spearman(STORED_PIX, STORED_RATES) == pytest.approx(18.5 / 27.5)


def test_the_permutation_p_counts_both_tails() -> None:
    """Of the six orderings of three points, the identity and its reverse
    are as far from zero as the observed rho of 1."""
    assert TD.permutation_p([1, 2, 3], [1, 2, 3]) == pytest.approx(2 / 6)


def test_the_stored_measures_have_p_near_a_tenth() -> None:
    """Seven points: a rho of 0.673 is as extreme as 540 of the 5040
    orderings."""
    assert TD.permutation_p(STORED_PIX, STORED_RATES) == pytest.approx(540 / 5040)


def test_the_permutation_p_refuses_a_sample_too_large_to_enumerate() -> None:
    with pytest.raises(ValueError):
        TD.permutation_p(list(range(TD.MAX_EXACT + 1)), list(range(TD.MAX_EXACT + 1)))


def test_spearman_against_values_worked_by_hand() -> None:
    assert TD.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert TD.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    # Ranks [1, 2.5, 2.5, 4] against [1, 2, 3, 4]: 4.5 over sqrt(4.5 * 5).
    assert TD.spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(math.sqrt(0.9))
    assert math.isnan(TD.spearman([1, 1, 1], [1, 2, 3]))


@pytest.mark.parametrize("rho, expected", [
    (0.6, "held"), (0.95, "held"), (0.5999, "undetermined"), (0.0001, "undetermined"),
    (0.0, "failed"), (-0.4, "failed"), (float("nan"), "undetermined"),
])
def test_the_registered_thresholds(rho, expected) -> None:
    assert TD.verdict(rho) == expected


def test_the_seven_renderings_and_the_one_kept_out() -> None:
    names = [r[0] for r in TD.RENDERINGS if r[4]]
    assert names == ["D4 position", "D4 appearance", "E1a", "E1b", "E2b",
                     "S3 position", "S3 appearance"]
    assert [r[0] for r in TD.RENDERINGS if not r[4]] == ["E2a"]
    for _, substrate, subgroup, _, _ in TD.RENDERINGS:
        G.glyph_action(substrate)
        assert len(G.subgroup_members(subgroup)) in (6, 8)


def test_each_sealed_cell_is_drawn_with_the_substrate_its_runs_used() -> None:
    """Cross-checked against escape_rates, which assigns runs to cells from
    their configs. A rendering paired with the wrong cell's rate would still
    produce a correlation, and a wrong one."""
    from loopvision.analysis import escape_rates as ER
    for name, substrate, subgroup, (kind, key), _ in TD.RENDERINGS:
        if kind != "cell":
            continue
        factor = subgroup if substrate == "native" else "full"
        assert ER.SUBSTRATE_ARMS[(factor, substrate)] == key, name


def test_each_d062_arm_is_drawn_with_its_own_substrate() -> None:
    expected = {"why_e1a_square": "d4_square", "why_e1b_scrambled": "d4_scrambled",
                "why_e2b_decoy": "d4_colour_decoy", "why_e2a_cue": "d4_colour_cue"}
    got = {key: substrate for _, substrate, _, (kind, key), _ in TD.RENDERINGS if kind == "stem"}
    assert got == expected


def _measures(values: dict[str, float]) -> dict:
    return {n: {"m_pix": v, "m_emb": -v} for n, v in values.items()}


def test_evaluate_uses_m_pix_for_the_verdict_and_leaves_e2a_out() -> None:
    names = [r[0] for r in TD.RENDERINGS]
    pix = {n: float(i) for i, n in enumerate(names)}
    rates = {n: (i, 10) for i, n in enumerate(names)}
    # E2a gets the highest M_pix and the lowest rate. Counted, it would pull rho below 1.
    pix["E2a"], rates["E2a"] = 100.0, (0, 10)
    out = TD.evaluate(_measures(pix), rates)
    assert out["S1"] == "held" and out["rho_pix"] == pytest.approx(1.0)
    assert out["rho_emb"] == pytest.approx(-1.0), "M_emb is reported, and does not decide"
    assert "E2a" not in out["tested"]
    # Each p belongs to its own correlation: seven points in perfect order are
    # matched by 2 of the 5040 orderings either way.
    assert out["p_pix"] == pytest.approx(2 / 5040)
    assert out["p_emb"] == pytest.approx(2 / 5040)
    assert out["S1"] == "held", "the p decides nothing"


def test_a_missing_rate_leaves_s1_pending() -> None:
    names = [r[0] for r in TD.RENDERINGS]
    rates = {n: (1, 10) for n in names}
    rates["E1a"] = None
    out = TD.evaluate(_measures({n: 1.0 for n in names}), rates)
    assert out["S1"] == "pending"


def test_rates_are_read_from_cells_and_stems() -> None:
    """Synthetic rows in the shape escape_rates.scan produces."""
    def row(rid, arm, outcome):
        return {"run_id": rid, "arm": arm, "done": True, "outcome": outcome, "accuracy": 1.0}
    rows = [row("famA_d2_d4only_s0", ("D4", "position"), "escaped"),
            row("famA_d2_d4only_s1", ("D4", "position"), "collapsed"),
            row("why_e1a_square_s200", None, "escaped"),
            row("why_e1a_square_s201", None, "escaped")]
    got = TD.escape_rates(rows)
    assert got["D4 position"] == (1, 2)
    assert got["E1a"] == (2, 2)
    assert got["D4 appearance"] is None
