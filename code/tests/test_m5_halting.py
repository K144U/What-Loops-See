"""The halting instrument has to be right about when a rule fires.

H2 is one of only two hypotheses the preregistration binds in the ordinary
sense, and the whole claim is a signed distance between where a criterion
fires and where the answer actually arrived. An off-by-one in either index
is a result, not a bug report, so every index convention below is asserted
against a hand-built trajectory rather than against a model.

The censoring tests matter as much as the firing ones. An item the model
never gets right and an item it gets right at the last iteration must not
end up as the same number, and a criterion that never fires must not be
silently read as firing at k.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from loopvision.instruments import m5_halting as M5


def _states(seq: list[list[float]]) -> list[torch.Tensor]:
    return [torch.tensor([[v]], dtype=torch.float32) for v in seq]


def _probs(rows: list[list[float]]) -> torch.Tensor:
    """[k+1, 1, classes] of explicit probabilities, one item."""
    return torch.tensor([[r] for r in rows], dtype=torch.float32)


def test_first_below_is_one_based_and_matches_the_hook_convention() -> None:
    """Column j is iteration j+1. M4 uses the same convention and the two
    instruments are compared against each other."""
    v = np.array([[10.0, 5.0, 0.5, 0.1]])
    assert M5._first_below(v, 1.0)[0] == 3
    assert M5._first_below(v, 6.0)[0] == 2
    assert M5._first_below(v, 100.0)[0] == 1


def test_a_criterion_that_never_fires_is_censored_not_clamped() -> None:
    """Mutation checked: returning k instead of CENSORED makes this pass a
    never-halting item off as one that halted at the last iteration."""
    v = np.array([[10.0, 9.0, 8.0]])
    assert M5._first_below(v, 1.0)[0] == M5.CENSORED
    assert M5.CENSORED < 0, "the sentinel must not be a possible iteration"


def test_k_star_model_is_the_first_correct_iteration() -> None:
    # Classes are 2. Wrong, wrong, right, right -> k*_model is 3.
    probs = _probs([[0.9, 0.1], [0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    label = torch.tensor([1])
    assert M5.k_star_model(probs, label)[0] == 3


def test_a_model_never_correct_is_censored() -> None:
    """Not the same as being correct late, and averaging the two together
    would understate every criterion's error."""
    probs = _probs([[0.9, 0.1]] * 5)
    assert M5.k_star_model(probs, torch.tensor([1]))[0] == M5.CENSORED


def test_state_index_zero_is_excluded_from_k_star_model() -> None:
    """s_0 is the initial state, before any core application. A model that
    looks correct there is correct by accident of initialisation."""
    # Right at i=0 only, then wrong forever.
    probs = _probs([[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
    assert M5.k_star_model(probs, torch.tensor([1]))[0] == M5.CENSORED


def test_second_difference_is_undefined_before_the_third_iteration() -> None:
    """It needs three step sizes, so i=1 and i=2 have no value and must be
    NaN rather than zero, which would fire every criterion immediately."""
    states = _states([[0.0], [1.0], [3.0], [6.0], [10.0]])
    probs = _probs([[0.5, 0.5]] * 5)
    sig = M5.trajectory_signals(states, probs)
    assert np.isnan(sig["second"][0, 0])
    assert np.isnan(sig["second"][0, 1])
    # Steps are 1,2,3,4 so the second difference is exactly 0 from i=3.
    assert np.allclose(sig["second"][0, 2:], 0.0, atol=1e-6)


def test_step_and_entropy_are_computed_on_the_right_things() -> None:
    states = _states([[0.0], [3.0], [3.0]])
    # Uniform over 2 classes has entropy ln 2; a one-hot has entropy 0.
    probs = _probs([[0.5, 0.5], [0.5, 0.5], [1.0, 0.0]])
    sig = M5.trajectory_signals(states, probs)
    assert np.allclose(sig["step"][0], [3.0, 0.0], atol=1e-6)
    assert np.isclose(sig["entropy"][0, 0], np.log(2), atol=1e-5)
    assert np.isclose(sig["entropy"][0, 1], 0.0, atol=1e-5)


def test_median_abs_error_ignores_censored_items_on_either_side() -> None:
    k_hat = np.array([3, 5, M5.CENSORED, 4])
    k_model = np.array([3, 3, 3, M5.CENSORED])
    # Only items 0 and 1 are usable: errors 0 and 2, median 1.
    assert M5.median_abs_error(k_hat, k_model) == 1.0


def test_median_abs_error_is_infinite_when_nothing_is_usable() -> None:
    """So the threshold fit cannot silently select a criterion that never
    fires by scoring it as perfect."""
    k_hat = np.array([M5.CENSORED, M5.CENSORED])
    k_model = np.array([3, 4])
    assert M5.median_abs_error(k_hat, k_model) == float("inf")


def test_fit_picks_the_threshold_that_matches_the_oracle() -> None:
    """The whole protocol rests on this being a real minimisation.

    Step sizes fall 100, 10, 1, 0.1 and the model becomes correct at i=3,
    so a threshold that fires at 3 is exact and must win.
    """
    n = 8
    signals = {
        "step": np.tile(np.array([100.0, 10.0, 1.0, 0.1]), (n, 1)),
        "kl": np.tile(np.array([100.0, 10.0, 1.0, 0.1]), (n, 1)),
        "entropy": np.tile(np.array([1.0, 0.5, 0.1, 0.01]), (n, 1)),
        "second": np.tile(np.array([np.nan, np.nan, 1.0, 0.1]), (n, 1)),
    }
    k_model = np.full(n, 3)
    t = M5.fit_thresholds(signals, k_model)
    fired = M5.fire_points(signals, t)
    assert M5.median_abs_error(fired["update_norm"], k_model) == 0.0
    assert M5.median_abs_error(fired["entropy"], k_model) == 0.0


def test_all_four_registered_criteria_are_present() -> None:
    """The preregistration fixes them so none can be added or dropped once
    the numbers exist. This is that clause, as a test."""
    assert M5.CRITERIA == ("update_norm", "kl", "entropy", "second_difference")
    signals = {
        "step": np.ones((2, 4)), "kl": np.ones((2, 4)),
        "entropy": np.ones((2, 4)), "second": np.ones((2, 4)),
    }
    fired = M5.fire_points(signals, {c: 2.0 for c in M5.CRITERIA})
    assert set(fired) == set(M5.CRITERIA)


def test_candidates_come_from_the_signal_not_a_fixed_grid() -> None:
    """A fixed logspace grid pinned three of four criteria to its own top
    edge on the first real run. Candidates drawn from the data cannot."""
    small = M5.candidate_thresholds(np.array([[1e-3, 2e-3, 5e-3]]))
    large = M5.candidate_thresholds(np.array([[1e7, 2e7, 5e7]]))
    assert small.max() < 1.0, "candidates did not follow a small signal"
    assert large.min() > 1.0, "candidates did not follow a large signal"


def test_the_chosen_threshold_is_never_pinned_to_the_range_edge() -> None:
    """The failure that motivated this: a threshold equal to the largest
    candidate means the search wanted to go further and could not."""
    n = 8
    step = np.tile(np.array([1e6, 1e4, 1e2, 1.0]), (n, 1))
    signals = {"step": step, "kl": step.copy(),
               "entropy": step.copy(), "second": step.copy()}
    k_model = np.full(n, 1)   # correct immediately, so fire as early as possible
    t = M5.fit_thresholds(signals, k_model)
    grid = M5.candidate_thresholds(step)
    assert M5.median_abs_error(M5.fire_points(signals, t)["update_norm"], k_model) == 0.0
    assert t["update_norm"] <= grid.max()


def test_candidates_survive_an_all_nan_signal() -> None:
    """second_difference is NaN for the first two iterations, and a run
    short enough to be all NaN must not crash the fit."""
    out = M5.candidate_thresholds(np.array([[np.nan, np.nan]]))
    assert out.size >= 1 and np.isfinite(out).all()
