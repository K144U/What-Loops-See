"""Loop-count sampling, IMPLEMENTATION.md Section 6.1.

The loop count is what the whole study measures, so a bug here would not
produce a crash, it would produce a loop-count curve for a schedule nobody
intended. These tests pin the shape of the distribution and, just as
importantly, its reproducibility across a resume.
"""

from __future__ import annotations

import math

import pytest
import torch

from loopvision.train import loop_schedule as L


def test_mu_ramps_then_holds() -> None:
    total = 100_000
    assert L.mu_at(0, total) == pytest.approx(math.log(2.0))
    # Ramp completes at 40 percent of the run and holds thereafter.
    end = math.log(L.DEFAULT_K_MEAN)
    assert L.mu_at(40_000, total) == pytest.approx(end)
    assert L.mu_at(70_000, total) == pytest.approx(end)
    assert L.mu_at(total, total) == pytest.approx(end)
    # Monotone during the ramp.
    vals = [L.mu_at(s, total) for s in range(0, 40_000, 4_000)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_k_stays_in_range() -> None:
    torch.manual_seed(0)
    for step in (0, 5_000, 50_000, 200_000):
        for _ in range(300):
            k = L.sample_loop_count(step, 200_000)
            assert 1 <= k <= L.K_MAX_TRAIN


def test_k_max_is_respected_even_with_a_wild_sigma() -> None:
    torch.manual_seed(1)
    for _ in range(2000):
        assert L.sample_loop_count(100_000, 200_000, sigma=3.0) <= L.K_MAX_TRAIN


def test_sigma_zero_is_genuinely_deterministic() -> None:
    """The H4 width-zero arm must have no jitter at all.

    Setting sigma to zero fixes tau but leaves the Poisson draw in place,
    so k would still vary batch to batch and the "no jitter" condition
    would not be jitter free. The implementation special cases it, and this
    is the test that keeps that special case honest.
    """
    torch.manual_seed(2)
    for step in (0, 40_000, 150_000):
        counts = {L.sample_loop_count(step, 200_000, sigma=0.0) for _ in range(200)}
        assert len(counts) == 1, f"sigma=0 gave {counts} at step {step}"


def test_jitter_actually_varies_the_count() -> None:
    torch.manual_seed(3)
    counts = {L.sample_loop_count(150_000, 200_000, sigma=0.5) for _ in range(300)}
    assert len(counts) > 5, f"default jitter is nearly deterministic: {counts}"


def test_sampled_mean_matches_the_closed_form() -> None:
    """E[k] = 1 + exp(mu + sigma^2 / 2), the lognormal mean.

    Note this is about 10.1 at k_mean_target = 8, not 8. The spec's
    k_mean_target names the median of tau, not the mean of k. Pinned here
    so the difference is deliberate rather than discovered later while
    interpreting a loop-count curve.
    """
    torch.manual_seed(4)
    step, total = 150_000, 200_000
    draws = [L.sample_loop_count(step, total) for _ in range(20_000)]
    empirical = sum(draws) / len(draws)
    assert empirical == pytest.approx(L.expected_k(step, total), rel=0.05)
    assert 9.5 < L.expected_k(step, total) < 10.5


def test_the_ramp_starts_low() -> None:
    """Early training must not sit at the full loop count.

    A model that begins at k=8 has to learn to use eight applications of an
    untrained core before any of them help, which in the text case was one
    of the things that decided whether a small looped model trained at all.
    """
    torch.manual_seed(5)
    early = [L.sample_loop_count(0, 200_000) for _ in range(2000)]
    late = [L.sample_loop_count(150_000, 200_000) for _ in range(2000)]
    assert sum(early) / len(early) < sum(late) / len(late) - 3


def test_draws_are_reproducible_from_the_global_rng() -> None:
    """Resume exactness depends on this.

    checkpoint.py captures the global torch RNG. If loop counts came from a
    private generator, a resumed run would follow a different loop schedule
    and nothing else would notice.
    """
    torch.manual_seed(99)
    first = [L.sample_loop_count(s, 200_000) for s in range(0, 5_000, 500)]
    torch.manual_seed(99)
    second = [L.sample_loop_count(s, 200_000) for s in range(0, 5_000, 500)]
    assert first == second


def test_eval_grid_covers_extrapolation_and_respects_the_cap() -> None:
    grid = L.eval_k_grid(64)
    assert grid[0] == 1
    assert 8 in grid, "the training mean must be on the curve"
    assert max(grid) == 64, "H2 needs points well past the training k_max"
    assert grid == sorted(set(grid))
    assert max(L.eval_k_grid(16)) == 16
