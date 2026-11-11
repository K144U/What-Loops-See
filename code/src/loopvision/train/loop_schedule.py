"""Loop-count sampling and curriculum.

IMPLEMENTATION.md Section 6.1. The loop count is drawn **per batch, not per
sample**, because a per-sample count would need ragged execution.

    tau = exp(normal(mu_t, sigma))
    k   = clip(1 + poisson(tau), 1, k_max)

``mu_t`` ramps linearly from log(2) to log(k_mean_target) over the first 40
percent of steps, then holds. Starting the ramp low matters: a model that
begins at k=8 has to learn to use eight applications of an untrained core
before any of them are useful, and in the text case that was one of the
things that decided whether a small looped model trained at all.

**Jitter is on by default and is not our idea.** Kuo et al. (arXiv
2606.29983) report that stochastic training-time loop counts sharply reduce
out-of-distribution variance. H4 tests whether the attractor version of the
effect replicates in vision; the width-zero arm is the comparison the prior
text-case sweep needed and lacked.

Draws come from the global torch RNG so that `train/checkpoint.py` captures
them, and a run resumed after a kill follows the same loop schedule it
would have followed uninterrupted. Using a private generator here would
silently break that.
"""

from __future__ import annotations

import math

import torch

K_MAX_TRAIN = 32
DEFAULT_SIGMA = 0.5
DEFAULT_K_MEAN = 8
RAMP_FRACTION = 0.4


def mu_at(step: int, total_steps: int, k_mean_target: int = DEFAULT_K_MEAN,
          ramp_fraction: float = RAMP_FRACTION) -> float:
    """Log-space centre of the loop-count distribution at a given step."""
    start, end = math.log(2.0), math.log(max(2.0, float(k_mean_target)))
    ramp_steps = max(1.0, ramp_fraction * total_steps)
    progress = min(1.0, step / ramp_steps)
    return start + (end - start) * progress


def sample_loop_count(
    step: int,
    total_steps: int,
    k_mean_target: int = DEFAULT_K_MEAN,
    sigma: float = DEFAULT_SIGMA,
    k_max: int = K_MAX_TRAIN,
    ramp_fraction: float = RAMP_FRACTION,
) -> int:
    """Draw this batch's loop count.

    ``sigma`` is the jitter width that H4 sweeps. **sigma = 0 is the
    deterministic arm**: the count becomes a fixed function of the step
    rather than merely a Poisson draw around a fixed mean. That
    distinction matters, because leaving the Poisson in place would still
    vary k from batch to batch and the "no jitter" condition would not
    actually be jitter free.
    """
    mu = mu_at(step, total_steps, k_mean_target, ramp_fraction)

    if sigma == 0.0:
        k = 1 + int(round(math.exp(mu)))
    else:
        tau = torch.exp(torch.randn(()) * sigma + mu)
        k = 1 + int(torch.poisson(tau.clamp(min=1e-6)).item())

    return max(1, min(k, k_max))


def expected_k(step: int, total_steps: int, k_mean_target: int = DEFAULT_K_MEAN,
               sigma: float = DEFAULT_SIGMA) -> float:
    """Mean loop count at a step, for logging and for sanity checks.

    E[k] = 1 + E[tau] and E[tau] = exp(mu + sigma^2 / 2) for a lognormal.
    """
    mu = mu_at(step, total_steps, k_mean_target)
    return 1.0 + math.exp(mu + sigma**2 / 2.0)


def eval_k_grid(k_max_eval: int = 64) -> list[int]:
    """Loop counts to evaluate at, for the M1 loop-count curve.

    Extends past the training k_max on purpose: H2's extrapolation arm asks
    what happens when a model trained at k_mean = 8 is run at 4x that and
    beyond, and the answer is only visible if we look there.
    """
    grid = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    return [k for k in grid if k <= k_max_eval]
