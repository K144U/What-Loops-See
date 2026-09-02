"""The shared capture and injection mechanism, IMPLEMENTATION.md Section 7.

Every instrument reads and writes recurrent state through `hooks.py`, so a
bug here is a bug in M2, M3 and M4 simultaneously, and it would be silent:
the model still trains, the heatmaps still render, and they describe a
computation that never happened.

The properties worth holding are that patching does what it says (that
position, that iteration, nothing else), that the batched position scan
agrees with the obvious slow version, and that the recovery metric is
anchored correctly at its two ends.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from loopvision.data import dataset as D
from loopvision.instruments import hooks
from loopvision.model.loopvit import LoopViT, LoopViTConfig


def build(**kw) -> LoopViT:
    torch.manual_seed(0)
    kw.setdefault("state_init", "zeros")
    return LoopViT(LoopViTConfig(d_model=64, vocab_size=D.VOCAB_SIZE, **kw)).eval()


def batch(n: int = 1):
    torch.manual_seed(1)
    return torch.randn(n, 3, 32, 32), torch.zeros(n, 8, dtype=torch.long)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def test_capture_indexes_states_by_core_applications() -> None:
    """states[i] is the state after i core steps. states[0] is the input state."""
    model = build()
    image, query = batch()
    logits, states = hooks.capture(model, image, query, k=5)
    assert len(states) == 6
    assert torch.allclose(logits, model.decode(states[-1]), atol=1e-5)


def test_hook_indices_match_the_states_list() -> None:
    """A hook at index i must see exactly the state capture calls states[i]."""
    model = build()
    image, query = batch()
    _, states = hooks.capture(model, image, query, k=4)

    seen: dict[int, torch.Tensor] = {}

    def spy(i, s):
        seen[i] = s.clone()
        return None

    with torch.no_grad():
        model(image, query, k=4, state_hook=spy)
    assert sorted(seen) == [0, 1, 2, 3, 4]
    for i, s in seen.items():
        assert torch.allclose(s, states[i], atol=1e-6), f"hook index {i} misaligned"


def test_shared_init_state_makes_two_runs_comparable() -> None:
    """Without a shared s_0 a counterfactual pair differs in two ways."""
    model = build(state_init="randn")
    image, query = batch()
    s0 = hooks.shared_init_state(model, image, query)
    a, _ = hooks.capture(model, image, query, k=3, s0=s0)
    b, _ = hooks.capture(model, image, query, k=3, s0=s0)
    assert torch.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


def test_patch_replaces_only_the_named_positions_and_iteration() -> None:
    model = build()
    image, query = batch()
    _, donor = hooks.capture(model, image, query, k=4)
    donor = [torch.randn_like(s) for s in donor]  # obviously distinct donor

    captured: dict[int, torch.Tensor] = {}

    def spy(i, s):
        captured[i] = s.clone()
        return None

    hook = hooks.compose_hooks(hooks.patch_hook(donor, iteration=2, positions=[5, 9]), spy)
    with torch.no_grad():
        model(image, query, k=4, state_hook=hook)

    patched_row = captured[2]
    for pos in (5, 9):
        assert torch.allclose(patched_row[:, pos, :], donor[2][:, pos, :], atol=1e-6)
    untouched = [p for p in range(model.cfg.seq_len) if p not in (5, 9)]
    assert not torch.allclose(
        patched_row[:, untouched, :], donor[2][:, untouched, :], atol=1e-3
    ), "positions outside the patch were also overwritten"


def test_patching_changes_the_output() -> None:
    """A patch that changes nothing measures nothing."""
    model = build()
    image, query = batch()
    _, donor = hooks.capture(model, image, query, k=4)
    donor = [torch.randn_like(s) for s in donor]
    with torch.no_grad():
        base = model(image, query, k=4)
        hooked = model(
            image, query, k=4,
            state_hook=hooks.patch_hook(donor, iteration=1, positions=[0, 1, 2]),
        )
    assert not torch.allclose(base, hooked, atol=1e-5)


def test_patch_at_the_final_state_still_reaches_the_coda() -> None:
    """Index k is what the coda decodes, so patching there must matter."""
    model = build()
    image, query = batch()
    _, donor = hooks.capture(model, image, query, k=3)
    donor = [torch.randn_like(s) for s in donor]
    with torch.no_grad():
        base = model(image, query, k=3)
        hooked = model(
            image, query, k=3,
            state_hook=hooks.patch_hook(donor, iteration=3, positions=list(range(10))),
        )
    assert not torch.allclose(base, hooked, atol=1e-5)


def test_position_scan_matches_patching_one_position_at_a_time() -> None:
    """The batched shortcut must equal the obvious slow version.

    The scan collapses the 64-position axis into the batch to turn the M2
    grid from k*64 forward passes into k. If it disagreed with the slow
    path, every heatmap would be wrong and nothing would say so.
    """
    model = build()
    image, query = batch()
    positions = [0, 3, 17, 64, 70]
    _, donor_single = hooks.capture(model, image, query, k=3)
    donor_single = [torch.randn_like(s) for s in donor_single]

    slow = []
    with torch.no_grad():
        for p in positions:
            slow.append(
                model(image, query, k=3,
                      state_hook=hooks.patch_hook(donor_single, 2, [p]))
            )
    slow = torch.cat(slow, dim=0)

    n = len(positions)
    rep_image = image.expand(n, -1, -1, -1).contiguous()
    rep_query = query.expand(n, -1).contiguous()
    rep_donor = [s.expand(n, -1, -1).contiguous() for s in donor_single]
    with torch.no_grad():
        fast = model(
            rep_image, rep_query, k=3,
            state_hook=hooks.position_scan_hook(rep_donor, 2, positions),
        )
    assert torch.allclose(slow, fast, atol=1e-5)


def test_position_scan_refuses_a_batch_mismatch() -> None:
    """A silent broadcast would patch the wrong positions and still look fine."""
    model = build()
    image, query = batch(2)
    _, donor = hooks.capture(model, image, query, k=2)
    with pytest.raises(ValueError, match="one replica per"):
        with torch.no_grad():
            model(image, query, k=2,
                  state_hook=hooks.position_scan_hook(donor, 1, [0, 1, 2]))


def test_donor_shape_mismatch_is_refused() -> None:
    model = build()
    image, query = batch()
    _, donor = hooks.capture(model, image, query, k=2)
    bad = [torch.randn(1, 5, 64) for _ in donor]
    with pytest.raises(ValueError, match="donor state"):
        with torch.no_grad():
            model(image, query, k=2, state_hook=hooks.patch_hook(bad, 1, [0]))


# ---------------------------------------------------------------------------
# The recovery metric
# ---------------------------------------------------------------------------


def test_recovery_is_anchored_at_zero_and_one() -> None:
    scale = hooks.LogitDiff(clean=4.0, corrupted=-2.0)
    assert scale.recovered(-2.0) == pytest.approx(0.0)
    assert scale.recovered(4.0) == pytest.approx(1.0)
    assert scale.recovered(1.0) == pytest.approx(0.5)


def test_recovery_is_nan_when_the_twin_moved_nothing() -> None:
    """Undefined, not zero.

    Returning 0.0 would fill a heatmap cell with a number meaning "no
    recovery" when the truth is "this counterfactual measured nothing",
    and the two are not the same claim.
    """
    scale = hooks.LogitDiff(clean=1.0, corrupted=1.0)
    assert np.isnan(scale.recovered(1.0))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_recovery_grid_has_the_right_shape_and_endpoints() -> None:
    """Patching every position at the last iteration must fully recover.

    Overwriting the entire final state with the clean one makes the coda
    see exactly what the clean run gave it, so recovery there is 1 by
    construction. That is the strongest available check that the grid is
    wired up the right way round.
    """
    model = build()
    clean_image, query = batch()
    torch.manual_seed(99)
    corrupt_image = torch.randn_like(clean_image)

    k = 3
    grid = hooks.recovery_grid(
        model, clean_image, corrupt_image, query,
        clean_label=0, corrupt_label=1, k=k,
    )
    assert grid.shape == (k + 1, model.cfg.seq_len)

    full = hooks.recovery_grid(
        model, clean_image, corrupt_image, query,
        clean_label=0, corrupt_label=1, k=k,
        positions=list(range(model.cfg.seq_len)),
    )
    assert torch.allclose(grid, full, atol=1e-5, equal_nan=True)

    # Row 0 must be exactly zero: both runs share s_0, so patching the
    # initial state from the donor is patching it with itself.
    assert torch.allclose(grid[0], torch.zeros_like(grid[0]), atol=1e-6)


def test_recovery_endpoints_are_exactly_zero_and_one() -> None:
    """The strongest available check that the grid is wired the right way round.

    Overwriting the entire final state with the clean one makes the coda
    see exactly what the clean run gave it, so recovery must be 1. Running
    the corrupted item untouched must give 0. If the donor and recipient
    were swapped, or the metric inverted, these two would come out
    reversed and every heatmap would be a mirror of the truth.
    """
    model = build()
    clean_image, query = batch()
    torch.manual_seed(99)
    corrupt_image = torch.randn_like(clean_image)
    k = 4

    s0 = hooks.shared_init_state(model, clean_image, query)
    clean_logits, clean_states = hooks.capture(model, clean_image, query, k, s0=s0)
    corrupt_logits, _ = hooks.capture(model, corrupt_image, query, k, s0=s0)
    scale = hooks.LogitDiff(
        clean=float(hooks.logit_difference(clean_logits, 0, 1)[0]),
        corrupted=float(hooks.logit_difference(corrupt_logits, 0, 1)[0]),
    )

    everywhere = list(range(model.cfg.seq_len))
    with torch.no_grad():
        fully = model(
            corrupt_image, query, k=k, s0=s0,
            state_hook=hooks.patch_hook(clean_states, k, everywhere),
        )
        untouched = model(corrupt_image, query, k=k, s0=s0)

    assert scale.recovered(float(hooks.logit_difference(fully, 0, 1)[0])) == pytest.approx(1.0, abs=1e-5)
    assert scale.recovered(float(hooks.logit_difference(untouched, 0, 1)[0])) == pytest.approx(0.0, abs=1e-5)
