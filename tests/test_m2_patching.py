"""M2's positive control is an identity, so it can be tested without training.

Patching every position at the last iteration hands the coda exactly the
clean state. The coda and head are deterministic functions of that state,
so the recovered logit difference must be the clean one and recovery must
be exactly 1. That holds for an untrained model with random weights, which
means it is a unit test rather than an experiment.

This exists because the first version of the control discarded the hook.
`patched()` yields a hook to hand to `forward(state_hook=...)` and does not
mutate the model, so a `with patched(...):` that ignores the yielded value
patches nothing. The control then read 0.0000 at every iteration, which
looks exactly like a real negative result and was very nearly reported as
one. See L-022.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from loopvision.instruments.m2_patching import full_state_control, item_grid
from loopvision.model.loopvit import LoopViT, LoopViTConfig


class FakeSample:
    def __init__(self, image, query, label):
        self.image = image
        self.query = query
        self.label = label


def tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = LoopViTConfig(
        d_model=64,
        prelude_blocks=1,
        core_blocks=1,
        coda_blocks=1,
        num_classes=8,
        vocab_size=16,
        query_len=2,
    )
    return LoopViT(cfg).eval()


def pair(seed: int = 0):
    rng = np.random.default_rng(seed)
    canvas = LoopViTConfig().canvas
    a = rng.integers(0, 255, size=(3, canvas, canvas), dtype=np.int64).astype(np.uint8)
    b = rng.integers(0, 255, size=(3, canvas, canvas), dtype=np.int64).astype(np.uint8)
    q = rng.integers(0, 15, size=(2,)).astype(np.int32)
    return FakeSample(a, q, 1), FakeSample(b, q, 3)


def test_control_recovers_exactly_one_at_the_last_iteration():
    """The identity. If this is not 1, the patch never reached the model."""
    model = tiny_model()
    clean, corrupt = pair()
    rec = full_state_control(model, clean, corrupt, k=4, device="cpu")
    assert len(rec) == 5
    assert rec[-1] == pytest.approx(1.0, abs=1e-4), (
        f"patching every position at the last iteration recovered {rec[-1]}, "
        f"not 1. The coda sees the clean state by construction, so anything "
        f"else means the hook is not being applied"
    )


def test_control_is_not_uniformly_one():
    """And earlier iterations must NOT be 1, or the hook ignores `iteration`.

    A hook that patched at every iteration regardless of which one was asked
    for would also pass the test above, and would make the whole heatmap
    meaningless in the opposite direction.
    """
    model = tiny_model()
    clean, corrupt = pair()
    rec = full_state_control(model, clean, corrupt, k=4, device="cpu")
    assert not all(r == pytest.approx(1.0, abs=1e-4) for r in rec[:-1]), (
        "every iteration recovered fully, so the patch is not respecting the "
        "iteration it was asked to act on"
    )


def test_a_twin_that_does_not_change_the_label_is_rejected():
    """Recovery is a ratio whose denominator is the clean-corrupt gap."""
    model = tiny_model()
    clean, corrupt = pair()
    corrupt.label = clean.label
    assert item_grid(model, clean, corrupt, k=4, device="cpu") is None
    assert full_state_control(model, clean, corrupt, k=4, device="cpu") == []


def test_grid_has_one_row_per_iteration_and_one_column_per_position():
    model = tiny_model()
    clean, corrupt = pair()
    g = item_grid(model, clean, corrupt, k=3, device="cpu")
    assert g.shape == (4, model.cfg.seq_len)
