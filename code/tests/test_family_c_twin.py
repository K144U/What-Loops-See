"""Family C's M2 counterfactual moves the count by one, at one location.

Family C is the breadth arm, and it is the family that turns H3 on. H3
predicts breadth tasks are loop-diffuse and spatially local, and "spatially
local" is only testable if the corruption sits at a single place in the
image. So the twin edits one attribute of one sprite and leaves position
alone, and these tests pin both halves: exactly one sprite differs, and the
pixels differ in a bounded region rather than everywhere.

Editing the query instead would have been easier and would have made the
locality prediction untestable, because changing what is being counted
changes which sprites matter everywhere at once.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from loopvision.data import dataset as D
from loopvision.data import family_c as FC

SPLIT = "iid_val"


def items(n: int):
    cfg = D.TaskConfig()
    for i in range(n):
        gi = D.global_index(SPLIT, i)
        clean = D.generate("C", gi, SPLIT, cfg)
        yield clean, FC.corrupted_twin(clean, cfg, split=SPLIT), cfg


def scene_set(program: str) -> set[str]:
    return set(program.split("scene=")[1].split(","))


def test_the_replay_guard_is_live():
    """Feed it a sample whose image and index disagree, and it must refuse."""
    cfg = D.TaskConfig()
    clean = D.generate("C", D.global_index(SPLIT, 0), SPLIT, cfg)
    other = D.generate("C", D.global_index(SPLIT, 11), SPLIT, cfg)
    FC.corrupted_twin(clean, cfg, split=SPLIT)  # the real thing replays fine
    if np.array_equal(other.image, clean.image):
        pytest.skip("indices 0 and 11 render identically")
    with pytest.raises(ValueError, match="drifted"):
        FC.corrupted_twin(replace(clean, image=other.image), cfg, split=SPLIT)


def test_exactly_one_sprite_differs():
    checked = 0
    for clean, twin, _ in items(60):
        if twin is None:
            continue
        a, b = scene_set(clean.program), scene_set(twin.program)
        assert len(a - b) == 1 and len(b - a) == 1, (
            f"{len(a - b)} sprites removed and {len(b - a)} added, expected "
            f"exactly one changed"
        )
        checked += 1
    assert checked > 0


def test_the_count_moves_by_exactly_one():
    checked = 0
    for clean, twin, _ in items(60):
        if twin is None:
            continue
        assert abs(twin.label - clean.label) == 1, (
            f"count moved by {twin.label - clean.label}, but flipping one "
            f"sprite's match status can only move it by one"
        )
        checked += 1
    assert checked > 0


def test_the_edit_is_spatially_local():
    """The property H3 needs. Pixels differ in one region, not everywhere.

    A sprite occupies a small part of the canvas, so a one sprite edit must
    leave the great majority of the image untouched. Without this the
    locality half of H3 cannot be read off the heatmap at all.
    """
    checked = 0
    for clean, twin, _ in items(60):
        if twin is None:
            continue
        differing = np.abs(twin.image.astype(int) - clean.image.astype(int)).sum(axis=0)
        frac = float((differing > 0).mean())
        assert 0 < frac < 0.25, (
            f"{frac:.1%} of the canvas changed. Zero means the edit is "
            f"invisible, and a quarter means it is not a one sprite edit"
        )
        checked += 1
    assert checked > 0


def test_the_query_is_untouched():
    """What is being counted must not change, only the scene."""
    checked = 0
    for clean, twin, _ in items(60):
        if twin is None:
            continue
        assert np.array_equal(np.asarray(clean.query), np.asarray(twin.query))
        checked += 1
    assert checked > 0


def test_the_cap_is_currently_out_of_reach():
    """Pins an assumption the twin's last guard depends on.

    `corrupted_twin` refuses when the edited count equals the original. That
    can only happen when COUNT_CAP truncates a difference away, and measured
    over 800 items the largest count the sampler produces is 7 against a cap
    of 10. So that branch is unreachable and no mutation of it can be
    detected, which is why it has no behavioural test.

    If the breadth range or the query subsets change, counts can reach the
    cap and this fails, which is the notice that the guard has become live
    and now needs one.
    """
    cfg = D.TaskConfig()
    labels = [
        D.generate("C", D.global_index(SPLIT, i), SPLIT, cfg).label for i in range(400)
    ]
    assert max(labels) < FC.COUNT_CAP, (
        f"counts now reach {max(labels)} against a cap of {FC.COUNT_CAP}, so the "
        f"cap guard in corrupted_twin is reachable and needs a real test"
    )
