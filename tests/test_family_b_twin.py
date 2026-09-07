"""Family B's M2 counterfactual edits one relation hop and nothing else.

`corrupted_twin` regenerates the scene by replaying the generator's random
stream from the sample index, because a scene is a pure function of that
index and `scenes.scene_program` has no parser. That replay duplicates the
opening lines of `generate`, and duplicated code drifts. The first test
here is the guard: if the replay and the generator ever disagree, the
regenerated image stops matching and this fails, rather than the twin
quietly patching against a scene the model never saw.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from loopvision.data import dataset as D
from loopvision.data import family_b as FB
from loopvision.data import scenes

SPLIT = "iid_val"


def items(n: int):
    cfg = D.TaskConfig()
    for i in range(n):
        gi = D.global_index(SPLIT, i)
        clean = D.generate("B", gi, SPLIT, cfg)
        yield clean, FB.corrupted_twin(clean, cfg, split=SPLIT), cfg


def test_the_replayed_scene_matches_the_generator():
    """The drift guard, and it must not be checkable by construction.

    An earlier version asserted `twin.image == clean.image`, which is true
    however wrong the replay is, because the twin reuses the original array.
    The binding check is inside `corrupted_twin`: it renders the replayed
    sprites and compares. This test confirms that check is live by feeding
    it a sample whose index points at a different scene.
    """
    cfg = D.TaskConfig()
    gi = D.global_index(SPLIT, 0)
    clean = D.generate("B", gi, SPLIT, cfg)

    # Replay must succeed on the real thing.
    FB.corrupted_twin(clean, cfg, split=SPLIT)

    # And must refuse when the image and the index disagree. dataclasses are
    # frozen, so build the mismatch rather than mutating.
    other = D.generate("B", D.global_index(SPLIT, 7), SPLIT, cfg)
    mismatched = replace(clean, image=other.image)
    if np.array_equal(other.image, clean.image):
        pytest.skip("indices 0 and 7 render identically, no mismatch to test")
    with pytest.raises(ValueError, match="drifted"):
        FB.corrupted_twin(mismatched, cfg, split=SPLIT)


def test_exactly_one_relation_token_changes():
    """One hop, not two, and nothing outside the relation block."""
    checked = 0
    for clean, twin, _ in items(40):
        if twin is None:
            continue
        a = np.asarray(clean.query)
        b = np.asarray(twin.query)
        assert a.shape == b.shape
        differing = np.flatnonzero(a != b)
        assert len(differing) == 1, (
            f"{len(differing)} query tokens changed, expected exactly one "
            f"relation hop"
        )
        # and the changed token must be a relation, not an attribute or anchor
        assert a[differing[0]] >= D.REL_BASE
        assert b[differing[0]] >= D.REL_BASE
        checked += 1
    assert checked > 0


def test_the_edited_chain_still_resolves():
    """A hop into empty space is not a harder question, it is not a question."""
    cfg = D.TaskConfig()
    checked = 0
    for i in range(40):
        gi = D.global_index(SPLIT, i)
        clean = D.generate("B", gi, SPLIT, cfg)
        twin = FB.corrupted_twin(clean, cfg, split=SPLIT)
        if twin is None:
            continue
        chain = twin.program.split("chain=")[1].split(";")[0]
        assert chain and chain != "-"
        for r in chain.split(">"):
            assert r in scenes.RELATIONS
        checked += 1
    assert checked > 0


def test_depth_one_has_no_hop_to_edit():
    """Depth 1 is the anchor alone, so there is no counterfactual."""
    cfg = D.TaskConfig(depths=(1,))
    seen_depth_one = 0
    for i in range(20):
        gi = D.global_index(SPLIT, i)
        clean = D.generate("B", gi, SPLIT, cfg)
        if clean.depth != 1:
            continue
        seen_depth_one += 1
        assert FB.corrupted_twin(clean, cfg, split=SPLIT) is None
    assert seen_depth_one > 0, "no depth 1 items generated, test proved nothing"
