"""Milestone 1: families B and C.

Family B is the relational chain task, family C the counting task and the
H1 null arm. The shared properties (determinism, split disjointness, range
adherence) are parametrised over all three families in test_data.py. What
lives here is what is specific to the sprite scenes.
"""

from __future__ import annotations

import numpy as np
import pytest

from loopvision.data import dataset as D
from loopvision.data import family_b as FB
from loopvision.data import family_c as FC
from loopvision.data import render, scenes


# ---------------------------------------------------------------------------
# Sprites
# ---------------------------------------------------------------------------


def test_all_sprites_render_distinctly() -> None:
    """Every (colour, shape, size) must be visually distinguishable.

    Sixty combinations, and if any two collided the corresponding
    questions would be unanswerable from the image for reasons that have
    nothing to do with relations or counting.
    """
    tiles = {
        render.render_sprite(c, sh, sz).tobytes()
        for c in range(render.N_SPRITE_COLOURS)
        for sh in render.SHAPES
        for sz in render.SIZES
    }
    assert len(tiles) == render.N_SPRITE_COLOURS * len(render.SHAPES) * len(render.SIZES)


def test_sprites_are_patch_aligned() -> None:
    """One sprite, one patch token. The same guarantee family A gives."""
    sample = D.generate("C", D.global_index("breadth_ood", 2), "breadth_ood")
    bg = np.array(render.BACKGROUND, dtype=np.uint8).reshape(3, 1, 1)
    occupied = (sample.image != bg).any(axis=0)
    cells = {
        (y // render.PATCH, x // render.PATCH)
        for y in range(sample.image.shape[1])
        for x in range(sample.image.shape[2])
        if occupied[y, x]
    }
    assert len(cells) == sample.breadth


def test_adjacent_sprites_never_merge() -> None:
    """Every sprite leaves its last row and column clear.

    Two same-coloured sprites in neighbouring patches would otherwise
    render as one connected blob. That is merely ugly for family B and
    actively wrong for family C, whose entire task is counting them.
    Caught by eyeballing a dump: two adjacent yellow triangles had merged
    into a single shape.
    """
    bg = np.array(render.BACKGROUND, dtype=np.uint8).reshape(3, 1)
    for shape in render.SHAPES:
        for size in render.SIZES:
            tile = render.render_sprite(0, shape, size)
            assert (tile[:, -1, :] == bg).all(), f"{shape} {size} touches the bottom edge"
            assert (tile[:, :, -1] == bg).all(), f"{shape} {size} touches the right edge"


def test_family_a_glyphs_also_leave_a_gap() -> None:
    """The same guarantee, for the family A tetromino."""
    from loopvision.data import groups as G

    bg = np.array(render.BACKGROUND, dtype=np.uint8).reshape(3, 1)
    for i in range(G.GROUP_ORDER):
        tile = render.render_element(i)
        clear_row = (tile[:, -1, :] == bg).all() or (tile[:, 0, :] == bg).all()
        clear_col = (tile[:, :, -1] == bg).all() or (tile[:, :, 0] == bg).all()
        assert clear_row and clear_col, f"element {G.element_name(i)} fills its tile"


def test_sprites_never_overlap() -> None:
    rng = np.random.default_rng(3)
    for breadth in (4, 9, 16):
        placed = scenes.place_sprites(rng, breadth, "train")
        assert len({(s.row, s.col) for s in placed}) == breadth


# ---------------------------------------------------------------------------
# Family B: relational chains
# ---------------------------------------------------------------------------


def test_relations_are_inverses_of_each_other() -> None:
    """left_of and right_of must undo each other, and above and below.

    If they did not, the chain semantics would not be the ones the query
    tokens describe, and the task would be unanswerable in a way no other
    test would reveal.
    """
    rng = np.random.default_rng(5)
    for _ in range(200):
        placed = scenes.place_sprites(rng, 9, "train")
        for sprite in placed:
            for forward, backward in (("left_of", "right_of"), ("above", "below")):
                nxt = scenes.relate(placed, sprite, forward)
                if nxt is not None:
                    assert scenes.relate(placed, nxt, backward) == sprite


def test_chain_length_matches_depth() -> None:
    for split in sorted(FB.SUPPORTED_SPLITS):
        for i in range(120):
            s = D.generate("B", D.global_index(split, i), split)
            chain = s.program.split("chain=")[1].split(";")[0]
            hops = 0 if chain == "-" else len(chain.split(">"))
            assert hops == s.depth - 1, f"{split}[{i}]: {hops} hops at depth {s.depth}"


def test_chain_visits_distinct_sprites() -> None:
    """A chain that revisits a sprite is not really deep.

    Without the visited set, left_of followed by right_of returns to the
    anchor, so a depth 3 question is exactly as hard as a depth 1 question
    while still being labelled depth 3. The depth axis would then be part
    signal and part noise, and H1 would be fitting the mixture.
    """
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(400):
        placed = scenes.place_sprites(rng, 9, "train")
        anchors = scenes.unique_anchors(placed, "size")
        if not anchors:
            continue
        walked = scenes.walk_chain(rng, placed, anchors[0], 3)
        if walked is None:
            continue
        chain, target = walked
        current = anchors[0]
        visited = [(current.row, current.col)]
        for relation in chain:
            current = scenes.relate(placed, current, relation)
            visited.append((current.row, current.col))
        assert len(set(visited)) == len(visited), f"chain revisits: {visited}"
        assert (current.row, current.col) == (target.row, target.col)
        checked += 1
    assert checked > 100, f"only {checked} chains exercised"


def parse_spec(text: str) -> dict:
    """Parse an `attr:value+attr:value` descriptor from a program string."""
    out = {}
    for part in text.split("+"):
        a, _, v = part.partition(":")
        out[a] = int(v) if a == "colour" else v
    return out


def sprite_attrs(token: str) -> dict:
    """Recover a sprite's attributes from its scene token."""
    attrs, _, _ = token.partition("@")
    return {
        "colour": int(attrs[0]),
        "shape": next(sh for sh in render.SHAPES if sh[0] == attrs[1]),
        "size": next(z for z in render.SIZES if z[0] == attrs[2]),
    }


def test_anchor_is_unique_in_the_scene() -> None:
    """An ambiguous anchor makes the whole question ill-posed."""
    for split in sorted(FB.SUPPORTED_SPLITS):
        for i in range(100):
            s = D.generate("B", D.global_index(split, i), split)
            desc = parse_spec(s.program.split("anchor=")[1].split(";")[0])
            matching = sum(
                all(sprite_attrs(t)[a] == v for a, v in desc.items())
                for t in s.program.split("scene=")[1].split(",")
            )
            assert matching == 1, f"{split}[{i}]: anchor matches {matching} sprites"


def test_anchor_never_names_the_attribute_being_asked_for() -> None:
    """The leak that made depth 1 answerable without looking at the image.

    At depth 1 the target IS the anchor, so describing the anchor by the
    attribute under question writes the answer into the question. Measured
    before the fix: 66.8 percent of depth 1 samples were answerable from the
    query alone, and a trained model scored 0.833 on a blank image.
    """
    for split in sorted(FB.SUPPORTED_SPLITS):
        for i in range(200):
            s = D.generate("B", D.global_index(split, i), split)
            attribute = s.program.split("attr=")[1].split(";")[0]
            desc = parse_spec(s.program.split("anchor=")[1].split(";")[0])
            assert attribute not in desc, (
                f"{split}[{i}]: asked for {attribute} and the query names it"
            )
            assert len(desc) == 2, "an anchor needs both remaining attributes"


def test_label_agrees_with_the_recorded_target() -> None:
    for split in sorted(FB.SUPPORTED_SPLITS):
        for i in range(120):
            s = D.generate("B", D.global_index(split, i), split)
            attribute, value = FB.decode_label(s.label)
            assert attribute in D.ATTRIBUTES
            target = s.program.split("target=")[1].split(";")[0]
            if attribute == "colour":
                assert int(target[0]) == value
            elif attribute == "shape":
                assert target[1] == value[0]
            else:
                assert target[2] == value[0]


@pytest.mark.parametrize(
    "split,depth,breadth",
    [
        ("train", 1, 6),
        ("train", 2, 9),
        ("depth_ood", 3, 6),
        ("depth_ood", 4, 6),
        ("depth_ood", 4, 9),
        ("breadth_ood", 2, 13),
        ("combo_ood", 2, 6),
    ],
)
def test_rejection_rate_is_acceptable(split: str, depth: int, breadth: int) -> None:
    """Above 0.30 the sampler skews the scene distribution.

    Required by IMPLEMENTATION.md Section 4.3. A high rejection rate does
    not merely waste time: it biases the surviving scenes toward whatever
    happens to satisfy the constraint, which correlates scene structure
    with depth and quietly contaminates the H1 contrast.
    """
    rate = FB.rejection_rate(split, depth, breadth, trials=120)
    assert rate < 0.30, f"{split} d={depth} b={breadth} rejects {rate:.3f}"


def test_structurally_empty_cells_are_refused_not_retried() -> None:
    """depth > breadth cannot be sampled, and must say so rather than hang."""
    with pytest.raises(ValueError, match="structurally empty"):
        FB.sample_scene(np.random.default_rng(0), depth=5, breadth=4, split="train")


def test_depth_and_breadth_ranges_are_crossed_not_confounded() -> None:
    """train and depth_ood must share one breadth range.

    If the deeper split also used broader scenes, depth and breadth would
    move together and H1 could not separate them. This is the single most
    important structural property of the split design.
    """
    train = D.split_spec("B", "train")
    deeper = D.split_spec("B", "depth_ood")
    assert train.breadth == deeper.breadth
    assert set(train.depth).isdisjoint(deeper.depth)

    broader = D.split_spec("B", "breadth_ood")
    assert broader.depth == train.depth
    assert set(broader.breadth).isdisjoint(train.breadth)


# ---------------------------------------------------------------------------
# Family C: counting, the H1 null arm
# ---------------------------------------------------------------------------


def test_label_is_the_true_count() -> None:
    for split in sorted(FC.SUPPORTED_SPLITS):
        for i in range(150):
            s = D.generate("C", D.global_index(split, i), split)
            spec = parse_spec(s.program.split("query=")[1].split(";")[0])
            counted = sum(
                all(sprite_attrs(t)[a] == v for a, v in spec.items())
                for t in s.program.split("scene=")[1].split(",")
            )
            assert s.label == min(counted, FC.COUNT_CAP)


def test_family_c_query_names_one_or_two_attributes() -> None:
    """Never three. A three-way conjunction almost never matches anything."""
    for i in range(200):
        s = D.generate("C", D.global_index("train", i), "train")
        spec = parse_spec(s.program.split("query=")[1].split(";")[0])
        assert 1 <= len(spec) <= 2, f"query names {len(spec)} attributes"


def test_family_c_is_depth_one_everywhere() -> None:
    """Family C is the H1 null arm. If it had depth it would not be one."""
    for split in sorted(FC.SUPPORTED_SPLITS):
        for i in range(80):
            assert D.generate("C", D.global_index(split, i), split).depth == 1


def test_family_c_has_no_depth_ood_split() -> None:
    """Depth 1 by construction, so there is nothing to extrapolate along."""
    with pytest.raises(ValueError, match="depth_ood"):
        D.generate("C", 0, "depth_ood")


def test_family_c_count_scales_with_breadth() -> None:
    """The property that makes family C a breadth stressor at all.

    An earlier version of this test only asserted that fewer than 75 percent
    of labels were zero and that three distinct counts appeared. It passed
    while the task was 93 percent binary and the count never exceeded 4,
    because a rare conjunction barely responds to scene size. Family C
    exists to stress breadth, so the thing to assert is that the answer
    actually moves when breadth does.
    """
    means = {}
    for breadth, split in ((4, "train"), (8, "train"), (16, "breadth_ood")):
        cfg = D.TaskConfig(breadths=(breadth,))
        labels = [
            D.generate("C", D.global_index(split, i), split, cfg).label
            for i in range(800)
        ]
        means[breadth] = sum(labels) / len(labels)
    assert means[4] < means[8] < means[16], f"count does not track breadth: {means}"
    assert means[16] > 2 * means[4], f"count barely responds to breadth: {means}"


def test_family_c_is_not_a_presence_detector() -> None:
    """No constant answer should score well, and the range must be used."""
    cfg = D.TaskConfig(breadths=(16,))
    labels = [
        D.generate("C", D.global_index("breadth_ood", i), "breadth_ood", cfg).label
        for i in range(1500)
    ]
    import collections

    dist = collections.Counter(labels)
    best_constant = dist.most_common(1)[0][1] / len(labels)
    assert best_constant < 0.35, (
        f"always answering {dist.most_common(1)[0][0]} scores {best_constant:.1%}"
    )
    assert len(dist) >= 6, f"only {len(dist)} distinct counts at breadth 16"
    assert max(labels) >= 6, f"the count never exceeds {max(labels)} of a cap of 10"


def test_family_c_count_is_capped() -> None:
    for i in range(400):
        s = D.generate("C", D.global_index("breadth_ood", i), "breadth_ood")
        assert 0 <= s.label <= FC.COUNT_CAP


# ---------------------------------------------------------------------------
# combo_ood
# ---------------------------------------------------------------------------


def test_combo_ood_contains_held_out_pairings() -> None:
    """An OOD split that is not out of distribution is worse than none."""
    for family in ("B", "C"):
        for i in range(60):
            s = D.generate(family, D.global_index("combo_ood", i), "combo_ood")
            present = set()
            for token in s.program.split("scene=")[1].split(","):
                attrs, _, _ = token.partition("@")
                shape = next(sh for sh in render.SHAPES if sh[0] == attrs[1])
                present.add((int(attrs[0]), shape))
            assert present & scenes.HELDOUT_PAIRS, (
                f"{family} combo_ood[{i}] contains no held-out pairing"
            )


def test_training_splits_never_leak_held_out_pairings() -> None:
    """Otherwise combo_ood is not out of distribution at all."""
    for family in ("B", "C"):
        for split in ("train", "iid_val", "breadth_ood"):
            for i in range(60):
                s = D.generate(family, D.global_index(split, i), split)
                for token in s.program.split("scene=")[1].split(","):
                    attrs, _, _ = token.partition("@")
                    shape = next(sh for sh in render.SHAPES if sh[0] == attrs[1])
                    assert (int(attrs[0]), shape) not in scenes.HELDOUT_PAIRS, (
                        f"{family} {split}[{i}] leaks a held-out pairing"
                    )
