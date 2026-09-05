"""The same group acting on geometry or on appearance.

The native glyph confounds two things: D4 moves the figure and S3 recolours
it, so "D4 is learnable and S3 is not" could be about the groups or about
geometry versus appearance. These two substrates separate them.

The design has one trap and these tests exist mostly to hold it shut. If a
group permutes three slots and three colours sit in bijection with those
slots, then moving contents between slots and relabelling colours in place
generate the same set of images, differing only by element versus inverse.
Inverse is a bijection, so the two tasks are isomorphic, equally hard, and
the experiment measures nothing while looking like it measured something.
"""

from __future__ import annotations

import pytest

from loopvision.data import groups as G


@pytest.mark.parametrize("name,order", [("s3_spatial", 6), ("d4_colour", 8)])
def test_the_action_does_not_collapse(name: str, order: int) -> None:
    """Every element must give a distinct picture.

    A collapsed action makes the task impossible, and the model failing it
    would be read as evidence about the model. This is the L-tromino
    failure and it is the first thing to check on any new glyph.
    """
    assert G.substrate_states(name) == order


@pytest.mark.parametrize("name", ["s3_spatial", "d4_colour"])
def test_the_action_is_faithful(name: str) -> None:
    action, sub = G.SUBSTRATE_ACTIONS[name]
    members = G.subgroup_members(sub)
    pictures = {action(g) for g in members}
    assert len(pictures) == len(members)


def test_s3_spatial_carries_no_colour_at_all() -> None:
    """If any real colour appeared, the task would be partly an appearance
    task and the contrast against the colour substrate would be blunted."""
    colours = {
        c for g in G.subgroup_members("s3") for _, _, c in G.act_s3_spatial(g)
    }
    assert colours == {G.NEUTRAL_COLOUR}, f"colour leaked in: {colours}"


def test_s3_spatial_varies_in_shape() -> None:
    """The whole point: the group changes the figure's geometry."""
    shapes = {
        frozenset((x, y) for x, y, _ in G.act_s3_spatial(g))
        for g in G.subgroup_members("s3")
    }
    assert len(shapes) == 6, f"only {len(shapes)} distinct shapes"


def test_d4_colour_never_moves_anything() -> None:
    """The appearance counterpart. Geometry must be constant across all
    eight elements, or the substrate is not purely appearance and the two
    arms of the crossing are not comparable."""
    shapes = {
        frozenset((x, y) for x, y, _ in G.act_d4_colour(g))
        for g in G.subgroup_members("d4")
    }
    assert len(shapes) == 1, f"geometry varies across {len(shapes)} shapes"


def test_d4_colour_varies_in_colour() -> None:
    colourings = {
        tuple(sorted(G.act_d4_colour(g))) for g in G.subgroup_members("d4")
    }
    assert len(colourings) == 8


def test_the_two_substrates_are_not_the_same_task_relabelled() -> None:
    """The trap, checked directly.

    One substrate varies only geometry and the other only colour, so no
    relabelling can carry one onto the other: they do not even share a
    feature to relabel. Stated as an assertion because "obviously
    different" is how the isomorphic version would have been shipped.
    """
    spatial_shapes = {
        frozenset((x, y) for x, y, _ in G.act_s3_spatial(g))
        for g in G.subgroup_members("s3")
    }
    colour_shapes = {
        frozenset((x, y) for x, y, _ in G.act_d4_colour(g))
        for g in G.subgroup_members("d4")
    }
    assert len(spatial_shapes) > 1 and len(colour_shapes) == 1, (
        "one substrate must vary geometry alone and the other colour alone"
    )


def test_the_corner_action_is_faithful_which_is_why_four_labels_suffice() -> None:
    """D4 has order 8 and acts on only four corners. If that action had a
    kernel, four colour labels could not carry the group and the appearance
    arm would be impossible rather than hard."""
    perms = {G.d4_corner_permutation(g) for g in G.subgroup_members("d4")}
    assert len(perms) == G.D4_ORDER, (
        f"the corner action collapses to {len(perms)} of {G.D4_ORDER}, so "
        f"four labels cannot represent D4"
    )
    for p in perms:
        assert sorted(p) == [0, 1, 2, 3], f"{p} is not a permutation of four corners"


def test_the_corner_permutation_composes_like_the_group() -> None:
    """A picture sequence only means anything if the action is a genuine
    homomorphism: applying two elements must equal applying their product."""
    members = G.subgroup_members("d4")
    for a in members:
        for b in members:
            pa = G.d4_corner_permutation(a)
            pb = G.d4_corner_permutation(b)
            composed = tuple(pa[pb[i]] for i in range(4))
            assert composed == G.d4_corner_permutation(G.multiply(a, b)), (
                "the corner action is not a homomorphism, so composing "
                "pictures does not correspond to composing group elements"
            )
