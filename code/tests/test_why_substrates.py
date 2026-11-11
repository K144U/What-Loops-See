"""E1 and E2 change only how D4's elements are drawn. Each test is a way the
new drawings could fail to be the experiment they claim to be.

A drawing that collapses two elements makes the task impossible. E1a is only
a test of the aspect-ratio account if aspect ratio truly carries nothing.
E1b is only a test if it is not a group action in disguise, and if no tile
feature still marks a character. E2 is only a test if the cue and the decoy
differ in nothing but whether the pixel composes.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter

import pytest

from loopvision.data import dataset as D
from loopvision.data import family_a as FA
from loopvision.data import groups as G
from loopvision.data import render

NEW = ("d4_square", "d4_scrambled", "d4_colour_cue", "d4_colour_decoy")
D4 = list(G.subgroup_members("d4"))
CH = G.D4_CHARACTERS


def _shape(cells):
    return frozenset((x, y) for x, y, _ in cells)


def _box(cells):
    return (max(x for x, _, _ in cells) + 1, max(y for _, y, _ in cells) + 1)


# --- every new drawing ---------------------------------------------------------

@pytest.mark.parametrize("name", NEW)
def test_no_two_elements_share_a_picture(name) -> None:
    assert G.substrate_states(name) == 8


@pytest.mark.parametrize("name", NEW)
def test_every_picture_fits_one_patch(name) -> None:
    for g in D4:
        assert render.render_element(g, substrate=name).shape == (3, 4, 4)


@pytest.mark.parametrize("name", NEW)
def test_the_substrate_reaches_the_pixels_and_labels_stay_in_d4(name) -> None:
    """The L-017 failure: a substrate that never reaches the renderer."""
    cfg = D.TaskConfig(substrate=name, depths=(2,), breadths=(1,))
    s = FA.generate(D.global_index("train", 0), "train", cfg)
    want = render.render_element(FA.parse_program(s.program)[0][0], substrate=name)
    assert (s.image[:, 0:4, 0:4] == want).all()
    labels = {FA.generate(D.global_index("train", i), "train", cfg).label for i in range(200)}
    assert labels == set(D4)


# --- E1a, the square glyph -------------------------------------------------------

def test_e1a_is_a_rigid_motion_with_eight_distinct_silhouettes() -> None:
    assert len({_shape(G.act_d4_square(g)) for g in D4}) == 8


def test_e1a_aspect_ratio_carries_nothing() -> None:
    assert {_box(G.act_d4_square(g)) for g in D4} == {(3, 3)}


def test_the_native_glyph_is_what_e1a_removes() -> None:
    """The positive control: on the native J aspect ratio IS rotflip."""
    tall = {g: _box(G.act_on_glyph(g))[1] > _box(G.act_on_glyph(g))[0] for g in D4}
    assert all(tall[g] == (CH["rotflip"](g) == 0) for g in D4)


# --- E1b, the scramble -------------------------------------------------------------

def test_e1b_uses_exactly_the_native_tiles() -> None:
    assert {G.act_d4_scrambled(g) for g in D4} == {G.act_on_glyph(g) for g in D4}


def test_e1b_moves_every_element() -> None:
    assert all(G.act_d4_scrambled(g) != G.act_on_glyph(g) for g in D4)


def _drawn_as() -> dict[int, int]:
    """Which native tile each element is actually rendered as, read from the
    pictures. Read from D4_SCRAMBLE instead, these tests passed against a
    renderer that ignored it, found by mutation."""
    native = {G.act_on_glyph(h): h for h in D4}
    return {g: native[G.act_d4_scrambled(g)] for g in D4}


def test_no_tile_feature_marks_a_character_under_e1b() -> None:
    """A tile's own characters (its aspect ratio among them) against the
    characters of the element it stands for: never equal, never opposite,
    at most 6 of 8 agreement."""
    target = _drawn_as()
    for f in CH.values():
        for c in CH.values():
            agree = sum(f(target[g]) == c(g) for g in D4)
            assert 2 <= agree <= 6


def _automorphisms():
    out = []
    rest = [g for g in D4 if g != G.IDENTITY]
    for perm in itertools.permutations(rest):
        phi = dict(zip(rest, perm))
        phi[G.IDENTITY] = G.IDENTITY
        if all(phi[G.multiply(a, b)] == G.multiply(phi[a], phi[b]) for a in D4 for b in D4):
            out.append(phi)
    return out


def test_e1b_is_not_a_group_action_in_disguise() -> None:
    """If the scramble were an affine map of D4, the scrambled pictures would
    still compose like the group, and E1b would be the native task relabelled."""
    scr = _drawn_as()
    autos = _automorphisms()
    assert len(autos) == 8
    for phi in autos:
        for a in D4:
            assert any(scr[g] != G.multiply(a, phi[g]) for g in D4)
            assert any(scr[g] != G.multiply(phi[g], a) for g in D4)


# --- E2, the cue and the decoy -------------------------------------------------------

def _bits(action):
    return {g: int((*G.CUE_PIXEL, G.NEUTRAL_COLOUR) in action(g)) for g in D4}


def test_the_cue_shows_rotflip() -> None:
    assert _bits(G.act_d4_colour_cue) == {g: CH["rotflip"](g) for g in D4}


def test_cue_and_decoy_differ_only_in_which_elements_carry_the_pixel() -> None:
    for g in D4:
        a = G.act_d4_colour_cue(g) - {(*G.CUE_PIXEL, G.NEUTRAL_COLOUR)}
        b = G.act_d4_colour_decoy(g) - {(*G.CUE_PIXEL, G.NEUTRAL_COLOUR)}
        assert a == b
    assert sum(_bits(G.act_d4_colour_cue).values()) == sum(_bits(G.act_d4_colour_decoy).values()) == 4


@pytest.mark.parametrize("name", ["d4_colour_cue", "d4_colour_decoy"])
def test_the_cue_never_moves_the_colour_block(name) -> None:
    action = G.SUBSTRATE_ACTIONS[name][0]
    for g in D4:
        cells = action(g)
        assert _box(cells) == (3, 3)
        block = {c for c in cells if c[2] != G.NEUTRAL_COLOUR}
        assert block == set(G.act_d4_colour(g))


def test_the_decoy_leaks_no_character_and_does_not_compose() -> None:
    bits = _bits(G.act_d4_colour_decoy)
    for c in CH.values():
        assert sum(bits[g] == c(g) for g in D4) == 4
    assert any(bits[G.multiply(a, b)] != (bits[a] + bits[b]) % 2 for a in D4 for b in D4)


def _mi_with_label(bits):
    joint = Counter()
    for s, o1, o2 in itertools.product(D4, repeat=3):
        joint[(bits[s], bits[o1], bits[o2], G.multiply(o2, G.multiply(o1, s)))] += 1
    n = sum(joint.values())
    pb, pl = Counter(), Counter()
    for (a, b, c, l), k in joint.items():
        pb[(a, b, c)] += k
        pl[l] += k
    return sum(k / n * math.log2((k / n) / ((pb[key[:3]] / n) * (pl[key[3]] / n)))
               for key, k in joint.items())


def test_what_the_three_cue_bits_say_about_the_label() -> None:
    """The cue carries exactly one bit about a depth 2 label, the decoy under
    a twentieth of one. That difference is the whole of E2."""
    assert _mi_with_label(_bits(G.act_d4_colour_cue)) == pytest.approx(1.0)
    assert _mi_with_label(_bits(G.act_d4_colour_decoy)) < 0.05


def test_the_old_substrates_are_unchanged() -> None:
    assert G.SUBSTRATE_ACTIONS["d4_colour"][0] is G.act_d4_colour
    assert G.SUBSTRATE_ACTIONS["s3_spatial"][0] is G.act_s3_spatial
