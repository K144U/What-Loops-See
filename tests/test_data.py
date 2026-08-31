"""Milestone 1 gate: the data layer.

Milestone 1's definition of done is that this file passes and that 64
samples per depth have been dumped and looked at.

The properties here are the ones the whole study rests on. Composition
depth must be exactly what we say it is, splits must not leak, and the
rendered image must carry the information the label depends on. A failure
in any of these produces a dataset that trains fine and means nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from loopvision.data import dataset as D
from loopvision.data import family_a as FA
from loopvision.data import groups as G
from loopvision.data import render

FAMILIES = ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_all_48_elements_render_distinctly() -> None:
    """The picture must carry everything the label depends on.

    If two group elements rendered identically, the task would be
    unanswerable from the image for those pairs and the ceiling would sit
    below 100 percent for reasons unrelated to composition.
    """
    tiles = {render.render_element(i).tobytes() for i in range(G.GROUP_ORDER)}
    assert len(tiles) == G.GROUP_ORDER


def test_marker_is_not_confusable_with_any_glyph() -> None:
    marker = render.render_marker().tobytes()
    tiles = {render.render_element(i).tobytes() for i in range(G.GROUP_ORDER)}
    assert marker not in tiles


def test_glyphs_are_patch_aligned() -> None:
    """Every glyph occupies exactly one 4x4 patch and never straddles two.

    This is what makes the M2 spatial axis interpretable: patch token p
    corresponds to exactly one semantic element. Verified by checking that
    every non-background pixel lies inside a patch that is entirely
    accounted for by one glyph.
    """
    sample = D.generate("A", D.global_index("depth_ood", 3), "depth_ood")
    bg = np.array(render.BACKGROUND, dtype=np.uint8).reshape(3, 1, 1)
    occupied = (sample.image != bg).any(axis=0)

    initial, strips, queried, depth, breadth = FA.parse_program(sample.program)
    expected = {(r, render.STATE_COL) for r in range(breadth)}
    for r in range(breadth):
        for j in range(len(strips[r])):
            expected.add((r, render.FIRST_OP_COL + j))
    expected.add((queried, render.MARKER_COL))

    found = {
        (y // render.PATCH, x // render.PATCH)
        for y in range(sample.image.shape[1])
        for x in range(sample.image.shape[2])
        if occupied[y, x]
    }
    assert found == expected, f"unexpected occupied patches: {found ^ expected}"


# ---------------------------------------------------------------------------
# Determinism and identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_generation_is_deterministic(family: str) -> None:
    """Same arguments, same sample, always. Nothing reads global RNG."""
    for idx in (0, 17, 4321):
        a = D.generate(family, idx, "train")
        b = D.generate(family, idx, "train")
        assert np.array_equal(a.image, b.image)
        assert a.program == b.program
        assert a.label == b.label
        assert a.layout_seed == b.layout_seed


@pytest.mark.parametrize("family", FAMILIES)
def test_master_seed_changes_the_data(family: str) -> None:
    a = D.generate(family, 0, "train", D.TaskConfig(master_seed=1))
    b = D.generate(family, 0, "train", D.TaskConfig(master_seed=2))
    assert a.program != b.program


@pytest.mark.parametrize("family", FAMILIES)
def test_distinct_indices_give_distinct_programs(family: str) -> None:
    programs = {
        D.generate(family, D.global_index("train", i), "train").program
        for i in range(500)
    }
    # A few collisions are possible at depth 1 where the program space is
    # small, but the bulk must be distinct.
    assert len(programs) > 450, f"only {len(programs)}/500 distinct programs"


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_splits_do_not_overlap(family: str) -> None:
    """Zero shared (program, layout_seed) pairs between any two splits.

    Guaranteed by construction, since the layout seed is derived from the
    global index and split index ranges are disjoint. Tested anyway,
    because "guaranteed by construction" is a claim about code that can
    stop being true when the code changes.
    """
    module = D.get_family(family)
    seen: dict[tuple[str, int], str] = {}
    for split in sorted(module.SUPPORTED_SPLITS):
        for i in range(300):
            s = D.generate(family, D.global_index(split, i), split)
            key = (s.program, s.layout_seed)
            assert key not in seen, (
                f"split {split} sample {i} collides with split {seen[key]}"
            )
            seen[key] = split


@pytest.mark.parametrize("family", FAMILIES)
def test_split_depth_and_breadth_ranges_are_respected(family: str) -> None:
    """The axes must be exactly what the split declares.

    H1 regresses k_min on depth and breadth. If a sample carried a depth
    outside its split's range, the regression would be fitting mislabelled
    points and no test downstream would notice.
    """
    module = D.get_family(family)
    for split in sorted(module.SUPPORTED_SPLITS):
        spec = D.split_spec(family, split)
        depths, breadths = set(), set()
        for i in range(400):
            s = D.generate(family, D.global_index(split, i), split)
            assert s.depth in spec.depth, f"{split}: depth {s.depth} not in {spec.depth}"
            assert s.breadth in spec.breadth
            depths.add(s.depth)
            breadths.add(s.breadth)
        # Every declared value must actually appear, or the split is
        # narrower than it claims and a cell of the H1 grid would be empty.
        assert depths == set(spec.depth), f"{split}: saw depths {depths}"
        assert breadths == set(spec.breadth), f"{split}: saw breadths {breadths}"


def test_depth_ood_is_deeper_than_train() -> None:
    assert max(D.SPLITS["train"].depth) < min(D.SPLITS["depth_ood"].depth)


def test_breadth_ood_is_broader_than_train() -> None:
    assert max(D.SPLITS["train"].breadth) < min(D.SPLITS["breadth_ood"].breadth)


def test_unsupported_split_raises_rather_than_improvising() -> None:
    """Family A has no combo_ood, and asking for it must say so.

    Silently returning something plausible would be worse than an error:
    an OOD result would be reported for a split that does not mean what
    its name says. See docs/decisions.md D-018.
    """
    with pytest.raises(ValueError, match="combo_ood"):
        D.generate("A", 0, "combo_ood")


# ---------------------------------------------------------------------------
# Family A semantics
# ---------------------------------------------------------------------------


def test_label_is_recomputable_from_the_program() -> None:
    """The label must follow from the program, not from a parallel path.

    This is the test that catches a divergence between what is drawn and
    what is scored, which is the failure mode that trains happily and
    means nothing.
    """
    for split in sorted(FA.SUPPORTED_SPLITS):
        for i in range(200):
            s = D.generate("A", D.global_index(split, i), split)
            initial, strips, queried, depth, breadth = FA.parse_program(s.program)
            recomputed = G.multiply(G.compose_sequence(strips[queried]), initial[queried])
            assert recomputed == s.label
            assert len(strips[queried]) == s.depth == depth
            assert len(strips) == s.breadth == breadth


def test_program_round_trips() -> None:
    for i in range(200):
        s = D.generate("A", D.global_index("train", i), "train")
        parsed = FA.parse_program(s.program)
        assert FA.build_program(*parsed) == s.program


def test_labels_are_uniform_over_48_classes() -> None:
    """Chance must be exactly 1/48 with no exploitable class prior."""
    counts = np.zeros(G.GROUP_ORDER, dtype=np.int64)
    n = 48 * 300
    for i in range(n):
        counts[D.generate("A", D.global_index("train", i), "train").label] += 1
    expected = n / G.GROUP_ORDER
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 92.0, f"labels not uniform, chi2={chi2:.1f}"


def test_distractor_strips_carry_no_information_about_the_label() -> None:
    """Unqueried objects must be independent of the answer.

    If a distractor's composite correlated with the label, breadth would
    leak depth information and the H1 null axis would be contaminated.
    """
    labels, distractor_composites = [], []
    for i in range(3000):
        s = D.generate("A", D.global_index("breadth_ood", i), "breadth_ood")
        initial, strips, queried, _, breadth = FA.parse_program(s.program)
        other = 0 if queried != 0 else 1
        labels.append(s.label)
        distractor_composites.append(G.compose_sequence(strips[other]))

    joint = np.zeros((G.GROUP_ORDER, G.GROUP_ORDER), dtype=np.int64)
    for lab, comp in zip(labels, distractor_composites):
        joint[lab, comp] += 1
    # Under independence every cell has the same expectation. 2303 degrees
    # of freedom, so the 0.999 critical value is about 2560.
    expected = len(labels) / (G.GROUP_ORDER**2)
    chi2 = float(((joint - expected) ** 2 / expected).sum())
    assert chi2 < 2560.0, f"distractors are not independent of the label, chi2={chi2:.1f}"


def test_image_is_a_valid_uint8_tensor() -> None:
    s = D.generate("A", 0, "train")
    assert s.image.dtype == np.uint8
    assert s.image.shape == (3, render.CANVAS, render.CANVAS)
    assert s.query.dtype == np.int32
    assert s.query.shape == (D.L_MAX["A"],)


def test_query_tokens_are_in_vocabulary() -> None:
    for split in sorted(FA.SUPPORTED_SPLITS):
        for i in range(100):
            s = D.generate("A", D.global_index(split, i), split)
            assert s.query.min() >= 0
            assert s.query.max() < D.VOCAB_SIZE


def test_tokens_presentation_moves_operators_out_of_the_image() -> None:
    """The tokens variant isolates composition from visual parsing."""
    cfg = D.TaskConfig(presentation="tokens")
    s = D.generate("A", D.global_index("train", 5), "train", cfg)
    initial, strips, queried, depth, breadth = FA.parse_program(s.program)

    assert (s.query[2 : 2 + depth] == [D.OP_BASE + o for o in strips[queried]]).all()

    bg = np.array(render.BACKGROUND, dtype=np.uint8).reshape(3, 1, 1)
    occupied = {
        (y // render.PATCH, x // render.PATCH)
        for y in range(s.image.shape[1])
        for x in range(s.image.shape[2])
        if (s.image != bg).any(axis=0)[y, x]
    }
    expected = {(r, render.STATE_COL) for r in range(breadth)}
    expected.add((queried, render.MARKER_COL))
    assert occupied == expected, "operator glyphs should be absent in the tokens variant"


def _decode_scene(image: np.ndarray) -> tuple[list[int], list[list[int]], int]:
    """Read a family A scene back out of its pixels.

    Matches every occupied patch against the 48 rendered element tiles and
    the marker tile, then reassembles the program. Exists so that the label
    can be checked against the *image* rather than against the program
    string that produced both.
    """
    tiles = {render.render_element(i).tobytes(): i for i in range(G.GROUP_ORDER)}
    marker = render.render_marker().tobytes()
    bg = render.blank_canvas(render.PATCH).tobytes()

    initial: list[int] = []
    strips: list[list[int]] = []
    queried = -1
    for row in range(render.GRID):
        cells = []
        for col in range(render.GRID):
            y0, x0 = row * render.PATCH, col * render.PATCH
            key = image[:, y0 : y0 + render.PATCH, x0 : x0 + render.PATCH].tobytes()
            if key == marker:
                queried = row
                cells.append(("marker", None))
            elif key == bg:
                cells.append(("empty", None))
            else:
                assert key in tiles, f"unrecognised tile at ({row}, {col})"
                cells.append(("glyph", tiles[key]))

        if cells[render.STATE_COL][0] != "glyph":
            continue  # no object on this row
        initial.append(cells[render.STATE_COL][1])
        ops = []
        for col in range(render.FIRST_OP_COL, render.MARKER_COL):
            kind, value = cells[col]
            if kind != "glyph":
                break
            ops.append(value)
        strips.append(ops)
    return initial, strips, queried


def test_label_is_recoverable_from_the_image_alone() -> None:
    """The end-to-end check: pixels in, label out.

    Every other test in this file compares the label against the program
    string, and the program string is what produced both the label and the
    image. That leaves a hole: if the renderer drew the operator strip in
    the wrong order, or on the wrong row, the label and the program would
    still agree with each other while the picture disagreed with both.
    A model would then be trained on unanswerable examples and would simply
    plateau below ceiling, with nothing to say why.

    This test closes that hole by decoding the scene back out of the
    pixels and recomputing the label from what is actually drawn.
    """
    for split in sorted(FA.SUPPORTED_SPLITS):
        for i in range(150):
            s = D.generate("A", D.global_index(split, i), split)
            initial, strips, queried = _decode_scene(s.image)

            assert queried >= 0, "no query marker found in the image"
            assert len(initial) == s.breadth
            assert all(len(ops) == s.depth for ops in strips)

            from_pixels = G.multiply(G.compose_sequence(strips[queried]), initial[queried])
            assert from_pixels == s.label, (
                f"{split}[{i}]: the image encodes {from_pixels} but the label "
                f"is {s.label}. The renderer and the label have diverged."
            )


def test_decoded_scene_matches_the_program() -> None:
    """Belt and braces: pixels, program and label must all be one story."""
    for i in range(100):
        s = D.generate("A", D.global_index("depth_ood", i), "depth_ood")
        p_initial, p_strips, p_queried, _, _ = FA.parse_program(s.program)
        i_initial, i_strips, i_queried = _decode_scene(s.image)
        assert i_initial == p_initial
        assert i_strips == p_strips
        assert i_queried == p_queried


# ---------------------------------------------------------------------------
# M2 counterfactuals
# ---------------------------------------------------------------------------


def test_corrupted_twin_differs_in_exactly_one_patch() -> None:
    """The M2 counterfactual must isolate one program element.

    If the twin differed anywhere else, a patching heatmap would be
    attributing an effect to a position that changed for an unrelated
    reason, and H3 would be measuring the renderer rather than the model.
    """
    for i in range(100):
        s = D.generate("A", D.global_index("train", i), "train")
        t = FA.corrupted_twin(s)

        differing = {
            (y // render.PATCH, x // render.PATCH)
            for y in range(s.image.shape[1])
            for x in range(s.image.shape[2])
            if (s.image[:, y, x] != t.image[:, y, x]).any()
        }
        assert len(differing) == 1, f"twin differs in {len(differing)} patches"

        row, col = next(iter(differing))
        _, _, queried, _, _ = FA.parse_program(s.program)
        assert row == queried, "the edit must land on the queried object"
        assert col >= render.FIRST_OP_COL, "the edit must land on an operator"


def test_corrupted_twin_changes_the_label() -> None:
    """A counterfactual that leaves the answer alone measures nothing."""
    changed = 0
    for i in range(100):
        s = D.generate("A", D.global_index("train", i), "train")
        t = FA.corrupted_twin(s)
        assert t.label != s.label
        changed += 1
    assert changed == 100


def test_corrupted_twin_is_deterministic() -> None:
    s = D.generate("A", D.global_index("train", 9), "train")
    assert np.array_equal(FA.corrupted_twin(s).image, FA.corrupted_twin(s).image)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


def test_dataset_indexing_and_bounds() -> None:
    ds = D.LoopVisionDataset("A", "train", length=64)
    assert len(ds) == 64
    assert ds[0].program == D.generate("A", D.global_index("train", 0), "train").program
    with pytest.raises(IndexError):
        ds[64]
