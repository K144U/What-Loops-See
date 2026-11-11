"""F1 draws real samples, and its captions must come from those samples.

The figure's entire claim is that depth and breadth move independently. If
the captions were typed they could assert that while the panels showed
something else, which is exactly how F9's title came to read "never" over a
panel plotting fifteen escapes.
"""

from __future__ import annotations

import pytest

from loopvision.analysis.plots import f1_tasks as F


def test_the_caption_is_read_off_the_samples_own_program() -> None:
    class FakeSample:
        program = "A;d=3;b=7;q=0;rest"

    assert F._db(FakeSample()) == "depth 3, breadth 7"


def test_a_program_without_the_fields_says_so_rather_than_guessing() -> None:
    class Odd:
        program = "A;q=0;nothing"

    assert F._db(Odd()) == "depth ?, breadth ?"


def test_family_a_varies_depth_at_fixed_breadth() -> None:
    """Row 1 of the figure, checked against the generator rather than the
    caller's request: the breadth must not move while depth does."""
    rows = F._panels()
    captions = [c for c, _ in rows[0][1]]
    assert [c.split(",")[0] for c in captions] == [
        "depth 1", "depth 2", "depth 3"]
    breadths = {c.split("breadth")[1].strip() for c in captions}
    assert len(breadths) == 1, f"breadth moved across the depth row: {breadths}"


def test_family_c_varies_breadth_at_fixed_depth() -> None:
    rows = F._panels()
    captions = [c for c, _ in rows[1][1]]
    depths = {c.split(",")[0] for c in captions}
    assert depths == {"depth 1"}, f"depth moved across the breadth row: {depths}"
    assert [c.split("breadth")[1].strip() for c in captions] == ["4", "6", "8"]


def test_the_substrate_row_holds_everything_but_the_substrate() -> None:
    """The decisive comparison of the paper. If depth or breadth differed
    between these three panels the figure would be arguing for the wrong
    thing."""
    rows = F._panels()
    tails = {c.splitlines()[-1] for c, _ in rows[2][1]}
    assert len(tails) == 1, f"the substrate panels disagree on depth or breadth: {tails}"


def test_every_panel_carries_a_real_image() -> None:
    for _label, panels in F._panels():
        for caption, sample in panels:
            assert sample.image.ndim == 3 and sample.image.shape[0] == 3, caption
            assert sample.image.any(), f"{caption} rendered an empty image"


def test_the_figure_carries_a_title_and_no_caption_of_its_own(tmp_path) -> None:
    """The caption lives in the draft. A seven-line note drawn under the
    panels as well made the paper print the caption twice, which a reader
    caught on 2026-10-22."""
    import matplotlib.pyplot as plt

    fig = F.draw(tmp_path / "f1.png")
    try:
        texts = [t.get_text() for t in fig.texts if t.get_text()]
        assert texts == ["Depth and breadth are crossed, and both are known exactly"]
    finally:
        plt.close(fig)
