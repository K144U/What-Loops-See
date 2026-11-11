"""F3 is the lead figure, so the ways it could quietly mislead matter most.

Four of them are specific to this grid. Position -1 is the full-state
positive control rather than a patch site, and drawing it would put a
saturated column among the real ones. The exploratory and confirmatory
grids are different evidence classes that the frozen preregistration keeps
apart. A colour scale fitted per panel would make three families look
alike when their peaks differ by a factor of three. And H3 is a contrast
between families, so a figure with one column would read as the result
while showing nothing of the kind.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loopvision.analysis.plots import f3_patching as F3


def _grid(tmp_path, run_id, family, *, peak=0.05, n_pos=8, n_query=2,
          iters=4, filename="m2_patching.parquet", control_top=1.0):
    d = tmp_path / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in range(iters):
        # The full-state control, at position -1, rising to control_top.
        rows.append({"run_id": run_id, "family": family, "depth": 2, "k": 8,
                     "iteration": it, "position": -1, "is_query_token": False,
                     "recovery_mean": control_top * it / (iters - 1),
                     "recovery_std": 0.0, "n_items": 64})
        for p in range(n_pos):
            rows.append({"run_id": run_id, "family": family, "depth": 2,
                         "k": 8, "iteration": it, "position": p,
                         "is_query_token": False,
                         "recovery_mean": peak if (it == iters - 1 and p == 1)
                         else 0.0,
                         "recovery_std": 0.0, "n_items": 64})
        for q in range(n_query):
            rows.append({"run_id": run_id, "family": family, "depth": 2,
                         "k": 8, "iteration": it, "position": n_pos + q,
                         "is_query_token": True, "recovery_mean": 0.0,
                         "recovery_std": 0.0, "n_items": 64})
    pd.DataFrame(rows).to_parquet(d / filename, index=False)
    return d


def test_the_full_state_control_is_not_drawn_as_a_position(tmp_path) -> None:
    """Position -1 is the patch that replaces the whole state, and it
    reaches 1.0 by construction. In the grid it would be a saturated column
    no reader could tell from a real patch site, and it would set the
    colour scale for every panel.

    Mutation checked: changing the filter to position >= -1 puts 1.0 into
    the grid and fails here.
    """
    _grid(tmp_path, "a0", "A", peak=0.05)
    _grid(tmp_path, "c0", "C", peak=0.03)
    panels = F3.load(tmp_path / "runs", "exploratory")
    for p in panels:
        assert -1 not in p["positions"], "the control entered the grid"
        assert np.nanmax(p["grid"]) <= 0.05 + 1e-9
        assert max(p["control"]) == pytest.approx(1.0)


def test_the_control_is_kept_and_not_merely_dropped(tmp_path) -> None:
    """Excluding it from the grid is only half the job. It is the evidence
    the patch landed at all, so it has to reach the margin."""
    _grid(tmp_path, "a0", "A")
    _grid(tmp_path, "c0", "C")
    p = F3.load(tmp_path / "runs", "exploratory")[0]
    assert len(p["control"]) == len(p["control_iterations"]) > 0


def test_exploratory_and_confirmatory_are_never_mixed(tmp_path) -> None:
    """Section 0 of the frozen preregistration excludes the m2_patching
    grids from the H3 confirmation set, and h3_contrast enforces that by
    being unable to load them. A figure that quietly concatenated both
    would put excluded data into a registered result.

    Mutation checked: globbing both filenames makes the confirm source
    return family A as well and fails here.
    """
    _grid(tmp_path, "a0", "A", filename="m2_patching.parquet")
    _grid(tmp_path, "b0", "B", filename="m2_confirm.parquet")

    expl = {p["family"] for p in F3.load(tmp_path / "runs", "exploratory")}
    conf = {p["family"] for p in F3.load(tmp_path / "runs", "confirm")}
    assert expl == {"A"}
    assert conf == {"B"}


def test_the_sources_name_different_files() -> None:
    assert F3.SOURCES["exploratory"] != F3.SOURCES["confirm"]
    assert len(set(F3.SOURCES.values())) == 2


def test_one_family_is_refused_rather_than_drawn(tmp_path) -> None:
    """H3 is a contrast between a depth family and a breadth family. On
    2026-09-09 the confirmation set holds family B alone, and a single
    column figure would be read as the registered result."""
    _grid(tmp_path, "b0", "B", filename="m2_confirm.parquet")
    rc = F3.main(["--runs", str(tmp_path / "runs"), "--source", "confirm",
                  "--out", str(tmp_path / "f3")])
    assert rc == 1
    assert not (tmp_path / "f3.pdf").exists(), "a one column figure was drawn"


def test_two_families_are_enough_to_draw(tmp_path) -> None:
    _grid(tmp_path, "a0", "A")
    _grid(tmp_path, "c0", "C")
    out = tmp_path / "fig" / "f3"
    assert F3.main(["--runs", str(tmp_path / "runs"), "--out", str(out)]) == 0
    assert out.with_suffix(".pdf").is_file()


def test_the_colour_scale_is_shared_across_families(tmp_path) -> None:
    """Family peaks differ by a factor of three in the real grids. Scaling
    each panel to its own maximum would make three different pictures look
    identical, which is the opposite of what the figure is for.

    Mutation checked: computing vmax inside the per panel loop fails here.
    """
    _grid(tmp_path, "a0", "A", peak=0.09)
    _grid(tmp_path, "c0", "C", peak=0.03)
    panels = F3.load(tmp_path / "runs", "exploratory")
    fig = F3.draw(panels, "exploratory")
    images = [im for ax in fig.axes for im in ax.get_images()]
    assert len(images) == 2
    clims = {im.get_clim() for im in images}
    assert len(clims) == 1, "each panel was scaled to its own maximum"
    assert max(c[1] for c in clims) == pytest.approx(0.09)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_families_are_ordered_depth_then_breadth(tmp_path) -> None:
    """The prediction is that depth families are the localised ones and
    the breadth family is the contrast, so the eye should travel that way.

    This test cannot fail while the families are named A, B and C, and it
    is kept for the day one is not. groupby already returns them in
    alphabetical order and FAMILY_ORDER happens to be alphabetical, so
    neutering the sort leaves this green. That was found by mutating the
    sort and watching every test pass; the case that does bite is below.
    """
    _grid(tmp_path, "c0", "C")
    _grid(tmp_path, "a0", "A")
    _grid(tmp_path, "b0", "B")
    got = [p["family"] for p in F3.load(tmp_path / "runs", "exploratory")]
    assert got == ["A", "B", "C"]
    assert list(F3.FAMILY_ORDER) == sorted(F3.FAMILY_ORDER), \
        "FAMILY_ORDER no longer matches alphabetical, so the test above " \
        "has become load bearing and the sort must be checked directly"


def test_a_family_the_order_does_not_name_goes_last(tmp_path) -> None:
    """The half of the sort that is load bearing today.

    A new family arriving without being added to FAMILY_ORDER must not
    land between the registered ones, where it would read as part of the
    contrast. It goes to the end, where it is visibly extra.

    Mutation checked: sending unknown families to -1 instead of 99 puts
    the stranger first and fails here.
    """
    _grid(tmp_path, "a0", "A")
    _grid(tmp_path, "c0", "C")
    _grid(tmp_path, "z0", "Z")
    got = [p["family"] for p in F3.load(tmp_path / "runs", "exploratory")]
    assert got == ["A", "C", "Z"]


def test_query_positions_are_found_so_the_divider_lands(tmp_path) -> None:
    _grid(tmp_path, "a0", "A", n_pos=8, n_query=3)
    _grid(tmp_path, "c0", "C")
    p = F3.load(tmp_path / "runs", "exploratory")[0]
    assert p["query_positions"] == [8, 9, 10]


def test_seeds_are_counted_and_averaged_within_a_family(tmp_path) -> None:
    """The confirmation set is many seeds of one family, so the panel has
    to say how many runs are behind it rather than showing one."""
    _grid(tmp_path, "b0", "B", peak=0.02, filename="m2_confirm.parquet")
    _grid(tmp_path, "b1", "B", peak=0.06, filename="m2_confirm.parquet")
    _grid(tmp_path, "c0", "C", peak=0.03, filename="m2_confirm.parquet")
    panels = {p["family"]: p for p in F3.load(tmp_path / "runs", "confirm")}
    assert panels["B"]["n_seeds"] == 2
    assert np.nanmax(panels["B"]["grid"]) == pytest.approx(0.04), \
        "two seeds were not averaged"


def test_a_missing_source_says_so(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    assert F3.main(["--runs", str(tmp_path / "runs"),
                    "--out", str(tmp_path / "f3")]) == 2


# ---------------------------------------------------------------------------
# The display gamma, added 2026-09-10. Family peaks differ by roughly six
# times, so on a linear shared scale family C renders almost black, and
# family C being the spatially uniform case is the surviving half of H3.
#
# The danger in fixing that is worse than the problem: a compressed scale
# makes small values look larger, so it must not reach the data, and it must
# not be readable as linear.
# ---------------------------------------------------------------------------


def test_the_gamma_never_touches_the_data(tmp_path) -> None:
    """The one that matters. CLAUDE.md says plot scripts do not compute
    statistics, and a gamma applied to the array rather than to the colour
    mapping would be exactly that: every number in the figure silently
    raised to a power, still labelled recovery."""
    _grid(tmp_path, "a0", "A", peak=0.09)
    _grid(tmp_path, "c0", "C", peak=0.03)
    panels = F3.load(tmp_path / "runs", "exploratory")
    before = [p["grid"].copy() for p in panels]

    fig = F3.draw(panels, "exploratory", gamma=0.5)
    images = [im for ax in fig.axes for im in ax.get_images()]
    for im, orig in zip(images, before):
        assert np.allclose(im.get_array(), orig), "the array itself was transformed"
    for p, orig in zip(panels, before):
        assert np.allclose(p["grid"], orig), "draw mutated its input"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_the_gamma_lifts_the_low_end_without_unsharing_the_scale(tmp_path) -> None:
    """The point of the change. Family C has to become visible, and the two
    panels have to stay on one scale while it happens."""
    from matplotlib.colors import PowerNorm

    _grid(tmp_path, "a0", "A", peak=0.09)
    _grid(tmp_path, "c0", "C", peak=0.03)
    panels = F3.load(tmp_path / "runs", "exploratory")
    fig = F3.draw(panels, "exploratory", gamma=0.5)
    images = [im for ax in fig.axes for im in ax.get_images()]

    assert len({im.get_clim() for im in images}) == 1, "the scale stopped being shared"
    for im in images:
        assert isinstance(im.norm, PowerNorm)
        assert im.norm.gamma == 0.5
    # 0.03 of 0.09 is a third of the range linearly and about 58 percent
    # under a square root, which is the difference between invisible and legible.
    assert images[0].norm(0.03) > 0.5
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_gamma_one_is_the_linear_scale_and_says_nothing(tmp_path) -> None:
    """The escape hatch has to actually be linear, and a linear figure must
    not carry a caption warning about a compression it does not have."""
    _grid(tmp_path, "a0", "A", peak=0.09)
    _grid(tmp_path, "c0", "C", peak=0.03)
    panels = F3.load(tmp_path / "runs", "exploratory")
    fig = F3.draw(panels, "exploratory", gamma=1.0)
    im = [i for ax in fig.axes for i in ax.get_images()][0]
    assert im.norm(0.03) == pytest.approx(1 / 3, abs=1e-6)
    text = " ".join(t.get_text() for t in fig.texts)
    assert "gamma" not in text.lower()
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_a_compressed_figure_declares_itself(tmp_path) -> None:
    """A gamma that the reader cannot see is a figure that overstates small
    values. The caption has to name it and has to send the reader to the
    bar rather than to the brightness."""
    _grid(tmp_path, "a0", "A", peak=0.09)
    _grid(tmp_path, "c0", "C", peak=0.03)
    panels = F3.load(tmp_path / "runs", "exploratory")
    fig = F3.draw(panels, "exploratory", gamma=0.5)
    text = " ".join(t.get_text() for t in fig.texts).lower()
    assert "gamma 0.5" in text
    assert "unevenly spaced" in text
    # Naming the gamma is not enough on its own. A reader who keeps judging
    # magnitude by brightness is misled by exactly the amount the gamma
    # lifts, so the caption has to redirect them to the bar.
    assert "off the bar" in text
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_the_colourbar_ticks_stay_in_data_units(tmp_path) -> None:
    """Under a gamma the ticks come out unevenly spaced, and that unevenness
    is the reader's only visible cue that the mapping is compressed. Ticks
    placed evenly in colour space would erase it."""
    _grid(tmp_path, "a0", "A", peak=0.09)
    _grid(tmp_path, "c0", "C", peak=0.03)
    panels = F3.load(tmp_path / "runs", "exploratory")
    fig = F3.draw(panels, "exploratory", gamma=0.5)
    bars = [ax for ax in fig.axes if ax.get_ylabel().startswith("recovery")]
    assert bars, "no colourbar found"
    ticks = list(bars[0].get_yticks())
    assert max(ticks) == pytest.approx(0.09, abs=1e-6)
    assert min(ticks) == pytest.approx(0.0, abs=1e-6)
    # Even in data units is the whole point. Ticks placed evenly in colour
    # space would land at 0, 0.006, 0.023, 0.051, 0.09 and look linear on
    # the bar, erasing the one visible cue that the mapping is not.
    gaps = np.diff(ticks)
    assert np.allclose(gaps, gaps[0]), "ticks are not evenly spaced in data units"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_the_shared_maximum_is_the_largest_family_not_the_first(tmp_path) -> None:
    """`load` returns families in name order, so taking panels[0]'s maximum
    happens to be right whenever family A is the largest, which it is in the
    real grids and in every other fixture here. This one puts the peak in
    the last family, where that shortcut gives the wrong scale and clips."""
    _grid(tmp_path, "a0", "A", peak=0.03)
    _grid(tmp_path, "c0", "C", peak=0.09)
    panels = F3.load(tmp_path / "runs", "exploratory")
    assert panels[0]["family"] == "A"
    fig = F3.draw(panels, "exploratory")
    images = [im for ax in fig.axes for im in ax.get_images()]
    assert len({im.get_clim() for im in images}) == 1
    assert max(im.get_clim()[1] for im in images) == pytest.approx(0.09)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_the_caption_is_wrapped_so_it_cannot_stretch_the_figure(tmp_path) -> None:
    """The caption sits outside the axes and the figure saves with a tight
    bounding box, so one long line widens the saved image and shrinks the
    panels to compensate. Declaring the gamma lengthened this text and took
    the PNG from 4139 to 6802 pixels wide before it was wrapped."""
    _grid(tmp_path, "a0", "A", peak=0.09)
    _grid(tmp_path, "c0", "C", peak=0.03)
    panels = F3.load(tmp_path / "runs", "exploratory")
    for gamma in (0.5, 1.0):
        fig = F3.draw(panels, "exploratory", gamma=gamma)
        caption = max((t.get_text() for t in fig.texts), key=len)
        longest = max(len(line) for line in caption.splitlines())
        assert longest <= 110, f"caption line of {longest} chars will stretch the figure"
        import matplotlib.pyplot as plt
        plt.close(fig)
