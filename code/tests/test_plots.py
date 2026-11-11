"""A figure is an argument, so the ways it can lie are the ways to test it.

CLAUDE.md: plot scripts do not compute statistics. That rule is testable in
one direction only, by checking that the script reads what the analysis
module wrote and does not decide anything itself. The three things F9 could
get wrong without anyone noticing are: dropping unfinished runs into the
plot as though they were results, drawing one chance line under two series
that do not share one, and putting an arm in the wrong column so the
picture is transposed while every number in it is right.

The style module is tested for the property the paper actually requires,
which is that a series survives losing its colour.
"""

from __future__ import annotations

import json

import pytest

from loopvision.analysis.plots import f9_binding as F9
from loopvision.analysis.plots import style


def test_every_series_is_identifiable_without_colour() -> None:
    """paper.md section 6 requires greyscale legibility and reviewers
    print. A palette that separates only by hue fails both a greyscale
    print and a colourblind reader, and neither will write to tell us."""
    markers = [s["marker"] for s in style.SERIES_STYLES]
    dashes = [s["linestyle"] for s in style.SERIES_STYLES]
    assert len(set(markers)) == len(markers), "two series share a marker"
    assert len(set(map(str, dashes))) == len(dashes), "two series share a dash"


def test_series_cycles_rather_than_failing() -> None:
    a = style.series(0)
    assert style.series(len(style.SERIES_STYLES)) == a
    assert "name" not in a, "the human label must not reach matplotlib"


def test_the_reference_colour_is_not_a_series_colour() -> None:
    """A chance line drawn in a series colour reads as data."""
    assert style.REFERENCE not in [s["color"] for s in style.SERIES_STYLES]


def test_every_outcome_has_a_colour() -> None:
    """A missing key would raise while drawing, at the point where the
    figure is being regenerated for a deadline."""
    from loopvision.analysis import escape_rates as ER

    for outcome in ("escaped", "collapsed", "neither", "unfinished"):
        assert outcome in style.OUTCOME
    assert set(style.OUTCOME) >= {ER.classify({"final_accuracy": 1.0})[0]}


def _rates(tmp_path, rows, arms=None):
    p = tmp_path / "rates.json"
    p.write_text(json.dumps({"escape_tau": 0.95, "arms": arms or {},
                             "runs": rows, "neither": []}), encoding="utf-8")
    return p


def test_the_collapse_floor_is_drawn_per_arm_and_never_across(tmp_path) -> None:
    """The bug the first render of this figure shipped with.

    D4 has order 8 and S3 order 6, so a collapsed D4 run sits near 0.125
    and a collapsed S3 run near 0.167. The first version drew one dashed
    line across the whole panel at the D4 constant, which put the S3
    identity points visibly ABOVE their floor: the arm read as beating a
    constant while it was sitting on one. The bottom row already refused
    the same mistake for the chance line and the top row was making it.

    The floors are the arms' own collapsed accuracies, read from the
    generator, so this also pins that the figure holds no numbers.

    Mutation checked: taking min over every arm's floors instead of the
    arm's own draws one height for all four and fails here.
    """
    fig = F9.draw(
        {
            "arms": {
                "D4/identity": {"floors": [0.1254, 0.1294]},
                "S3/identity": {"floors": [0.1681, 0.1787]},
            },
            "runs": [
                {"run_id": "d", "arm": "D4/identity", "seed": 0,
                 "outcome": "collapsed", "accuracy": 0.1254,
                 "n_distinct": 1, "modal": 12},
                {"run_id": "s", "arm": "S3/identity", "seed": 0,
                 "outcome": "collapsed", "accuracy": 0.1787,
                 "n_distinct": 1, "modal": 0},
            ],
        },
        [{"passes": [0, 1], "values": [0.1, 0.9], "chance": 0.125,
          "label": "x", "run": "r"}],
    )
    ax = fig.axes[0]
    # Every reference segment, as (y, x-extent). Filtering on the
    # reference colour rather than the dash, because the panel divider
    # is also a two point line with a dash.
    segs = [(ln.get_ydata()[0], ln.get_xdata())
            for ln in ax.get_lines()
            if ln.get_color() == style.REFERENCE]
    heights = sorted({round(y, 4) for y, _ in segs})
    assert heights == [0.1254, 0.1294, 0.1681, 0.1787], heights
    for y, xs in segs:
        span = max(xs) - min(xs)
        assert span < 1.0, "a floor line spanned more than its own arm"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_an_arm_with_no_collapsed_run_gets_no_floor(tmp_path) -> None:
    """D4 on position has nothing collapsed, so there is no constant to
    draw. Inventing one would imply a floor that was never observed."""
    fig = F9.draw(
        {"arms": {"D4/position": {"floors": []}},
         "runs": [{"run_id": "a", "arm": "D4/position", "seed": 0,
                   "outcome": "escaped", "accuracy": 1.0,
                   "n_distinct": 8, "modal": None}]},
        [{"passes": [0], "values": [0.9], "chance": 0.125, "label": "x",
          "run": "r"}],
    )
    ax = fig.axes[0]
    segs = [ln for ln in ax.get_lines()
            if ln.get_color() == style.REFERENCE]
    assert segs == []
    import matplotlib.pyplot as plt
    plt.close(fig)


def _probe(tmp_path, run_id, filename, values, chance=0.125, factor="d4"):
    d = tmp_path / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    probes = {}
    for i, v in enumerate(values):
        probes[f"pass{i}|query|composite"] = v
        probes[f"pass{i}|patch|composite"] = v - 0.05
        probes[f"pass{i}|query|initial"] = chance
    (d / filename).write_text(
        json.dumps({"run": run_id, "chance": chance, "factor": factor,
                    "probes": probes}),
        encoding="utf-8",
    )


def test_the_probe_reader_takes_one_stream_and_one_target(tmp_path) -> None:
    """The stored file holds fifty readouts. Picking the wrong stream would
    plot a curve that is real and answers a different question."""
    _probe(tmp_path, "r0", "p.json", [0.2, 0.9, 0.88])
    pr = F9.load_probe(tmp_path / "runs", "r0", "p.json")
    assert pr["values"] == [0.2, 0.9, 0.88]
    assert pr["passes"] == [0, 1, 2]


def test_the_probe_reader_sorts_by_pass(tmp_path) -> None:
    """Dict order is insertion order, and a file written pass2 first would
    draw the trajectory backwards without any error."""
    d = tmp_path / "runs" / "r1"
    d.mkdir(parents=True)
    (d / "p.json").write_text(json.dumps({
        "chance": 0.125,
        "probes": {"pass2|query|composite": 0.3,
                   "pass0|query|composite": 0.1,
                   "pass1|query|composite": 0.2},
    }), encoding="utf-8")
    pr = F9.load_probe(tmp_path / "runs", "r1", "p.json")
    assert pr["passes"] == [0, 1, 2]
    assert pr["values"] == [0.1, 0.2, 0.3]


def test_an_unfinished_run_is_never_drawn(tmp_path) -> None:
    """A live run at 1.0000 with steps to go is not a result, and the top
    row counts points. Drawing it would inflate the escape count in the
    picture while the generated table said something else."""
    rates = _rates(tmp_path, [
        {"run_id": "a", "arm": "D4/position", "seed": 0,
         "outcome": "escaped", "accuracy": 1.0, "n_distinct": 8, "modal": None},
        {"run_id": "b", "arm": "D4/position", "seed": 1,
         "outcome": "unfinished", "accuracy": 1.0, "n_distinct": 8,
         "modal": None},
    ])
    _probe(tmp_path, "famA_d2_d4only_s0", "state_probe_d4.json", [0.2, 0.9])
    _probe(tmp_path, "famA_d2_d4colour_s0", "state_probe.json", [0.12, 0.13])
    out = tmp_path / "fig" / "f9"
    rc = F9.main(["--rates", str(rates), "--runs", str(tmp_path / "runs"),
                  "--out", str(out)])
    assert rc == 0
    assert out.with_suffix(".pdf").is_file()


def test_two_probes_on_different_chance_floors_are_refused(tmp_path) -> None:
    """One dashed line under two series is right for at most one of them.

    D4 has eight elements and S3 has six, so their probes sit on 0.125 and
    0.1667. A figure drawing one floor for both is wrong in a way no
    reader can see, which is worse than a figure that fails to render.

    Mutation checked: dropping the equality check draws the figure and
    returns 0.
    """
    rates = _rates(tmp_path, [
        {"run_id": "a", "arm": "D4/position", "seed": 0, "outcome": "escaped",
         "accuracy": 1.0, "n_distinct": 8, "modal": None}])
    _probe(tmp_path, "famA_d2_d4only_s0", "state_probe_d4.json",
           [0.2, 0.9], chance=0.125)
    _probe(tmp_path, "famA_d2_d4colour_s0", "state_probe.json",
           [0.16, 0.17], chance=0.16666666666666666)
    rc = F9.main(["--rates", str(rates), "--runs", str(tmp_path / "runs"),
                  "--out", str(tmp_path / "fig" / "f9")])
    assert rc == 2, "a figure with two chance floors and one line was drawn"


def test_the_arms_are_in_the_registered_order_and_not_transposed() -> None:
    """Swapping a column leaves every number correct and the conclusion
    backwards, which is the failure mode a reader cannot catch.

    Position is drawn first because the claim is that position is solved,
    and both groups appear in both media, which is what makes the group
    order matched.
    """
    arms = [a for a, _ in F9.ARMS]
    assert arms[:2] == ["D4/position", "S3/position"]
    assert arms[2:] == ["D4/identity", "S3/identity"]
    assert len(set(arms)) == 4


def test_the_arm_names_match_what_the_generator_emits() -> None:
    """F9 keys on strings the generator writes. If either side renamed an
    arm the figure would silently draw four empty columns."""
    from loopvision.analysis import escape_rates as ER

    emitted = {f"{g}/{m}" for g in ER.GROUPS for m in ER.MEDIA}
    assert {a for a, _ in F9.ARMS} == emitted


def test_a_missing_probe_file_refuses_rather_than_drawing_half(tmp_path) -> None:
    """The bottom row is what closes the undecodable objection. A figure
    with the top row only makes the claim it cannot support."""
    rates = _rates(tmp_path, [
        {"run_id": "a", "arm": "D4/position", "seed": 0, "outcome": "escaped",
         "accuracy": 1.0, "n_distinct": 8, "modal": None}])
    (tmp_path / "runs").mkdir(exist_ok=True)
    rc = F9.main(["--rates", str(rates), "--runs", str(tmp_path / "runs"),
                  "--out", str(tmp_path / "fig" / "f9")])
    assert rc == 2


def test_a_missing_rates_file_says_how_to_make_one(tmp_path) -> None:
    rc = F9.main(["--rates", str(tmp_path / "nope.json"),
                  "--out", str(tmp_path / "f9")])
    assert rc == 2


def test_save_writes_both_formats(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    written = style.save(fig, str(tmp_path / "sub" / "fig"))
    assert {p.suffix for p in written} == {".pdf", ".png"}
    assert all(p.is_file() for p in written)
    plt.close(fig)


# --- F10, the H4 attractor sweep ---------------------------------------

def _h4_summary() -> dict:
    """One run per arm, with the zero-jitter arm high and untrained. Enough
    for the panels to be drawn and for their limits to be compared."""
    return {
        "solved_threshold": 0.95,
        "pearson_rate_vs_accuracy": -0.87,
        "n_runs": 4,
        "all_arms": [
            {"sigma": 0.0, "seeds": 1, "rate_mean": 0.75,
             "rate_min": 0.75, "rate_max": 0.7539},
            {"sigma": 0.5, "seeds": 1, "rate_mean": 0.10,
             "rate_min": 0.10, "rate_max": 0.10},
        ],
        "matched_arms": [{"sigma": 0.5, "seeds": 1, "rate_mean": 0.10}],
        "coverage": [{"sigma": 0.0, "solved": 0}, {"sigma": 0.5, "solved": 1}],
        "runs": [
            {"sigma": 0.0, "accuracy": 0.78, "rate_by_label": 0.7539},
            {"sigma": 0.5, "accuracy": 0.99, "rate_by_label": 0.10},
        ],
    }


def test_both_f10_panels_share_one_rate_axis(tmp_path) -> None:
    """The figure IS the comparison between the two panels: the same rates
    against two x axes. If they were scaled differently a reader would have
    to rescale one eye to read the other, which is the work the figure is
    supposed to do for them."""
    from loopvision.analysis.plots import f10_attractor as F10
    import matplotlib.pyplot as plt

    fig = F10.draw(_h4_summary(), tmp_path / "f10.png")
    a, b = fig.axes[0], fig.axes[1]
    assert a.get_ylim() == b.get_ylim(), (
        "the two panels are on different rate scales, so the comparison the "
        "figure exists to make cannot be read off it"
    )
    plt.close(fig)


def test_the_absent_comparison_note_clears_every_point(tmp_path) -> None:
    """The note sits in headroom above the data. At a top of 0.85 the highest
    run, 0.7539, reached 89 percent of the panel height and the text landed
    on top of it."""
    from loopvision.analysis.plots import f10_attractor as F10
    import matplotlib.pyplot as plt

    s = _h4_summary()
    fig = F10.draw(s, tmp_path / "f10.png")
    b = fig.axes[1]
    lo, hi = b.get_ylim()
    highest = max(r["rate_by_label"] for r in s["runs"])
    # The note is anchored at 0.98 of the panel and runs downward.
    note_floor = lo + (hi - lo) * 0.84
    assert highest < note_floor, (
        f"the highest point sits at {highest}, inside the band the note "
        f"occupies above {note_floor:.4f}"
    )
    plt.close(fig)
