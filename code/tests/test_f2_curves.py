"""F2 is the H1 figure, so the ways it could manufacture a dissociation
are the ways to test it.

Three of them are specific to this curve. A censored cell is recorded at
33, so leaving one in a quantile puts a spike on the depth line where
there is no measurement and makes depth look worse than it is. The two
task definitions that carry H1 are named explicitly, because the store
also holds family B runs from before D-028 moved the breadth range and
pooling them would put a task redefinition on the x axis. And the summary
is taken in `m1_curves`, where it is tested, rather than in the plot.
"""

from __future__ import annotations

import json

import pytest

from loopvision.analysis import m1_curves as M1
from loopvision.analysis.plots import f2_curves as F2


def _frame(rows):
    import pandas as pd

    df = pd.DataFrame(rows)
    df["censored"] = df.k_min == M1.CENSORED
    return df


def test_a_censored_cell_is_excluded_from_the_quantiles_and_counted() -> None:
    """33 is one past the measurable range. A median that includes it is
    not a k_min at all, and on the depth curve it would draw a spike.

    Mutation checked: quantiling over all cells instead of the uncensored
    ones returns 33 here.
    """
    rows = [{"run_id": "r", "seed": 0, "task_key": "T", "depth": d,
             "breadth": 12, "k_min": 2}
            for d in (1, 2)]
    rows += [{"run_id": "r", "seed": 1, "task_key": "T", "depth": 1,
              "breadth": 12, "k_min": M1.CENSORED}]
    s = M1.curve_summary(_frame(rows))["T"]["axes"]["depth"]
    at1 = [r for r in s if r["depth"] == 1][0]
    assert at1["median"] == pytest.approx(2.0)
    assert at1["censored"] == 1
    assert at1["n"] == 1


def test_a_cell_that_is_entirely_censored_reports_no_median(tmp_path) -> None:
    """Reporting 33, or 0, would both be inventions. None says the curve
    has no measurement there."""
    rows = [{"run_id": "r", "seed": 0, "task_key": "T", "depth": 1,
             "breadth": 12, "k_min": M1.CENSORED},
            {"run_id": "r", "seed": 0, "task_key": "T", "depth": 2,
             "breadth": 12, "k_min": 3}]
    s = M1.curve_summary(_frame(rows))["T"]["axes"]["depth"]
    at1 = [r for r in s if r["depth"] == 1][0]
    assert at1["median"] is None and at1["censored"] == 1


def test_an_axis_that_is_not_swept_is_not_a_curve() -> None:
    """Family C holds depth at 1. A one point depth curve would draw as a
    flat line and read as evidence that depth does nothing."""
    rows = [{"run_id": "r", "seed": 0, "task_key": "C", "depth": 1,
             "breadth": b, "k_min": 1} for b in (4, 5, 6)]
    axes = M1.curve_summary(_frame(rows))["C"]["axes"]
    assert "breadth" in axes
    assert "depth" not in axes, "a single depth was drawn as a curve"


def test_the_summary_keeps_tasks_apart() -> None:
    """D-028 moved family B's breadths from 6 to 9 up to 12 to 16. Pooling
    the two would put a task redefinition on the breadth axis."""
    rows = [{"run_id": "old", "seed": 0, "task_key": "B old", "depth": d,
             "breadth": 6, "k_min": 1} for d in (1, 2)]
    rows += [{"run_id": "new", "seed": 1, "task_key": "B new", "depth": d,
              "breadth": 12, "k_min": 3} for d in (1, 2)]
    s = M1.curve_summary(_frame(rows))
    assert set(s) == {"B old", "B new"}
    assert s["B old"]["axes"]["depth"][0]["median"] == pytest.approx(1.0)
    assert s["B new"]["axes"]["depth"][0]["median"] == pytest.approx(3.0)


def test_the_figure_names_only_the_two_tasks_that_carry_h1() -> None:
    """Every other task_key in the store is family A at one depth, or a
    family B run from before the breadth range moved."""
    assert F2.DEPTH_TASK.startswith("B d(1, 2, 3, 4, 5, 6)")
    keys = [k for k, _ in F2.BREADTH_TASKS]
    assert F2.DEPTH_TASK in keys, "family B's own breadth arm is the control"
    assert any(k.startswith("C ") for k in keys), "family C is the second arm"


def test_family_b_supplies_both_curves_from_the_same_runs() -> None:
    """The strongest form of the dissociation: one task, one set of
    weights, one set of seeds, depth rising and breadth flat. If the depth
    curve and the breadth control came from different tasks, everything
    that differs between them would be uncontrolled."""
    assert F2.BREADTH_TASKS[0][0] == F2.DEPTH_TASK


def _summary(tmp_path, depth_med, breadth_med, censored=0):
    s = {
        "tau": 0.90, "censored_at": 33,
        "tasks": {
            F2.DEPTH_TASK: {
                "runs": 20, "seeds": 8, "cells": 600,
                "axes": {
                    "depth": [{"depth": i + 1, "median": m, "q25": m,
                               "q75": m, "n": 100,
                               "censored": censored if i == 5 else 0}
                              for i, m in enumerate(depth_med)],
                    "breadth": [{"breadth": 12 + i, "median": m, "q25": m,
                                 "q75": m, "n": 120, "censored": 0}
                                for i, m in enumerate(breadth_med)],
                },
            },
            "C d(1,) b(4, 5, 6, 7, 8)": {
                "runs": 5, "seeds": 5, "cells": 25,
                "axes": {"breadth": [{"breadth": 4 + i, "median": 1.0,
                                      "q25": 1.0, "q75": 1.0, "n": 5,
                                      "censored": 0} for i in range(5)]},
            },
        },
    }
    p = tmp_path / "m1_summary.json"
    p.write_text(json.dumps(s), encoding="utf-8")
    return p


def test_the_figure_draws_from_the_stored_summary(tmp_path) -> None:
    p = _summary(tmp_path, [1, 1, 2, 2, 2, 3], [2, 2, 2, 2, 2], censored=7)
    out = tmp_path / "fig" / "f2"
    assert F2.main(["--summary", str(p), "--out", str(out)]) == 0
    assert out.with_suffix(".pdf").is_file()


def test_a_missing_task_refuses_rather_than_drawing_one_arm(tmp_path) -> None:
    """A figure with the depth curve and no breadth control is the depth
    trend, which is close to known in text. The dissociation is the
    contribution and half of it is not a weaker version of it."""
    s = json.loads(_summary(tmp_path, [1, 2], [2, 2]).read_text())
    del s["tasks"]["C d(1,) b(4, 5, 6, 7, 8)"]
    p = tmp_path / "partial.json"
    p.write_text(json.dumps(s), encoding="utf-8")
    assert F2.main(["--summary", str(p), "--out", str(tmp_path / "f2")]) == 2


def test_the_censored_label_names_every_depth_that_has_them(tmp_path) -> None:
    """It said "all at depth" the worst one, which the registered arm made
    false: its censored cells sit at depths 5 and 6. D-074."""
    s = json.loads(_summary(tmp_path, [1, 1, 2, 2, 2, 3], [2] * 5).read_text())
    rows = s["tasks"][F2.DEPTH_TASK]["axes"]["depth"]
    rows[4]["censored"], rows[5]["censored"] = 3, 5
    depth = F2.series_for(s, F2.DEPTH_TASK, "depth")
    breadths = [(F2.series_for(s, k, "breadth"), label)
                for k, label in F2.BREADTH_TASKS]
    fig = F2.draw(depth, breadths, s["tau"])
    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "8 of" in texts
    assert "3 at depth 5" in texts and "5 at depth 6" in texts
    assert "all at" not in texts


def test_a_missing_summary_says_how_to_make_one(tmp_path) -> None:
    assert F2.main(["--summary", str(tmp_path / "nope.json"),
                    "--out", str(tmp_path / "f2")]) == 2


def test_series_for_raises_on_an_axis_the_task_does_not_sweep(tmp_path) -> None:
    s = json.loads(_summary(tmp_path, [1, 2], [2, 2]).read_text())
    with pytest.raises(KeyError):
        F2.series_for(s, "C d(1,) b(4, 5, 6, 7, 8)", "depth")
