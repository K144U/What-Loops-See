"""F6 puts two different quantities on one axis, so it has to be exact.

The top row is accuracy when every item runs the same fixed k. The bottom
row is where per-item rules stop. They share an x and they are not
comparable point for point, which is a misreading the figure has to make
hard rather than invite. Beyond that, the summary it reads must exclude
the censored sentinel from its quantiles, keep task definitions apart, and
carry both oracles rather than silently picking the one prior work has.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from loopvision.analysis import m5_h2 as M5
from loopvision.analysis.plots import f6_halting as F6
from loopvision.instruments.m5_halting import CENSORED, CRITERIA


def _items(n=8, k_model=1, k_task=4, fires=None, seed=0):
    fires = fires or {c: 2 for c in CRITERIA}
    rows = []
    for i in range(n):
        row = {"run_id": f"r{seed}", "seed": seed, "task_key": "T",
               "k_task": k_task, "k_model": k_model}
        for c in CRITERIA:
            row[f"k_hat_{c}"] = fires[c]
        rows.append(row)
    return rows


def test_the_censored_sentinel_never_enters_a_quantile(tmp_path) -> None:
    """A rule that never fires is recorded at -1. Averaging that in would
    drag its median below every real firing point and make the worst
    behaved rule look like the earliest.

    Mutation checked: quantiling the raw column returns a negative median.
    """
    rows = _items(4, fires={c: 3 for c in CRITERIA})
    rows += _items(4, fires={**{c: 3 for c in CRITERIA}, "kl": CENSORED})
    s = M5.halting_summary(pd.DataFrame(rows), tmp_path, [])
    assert s["criteria"]["kl"]["median"] == pytest.approx(3.0)
    assert s["criteria"]["kl"]["never_fired"] == 4
    assert s["criteria"]["kl"]["n"] == 4


def test_a_rule_that_never_fires_at_all_reports_no_median(tmp_path) -> None:
    rows = _items(4, fires={**{c: 2 for c in CRITERIA}, "kl": CENSORED})
    s = M5.halting_summary(pd.DataFrame(rows), tmp_path, [])
    assert s["criteria"]["kl"].get("median") is None
    assert s["criteria"]["kl"]["never_fired"] == 4


def test_both_oracles_are_carried(tmp_path) -> None:
    """Prior work has only k*_model. Dropping k*_task would make this
    figure the same comparison everyone else can already draw."""
    s = M5.halting_summary(pd.DataFrame(_items()), tmp_path, [])
    assert s["oracles"]["k_model"]["median"] == pytest.approx(1.0)
    assert s["oracles"]["k_task"]["median"] == pytest.approx(4.0)


def test_the_accuracy_curve_is_read_from_the_runs(tmp_path) -> None:
    """Averaged in the module, not the figure, because a plot script may
    not take a statistic."""
    d = tmp_path / "r0"
    d.mkdir()
    rows = []
    for k, v in ((1, 0.5), (1, 0.7), (2, 0.9)):
        rows.append({"metric_name": "sweep_accuracy", "k": k, "depth": 2,
                     "breadth": 12, "value": v})
    rows.append({"metric_name": "loss", "k": 1, "depth": 2, "breadth": 12,
                 "value": 99.0})
    pd.DataFrame(rows).to_parquet(d / "metrics.parquet", index=False)
    curve = M5.accuracy_curve(tmp_path, ["r0"])
    got = {c["k"]: c["accuracy"] for c in curve}
    assert got[1] == pytest.approx(0.6), "the two k=1 cells were not averaged"
    assert got[2] == pytest.approx(0.9)
    assert 99.0 not in got.values(), "a non-accuracy metric was averaged in"


def test_a_run_without_metrics_is_skipped_not_fatal(tmp_path) -> None:
    assert M5.accuracy_curve(tmp_path, ["missing"]) == []


def _summary(tmp_path, tasks=1, curve=True):
    task = {
        "n_seeds": 8, "n_items": 2048, "runs": ["r0"],
        "criteria": {c: {"median": 1.0 + i, "q25": 1.0, "q75": 2.0 + i,
                         "n": 100, "never_fired": 0}
                     for i, c in enumerate(CRITERIA)},
        "oracles": {"k_model": {"median": 1.0, "q25": 1.0, "q75": 2.0, "n": 8},
                    "k_task": {"median": 4.0, "q25": 2.0, "q75": 5.0, "n": 8}},
        "accuracy_curve": ([{"k": k, "accuracy": a, "cells": 10}
                            for k, a in ((1, 0.72), (4, 0.99), (8, 0.997),
                                         (64, 0.82))] if curve else []),
    }
    d = {"alpha": 0.05, "family_size": 14,
         "criteria_order": list(CRITERIA), "tasks": {"T": task}}
    if tasks > 1:
        d["tasks"]["T2"] = json.loads(json.dumps(task))
    p = tmp_path / "m5_summary.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_two_task_definitions_are_never_pooled_silently(tmp_path) -> None:
    """The fault M1 shipped with once: two task definitions in one fit.
    Here it would be two task definitions in one panel.

    Mutation checked: picking the first task when several exist draws a
    figure and returns 0.
    """
    p = _summary(tmp_path, tasks=2)
    assert F6.main(["--summary", str(p), "--out", str(tmp_path / "f6")]) == 2
    assert not (tmp_path / "f6.pdf").exists()


def test_naming_the_task_resolves_it(tmp_path) -> None:
    p = _summary(tmp_path, tasks=2)
    out = tmp_path / "fig" / "f6"
    assert F6.main(["--summary", str(p), "--task", "T",
                    "--out", str(out)]) == 0
    assert out.with_suffix(".pdf").is_file()


def test_a_single_task_needs_no_naming(tmp_path) -> None:
    p = _summary(tmp_path)
    out = tmp_path / "fig" / "f6"
    assert F6.main(["--summary", str(p), "--out", str(out)]) == 0


def test_a_missing_accuracy_curve_refuses_half_the_figure(tmp_path) -> None:
    """Without the curve the bottom row is a set of offsets with nothing
    at stake, and the practical point of the figure is the cost."""
    p = _summary(tmp_path, curve=False)
    assert F6.main(["--summary", str(p), "--out", str(tmp_path / "f6")]) == 2


def test_the_criteria_keep_the_registered_order(tmp_path) -> None:
    """The preregistration names four criteria in an order and the panel
    reads top to bottom. Sorting them alphabetically would put the best
    published rule in the middle for no reason."""
    p = _summary(tmp_path)
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["criteria_order"] == list(CRITERIA)
    assert list(F6.LABELS) == list(CRITERIA)


def test_every_criterion_has_a_printed_name() -> None:
    """A missing key would fall back to the raw column name mid-figure."""
    for c in CRITERIA:
        assert c in F6.LABELS and F6.LABELS[c]


def test_a_missing_summary_says_how_to_make_one(tmp_path) -> None:
    assert F6.main(["--summary", str(tmp_path / "nope.json"),
                    "--out", str(tmp_path / "f6")]) == 2


def test_the_order_the_figure_receives_is_the_registered_one(tmp_path) -> None:
    """The gap the previous test left.

    It checked the summary's constant and the label table, neither of
    which is what draw() is handed, so sorting the criteria alphabetically
    inside main passed every test in this file. Alphabetical puts the best
    published rule in the middle of the panel for no reason.

    Mutation checked: returning sorted(...) unconditionally fails here.
    """
    summary = {"criteria_order": list(CRITERIA), "tasks": {}}
    task = {"criteria": {c: {} for c in CRITERIA}}
    assert F6.order_for(summary, task) == list(CRITERIA)
    assert F6.order_for(summary, task) != sorted(CRITERIA), \
        "the registered order happens to be alphabetical, so this test " \
        "no longer distinguishes anything and needs a different fixture"


def test_a_summary_without_an_order_falls_back_predictably(tmp_path) -> None:
    task = {"criteria": {c: {} for c in CRITERIA}}
    assert F6.order_for({"tasks": {}}, task) == sorted(CRITERIA)
