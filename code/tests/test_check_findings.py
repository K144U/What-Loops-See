"""The checker has to catch the fault it was written for.

The 2026-09-08 audit found findings.md quoting 0.1249 where the run
recorded 0.1234. The first test below is that exact shape: a table naming a
run, quoting a number one whisker away from the truth. If the checker
cannot fail on that, it is decoration.

The rest guard the ways a checker like this goes quietly useless: firing on
a chance column that is not an accuracy at all, missing a row because the
run id is not in the first cell, and reporting a clean pass over rows it
never actually attributed to anything.
"""

from __future__ import annotations

import json

import pytest

from loopvision.analysis import check_findings as CF


def _run(tmp_path, run_id: str, final: float, cells: dict[str, float] | None = None):
    d = tmp_path / "runs" / run_id
    d.mkdir(parents=True)
    (d / "summary.json").write_text(
        json.dumps({"final_accuracy": final, "by_cell": cells or {}}),
        encoding="utf-8",
    )
    return d


def _doc(tmp_path, body: str):
    p = tmp_path / "findings.md"
    p.write_text(body, encoding="utf-8")
    return p


def test_it_catches_a_number_that_is_close_but_not_recorded(tmp_path) -> None:
    """The fault this module exists for, in its original shape.

    Mutation checked: comparing at 2dp instead of the quoted precision, or
    dropping the `quoted not in got` branch, both make this pass.
    """
    _run(tmp_path, "famA_x_s0", 0.123413)
    doc = _doc(
        tmp_path,
        "| run | final accuracy |\n"
        "|---|---|\n"
        "| `famA_x_s0` | 0.1249 |\n",
    )
    r = CF.check(doc, tmp_path / "runs")
    assert len(r["mismatches"]) == 1
    m = r["mismatches"][0]
    assert m["run"] == "famA_x_s0"
    assert m["quoted"] == "0.1249"


def test_the_correct_number_passes(tmp_path) -> None:
    """The positive control. A right number must not be reported."""
    _run(tmp_path, "famA_x_s0", 0.123413)
    doc = _doc(
        tmp_path,
        "| run | final accuracy |\n|---|---|\n| `famA_x_s0` | 0.1234 |\n",
    )
    r = CF.check(doc, tmp_path / "runs")
    assert r["mismatches"] == []
    assert r["checked"] == 1


def test_a_per_cell_value_is_accepted(tmp_path) -> None:
    """Rows legitimately quote a cell rather than the pooled figure."""
    _run(tmp_path, "famA_x_s0", 0.123413, {"2x1": 0.127123, "2x2": 0.125908})
    doc = _doc(
        tmp_path,
        "| run | final accuracy |\n|---|---|\n| `famA_x_s0` | 0.1271 |\n",
    )
    r = CF.check(doc, tmp_path / "runs")
    assert r["mismatches"] == []


def test_a_chance_column_is_not_treated_as_an_accuracy(tmp_path) -> None:
    """0.1250 next to a real 0.1234 is the false positive that would make
    this checker unusable, so the column header decides what is checked."""
    _run(tmp_path, "famA_x_s0", 0.123413)
    doc = _doc(
        tmp_path,
        "| run | final accuracy | chance | loss |\n"
        "|---|---|---|---|\n"
        "| `famA_x_s0` | 0.1234 | 0.1250 | 2.0794 |\n",
    )
    r = CF.check(doc, tmp_path / "runs")
    assert r["mismatches"] == []
    assert r["checked"] == 1


def test_the_run_id_need_not_be_in_the_first_cell(tmp_path) -> None:
    _run(tmp_path, "famA_x_s0", 0.123413)
    doc = _doc(
        tmp_path,
        "| arm | run | final accuracy |\n"
        "|---|---|---|\n"
        "| feedforward | `famA_x_s0` | 0.1249 |\n",
    )
    r = CF.check(doc, tmp_path / "runs")
    assert len(r["mismatches"]) == 1


def test_a_row_naming_no_run_is_counted_not_silently_passed(tmp_path) -> None:
    """A checker that reports zero mismatches over rows it never attributed
    is worse than no checker, so the uncovered rows are counted."""
    doc = _doc(
        tmp_path,
        "| arm | final accuracy |\n|---|---|\n| feedforward | 0.1249 |\n",
    )
    (tmp_path / "runs").mkdir()
    r = CF.check(doc, tmp_path / "runs")
    assert r["mismatches"] == []
    assert r["unattributed"] == 1
    assert r["checked"] == 0


def test_an_unfinished_run_is_not_a_mismatch(tmp_path) -> None:
    (tmp_path / "runs" / "famA_x_s0").mkdir(parents=True)
    doc = _doc(
        tmp_path,
        "| run | final accuracy |\n|---|---|\n| `famA_x_s0` | 0.1249 |\n",
    )
    r = CF.check(doc, tmp_path / "runs")
    assert r["mismatches"] == []
    assert r["unfinished"] == 1


def test_a_table_can_opt_out(tmp_path) -> None:
    """The CORRECTION entry quotes the wrong numbers on purpose."""
    _run(tmp_path, "famA_x_s0", 0.123413)
    doc = _doc(
        tmp_path,
        "<!-- check-findings: skip -->\n\n"
        "| run | final accuracy |\n|---|---|\n| `famA_x_s0` | 0.1249 |\n",
    )
    r = CF.check(doc, tmp_path / "runs")
    assert r["mismatches"] == []
    assert r["skipped_tables"] == 1


def test_precision_follows_the_document(tmp_path) -> None:
    """A table quoting 0.125 must not be failed for a run at 0.1254."""
    _run(tmp_path, "famA_x_s0", 0.125427)
    doc = _doc(
        tmp_path,
        "| run | final accuracy |\n|---|---|\n| `famA_x_s0` | 0.125 |\n",
    )
    r = CF.check(doc, tmp_path / "runs")
    assert r["mismatches"] == []


def test_main_exits_nonzero_on_a_mismatch(tmp_path) -> None:
    """The exit code is what a hook or a CI step would read."""
    _run(tmp_path, "famA_x_s0", 0.123413)
    doc = _doc(
        tmp_path,
        "| run | final accuracy |\n|---|---|\n| `famA_x_s0` | 0.1249 |\n",
    )
    code = CF.main(["--doc", str(doc), "--runs", str(tmp_path / "runs")])
    assert code == 1
    code = CF.main(
        ["--doc", str(tmp_path / "nope.md"), "--runs", str(tmp_path / "runs")]
    )
    assert code == 2
