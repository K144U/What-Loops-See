"""H4 aggregation, and the confound check that decides whether it says anything.

The raw sweep looks like strong support. The whole value of this module is
the second look, so these tests are mostly about that: that the correlation
is computed, that an arm with no trained run is reported as an absent
comparison rather than a low score, and that the matched table drops runs
that did not train.
"""

from __future__ import annotations

import json

import pytest

from loopvision.analysis import m6_h4 as H


def _row(run, sigma, rate, acc, agree=True):
    return {"run": run, "jitter_sigma": sigma, "rate_by_label": rate,
            "rate_by_prediction": rate, "readings_agree": agree}


def _written(tmp_path, rows):
    p = tmp_path / "rates.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return p


def _runs(tmp_path, accs):
    for name, acc in accs.items():
        d = tmp_path / "runs" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(
            json.dumps({"final_accuracy": acc}), encoding="utf-8")
    return tmp_path / "runs"


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

def test_the_width_axis_is_not_part_of_the_jitter_sweep(tmp_path) -> None:
    """famB_curve111_d256 and friends vary WIDTH. Section 5 registers a
    jitter sweep, and nine of those runs sat in the first output looking
    like sweep members."""
    rates = _written(tmp_path, [
        _row("famB_curve111_s0", 0.5, 0.1, None),
        _row("famB_curve111_d256_s0", 0.5, 0.9, None),
    ])
    runs = _runs(tmp_path, {"famB_curve111_s0": 0.99, "famB_curve111_d256_s0": 0.99})
    got = H.load(rates, runs)
    assert [r["run"] for r in got] == ["famB_curve111_s0"]


def test_the_untagged_runs_are_the_half_sigma_arm(tmp_path) -> None:
    """0.5 is the default jitter in cli.py, so the base runs ARE that arm."""
    assert H.H4_ARM.fullmatch("famB_curve111_s7")
    assert H.H4_ARM.fullmatch("famB_curve111_jit100_s0")


# ---------------------------------------------------------------------------
# The confound, which is the point of the module
# ---------------------------------------------------------------------------

def test_a_rate_that_tracks_accuracy_is_reported_as_correlated(tmp_path) -> None:
    """The finding. If the rate simply follows accuracy it is the training
    outcome in disguise, and the summary has to say so numerically."""
    rows = [_row(f"famB_curve111_jit000_s{i}", 0.0, 0.9 - 0.1 * i, None)
            for i in range(4)]
    rates = _written(tmp_path, rows)
    runs = _runs(tmp_path, {f"famB_curve111_jit000_s{i}": 0.3 + 0.1 * i
                            for i in range(4)})
    s = H.summarise(H.load(rates, runs))
    assert s["pearson_rate_vs_accuracy"] == pytest.approx(-1.0, abs=1e-9)


def test_an_uncorrelated_rate_is_not_reported_as_correlated(tmp_path) -> None:
    """The control. Without it the correlation could be hardwired negative
    and the test above would still pass."""
    rows = [_row(f"famB_curve111_s{i}", 0.5, 0.2, None) for i in range(4)]
    rates = _written(tmp_path, rows)
    runs = _runs(tmp_path, {f"famB_curve111_s{i}": 0.3 + 0.1 * i for i in range(4)})
    s = H.summarise(H.load(rates, runs))
    assert s["pearson_rate_vs_accuracy"] is None, "a flat rate has no correlation"


def test_an_arm_with_no_trained_run_is_an_absent_comparison(tmp_path) -> None:
    """Not a low score. The zero-jitter arm has 0 of 5 above threshold, so
    the comparison H4 asks for does not exist in this sweep, and saying that
    is different from saying jitter helped."""
    rows = ([_row(f"famB_curve111_jit000_s{i}", 0.0, 0.6, None) for i in range(3)]
            + [_row(f"famB_curve111_s{i}", 0.5, 0.08, None) for i in range(3)])
    rates = _written(tmp_path, rows)
    accs = {f"famB_curve111_jit000_s{i}": 0.60 for i in range(3)}
    accs.update({f"famB_curve111_s{i}": 0.999 for i in range(3)})
    s = H.summarise(H.load(rates, _runs(tmp_path, accs)))

    zero = [c for c in s["coverage"] if c["sigma"] == 0.0][0]
    assert zero["solved"] == 0 and zero["seeds"] == 3
    assert 0.0 not in [a["sigma"] for a in s["matched_arms"]], (
        "an arm with no trained run must not appear in the matched table")


def test_the_matched_table_drops_runs_that_did_not_train(tmp_path) -> None:
    rows = [_row("famB_curve111_s0", 0.5, 0.90, None),
            _row("famB_curve111_s1", 0.5, 0.05, None)]
    rates = _written(tmp_path, rows)
    runs = _runs(tmp_path, {"famB_curve111_s0": 0.40, "famB_curve111_s1": 0.999})
    s = H.summarise(H.load(rates, runs))
    matched = [a for a in s["matched_arms"] if a["sigma"] == 0.5][0]
    assert matched["seeds"] == 1
    assert matched["rate_mean"] == pytest.approx(0.05)


def test_every_arm_still_appears_in_the_unmatched_table(tmp_path) -> None:
    """The raw sweep is reported too. Showing only the matched view would
    hide the number that looks like support, and the reader is owed both."""
    rows = ([_row("famB_curve111_jit000_s0", 0.0, 0.6, None)]
            + [_row("famB_curve111_s0", 0.5, 0.08, None)])
    rates = _written(tmp_path, rows)
    s = H.summarise(H.load(rates, _runs(
        tmp_path, {"famB_curve111_jit000_s0": 0.60, "famB_curve111_s0": 0.999})))
    assert sorted(a["sigma"] for a in s["all_arms"]) == [0.0, 0.5]


def test_a_run_with_no_summary_does_not_break_the_aggregation(tmp_path) -> None:
    """Accuracy can be missing. It must drop out of the correlation rather
    than being treated as zero, which would manufacture the confound."""
    rates = _written(tmp_path, [_row("famB_curve111_s0", 0.5, 0.08, None)])
    (tmp_path / "runs").mkdir(exist_ok=True)
    s = H.summarise(H.load(rates, tmp_path / "runs"))
    assert s["n_scored"] == 0
    assert s["pearson_rate_vs_accuracy"] is None


def test_a_positive_correlation_is_reported_as_positive(tmp_path) -> None:
    """Every other correlation test here expects -1.0 or None, which a
    function hardwired to return -1.0 satisfies. This one does not: the rate
    rises with accuracy, so the sign has to be computed."""
    rows = [_row(f"famB_curve111_s{i}", 0.5, 0.1 * i, None) for i in range(4)]
    rates = _written(tmp_path, rows)
    runs = _runs(tmp_path, {f"famB_curve111_s{i}": 0.3 + 0.1 * i for i in range(4)})
    s = H.summarise(H.load(rates, runs))
    assert s["pearson_rate_vs_accuracy"] == pytest.approx(1.0, abs=1e-9)


def test_a_partial_correlation_is_neither_one_nor_minus_one(tmp_path) -> None:
    """Pins the magnitude too, so the function cannot return the sign alone."""
    rates = _written(tmp_path, [
        _row("famB_curve111_s0", 0.5, 0.0, None),
        _row("famB_curve111_s1", 0.5, 1.0, None),
        _row("famB_curve111_s2", 0.5, 0.0, None),
        _row("famB_curve111_s3", 0.5, 1.0, None),
    ])
    runs = _runs(tmp_path, {"famB_curve111_s0": 0.10, "famB_curve111_s1": 0.20,
                            "famB_curve111_s2": 0.30, "famB_curve111_s3": 0.40})
    got = H.summarise(H.load(rates, runs))["pearson_rate_vs_accuracy"]
    assert got == pytest.approx(0.4472135955, abs=1e-6)
