"""G2 must be able to fail, and must refuse to answer before it can.

The gate was moved from family A depth 3 because there it compared two
models both sitting on the chance floor, so every outcome was identical.
The replacement is only worth having if it can still come out either way,
and if a missing baseline reads as "not yet" rather than as a pass.
"""

from __future__ import annotations

import pandas as pd
import pytest

from loopvision.analysis import gate_g2 as G2


def _write(root, run_id, acc):
    d = root / run_id
    d.mkdir(parents=True)
    pd.DataFrame({
        "step": [300000] * 3, "k": [4] * 3, "split": ["iid_val"] * 3,
        "depth": [1, 2, 3], "breadth": [12] * 3,
        "metric_name": ["final_accuracy"] * 3, "value": [acc] * 3, "seed": [0] * 3,
    }).to_parquet(d / "metrics.parquet")


def test_a_missing_baseline_is_not_a_pass(tmp_path) -> None:
    """The failure that would matter most: a gate that reports success
    because the thing it compares against was never run."""
    for r in G2.LOOPED_RUNS:
        _write(tmp_path, r, 0.99)
    checks, passed = G2.run(tmp_path)
    assert passed is None, "a gate with no baseline must not report a verdict"
    assert all(c["passed"] is None for c in checks)


def test_the_gate_fails_when_the_baseline_wins(tmp_path) -> None:
    """This is the outcome the recent literature says is plausible, so it
    has to be reachable."""
    for r in G2.LOOPED_RUNS:
        _write(tmp_path, r, 0.70)
    for r in G2.FFWD_RUNS + G2.ECHO_RUNS:
        _write(tmp_path, r, 0.95)
    checks, passed = G2.run(tmp_path)
    assert passed is False
    assert all(c["passed"] is False for c in checks)


def test_the_gate_passes_when_loops_win(tmp_path) -> None:
    for r in G2.LOOPED_RUNS:
        _write(tmp_path, r, 0.99)
    for r in G2.FFWD_RUNS + G2.ECHO_RUNS:
        _write(tmp_path, r, 0.60)
    _, passed = G2.run(tmp_path)
    assert passed is True


def test_a_win_inside_the_margin_does_not_count(tmp_path) -> None:
    """Two runs of anything differ. A gate that trips on noise is worse
    than no gate, so the margin has to bite."""
    for r in G2.LOOPED_RUNS:
        _write(tmp_path, r, 0.9000)
    for r in G2.FFWD_RUNS + G2.ECHO_RUNS:
        _write(tmp_path, r, 0.8950)
    _, passed = G2.run(tmp_path)
    assert passed is False, "a 0.005 win is inside the 0.02 margin"


def test_one_arm_failing_fails_the_gate(tmp_path) -> None:
    """Both checks must pass. Beating feedforward while losing to echo
    would mean the loops add nothing over simply refreshing the state."""
    for r in G2.LOOPED_RUNS:
        _write(tmp_path, r, 0.99)
    for r in G2.FFWD_RUNS:
        _write(tmp_path, r, 0.60)
    for r in G2.ECHO_RUNS:
        _write(tmp_path, r, 0.99)
    _, passed = G2.run(tmp_path)
    assert passed is False
