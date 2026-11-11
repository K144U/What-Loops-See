"""Per iteration trajectory shape, the aggregation F5 reads.

The figure is not allowed to take a statistic, so the medians and quartiles
are computed here and tested here. The one thing that must not drift is which
metrics lead: RMSNorm before the head makes the raw norm invisible to the
model, so a figure that led with it would show a flat line and imply nothing
happens, when the direction is converging geometrically underneath.
"""

from __future__ import annotations

import pytest

from loopvision.analysis import m4_shape as S


@pytest.fixture
def parquet(tmp_path):
    """One run, two metrics, three items over two iterations."""
    pd = pytest.importorskip("pandas")

    rows = []
    # unit_step_norm: medians 2.0 then 1.0, so a fall is visible
    for it, vals in ((1, [1.0, 2.0, 3.0]), (2, [0.5, 1.0, 1.5])):
        for item, v in enumerate(vals):
            rows.append(("r0", item, it, 8, "unit_step_norm", v))
    # norm: flat, which is the trap
    for it in (1, 2):
        for item, v in enumerate([100.0, 100.0, 100.0]):
            rows.append(("r0", item, it, 8, "norm", v))
    df = pd.DataFrame(rows, columns=["run_id", "item", "iteration", "k",
                                     "metric_name", "value"])
    p = tmp_path / "m4_geometry.parquet"
    df.to_parquet(p)
    return p


def test_the_median_and_quartiles_are_per_iteration(parquet) -> None:
    got = S.summarise_run(parquet)
    m = got["metrics"]["unit_step_norm"]
    assert m["iteration"] == [1, 2]
    assert m["median"] == pytest.approx([2.0, 1.0])
    assert m["q1"] == pytest.approx([1.5, 0.75])
    assert m["q3"] == pytest.approx([2.5, 1.25])


def test_the_scale_free_metrics_lead_and_the_norm_follows() -> None:
    """The order is the order F5 draws in, so the leading metric is decided
    here rather than in the plot. RMSNorm makes the raw norm invisible to the
    model, and a figure that led with it would read as a null result."""
    assert S.DIRECTIONAL[0] == "unit_step_norm"
    assert "norm" in S.MAGNITUDE
    assert not set(S.DIRECTIONAL) & set(S.MAGNITUDE)


def test_both_kinds_are_still_reported(parquet) -> None:
    """Leading with direction does not mean hiding the norm. A reader who
    wants to check the trap has to be able to see it."""
    got = S.summarise_run(parquet)
    assert "unit_step_norm" in got["metrics"]
    assert "norm" in got["metrics"]


def test_an_undefined_point_stays_a_gap(tmp_path) -> None:
    """cos_step is undefined at i = 1 and ratio is gated. NaN is not JSON,
    and turning it into a number would let a reader draw a line across a
    place where nothing was measured."""
    pd = pytest.importorskip("pandas")
    rows = [("r0", 0, 1, 8, "cos_state", float("nan")),
            ("r0", 0, 2, 8, "cos_state", 0.5)]
    df = pd.DataFrame(rows, columns=["run_id", "item", "iteration", "k",
                                     "metric_name", "value"])
    p = tmp_path / "m4_geometry.parquet"
    df.to_parquet(p)
    m = S.summarise_run(p)["metrics"]["cos_state"]
    assert m["median"][0] is None
    assert m["median"][1] == pytest.approx(0.5)


def test_a_missing_file_returns_nothing_rather_than_raising(tmp_path) -> None:
    assert S.summarise_run(tmp_path / "absent.parquet") is None


def test_an_empty_summary_claims_nothing(capsys) -> None:
    rc = S.report({"runs": [], "directional": [], "magnitude": []})
    assert rc == 2
    assert "nothing is claimed" in capsys.readouterr().out.lower()


def test_the_summary_carries_which_metrics_are_inert(parquet) -> None:
    """F5 reads the split from the summary rather than hardcoding it, so the
    two cannot disagree about which panel is the misleading one."""
    s = S.summarise([parquet])
    assert s["directional"] == list(S.DIRECTIONAL)
    assert s["magnitude"] == list(S.MAGNITUDE)


def test_the_centre_is_a_median_and_not_a_mean(tmp_path) -> None:
    """Every other fixture here is symmetric, where the two agree, so a mean
    would pass them all. Trajectories are not symmetric: a handful of items
    that never converge drag a mean and leave a median alone, which is the
    whole reason this reports medians."""
    pd = pytest.importorskip("pandas")
    rows = [("r0", i, 1, 8, "unit_step_norm", v)
            for i, v in enumerate([1.0, 1.0, 1.0, 1.0, 96.0])]
    df = pd.DataFrame(rows, columns=["run_id", "item", "iteration", "k",
                                     "metric_name", "value"])
    p = tmp_path / "m4_geometry.parquet"
    df.to_parquet(p)
    got = S.summarise_run(p)["metrics"]["unit_step_norm"]
    assert got["median"][0] == pytest.approx(1.0), "median ignores the outlier"
    assert got["median"][0] != pytest.approx(20.0), "a mean would read 20.0"
