"""F8 asserts that two instruments agree about what they measured.

The figure's whole claim is that a lens and a probe on the SAME states
disagree in an informative way. If they are quietly on different runs,
different loop counts, different item counts or different factors, the
panel still renders and still looks like a comparison. So every one of
those has to stop the figure rather than decorate it.

The second hazard is the probe site. `state_probe` trains at query and at
patch; `decode` pools at query. Reading the patch site beside the lens
would compare what the head sees against somewhere it cannot see.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from loopvision.analysis.plots import f8_absent as F8

TARGETS = ["initial", "op1", "op2", "partial", "composite"]


def _lens(run="r0", factor="d4", k=4, n=2560, chance=0.125, grid=None):
    return {"run": run, "factor": factor, "k": k, "n": n, "chance": chance,
            "targets": TARGETS,
            "grid": grid or [[0.1] * 5 for _ in range(5)]}


def _probe(run="r0", factor="d4", k=4, n=2560, chance=0.125,
           query=0.3, patch=0.9):
    probes = {}
    for i in range(5):
        for t in TARGETS:
            probes[f"pass{i}|query|{t}"] = query
            probes[f"pass{i}|patch|{t}"] = patch
    return {"run": run, "factor": factor, "k": k, "n": n, "chance": chance,
            "probes": probes}


def test_matching_instruments_pass() -> None:
    assert F8.same_states(_lens(), _probe()) == []


@pytest.mark.parametrize("field,value", [
    ("run", "someone_else"),
    ("factor", "s3"),
    ("k", 8),
    ("n", 1280),
    ("chance", 1 / 6),
])
def test_any_disagreement_stops_the_figure(field, value) -> None:
    """Each of these renders a perfectly plausible panel that is not a
    comparison. A different k is a different number of states; a different
    n is a different evaluation set; a different factor is a different
    question entirely.

    Mutation checked: dropping any single check from same_states fails the
    corresponding case here and nothing else.
    """
    bad = F8.same_states(_lens(), _probe(**{field: value}))
    assert bad, f"a mismatched {field} was allowed through"


def test_the_probe_is_read_at_the_query_site() -> None:
    """decode pools over the query tokens, so query is the only site the
    head could read. Taking the patch site would put the head's reading
    beside somewhere the head cannot see.

    Mutation checked: defaulting SITE to patch returns the patch values.
    """
    g = F8.probe_grid(_probe(query=0.30, patch=0.90), TARGETS, F8.SITE)
    assert F8.SITE == "query"
    assert np.allclose(g, 0.30), "the patch site was read"


def test_the_probe_grid_is_ordered_by_pass() -> None:
    """Dict order is insertion order. A probe file written pass2 first
    would draw the trajectory upside down with no error."""
    p = _probe()
    for t in TARGETS:
        p["probes"][f"pass0|query|{t}"] = 0.11
        p["probes"][f"pass4|query|{t}"] = 0.44
    g = F8.probe_grid(p, TARGETS, "query")
    assert g[0][0] == pytest.approx(0.11)
    assert g[-1][0] == pytest.approx(0.44)


def test_the_probe_grid_columns_follow_the_lens_target_order() -> None:
    """Both grids are drawn on the same x axis, so a probe column order
    that differed from the lens would label one row wrongly."""
    p = _probe()
    for i in range(5):
        p["probes"][f"pass{i}|query|composite"] = 0.9
    g = F8.probe_grid(p, TARGETS, "query")
    assert g[0][TARGETS.index("composite")] == pytest.approx(0.9)
    assert g[0][TARGETS.index("initial")] == pytest.approx(0.3)


def test_the_probe_is_found_by_factor_not_by_filename(tmp_path) -> None:
    """The two runs store the probe under different names. Matching the
    filename picks the wrong factor on one of them, and an S3 probe beside
    a D4 lens is two different questions on one axis."""
    d = tmp_path / "runs" / "r0"
    d.mkdir(parents=True)
    (d / "state_probe.json").write_text(
        json.dumps(_probe(factor="s3", chance=1 / 6)), encoding="utf-8")
    (d / "state_probe_d4.json").write_text(
        json.dumps(_probe(factor="d4")), encoding="utf-8")
    got = F8.load_probe(d, "d4")
    assert got["factor"] == "d4"
    assert got["path"].endswith("state_probe_d4.json")


def test_a_missing_probe_for_that_factor_raises(tmp_path) -> None:
    d = tmp_path / "runs" / "r0"
    d.mkdir(parents=True)
    (d / "state_probe.json").write_text(
        json.dumps(_probe(factor="s3")), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        F8.load_probe(d, "d4")


def test_the_result_comes_first_and_the_control_is_not_an_inset() -> None:
    """paper.md section 6 requires the control to be as prominent as the
    result. Equal panels in a two by two is how that is met, and the order
    puts the null first because that is the order the argument is made in.
    """
    runs = [r for r, _ in F8.COLUMNS]
    assert runs == ["famA_d2_d4colour_s0", "famA_d2_d4only_s0"]
    assert "failed" in F8.COLUMNS[0][1]
    assert "solved" in F8.COLUMNS[1][1]


def test_every_panel_shares_one_scale() -> None:
    """Four panels scaled independently would make a model at chance and a
    model at 1.0 look equally bright, which inverts the whole figure."""
    panels = [
        {"label": "a", "lens_grid": np.full((5, 5), 0.12),
         "probe_grid": np.full((5, 5), 0.13), "run": "a", "n": 1, "k": 4},
        {"label": "b", "lens_grid": np.full((5, 5), 1.0),
         "probe_grid": np.full((5, 5), 0.9), "run": "b", "n": 1, "k": 4},
    ]
    fig = F8.draw(panels, 0.125, TARGETS)
    clims = {im.get_clim() for ax in fig.axes for im in ax.get_images()}
    assert clims == {(0.0, 1.0)}, clims
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_main_refuses_when_a_run_lacks_an_instrument(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    assert F8.main(["--runs", str(tmp_path / "runs"),
                    "--out", str(tmp_path / "f8")]) == 2
