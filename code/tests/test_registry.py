"""The run index must not quietly hide runs.

CLAUDE.md routes every count in the documents through this module, so a
run it drops is a run that vanishes from the paper's totals. The tests that
matter are therefore about what it refuses to omit, not about formatting.
"""

from __future__ import annotations

import yaml

from loopvision.analysis import registry as R


def _run(root, name, *, cfg=None, done=False, metrics=False):
    d = root / name
    d.mkdir(parents=True)
    if cfg is not None:
        (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    if done:
        (d / "DONE").write_text("", encoding="utf-8")
    if metrics:
        import pandas as pd

        pd.DataFrame({
            "step": [100, 200], "k": [8, 8],
            "split": ["iid_val", "iid_val"], "depth": [2, 2], "breadth": [4, 4],
            "metric_name": ["accuracy", "accuracy"], "value": [0.5, 0.75],
            "seed": [0, 0],
        }).to_parquet(d / "metrics.parquet")
    return d


def test_an_empty_run_is_reported_not_skipped(tmp_path) -> None:
    """A submitted run that produced nothing is the case most worth seeing.

    Filtering it out would make a failed launch indistinguishable from a
    launch that never happened.
    """
    _run(tmp_path, "launched_and_died", cfg={"family": "A"})
    rows = R.scan(tmp_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "empty"
    assert R.counts(rows)["by_status"] == {"empty": 1}


def test_a_run_without_a_config_still_appears(tmp_path) -> None:
    """Unparseable metadata must not remove a run from the totals."""
    _run(tmp_path, "no_config", cfg=None)
    (tmp_path / "broken_config").mkdir()
    (tmp_path / "broken_config" / "config.yaml").write_text(
        "family: [unclosed", encoding="utf-8")
    rows = R.scan(tmp_path)
    assert {r["run_id"] for r in rows} == {"no_config", "broken_config"}
    assert R.counts(rows)["total"] == 2


def test_status_comes_from_the_filesystem(tmp_path) -> None:
    _run(tmp_path, "a_done", cfg={"family": "A"}, done=True, metrics=True)
    _run(tmp_path, "b_live", cfg={"family": "B"}, metrics=True)
    _run(tmp_path, "c_empty", cfg={"family": "C"})
    got = {r["run_id"]: r["status"] for r in R.scan(tmp_path)}
    assert got == {"a_done": "done", "b_live": "live", "c_empty": "empty"}


def test_a_done_run_is_done_even_with_no_metrics(tmp_path) -> None:
    _run(tmp_path, "sentinel_only", cfg={"family": "A"}, done=True)
    assert R.scan(tmp_path)[0]["status"] == "done"


def test_latest_step_is_the_maximum_not_the_last_row(tmp_path) -> None:
    """Rows are appended by several writers and are not ordered by step."""
    import pandas as pd

    d = _run(tmp_path, "unordered", cfg={"family": "A"})
    pd.DataFrame({
        "step": [900, 100, 500], "k": [8, 8, 8],
        "split": ["iid_val"] * 3, "depth": [2] * 3, "breadth": [4] * 3,
        "metric_name": ["accuracy"] * 3, "value": [0.9, 0.1, 0.5],
        "seed": [0] * 3,
    }).to_parquet(d / "metrics.parquet")
    r = R.scan(tmp_path)[0]
    assert r["step"] == 900
    assert r["accuracy"] == 0.9


def test_factor_defaults_to_full_for_older_runs(tmp_path) -> None:
    """Runs predating the factor option must not be read as restricted."""
    _run(tmp_path, "old", cfg={"family": "A", "depths": [2]}, metrics=True)
    assert R.scan(tmp_path)[0]["factor"] == "full"


def test_counts_add_up(tmp_path) -> None:
    _run(tmp_path, "a", cfg={"family": "A"}, done=True)
    _run(tmp_path, "b", cfg={"family": "A"}, metrics=True)
    _run(tmp_path, "c", cfg={"family": "B"})
    c = R.counts(R.scan(tmp_path))
    assert c["total"] == 3
    assert sum(c["by_status"].values()) == 3
    assert sum(c["by_family"].values()) == 3
    assert c["by_family"] == {"A": 2, "B": 1}
