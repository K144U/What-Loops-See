"""A config checker is only useful if it separates the three cases.

Broken, historical, and fine are different things, and collapsing any two
of them makes the checker either cry wolf or stay silent. The distinction
that carries the weight is derived rather than listed: a config that cannot
run today is historical exactly when a finished run matches it, because the
run is the evidence it once worked. A hand maintained allowlist would be
the same fault the run watcher had, where the list falls behind reality and
nobody notices.
"""

from __future__ import annotations

import json

from pathlib import Path

import pytest
import yaml

from loopvision.analysis import check_configs as CC


def _cfg(tmp_path, name, **over):
    base = {
        "family": "A", "d_model": 384, "depths": [2], "breadths": [1, 2, 3, 4],
        "arch": "looped", "factor": "full", "substrate": "native",
        "k_schedule": "sampled", "steps": 10, "batch_size": 8,
    }
    base.update(over)
    d = tmp_path / "configs"
    d.mkdir(exist_ok=True)
    p = d / f"{name}.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")
    return p


def _run(tmp_path, name, done=True, **over):
    base = {
        "family": "A", "d_model": 384, "depths": [2], "breadths": [1, 2, 3, 4],
        "arch": "looped", "factor": "full", "substrate": "native",
        "k_schedule": "sampled", "steps": 10, "batch_size": 8,
    }
    base.update(over)
    d = tmp_path / "runs" / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    if done:
        (d / "DONE").write_text("done\n", encoding="utf-8")
    return d


def test_a_good_config_is_not_reported(tmp_path) -> None:
    _cfg(tmp_path, "fine")
    (tmp_path / "runs").mkdir()
    r = CC.check(tmp_path / "configs", tmp_path / "runs", models=False)
    assert r["broken"] == [] and r["historical"] == []
    assert r["checked"] == 1


def test_a_config_outside_its_split_is_broken(tmp_path) -> None:
    """The fault that cost thirteen hours of a job slot."""
    _cfg(tmp_path, "toodeep", depths=[5])
    (tmp_path / "runs").mkdir()
    r = CC.check(tmp_path / "configs", tmp_path / "runs", models=False)
    assert len(r["broken"]) == 1
    assert "toodeep" in r["broken"][0][0]


def test_a_broken_config_with_a_finished_run_is_historical(tmp_path) -> None:
    """The distinction that keeps this checker usable. Same config, but now
    a finished run proves it was valid when it ran."""
    _cfg(tmp_path, "wasfine", depths=[5])
    _run(tmp_path, "oldrun", depths=[5])
    r = CC.check(tmp_path / "configs", tmp_path / "runs", models=False)
    assert r["broken"] == []
    assert len(r["historical"]) == 1


def test_an_unfinished_run_is_not_evidence(tmp_path) -> None:
    """A run that started and died proves nothing. Only DONE counts."""
    _cfg(tmp_path, "neverworked", depths=[5])
    _run(tmp_path, "diedrun", done=False, depths=[5])
    r = CC.check(tmp_path / "configs", tmp_path / "runs", models=False)
    assert len(r["broken"]) == 1, "a dead run was treated as evidence"
    assert r["historical"] == []


def test_a_different_run_is_not_evidence(tmp_path) -> None:
    """The match has to be on the experiment's identity, not on any run
    existing at all, or every broken config becomes historical."""
    _cfg(tmp_path, "toodeep", depths=[5])
    _run(tmp_path, "unrelated", depths=[2])
    r = CC.check(tmp_path / "configs", tmp_path / "runs", models=False)
    assert len(r["broken"]) == 1
    assert r["historical"] == []


def test_run_snapshots_closed_to_reanalysis_are_reported(tmp_path) -> None:
    """Seven real runs are in this state. It is not an error and must not
    be silent either."""
    _cfg(tmp_path, "fine")
    _run(tmp_path, "oldsplit", depths=[5])
    r = CC.check(tmp_path / "configs", tmp_path / "runs", models=False)
    assert len(r["closed"]) == 1
    assert r["closed"][0][0] == "oldsplit"


def test_identity_ignores_seed_and_workers(tmp_path) -> None:
    """Runs of one config differ in seed. If identity included it, no run
    would ever match its own config."""
    a = {"family": "A", "depths": [2], "seed": 0, "num_workers": 3}
    b = {"family": "A", "depths": [2], "seed": 7, "num_workers": 0}
    assert CC._identity(a) == CC._identity(b)


def test_identity_separates_different_experiments() -> None:
    assert CC._identity({"family": "A", "d_model": 384}) != CC._identity(
        {"family": "A", "d_model": 768}
    )
    assert CC._identity({"family": "A", "substrate": "native"}) != CC._identity(
        {"family": "A", "substrate": "d4_colour"}
    )


def test_main_exits_nonzero_only_on_broken(tmp_path) -> None:
    """The exit code is what a pre-submit hook would read, so historical
    configs must not make it fail forever."""
    _cfg(tmp_path, "wasfine", depths=[5])
    _run(tmp_path, "oldrun", depths=[5])
    args = ["--configs", str(tmp_path / "configs"), "--runs",
            str(tmp_path / "runs"), "--skip-models"]
    assert CC.main(args) == 0

    _cfg(tmp_path, "neverworked", depths=[6])
    assert CC.main(args) == 1


def test_both_splits_are_checked_not_just_train(monkeypatch) -> None:
    """Family A and family B currently define train and iid_val over the
    same ranges, so a checker that looked only at train would pass every
    real config and look correct. It would also silently stop working the
    day the two ranges diverge, which is the whole reason split_spec takes
    a split argument. Ranges are forced apart here so the loop is actually
    exercised.

    Mutation checked: narrowing the loop to ("train",) fails this and
    nothing else in the file, which is how it was found.
    """
    from loopvision.data import dataset as D

    real = D.split_spec

    def only_train_is_wide(family, split, cfg=None):
        if split == "iid_val" and cfg is not None and cfg.depths == (2,):
            raise ValueError("B/iid_val: requested depths [2] are outside")
        return real(family, split, cfg)

    monkeypatch.setattr(D, "split_spec", only_train_is_wide)
    reason = CC.check_data(
        {"family": "A", "depths": [2], "breadths": [1, 2, 3, 4]}
    )
    assert reason is not None, "a config broken only on iid_val was passed"
    assert "iid_val" in reason


def test_a_data_failure_is_reported_not_swallowed() -> None:
    """check_data must return the reason, never None, when split_spec
    raises. Returning None would make every broken config look fine."""
    from loopvision.data import dataset as D

    reason = CC.check_data({"family": "A", "depths": [99]})
    assert reason is not None and "outside" in reason


def test_a_run_whose_model_cannot_be_built_is_reported(tmp_path) -> None:
    """The gap a geometry job found by failing on nine runs at once.

    The vark_* snapshots predate prelude_blocks, core_blocks and
    coda_blocks. They generate their data without complaint and then die
    in build_model with a KeyError, so a checker that only exercised
    split_spec called them healthy. Every instrument builds the model, so
    the check has to as well.

    Mutation checked: dropping the check_model call from the closed-run
    loop passes this run as fine.
    """
    _cfg(tmp_path, "fine")
    d = _run(tmp_path, "oldsnapshot")
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
    cfg.pop("d_model", None)
    cfg["d_model"] = 7  # not divisible by the head count
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    r = CC.check(tmp_path / "configs", tmp_path / "runs", models=True)
    names = [n for n, _ in r["closed"]]
    assert "oldsnapshot" in names, "a run that cannot build its model passed"


def test_skip_models_does_not_pretend_to_have_checked(tmp_path) -> None:
    """With --skip-models the model check is not run, so a snapshot that
    would fail it must not be reported as closed either. Reporting it
    would claim a check that did not happen."""
    _cfg(tmp_path, "fine")
    d = _run(tmp_path, "oldsnapshot")
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
    cfg["d_model"] = 7
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    r = CC.check(tmp_path / "configs", tmp_path / "runs", models=False)
    assert [n for n, _ in r["closed"]] == []


# --- D-044's recorded unrunnable set ------------------------------------

def _cfg_tree(tmp_path, name: str, depths: list[int]):
    """A config tree holding one family A config at the given depths."""
    import yaml
    root = tmp_path / "configs"
    (root / "m3").mkdir(parents=True, exist_ok=True)
    (root / "m3" / name).write_text(
        yaml.safe_dump({"family": "A", "depths": depths, "factor": "d4"}),
        encoding="utf-8")
    return root


def test_the_recorded_set_is_not_counted_as_broken(tmp_path) -> None:
    """The three D-044 configs failed this check every single time it ran,
    and CLAUDE.md tells you to run it before every submission. A check that
    is always red is one that stops being read, which is L-029."""
    root = _cfg_tree(tmp_path, "famA_d4_d4only.yaml", [4])
    r = CC.check(root, tmp_path / "runs", models=False)
    assert [Path(p).name for p, _ in r["recorded"]] == ["famA_d4_d4only.yaml"]
    assert r["broken"] == []


def test_a_config_outside_the_recorded_set_is_still_broken(tmp_path) -> None:
    """The whole value of the list is that everything not on it still fails.
    A blanket "ignore family A depth configs" would have swallowed this."""
    root = _cfg_tree(tmp_path, "famA_d4_somethingelse.yaml", [4])
    r = CC.check(root, tmp_path / "runs", models=False)
    assert r["recorded"] == []
    assert [Path(p).name for p, _ in r["broken"]] == ["famA_d4_somethingelse.yaml"]


def test_a_recorded_config_that_starts_running_is_a_failure(tmp_path) -> None:
    """Listing them makes this check STRICTER. Family A's depth axis would
    have to have changed for a D-044 config to run, and no such change is
    recorded. A split that moves silently is how two runs end up
    incomparable with nobody noticing."""
    root = _cfg_tree(tmp_path, "famA_d4_d4only.yaml", [2])   # depth 2 runs
    r = CC.check(root, tmp_path / "runs", models=False)
    assert [Path(p).name for p, _ in r["revived"]] == ["famA_d4_d4only.yaml"]


def test_the_recorded_set_names_exactly_the_three_from_d044() -> None:
    """Pinned so the list cannot quietly grow into a place to hide failures."""
    assert CC.RECORDED_UNRUNNABLE == frozenset({
        "m3/famA_d4_d4only.yaml",
        "m3/famA_d5_d4only.yaml",
        "m3/famA_d6_d4only.yaml",
    })
