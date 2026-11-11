"""A resumed run continues under the config it started with.

L-025. `multirun.pbs` passes --config on every relaunch, and `cli.train`
read that file rather than the run's own snapshot, so a config edited while
a run was in flight would change what that run computed at its next wall
kill. The run directory holds a `config.yaml` written at step 0 precisely
so the run is reproducible, and resume ignored it.

The first test is the one that matters: the architecture keys must come
from the run, not from the file that has since changed.
"""

from __future__ import annotations

import yaml

from loopvision.train import cli


def test_drift_is_detected_on_architecture_keys() -> None:
    """Mutation checked: dropping the `k not in RELAUNCH_KEYS` guard, or
    comparing the dicts with == instead of per key, both break this."""
    source = {"d_model": 768, "core_blocks": 4, "steps": 300000}
    snapshot = {"d_model": 384, "core_blocks": 2, "steps": 300000}
    drift = cli.config_drift(source, snapshot)
    assert drift == {"d_model": (384, 768), "core_blocks": (2, 4)}


def test_the_runs_value_is_reported_first() -> None:
    """The tuple order is load bearing: the run's value is what gets used,
    so it reads as 'run has X, the file now says Y'."""
    drift = cli.config_drift({"d_model": 768}, {"d_model": 384})
    was, now = drift["d_model"]
    assert (was, now) == (384, 768)


def test_num_workers_is_not_drift() -> None:
    """multirun computes it from the cores the job got, so it differs
    between launches of the same run and must not be reported."""
    source = {"num_workers": 3, "d_model": 384}
    snapshot = {"num_workers": 0, "d_model": 384}
    assert cli.config_drift(source, snapshot) == {}


def test_identical_configs_produce_no_drift() -> None:
    """The positive control. The common case must be silent."""
    cfg = {"d_model": 384, "core_blocks": 2, "steps": 300000}
    assert cli.config_drift(cfg, dict(cfg)) == {}


def test_a_key_added_to_the_source_counts_as_drift() -> None:
    """The d4colour_deep arm was mis-specified by ADDING keys that restated
    the defaults, so a key present on one side only has to be caught."""
    drift = cli.config_drift({"d_model": 384, "core_blocks": 4}, {"d_model": 384})
    assert drift == {"core_blocks": (None, 4)}


def test_resume_loads_the_snapshot_not_the_source(tmp_path, monkeypatch, capsys) -> None:
    """End to end, at the level that actually bit: the source config has
    been edited to a different architecture and the run must ignore it."""
    runs = tmp_path / "runs"
    run_dir = runs / "somerun"
    run_dir.mkdir(parents=True)
    started_with = {"d_model": 384, "core_blocks": 2, "steps": 10, "num_workers": 0}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(started_with), encoding="utf-8")

    edited = tmp_path / "edited.yaml"
    edited.write_text(
        yaml.safe_dump({"d_model": 768, "core_blocks": 4, "steps": 10}), encoding="utf-8"
    )

    seen = {}

    def fake_build_model(cfg, device):
        seen.update(cfg)
        raise SystemExit(0)

    monkeypatch.setattr(cli, "build_model", fake_build_model)

    class Args:
        config = str(edited)
        run_id = "somerun"
        runs_root = str(runs)
        steps = seed = k = batch_size = num_workers = None
        force = False
        resume = "auto"
        max_hours = None

    try:
        cli.train(Args())
    except SystemExit:
        pass

    assert seen["d_model"] == 384, "resume used the edited source config"
    assert seen["core_blocks"] == 2
    out = capsys.readouterr().out
    assert "config drift" in out.lower()
    assert "d_model" in out
