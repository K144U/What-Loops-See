"""Checkpoint retention: two while training, one once a run is DONE.

The disk reached 6 TB once and had to be cleared by hand. Pruning to the
last two had existed since the first commit, so the likeliest cause is a
prune that failed and said nothing: `_prune_checkpoints` swallowed every
OSError, and a run that checkpoints every 30 minutes grows by a full
checkpoint each time a prune fails. These pin both halves of the fix: a
finished run keeps exactly the checkpoint its readers load, and a failed
prune is loud.
"""

from __future__ import annotations

from pathlib import Path

import torch

from loopvision.train import checkpoint as C

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fake_run(tmp_path: Path, steps: list[int], latest: int | None) -> Path:
    """A run directory holding dummy step files and, optionally, a pointer."""
    run = tmp_path / "run"
    ckpt = run / "checkpoints"
    ckpt.mkdir(parents=True)
    for s in steps:
        (ckpt / f"step_{s:09d}.pt").write_bytes(b"x" * 16)
    if latest is not None:
        (ckpt / "LATEST").write_text(f"step_{latest:09d}.pt", encoding="utf-8")
    return run


def _left(run: Path) -> list[str]:
    return sorted(p.name for p in (run / "checkpoints").glob("step_*.pt"))


def test_prune_to_latest_keeps_exactly_the_pointed_checkpoint(tmp_path) -> None:
    run = _fake_run(tmp_path, [100, 200, 300], latest=300)
    assert C.prune_to_latest(run) == 2
    assert _left(run) == ["step_000000300.pt"]
    assert (run / "checkpoints" / "LATEST").read_text(encoding="utf-8") == "step_000000300.pt"


def test_prune_follows_the_pointer_and_not_the_newest_name(tmp_path) -> None:
    """What survives is what every reader loads, which is the pointer's
    target, so pruning must agree with find_latest_checkpoint exactly."""
    run = _fake_run(tmp_path, [100, 200], latest=100)
    C.prune_to_latest(run)
    assert _left(run) == ["step_000000100.pt"]
    assert C.find_latest_checkpoint(run).name == "step_000000100.pt"


def test_prune_without_a_pointer_keeps_the_newest(tmp_path) -> None:
    run = _fake_run(tmp_path, [100, 200], latest=None)
    C.prune_to_latest(run)
    assert _left(run) == ["step_000000200.pt"]


def test_prune_on_a_run_with_no_checkpoints_does_nothing(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    assert C.prune_to_latest(tmp_path / "empty") == 0


def test_a_failed_prune_warns_and_does_not_raise(tmp_path, monkeypatch, capsys) -> None:
    run = _fake_run(tmp_path, [100, 200], latest=200)

    def refuse(self, *a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "unlink", refuse)
    assert C.prune_to_latest(run) == 0
    assert "WARNING checkpoint prune failed" in capsys.readouterr().err


def test_interval_pruning_keeps_two_and_is_loud_when_it_fails(tmp_path, monkeypatch, capsys) -> None:
    """The training-time half: save_checkpoint keeps the last two, and a
    prune that cannot delete says so instead of passing silently."""
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    run = tmp_path / "train"
    for step in (10, 20, 30):
        C.save_checkpoint(run, step, model, opt, None, data_cursor=step)
    assert _left(run) == ["step_000000020.pt", "step_000000030.pt"]

    def refuse(self, *a, **k):
        raise OSError("stale file handle")

    monkeypatch.setattr(Path, "unlink", refuse)
    C.save_checkpoint(run, 40, model, opt, None, data_cursor=40)
    assert "WARNING checkpoint prune failed" in capsys.readouterr().err


def test_the_real_trainer_prunes_right_after_done() -> None:
    """cli.py is the trainer every cluster run uses, and running it end to
    end here is too slow, so pin the order in its source: DONE first, so a
    crash during pruning still leaves a finished run, then the prune."""
    src = (REPO_ROOT / "src" / "loopvision" / "train" / "cli.py").read_text(encoding="utf-8")
    done = src.index('done_path.write_text("done\\n", encoding="utf-8")')
    prune = src.index("prune_to_latest(run_dir)")
    assert done < prune, "prune_to_latest must come after DONE is written"
    assert src[done:prune].count("\n") <= 4, "prune_to_latest must follow DONE directly"
