"""Milestone 0 gate: the self-chaining job logic.

The PBS half of this cannot run off the cluster, so these tests cover the
part that can: the wall-clock self-stop, the DONE sentinel that terminates
the chain, and the guarantee that a run split across many jobs produces the
same result as one that ran straight through.

What is verified here is the decision logic that submit.pbs depends on.
What is not verified here is qsub, PBS dependencies, and pick_gpu.sh against
real hardware. Those are cluster-only and are checked by the milestone 0
sign off run, recorded in docs/progress.md.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Kept small on purpose. The chain test runs --max-hours 0, which stops
# after one step, so the step budget is also the number of processes
# spawned, and every process pays a fresh torch import. Twelve is enough to
# be a genuine multi-job chain and cheap enough that nobody is tempted to
# skip the suite. Raise it only if the chain logic gets more complicated.
STEPS = 12
CHECKPOINT_EVERY = 3


def _invoke(run_id: str, runs_root: Path, extra: list[str]) -> subprocess.CompletedProcess:
    """One job in the chain."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "loopvision.train.smoke",
            "--run-id",
            run_id,
            "--runs-root",
            str(runs_root),
            "--steps",
            str(STEPS),
            "--checkpoint-every",
            str(CHECKPOINT_EVERY),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )


def _trace(runs_root: Path, run_id: str) -> list[tuple[int, str]]:
    path = runs_root / run_id / "metrics.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [(r["step"], r["loss"]) for r in rows]


def test_max_hours_stops_cleanly_without_done(tmp_path) -> None:
    """The self-stop must checkpoint and must NOT write DONE.

    If a wall-clock stop wrote DONE, the chain would terminate believing the
    run had finished, and the run would be silently truncated. That failure
    is invisible in the metrics: the file just ends early.
    """
    proc = _invoke("walled", tmp_path, ["--max-hours", "0"])
    assert proc.returncode == 0, proc.stderr
    assert "hit max-hours" in proc.stdout
    assert not (tmp_path / "walled" / "DONE").is_file(), (
        "a wall-clock stop wrote DONE, which would terminate the chain early"
    )
    assert list((tmp_path / "walled" / "checkpoints").glob("step_*.pt")), (
        "stopped at the wall without leaving a checkpoint, so the successor "
        "job would restart from zero"
    )


def test_chain_of_many_jobs_matches_one_uninterrupted_run(tmp_path) -> None:
    """A run split across many jobs equals a run that was never split.

    This is the property the 24 hour wall makes unavoidable. Each iteration
    below is one PBS job: it resumes, works, stops at its wall, and leaves a
    checkpoint for its successor. The loop is the chain.
    """
    reference = _invoke("straight", tmp_path, [])
    assert reference.returncode == 0, reference.stderr
    assert (tmp_path / "straight" / "DONE").is_file()

    jobs = 0
    while not (tmp_path / "chained" / "DONE").is_file():
        jobs += 1
        assert jobs < 60, "chain did not terminate, DONE sentinel logic is broken"
        proc = _invoke("chained", tmp_path, ["--resume", "auto", "--max-hours", "0"])
        assert proc.returncode == 0, proc.stderr

    assert jobs >= 2, f"expected a multi-job chain, completed in {jobs}"
    assert _trace(tmp_path, "straight") == _trace(tmp_path, "chained"), (
        "a chained run diverged from an uninterrupted one"
    )


def test_done_run_is_a_noop(tmp_path) -> None:
    """The chain terminates because a finished run does nothing."""
    first = _invoke("once", tmp_path, [])
    assert first.returncode == 0
    before = _trace(tmp_path, "once")

    second = _invoke("once", tmp_path, ["--resume", "auto"])
    assert second.returncode == 0
    assert "already has a DONE sentinel" in second.stdout
    assert _trace(tmp_path, "once") == before


def test_resume_survives_a_missing_latest_pointer(tmp_path) -> None:
    """A kill between the checkpoint write and the pointer write.

    save_checkpoint writes the checkpoint first and the LATEST pointer
    second, so this interleaving is reachable in practice. Resume must fall
    back to scanning the directory rather than treating it as a cold start,
    which would silently discard hours of work.
    """
    walled = _invoke("nopointer", tmp_path, ["--max-hours", "0"])
    assert walled.returncode == 0

    pointer = tmp_path / "nopointer" / "checkpoints" / "LATEST"
    assert pointer.is_file()
    pointer.unlink()

    resumed = _invoke("nopointer", tmp_path, ["--resume", "auto"])
    assert resumed.returncode == 0, resumed.stderr
    assert "resumed from" in resumed.stdout, (
        "a missing LATEST pointer was treated as a cold start"
    )


def test_resume_refuses_a_mismatched_checkpoint_format(tmp_path) -> None:
    """A stale format must raise, not be reinterpreted.

    Silently loading a checkpoint whose layout has changed is how a run
    continues with the wrong optimizer state and nobody notices.
    """
    import torch

    from loopvision.train.checkpoint import load_checkpoint
    from loopvision.train.smoke import SmokeModel

    walled = _invoke("badformat", tmp_path, ["--max-hours", "0"])
    assert walled.returncode == 0

    ckpt = sorted((tmp_path / "badformat" / "checkpoints").glob("step_*.pt"))[-1]
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    payload["format_version"] = 999
    torch.save(payload, ckpt)

    model = SmokeModel()
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(RuntimeError, match="format version"):
        load_checkpoint(ckpt, model, optimizer, None)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("script", ["scripts/pick_gpu.sh", "scripts/submit.pbs"])
def test_cluster_scripts_are_syntactically_valid(script: str) -> None:
    """bash -n parses without executing.

    These scripts only ever run on the cluster, where a syntax error costs a
    queue wait to discover. This catches that class of error locally.
    """
    proc = subprocess.run(
        ["bash", "-n", str(REPO_ROOT / script)], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"{script} failed to parse: {proc.stderr}"
