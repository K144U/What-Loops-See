"""Milestone 0 gate: a killed run resumes bit exactly.

This is the test the whole cluster strategy rests on. `qalter` is blocked,
so jobs die at the wall with no warning and no extension. If resume is
merely approximate, every long run silently becomes a different experiment
each time it is restarted, and nothing downstream is trustworthy.

"Bit exact" here means exact: identical final parameters compared with
torch.equal, and an identical per step loss trace compared as strings, not
as floats within a tolerance. A tolerance would hide precisely the drift
this test exists to catch.

Scope. This asserts CPU determinism. GPU determinism is a separate problem
and is tested at milestone 3 against the real model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

STEPS = 60
CHECKPOINT_EVERY = 7  # deliberately not a divisor of STEPS


def _run(run_id: str, runs_root: Path, extra: list[str]) -> subprocess.CompletedProcess:
    cmd = [
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
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def _loss_trace(runs_root: Path, run_id: str) -> list[tuple[int, str]]:
    """Per step loss as stored strings, so comparison is exact.

    The trainer writes losses with 17 significant digits, which round trips
    a float64 exactly. Comparing the strings avoids anyone later relaxing
    this into an approximate comparison without noticing.
    """
    path = runs_root / run_id / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(r["step"], r["loss"]) for r in rows]


def _final_params(runs_root: Path, run_id: str) -> dict[str, torch.Tensor]:
    ckpt = sorted((runs_root / run_id / "checkpoints").glob("step_*.pt"))[-1]
    return torch.load(ckpt, map_location="cpu", weights_only=False)["model"]


def _assert_identical(reference: str, candidate: str, runs_root: Path) -> None:
    ref_trace = _loss_trace(runs_root, reference)
    cand_trace = _loss_trace(runs_root, candidate)

    assert [s for s, _ in ref_trace] == list(range(1, STEPS + 1)), (
        "reference run did not record every step exactly once"
    )
    assert [s for s, _ in cand_trace] == list(range(1, STEPS + 1)), (
        "resumed run did not record every step exactly once. Duplicate or "
        "missing rows mean the metrics truncation on resume is wrong."
    )

    for (ref_step, ref_loss), (cand_step, cand_loss) in zip(ref_trace, cand_trace):
        assert ref_step == cand_step
        assert ref_loss == cand_loss, (
            f"loss diverged at step {ref_step}: uninterrupted {ref_loss} "
            f"versus resumed {cand_loss}. Resume is not bit exact."
        )

    ref_params = _final_params(runs_root, reference)
    cand_params = _final_params(runs_root, candidate)
    assert ref_params.keys() == cand_params.keys()
    for name in ref_params:
        assert torch.equal(ref_params[name], cand_params[name]), (
            f"parameter {name} differs after resume. Resume is not bit exact."
        )


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> tuple[Path, str]:
    """One uninterrupted run, used as ground truth by every other test."""
    runs_root = tmp_path_factory.mktemp("runs")
    proc = _run("reference", runs_root, [])
    assert proc.returncode == 0, proc.stderr
    assert (runs_root / "reference" / "DONE").is_file()
    return runs_root, "reference"


def test_reference_run_completes(reference) -> None:
    runs_root, run_id = reference
    trace = _loss_trace(runs_root, run_id)
    assert len(trace) == STEPS
    # Sanity: the smoke model should actually be learning, otherwise the
    # test would pass just as happily on a trainer that does nothing.
    first = float(trace[0][1])
    last = float(trace[-1][1])
    assert last < first, f"smoke model did not learn: {first} to {last}"


def test_resume_after_abrupt_death_is_bit_exact(reference, tmp_path) -> None:
    """Kill at a step that is not a checkpoint boundary, then resume.

    Dying at step 25 with checkpoints every 7 means the last checkpoint is
    at step 21, so the resume has to redo steps 22 through 25. If the data
    cursor or any RNG stream were restored incorrectly, those four steps
    would differ and the trace comparison would catch it.
    """
    runs_root, ref_id = reference
    die_at = 25
    assert die_at % CHECKPOINT_EVERY != 0, "the kill must not land on a checkpoint"

    first = _run("killed", runs_root, ["--die-at", str(die_at)])
    assert first.returncode != 0, "the process was supposed to die abruptly"
    assert not (runs_root / "killed" / "DONE").is_file()

    second = _run("killed", runs_root, ["--resume", "auto"])
    assert second.returncode == 0, second.stderr
    assert "resumed from" in second.stdout
    assert (runs_root / "killed" / "DONE").is_file()

    _assert_identical(ref_id, "killed", runs_root)


def test_resume_after_external_kill_is_bit_exact(reference) -> None:
    """The honest version: an external kill at an arbitrary wall clock moment.

    subprocess.Popen.kill maps to SIGKILL on POSIX and TerminateProcess on
    Windows. Neither can be caught or cleaned up after, which is the point.
    Unlike the --die-at test, the kill lands wherever it lands, including
    potentially in the middle of a checkpoint write. The atomic write is
    what makes that survivable.
    """
    runs_root, ref_id = reference
    cmd = [
        sys.executable,
        "-m",
        "loopvision.train.smoke",
        "--run-id",
        "hardkilled",
        "--runs-root",
        str(runs_root),
        "--steps",
        str(STEPS),
        "--checkpoint-every",
        str(CHECKPOINT_EVERY),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait until at least two checkpoints exist, then kill without warning.
    ckpt_dir = runs_root / "hardkilled" / "checkpoints"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if ckpt_dir.is_dir() and len(list(ckpt_dir.glob("step_*.pt"))) >= 2:
            break
        if proc.poll() is not None:
            break
        time.sleep(0.01)
    proc.kill()
    proc.wait(timeout=30)

    if (runs_root / "hardkilled" / "DONE").is_file():
        pytest.skip("run finished before the kill landed, timing dependent")

    resumed = _run("hardkilled", runs_root, ["--resume", "auto"])
    assert resumed.returncode == 0, resumed.stderr
    assert (runs_root / "hardkilled" / "DONE").is_file()

    _assert_identical(ref_id, "hardkilled", runs_root)


def test_repeated_kills_are_bit_exact(reference, tmp_path) -> None:
    """Three kills in one run. A single resume working is weaker evidence
    than a run that survives being killed over and over, which is what a
    long job on a 24 hour wall actually experiences."""
    runs_root, ref_id = reference
    for die_at in (9, 23, 44):
        proc = _run("thrice", runs_root, ["--resume", "auto", "--die-at", str(die_at)])
        assert proc.returncode != 0, f"expected abrupt death at {die_at}"

    final = _run("thrice", runs_root, ["--resume", "auto"])
    assert final.returncode == 0, final.stderr
    assert (runs_root / "thrice" / "DONE").is_file()

    _assert_identical(ref_id, "thrice", runs_root)


def test_done_sentinel_stops_the_chain(reference) -> None:
    """A finished run must be a no-op, or the PBS chain never terminates."""
    runs_root, ref_id = reference
    again = _run(ref_id, runs_root, ["--resume", "auto"])
    assert again.returncode == 0
    assert "already has a DONE sentinel" in again.stdout
    # And it must not have appended anything.
    assert len(_loss_trace(runs_root, ref_id)) == STEPS


def test_checkpoint_pruning_keeps_last_two(reference) -> None:
    runs_root, ref_id = reference
    ckpts = sorted((runs_root / ref_id / "checkpoints").glob("step_*.pt"))
    assert len(ckpts) == 2, f"expected 2 retained checkpoints, found {len(ckpts)}"
    assert not list((runs_root / ref_id / "checkpoints").glob("*.tmp")), (
        "a temporary checkpoint file was left behind, so a write was not atomic"
    )
