"""Every queued run, checked before it costs a slot.

The failure this exists for is D-044: job 5275 held a GPU slot for thirteen
hours with three of four runs dead seconds after launch, and qstat showed it
running the whole time. check_configs could not have caught it, because the
question is not "is this config valid" but "is the thing I already queued
going to start".
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from loopvision.analysis import check_queue as Q

REAL_CFG = Path(__file__).resolve().parents[1] / "configs"


def test_triples_are_pulled_out_of_a_qstat_blob() -> None:
    blob = ("Variable_List=PBS_O_HOME=/home/x,RUNS=famA_d2_d4only_s5|m3/famA_d2_d4only.yaml|5 "
            "famA_d2_d4only_s6|m3/famA_d2_d4only.yaml|6,ITEMS=256")
    got = Q.triples_from_text(blob)
    assert got == [("famA_d2_d4only_s5", "m3/famA_d2_d4only.yaml", 5),
                   ("famA_d2_d4only_s6", "m3/famA_d2_d4only.yaml", 6)]


def test_a_run_in_two_jobs_is_checked_once() -> None:
    """D-041 records a qdel that resubmitted a remainder which already
    existed, so two jobs carried one run id. Counting it twice would make the
    report disagree with the queue."""
    blob = "a|m3/x.yaml|0 b|m3/y.yaml|1 a|m3/x.yaml|0"
    assert [r[0] for r in Q.triples_from_text(blob)] == ["a", "b"]


def test_a_config_that_predates_a_required_key_is_caught(tmp_path) -> None:
    """The KeyError class. A launch config is read with DEFAULTS layered, so
    this has to be a key DEFAULTS does not supply."""
    cfgs = tmp_path / "configs" / "m3"
    cfgs.mkdir(parents=True)
    (cfgs / "broken.yaml").write_text("family: A\narch: nonsense\n", encoding="utf-8")
    ok, why = Q.check_one("r", "m3/broken.yaml", 0, tmp_path / "configs")
    assert not ok
    assert "model will not build" in why


def test_a_config_asking_outside_its_split_is_caught(tmp_path) -> None:
    """The D-044 class, and the one that matters: the model builds fine and
    the failure appears only when data is first requested."""
    cfgs = tmp_path / "configs" / "m3"
    cfgs.mkdir(parents=True)
    src = yaml.safe_load((REAL_CFG / "m3" / "famA_d2_d4only.yaml").read_text())
    src["depths"] = [99]                       # outside family A's split
    (cfgs / "outside.yaml").write_text(yaml.safe_dump(src), encoding="utf-8")

    ok, why = Q.check_one("r", "m3/outside.yaml", 0, tmp_path / "configs")
    assert not ok, "a config asking outside its split was reported healthy"
    assert "cannot generate a batch" in why, why


def test_a_healthy_config_passes_both_halves() -> None:
    """The positive control. Without it every assertion above passes against
    a checker hardwired to fail."""
    ok, why = Q.check_one("famA_d2_d4only_s5", "m3/famA_d2_d4only.yaml", 5, REAL_CFG)
    assert ok, why
    assert "params" in why


def test_main_returns_one_when_any_queued_run_cannot_start(tmp_path, capsys) -> None:
    """It has to fail the command. A checker that reports and exits zero gets
    read once and then never run again."""
    cfgs = tmp_path / "configs" / "m3"
    cfgs.mkdir(parents=True)
    src = yaml.safe_load((REAL_CFG / "m3" / "famA_d2_d4only.yaml").read_text())
    (cfgs / "good.yaml").write_text(yaml.safe_dump(src), encoding="utf-8")
    src["depths"] = [99]
    (cfgs / "bad.yaml").write_text(yaml.safe_dump(src), encoding="utf-8")

    tri = tmp_path / "t.txt"
    tri.write_text("g|m3/good.yaml|0 b|m3/bad.yaml|1", encoding="utf-8")
    rc = Q.main(["--triples", str(tri), "--configs", str(tmp_path / "configs")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL b" in out
    assert "OK   g" in out
    assert "before they cost a slot" in out


def test_an_empty_queue_claims_nothing(tmp_path, capsys) -> None:
    """Off cluster there is no qstat. Reporting success there would be a
    clean check that checked nothing, which is the failure check_findings
    warns about in its own output."""
    tri = tmp_path / "empty.txt"
    tri.write_text("nothing here\n", encoding="utf-8")
    rc = Q.main(["--triples", str(tri)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "nothing is claimed" in out.lower()


# --------------------------------------------------------------------------
# Queue ORDER. L-030: "will it start" and "when will it start" are different
# questions, and the second one is what buried the headline 2x2 behind 146
# runs of supporting work with nothing in qstat looking wrong.
# --------------------------------------------------------------------------

#: Shaped like the blob `queued_text` builds: per-job `qstat -f` blocks
#: concatenated in the order `qstat -u` listed them, which is queue rank
#: order. Two donors, then a headline 2x2 seed behind them.
BLOB = """
Job Id: 5001.master
    job_state = R
    Variable_List = RUNS=famA_d2_d4only_s0|m3/famA_d2_d4only.yaml|0

Job Id: 5002.master
    job_state = Q
    Variable_List = RUNS=famA_d1_d4only_s4|m3/famA_d1_d4only.yaml|4 famA_d1_d4only_s5|m3/famA_d1_d4only.yaml|5

Job Id: 5003.master
    job_state = Q
    Variable_List = RUNS=famA_d2_d4colour_s15|m3/famA_d2_d4colour.yaml|15
"""


def test_jobs_are_read_in_the_order_the_blob_carries_them() -> None:
    """Blob order is queue order, and that is the only reason any of this can
    be read. If jobs_from_text sorted, every position below would be wrong
    and still look plausible."""
    got = Q.jobs_from_text(BLOB)
    assert [j for j, _s, _t in got] == ["5001", "5002", "5003"]
    assert [s for _j, s, _t in got] == ["R", "Q", "Q"]
    assert [len(t) for _j, _s, t in got] == [1, 2, 1]


def test_a_running_run_has_nothing_ahead_of_it() -> None:
    """It already holds a slot. Counting it as queued would make every
    position downstream too large by one per running job."""
    assert Q.queue_order(BLOB)["famA_d2_d4only_s0"] == 0


def test_position_counts_the_runs_ahead_and_not_the_jobs() -> None:
    """Job 5002 carries two runs, so the headline seed behind it waits on two
    runs and not on one job. max_run counts jobs, the GPU counts runs, and it
    is the runs that take the time."""
    pos = Q.queue_order(BLOB)
    assert pos["famA_d1_d4only_s4"] == 0
    assert pos["famA_d1_d4only_s5"] == 1
    assert pos["famA_d2_d4colour_s15"] == 2


def test_a_held_run_counts_ahead_because_its_hold_clears_into_this_queue() -> None:
    """Every hold in this project is afterany on a job in the same queue, so
    it releases while the queue is still draining. Skipping held runs would
    under-report the backlog by the whole 44 run curriculum block."""
    held = BLOB.replace(
        "Job Id: 5003.master", """Job Id: 5009.master
    job_state = H
    Variable_List = RUNS=famA_d2_d4colour_curr_s1|m3/famA_d2_d4colour_curriculum_s1.yaml|1

Job Id: 5003.master""")
    assert Q.queue_order(held)["famA_d2_d4colour_s15"] == 3


def test_headline_backlog_names_only_registered_2x2_runs() -> None:
    """The donors are depth 1 and outside the registered design, so they are
    not headline runs however long they take. The cell set comes from
    escape_rates.arm_of, not from a list kept here by hand."""
    got = Q.headline_backlog(BLOB, REAL_CFG)
    assert [r[0] for r in got] == ["famA_d2_d4only_s0", "famA_d2_d4colour_s15"]
    assert dict((r[0], r[1]) for r in got)["famA_d2_d4colour_s15"] == ("D4", "identity")


def test_a_buried_headline_run_fails_the_check(capsys) -> None:
    rc = Q.report_order(BLOB, REAL_CFG, budget=1)
    out = capsys.readouterr().out
    assert rc == 1
    assert "buried" in out and "L-030" in out
    assert "qorder" in out


def test_a_correct_ordering_passes(capsys) -> None:
    """A donor stage legitimately precedes the headline, so a budget that
    fires on it would be noise and would then get turned off. See L-029."""
    rc = Q.report_order(BLOB, REAL_CFG, budget=Q.AHEAD_BUDGET)
    assert rc == 0
    assert "within the" in capsys.readouterr().out


def test_order_mode_claims_nothing_when_the_queue_cannot_be_read(tmp_path, capsys) -> None:
    empty = tmp_path / "q.txt"
    empty.write_text("", encoding="utf-8")
    rc = Q.main(["--order", "--triples", str(empty)])
    assert rc == 0
    assert "nothing is claimed" in capsys.readouterr().out.lower()


def test_order_mode_is_reachable_from_the_command_line(tmp_path, capsys) -> None:
    """The flag has to be wired up, not just the function written. A check
    nobody can run is the L-013 failure."""
    blob = tmp_path / "q.txt"
    blob.write_text(BLOB, encoding="utf-8")
    rc = Q.main(["--order", "--triples", str(blob), "--configs", str(REAL_CFG),
                 "--budget", "1"])
    assert rc == 1
    assert "runs ahead" in capsys.readouterr().out


def test_a_headline_run_waiting_on_its_own_siblings_is_not_buried(capsys) -> None:
    """The nine famA_d2_d4colour_deep seeds are D4/identity members by D-043,
    so a 2x2 block is routinely longer than the budget. Counting siblings
    would make a correct ordering fail, and a check that fires on a correct
    ordering gets turned off. See L-029."""
    jobs = ["""
Job Id: 6000.master
    job_state = Q
    Variable_List = RUNS=famA_d2_d4colour_s%d|m3/famA_d2_d4colour.yaml|%d
""" % (n, n) for n in range(40, 52)]
    rc = Q.report_order("".join(jobs), REAL_CFG, budget=0)
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("D4/identity") == 12
    assert "buried" not in out


def test_supporting_work_in_front_of_the_headline_is_what_counts(capsys) -> None:
    """One donor ahead of one headline seed is one unit of backlog, and the
    second headline seed behind it adds none."""
    blob = BLOB + """
Job Id: 5004.master
    job_state = Q
    Variable_List = RUNS=famA_d2_d4colour_s16|m3/famA_d2_d4colour.yaml|16
"""
    got = dict((r[0], r[2]) for r in Q.headline_backlog(blob, REAL_CFG))
    assert got["famA_d2_d4colour_s15"] == 2
    assert got["famA_d2_d4colour_s16"] == 2


def test_a_running_headline_run_reports_no_backlog_wherever_it_is_listed() -> None:
    """It already holds a slot, so nothing is in front of it whatever the
    listing order says. Without this the check would report a run that is
    training right now as buried behind work it is not waiting for."""
    blob = """
Job Id: 7001.master
    job_state = Q
    Variable_List = RUNS=famA_d1_d4only_s4|m3/famA_d1_d4only.yaml|4

Job Id: 7002.master
    job_state = R
    Variable_List = RUNS=famA_d2_d4colour_s30|m3/famA_d2_d4colour.yaml|30
"""
    got = dict((r[0], r[2]) for r in Q.headline_backlog(blob, REAL_CFG))
    assert got["famA_d2_d4colour_s30"] == 0
