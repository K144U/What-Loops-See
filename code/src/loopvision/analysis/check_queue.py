"""Every run sitting in the queue, checked before it ever gets a slot.

`check_configs` answers "is this config valid". This answers the question that
actually costs compute: **is every run I have already queued going to start?**
Those differ, because a job carries `id|config|seed` triples chosen at submit
time and nothing re-reads them until the run begins. Job 5275 held a GPU slot
for thirteen hours with three of its four runs dead seconds after launch, and
`qstat` showed it healthy throughout. See D-044.

WHAT IT CHECKS, and why both halves are needed:

  build   the model can be constructed from the config as the trainer reads
          it. This is the `KeyError: 'prelude_blocks'` class, a config that
          predates a key the model now requires.
  batch   one real batch generates from the training split. This is the
          D-044 class, a config asking for depths or breadths outside its
          own split, which raises only when data is first requested.

A config can pass the first and fail the second, which is exactly how 5275
looked healthy: its models built.

READ THE CONFIG THE WAY THE TRAINER READS IT. `load_config` layers `DEFAULTS`
over the file and a launch config relies on that; reading one with a bare
`yaml.safe_load` reports failures that will not happen. That is L-027 in
reverse, and the first version of this check made exactly that mistake and
declared all 35 queued runs dead while five of them were visibly training.
A run snapshot under `runs/` is the opposite case and is read raw, which is
what `check_configs` does and why the two modules do not share a reader.

WILL IT START is not WHEN WILL IT START, and the gap between those two
questions cost ten days once. `--order` answers the second. A run that hits
its wall self-chains, and the chained job is a fresh submission with a fresh
job number, so it re-enters at the BACK. Priority here is arrival order among
equal resource requests, so queue position decays in proportion to how often
a run has been interrupted, and the runs that fall furthest are the ones
nearest finishing. On 2026-09-16 all nine unfinished seeds of the registered
2x2, the number the paper leads with, sat behind about 146 runs with 23 GPU
hours of work left between them. Nothing was broken and nothing in `qstat`
looked wrong. See L-030.

DID IT START is the third question, and `--launched` answers it. A job packs
five runs, and a job is running if one of them is. On 2026-10-11 five jobs
started together, nine of their 25 runs died of CUDA out of memory inside two
minutes, and every job stayed in `qstat` in state R because the other runs
lived. The check above had passed, correctly: every run could build a model
and a batch. What it cannot know is whether the card had room. So this reads
each run's OWN log for the job that is carrying it now, and says which runs
are training, which are dead, and which the picker never had a card for. A
dead run is not lost, it chains when its job ends, but it re-enters at the
back, which is L-030 again by a third route. See L-036.

    python -m loopvision.analysis.check_queue
    python -m loopvision.analysis.check_queue --triples queued.txt
    python -m loopvision.analysis.check_queue --order
    python -m loopvision.analysis.check_queue --launched
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

#: id|config|seed, the form multirun.pbs carries in its RUNS variable.
TRIPLE = re.compile(r"([A-Za-z0-9_]+)\|([A-Za-z0-9_/]+\.yaml)\|(\d+)")


def triples_from_text(text: str) -> list[tuple[str, str, int]]:
    """Every id|config|seed in a blob, de-duplicated, order preserved.

    De-duplication matters: a run can appear in two jobs, which D-041 records
    as a real near miss where a `qdel` resubmitted a remainder that already
    existed and two processes would have shared one run directory.
    """
    out: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for rid, cfg, seed in TRIPLE.findall(text):
        if rid in seen:
            continue
        seen.add(rid)
        out.append((rid, cfg, int(seed)))
    return out


def queued_text(user: str | None = None) -> str:
    """Raw `qstat -f` for this user's jobs, or an empty string off cluster."""
    try:
        ids = subprocess.run(
            ["qstat", "-u", user or subprocess.run(
                ["whoami"], capture_output=True, text=True).stdout.strip()],
            capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    jobs = [ln.split(".")[0] for ln in ids.splitlines()[5:] if ln.strip()]
    blob = []
    for j in jobs:
        try:
            blob.append(subprocess.run(["qstat", "-f", j], capture_output=True,
                                       text=True, timeout=60).stdout)
        except (OSError, subprocess.SubprocessError):
            continue
    return "\n".join(blob).replace("\n\t", "")


#: The largest block of SUPPORTING work that may legitimately precede the
#: stage is 26 runs and it gates 44 curriculum runs, so an ordering that puts
#: it first is correct and must not fire. Anything structurally larger than
#: that is the L-030 failure. A tighter budget would fire on a correct queue,
#: which is how a check becomes noise and then gets turned off. See L-029.
AHEAD_BUDGET = 30

#: `Job Id: 5688.<cluster>-master`, one per `qstat -f` block.
JOB_ID = re.compile(r"Job Id:\s*(\d+)")
JOB_STATE = re.compile(r"job_state\s*=\s*(\w)")


def jobs_from_text(text: str) -> list[tuple[str, str, list[tuple[str, str, int]]]]:
    """Every job as (id, state, triples), in the order the blob carries them.

    `qstat -u` lists jobs in queue rank order and `queued_text` concatenates
    the per-job `qstat -f` blocks in that same order, so blob order IS queue
    order. That is the whole reason this can be read at all.
    """
    out = []
    for chunk in text.split("Job Id:")[1:]:
        jid = JOB_ID.search("Job Id:" + chunk)
        st = JOB_STATE.search(chunk)
        out.append((
            jid.group(1) if jid else "?",
            st.group(1) if st else "?",
            triples_from_text(chunk),
        ))
    return out


def queue_order(text: str) -> dict[str, int]:
    """run_id -> how many not-yet-started runs the scheduler reaches first.

    A running run is at 0: it already holds a slot. Held runs are counted,
    because every hold in this project is `afterany` on a job in this same
    queue, so the hold clears while the queue is still draining and its runs
    really do land ahead of anything behind them.
    """
    ahead = 0
    pos: dict[str, int] = {}
    for _jid, state, triples in jobs_from_text(text):
        for rid, _cfg, _seed in triples:
            if state == "R":
                pos[rid] = 0
                continue
            pos.setdefault(rid, ahead)
            ahead += 1
    return pos


def arms_by_run(text: str, configs_root: Path) -> dict[str, tuple | None]:
    """run_id -> its 2x2 cell, or None for everything outside the design.

    The cell set comes from `escape_rates.arm_of`, which reads the registered
    factors off the config, so a cell cannot be added here by editing a list.
    That is how `famA_d2_d4colour_deep` turns out to be a D4/identity seed
    rather than a depth arm: D-043 found the config duplicates the baseline,
    and the generator has counted it in the cell ever since. A run whose
    config will not read is left out rather than guessed at; `check_one` is
    what reports an unreadable config.
    """
    from loopvision.analysis.escape_rates import arm_of
    from loopvision.train.cli import load_config

    out: dict[str, tuple | None] = {}
    for rid, cfg_path, seed in triples_from_text(text):
        try:
            cfg = load_config(str(configs_root / cfg_path), {"seed": seed})
        except Exception:
            continue
        out[rid] = arm_of(cfg)
    return out


def headline_backlog(text: str, configs_root: Path) -> list[tuple[str, tuple, int]]:
    """Every queued 2x2 run, with how much SUPPORTING work precedes it.

    Siblings do not count. A headline run waiting on the other nine seeds of
    its own cell is a correct ordering, and a check that fires on a correct
    ordering becomes noise and then gets turned off, which is L-029. What
    this measures is the thing that actually went wrong in L-030: work that
    is not the headline, sitting in front of the headline.
    """
    arms = arms_by_run(text, configs_root)
    out = []
    ahead = 0
    for _jid, state, triples in jobs_from_text(text):
        for rid, _cfg, _seed in triples:
            arm = arms.get(rid)
            if state == "R":
                if arm is not None:
                    out.append((rid, arm, 0))
                continue
            if arm is not None:
                out.append((rid, arm, ahead))
            else:
                ahead += 1
    return out


def report_order(text: str, configs_root: Path, budget: int = AHEAD_BUDGET) -> int:
    """Print the headline backlog. Non-zero when the 2x2 is buried."""
    rows = headline_backlog(text, configs_root)
    if not rows:
        print("No queued run belongs to the registered 2x2. Either it is "
              "complete or the queue is empty. Nothing is claimed.")
        return 0
    print("Unfinished runs of the registered 2x2, and what sits ahead of each:")
    worst = 0
    for rid, arm, ahead in sorted(rows, key=lambda r: r[2]):
        flag = "  <-- buried" if ahead > budget else ""
        print(f"  {rid:<34} {arm[0]}/{arm[1]:<9} {ahead:>4} runs ahead{flag}")
        worst = max(worst, ahead)
    if worst > budget:
        print(f"\nA headline run sits behind {worst} runs, over the {budget} "
              f"a correct ordering needs. This is L-030: self-chaining sends a "
              f"resumed run to the back, so the seeds nearest finishing fall "
              f"furthest. Fix with `qorder <headline_job> <job_ahead>`, which "
              f"works here although `qalter` does not, then read the `qstat` "
              f"listing back: qorder returns 0 whether or not it moved anything.")
        return 1
    print(f"\nWorst headline backlog is {worst} runs, within the {budget} a "
          f"correct ordering needs.")
    return 0


def check_one(rid: str, cfg_path: str, seed: int, configs_root: Path) -> tuple[bool, str]:
    """Build the model and generate one batch, the way the trainer would."""
    import torch

    from loopvision.train.cli import build_model, load_config, make_batch

    try:
        cfg = load_config(str(configs_root / cfg_path), {"seed": seed})
    except Exception as e:
        return False, f"config unreadable, {type(e).__name__}: {e}"
    try:
        model = build_model(cfg, torch.device("cpu"))
        n = sum(p.numel() for p in model.parameters())
    except Exception as e:
        return False, f"model will not build, {type(e).__name__}: {e}"
    try:
        make_batch(cfg, "train", 0, torch.device("cpu"))
    except Exception as e:
        return False, f"cannot generate a batch, {type(e).__name__}: {e}"
    return True, f"{n / 1e6:.1f}M params"


#: A run writes a checkpoint every 30 minutes. One that has shown nothing for
#: twice that, in a job that has been running that long, is not just slow to
#: start. Slow is real here: at pixel scale 2 on a shared card the first
#: evaluation line took 223 minutes on 2026-10-11, which is why the first
#: checkpoint is read as well and a missing evaluation line alone is not.
SILENT_AFTER_MIN = 60

WALLTIME = re.compile(r"resources_used\.walltime\s*=\s*(\d+):(\d+):(\d+)")
STEP_LINE = re.compile(r"^step\s+(\d+)", re.M)
ERROR_LINE = re.compile(r"^[\w.]*(?:Error|Exception)\b.*$", re.M)

#: States that mean a run in a RUNNING job is not being trained.
NOT_TRAINING = ("dead", "silent")


def launch_state(rid: str, jid: str, job_minutes: float, runs_root: Path,
                 logs_root: Path, now: float | None = None) -> tuple[str, str]:
    """What one run of a running job is doing, read from its own files.

    Only the log this job wrote counts, `<run>.<job id>.<host>.log`. A run's
    older logs belong to jobs that are gone, and the first submission of
    2026-10-11 left a traceback in one for every run it carried.
    """
    now = time.time() if now is None else now
    run_dir = runs_root / rid
    if (run_dir / "DONE").is_file():
        return "done", ""
    logs = sorted(list(logs_root.glob(f"{rid}.{jid}.*.log")) + list(logs_root.glob(f"{rid}.{jid}.log")),
                  key=lambda p: p.stat().st_mtime)
    if not logs:
        return "unlaunched", "no log from this job: the picker had no card for it"
    log = logs[-1]
    text = log.read_text(encoding="utf-8", errors="replace")
    if "Traceback (most recent call last)" in text:
        errors = ERROR_LINE.findall(text)
        return "dead", (errors[-1] if errors else "traceback with no error line")[:110]
    steps = STEP_LINE.findall(text)
    if steps:
        return "training", f"step {int(steps[-1])}"
    # No evaluation yet. A checkpoint or a metrics flush written after this
    # job's log last moved is the run itself, alive after launch; files a
    # resumed run inherited from an earlier job are older than the log.
    written = [p.stat().st_mtime for p in
               [run_dir / "metrics.parquet", *(run_dir / "checkpoints").glob("*")]
               if p.exists()]
    if written and max(written) > log.stat().st_mtime:
        return "training", "no evaluation yet"
    quiet = (now - log.stat().st_mtime) / 60
    if job_minutes >= SILENT_AFTER_MIN and quiet >= SILENT_AFTER_MIN:
        return "silent", f"nothing written for {quiet:.0f} min"
    return "starting", f"launched {quiet:.0f} min ago"


def report_launched(text: str, runs_root: Path, logs_root: Path,
                    now: float | None = None) -> int:
    """Every run of every RUNNING job. Exit 1 if any is dead or silent."""
    tally: dict[str, int] = {}
    rows = []
    running = 0
    for chunk in text.split("Job Id:")[1:]:
        jid = JOB_ID.search("Job Id:" + chunk)
        st = JOB_STATE.search(chunk)
        if not jid or not st or st.group(1) != "R":
            continue
        triples = triples_from_text(chunk)
        if not triples:
            # A running job of ours that trains nothing, the test suite on
            # workq for one. Counting it made the first real reading say six
            # running jobs carried 25 runs when five did.
            continue
        running += 1
        wall = WALLTIME.search(chunk)
        minutes = (int(wall.group(1)) * 60 + int(wall.group(2))) if wall else 0.0
        for rid, _, _ in triples:
            state, why = launch_state(rid, jid.group(1), minutes, runs_root, logs_root, now)
            tally[state] = tally.get(state, 0) + 1
            rows.append((state, rid, jid.group(1), why))
    if not running:
        print("No running job found. Off cluster, or nothing has a slot. "
              "Nothing is checked and nothing is claimed.")
        return 0

    for state, rid, jid, why in rows:
        if state in ("training", "done"):
            continue
        print(f"  {state.upper():<10} {rid:<34} job {jid}  {why}")
    total = len(rows)
    print(f"\n{running} running job(s) carry {total} run(s): "
          + ", ".join(f"{tally.get(s, 0)} {s}" for s in
                      ("training", "done", "starting", "unlaunched", "dead", "silent")) + ".")
    bad = sum(tally.get(s, 0) for s in NOT_TRAINING)
    if bad:
        print(f"\n{bad} run(s) hold a place in a running job and are not being trained. "
              "Nothing is lost: each restarts when its job ends and chains it. But the "
              "chained job joins the BACK of the queue, so check what waits on these "
              "runs and move the successor with `qorder` when it appears. See L-036.")
        return 1
    if tally.get("unlaunched"):
        print("\nUnlaunched runs chain when their job ends, as designed. They are "
              "listed because they are also places in a job that train nothing.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Every queued run, before it starts")
    ap.add_argument("--triples", default=None, type=Path,
                    help="a file of id|config|seed, instead of reading qstat")
    ap.add_argument("--configs", default=Path("configs"), type=Path)
    ap.add_argument("--order", action="store_true",
                    help="where the registered 2x2 sits in the queue, not "
                         "whether it will start")
    ap.add_argument("--budget", type=int, default=AHEAD_BUDGET,
                    help="runs that may precede a headline run before it "
                         "counts as buried")
    ap.add_argument("--launched", action="store_true",
                    help="whether every run of every RUNNING job is in fact "
                         "training, read from each run's own log")
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    ap.add_argument("--logs", default=Path("logs"), type=Path)
    args = ap.parse_args(argv)

    text = args.triples.read_text(encoding="utf-8") if args.triples else queued_text()
    if args.launched:
        return report_launched(text, args.runs, args.logs)
    if args.order:
        if not text.strip():
            print("No queue readable. Off cluster, or the queue is empty. "
                  "Nothing is checked and nothing is claimed.")
            return 0
        return report_order(text, args.configs, args.budget)
    rows = triples_from_text(text)
    if not rows:
        print("No queued runs found. Off cluster, or the queue is empty. "
              "Nothing is checked and nothing is claimed.")
        return 0

    bad = []
    for rid, cfg, seed in rows:
        ok, why = check_one(rid, cfg, seed, args.configs)
        print(f"  {'OK  ' if ok else 'FAIL'} {rid:<34} {why}")
        if not ok:
            bad.append((rid, why))

    print(f"\n{len(rows) - len(bad)} of {len(rows)} queued runs can build a model "
          f"and generate a batch.")
    if bad:
        print("\nThese will die seconds after they get a slot, while qstat shows "
              "the job running. Fix or qdel them before they cost a slot:")
        for rid, why in bad:
            print(f"  {rid}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
