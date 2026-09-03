"""Plan and submit a sweep, respecting the cluster's actual constraints.

    python -m loopvision.analysis.orchestrator --sweep configs/sweep/m1.yaml
    python -m loopvision.analysis.orchestrator --sweep ... --submit

Prints a plan by default and only submits when asked, because a sweep is
tens of GPU-days and a typo in a config should not cost that.

WHAT THIS CAN AND CANNOT DO, since "orchestrator" sounds like it can
conjure capacity.

It cannot exceed the queue's 16 concurrent core cap per user. PBS enforces
that and no amount of scheduling logic gets past it.

What it can do is spend that allocation better. Measured on 2026-09-03, the
GPU node had 94 of 96 cores allocated while GPUs 2 and 5 sat at 0 percent:
the cards were idle because nothing had cores to feed them. Submitting more
jobs would simply have queued them. Packing several runs into one job's
core allocation, each pinned to its own card, converts spare GPU capacity
into throughput without asking for more cores.

At `runs_per_job=4` and `ncpus=8`, the cap allows two such jobs, so eight
runs are in flight against the four that `submit.pbs` would have managed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

# Hard limits read from the gpu queue with `qstat -Qf gpu` on 2026-09-03.
# None of these are ours to change, and a request over any of them is
# rejected outright with "Job violates queue and/or server resource limits",
# which names no specific resource, so they are recorded here rather than
# rediscovered by trial and error.
CORE_CAP = 16          # max_run_res.ncpus, per user, across all running jobs
MAX_JOBS = 5           # max_run, per user
MAX_MEM_GB_PER_JOB = 64  # resources_max.mem
MAX_CPUS_PER_JOB = 16    # resources_max.ncpus


@dataclass(frozen=True)
class Run:
    run_id: str
    config: str  # path relative to configs/
    seed: int

    def spec(self) -> str:
        return f"{self.run_id}|{self.config}|{self.seed}"


def load_sweep(path: Path) -> list[Run]:
    """A sweep file is a list of runs, or a template crossed with seeds.

    template form:
        config: m3/vark_famA_d2.yaml
        run_prefix: vark_famA_d2
        seeds: [0, 1, 2, 3, 4]

    explicit form:
        runs:
          - {run_id: foo_s0, config: a.yaml, seed: 0}
    """
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "runs" in spec:
        return [Run(r["run_id"], r["config"], int(r["seed"])) for r in spec["runs"]]

    prefix = spec["run_prefix"]
    config = spec["config"]
    return [Run(f"{prefix}_s{s}", config, int(s)) for s in spec["seeds"]]


def pending(runs: list[Run], runs_root: Path) -> list[Run]:
    """Drop runs that already have a DONE sentinel.

    The registry refuses to restart a finished run anyway, but filtering
    here keeps the plan honest about how much work is actually left.
    """
    return [r for r in runs if not (runs_root / r.run_id / "DONE").is_file()]


def batch(runs: list[Run], runs_per_job: int) -> list[list[Run]]:
    return [runs[i : i + runs_per_job] for i in range(0, len(runs), runs_per_job)]


def qsub_command(group: list[Run], ncpus: int, max_hours: float, mem: str) -> str:
    specs = " ".join(r.spec() for r in group)
    return (
        f'qsub -l select=1:ncpus={ncpus}:mem={mem} '
        f'-v RUNS="{specs}",MAX_HOURS={max_hours},NCPUS={ncpus} '
        f'scripts/multirun.pbs'
    )


def plan(
    runs: list[Run],
    runs_root: Path,
    runs_per_job: int,
    ncpus: int,
    max_hours: float,
    mem: str,
) -> dict:
    todo = pending(runs, runs_root)
    groups = batch(todo, runs_per_job)
    # Two independent caps: total cores, and total jobs.
    concurrent_jobs = min(max(1, CORE_CAP // ncpus), MAX_JOBS)
    return {
        "total": len(runs),
        "done": len(runs) - len(todo),
        "pending": len(todo),
        "jobs": len(groups),
        "runs_per_job": runs_per_job,
        "ncpus_per_job": ncpus,
        "concurrent_jobs_allowed": concurrent_jobs,
        "runs_in_flight": concurrent_jobs * runs_per_job,
        "groups": groups,
        "commands": [qsub_command(g, ncpus, max_hours, mem) for g in groups],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", required=True, help="sweep yaml")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--runs-per-job", type=int, default=4)
    p.add_argument("--ncpus", type=int, default=8)
    p.add_argument("--mem", default="48gb",
                   help=f"per job. The queue rejects anything over "
                        f"{MAX_MEM_GB_PER_JOB}gb")
    p.add_argument("--max-hours", type=float, default=20.0)
    p.add_argument("--submit", action="store_true", help="actually qsub")
    p.add_argument("--limit", type=int, default=None, help="submit at most N jobs now")
    p.add_argument("--json", help="write the plan here")
    args = p.parse_args()

    if int(args.mem.rstrip("gb")) > MAX_MEM_GB_PER_JOB:
        raise SystemExit(
            f"--mem {args.mem} exceeds the queue's {MAX_MEM_GB_PER_JOB}gb per-job "
            f"limit. PBS would reject this without saying which resource was at "
            f"fault."
        )
    if args.ncpus > MAX_CPUS_PER_JOB:
        raise SystemExit(f"--ncpus {args.ncpus} exceeds the queue's {MAX_CPUS_PER_JOB}")

    runs = load_sweep(Path(args.sweep))
    result = plan(
        runs, Path(args.runs_root), args.runs_per_job, args.ncpus,
        args.max_hours, args.mem,
    )

    print(f"\nsweep: {args.sweep}")
    print(f"  {result['total']} run(s), {result['done']} done, {result['pending']} pending")
    print(f"  {result['jobs']} job(s) of {result['runs_per_job']} run(s) each, "
          f"{result['ncpus_per_job']} cores per job")
    print(f"  core cap {CORE_CAP} allows {result['concurrent_jobs_allowed']} concurrent "
          f"job(s), so {result['runs_in_flight']} run(s) in flight")
    print()
    for i, (group, cmd) in enumerate(zip(result["groups"], result["commands"]), 1):
        mark = "" if args.limit is None or i <= args.limit else "   (deferred)"
        print(f"  job {i}: {', '.join(r.run_id for r in group)}{mark}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps({k: v for k, v in result.items() if k != "groups"}, indent=2),
            encoding="utf-8",
        )

    if not args.submit:
        print("\n  dry run. Add --submit to queue these.\n")
        return 0

    limit = args.limit if args.limit is not None else result["concurrent_jobs_allowed"]
    submitted = 0
    for cmd in result["commands"][:limit]:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"  {out.stdout.strip() or out.stderr.strip()}")
        if out.returncode == 0:
            submitted += 1
    print(f"\n  submitted {submitted} job(s). "
          f"Re-run with --submit as they finish to queue the rest.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
