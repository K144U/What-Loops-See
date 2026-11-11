"""When a single-factor family A run escapes, and whether it climbs a staircase.

E0.2 of docs/why-experiments.md. It replaces the scratch pass stored at
runs/_why_scratch_20260924/, which was written while looking at the data and
never tested. Nothing that pass printed is quotable; what this prints is.

    python -m loopvision.analysis.escape_timing
    python -m loopvision.analysis.escape_timing --json runs/_why_timing/timing.json
    python -m loopvision.analysis.escape_timing --registered

TWO QUANTITIES, AND WHY THE SCRATCH PASS NEEDED SPLITTING INTO THEM. The
scratch pass called a run escaped when its smoothed loss fell below half of
ln C. For D4 position that fires when the run ENTERS its ln 2 plateau, which
can be 100k steps before it solves, so one number was measuring two events.

  solve step   the first logged step at which the trailing 1000 step mean of
               train loss is below SOLVE_FRACTION * ln C. Below every
               quotient level, so it means the whole group.
  plateau      a stretch of at least PLATEAU_MIN_STEPS during which the
               trailing 2000 step mean stays within PLATEAU_BAND of
               ln(C / q), for q a proper divisor of C greater than 1. A model
               that knows the answer's image in a quotient of order q, and
               nothing more, sits at exactly that loss.

C is the order of the subgroup the task draws from, read from the config
through groups.substrate_subgroup, never inferred from a run name.

A plateau level says a loss value, not a structure. On 1 September this
project read a 3.19 plateau as ln 24 and one learned bit, and the bits were
at chance (findings.md, "SUPERSEDED 2026-09-04"). So a plateau here is
reported as a level only. Whether a model on it knows a quotient is a
separate measurement, E0.3.

THE SMOOTHING IS TRAILING, NEVER CENTRED. Each logged loss is one batch at a
sampled k, so it has to be averaged, and a centred window would let a drop
at step S register before S. A trailing mean is only defined once its window
is full, so the first reading cannot come from three batches.

CONFIRMATION SEEDS ARE NEVER POOLED WITH THE DATA THEY TEST. They are grouped
under their own key and only `--registered` reads them. D-057 is why.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from loopvision.analysis import escape_rates as ER
from loopvision.data import groups as G

#: Trailing window for the solve step, in training steps.
SOLVE_WINDOW = 1000
#: Solved means below this fraction of ln C. 0.1 puts it under ln 2 for both
#: groups, so no quotient plateau can pass for a solve.
SOLVE_FRACTION = 0.1

#: Trailing window for plateau detection, in training steps.
PLATEAU_WINDOW = 2000
#: Half width of the band around a quotient level. At 0.15 the bands for
#: ln 4 and ln 2, and for ln 3 and ln 2, do not overlap.
PLATEAU_BAND = 0.15
#: Minimum duration of a plateau, first to last step inside the band.
PLATEAU_MIN_STEPS = 10000
#: A gap in logging wider than this many typical intervals breaks a stretch,
#: so a resumed run cannot join two separate visits into one plateau.
MAX_GAP_INTERVALS = 10

#: The registered timing predictions check this step. See D-059.
WINDOW_LIMIT = 40000

#: Only these orders get quotient levels. The full 48 element group has so
#: many divisors that the bands would overlap, so it is out of scope here.
SUPPORTED_ORDERS = (6, 8)


def group_order(cfg: dict) -> int:
    """How many classes the task has, from the config's substrate and factor."""
    sub = cfg.get("substrate") or "native"
    factor = cfg.get("factor", "full")
    return len(G.subgroup_members(G.substrate_subgroup(sub, factor)))


def quotient_levels(order: int) -> dict[str, float]:
    """Loss at each proper quotient, keyed by a readable name.

    Knowing the image in a quotient of order q leaves order / q candidates,
    so uniform confidence over them costs ln(order / q).
    """
    out = {}
    for q in range(2, order):
        if order % q == 0:
            out[f"ln{order // q}"] = math.log(order / q)
    return out


def trailing_mean(steps: np.ndarray, values: np.ndarray, window: int) -> np.ndarray:
    """Mean of the values logged in (step - window, step], NaN until full.

    Cadence independent: the window is in steps, not in points, so a run
    logged every 50 steps and one logged every 100 give the same meaning.
    """
    steps = np.asarray(steps, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    out = np.full(len(steps), np.nan)
    if len(steps) == 0:
        return out
    cadence = int(np.median(np.diff(steps))) if len(steps) > 1 else window
    csum = np.concatenate([[0.0], np.cumsum(values)])
    for i, s in enumerate(steps):
        if s - window < steps[0] - cadence:
            continue
        j = int(np.searchsorted(steps, s - window, side="right"))
        out[i] = (csum[i + 1] - csum[j]) / (i + 1 - j)
    return out


def train_loss(metrics: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Logged train loss, sorted by step, one value per step.

    A resumed run can log a step twice. The later write wins, because it is
    the one the surviving checkpoint descends from.
    """
    tl = metrics[(metrics.metric_name == "loss") & (metrics.split == "train")]
    tl = tl.drop_duplicates(subset="step", keep="last").sort_values("step")
    return tl["step"].to_numpy(), tl["value"].to_numpy()


def solve_step(steps: np.ndarray, values: np.ndarray, order: int) -> int | None:
    """First step whose trailing mean is below SOLVE_FRACTION * ln(order)."""
    sm = trailing_mean(steps, values, SOLVE_WINDOW)
    hit = np.nonzero(sm < SOLVE_FRACTION * math.log(order))[0]
    return int(steps[hit[0]]) if len(hit) else None


def leave_step(steps: np.ndarray, values: np.ndarray, order: int) -> int | None:
    """First step of a sustained departure from chance.

    The trailing PLATEAU_WINDOW mean has to fall below ln(order) minus
    PLATEAU_BAND and stay below it for PLATEAU_MIN_STEPS. A single dip is not
    a departure: collapsed runs wander around ln C for their whole budget.

    WHY THIS EXISTS BESIDE solve_step. Some identity runs leave chance early
    and then sit on a plateau near 1.0 for tens of thousands of steps before
    solving, so their solve step is late while their escape was not. That
    plateau is not a quotient: eval accuracy there is 1/b + (1 - 1/b)/C at
    breadth b, which is a model composing correctly for an object it has not
    learned to locate.
    """
    sm = trailing_mean(steps, values, PLATEAU_WINDOW)
    steps = np.asarray(steps)
    below = sm < math.log(order) - PLATEAU_BAND
    i = 0
    while i < len(steps):
        if not below[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(steps) and below[j + 1]:
            j += 1
        if steps[j] - steps[i] >= PLATEAU_MIN_STEPS or j == len(steps) - 1:
            return int(steps[i])
        i = j + 1
    return None


def plateaus(steps: np.ndarray, values: np.ndarray, order: int,
             before: int | None = None) -> list[dict]:
    """Every stretch of at least PLATEAU_MIN_STEPS inside a quotient band.

    `before` stops the search at a step, so plateaus after a solve are not
    reported: a solved run sitting at zero is not on any level.
    """
    sm = trailing_mean(steps, values, PLATEAU_WINDOW)
    steps = np.asarray(steps)
    if before is not None:
        keep = steps < before
        steps, sm = steps[keep], sm[keep]
    if len(steps) < 2:
        return []
    cadence = int(np.median(np.diff(steps)))
    found = []
    for name, level in quotient_levels(order).items():
        inside = np.abs(sm - level) <= PLATEAU_BAND
        start = None
        for i in range(len(steps) + 1):
            broken = (
                i == len(steps)
                or not inside[i]
                or (start is not None
                    and steps[i] - steps[i - 1] > MAX_GAP_INTERVALS * cadence)
            )
            if not broken and start is None:
                start = i
            elif broken and start is not None:
                end = i - 1
                if steps[end] - steps[start] >= PLATEAU_MIN_STEPS:
                    found.append({"level": name, "start": int(steps[start]),
                                  "end": int(steps[end])})
                start = i if (i < len(steps) and inside[i]) else None
    return sorted(found, key=lambda p: p["start"])


def group_key(row: dict, cfg: dict) -> str:
    """The run id without its seed, marked when it had a donor or is held out."""
    key = re.sub(r"_s\d+$", "", row["run_id"])
    if cfg.get("init_from"):
        key += " [init_from]"
    if row["confirm_arm"]:
        key += " [confirmation]"
    return key


def scan(runs_root: Path) -> tuple[list[dict], int]:
    """Timing for every finished single-factor family A run.

    Returns the rows and the number of in-scope runs skipped as unfinished.
    """
    rows, unfinished = [], 0
    for r in ER.scan(runs_root):
        run_dir = runs_root / r["run_id"]
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
        if cfg.get("family") != "A":
            continue
        try:
            order = group_order(cfg)
        except ValueError:
            continue
        if order not in SUPPORTED_ORDERS:
            continue
        if not r["done"]:
            unfinished += 1
            continue
        mp = run_dir / "metrics.parquet"
        if not mp.is_file():
            continue
        steps, values = train_loss(pd.read_parquet(mp))
        solved = solve_step(steps, values, order)
        left = leave_step(steps, values, order)
        rows.append({
            "run_id": r["run_id"],
            "group": group_key(r, cfg),
            "arm": list(r["arm"]) if r["arm"] else None,
            "confirm_arm": list(r["confirm_arm"]) if r["confirm_arm"] else None,
            "order": order,
            "outcome": r["outcome"],
            "final_accuracy": r["accuracy"],
            "leave_step": left,
            "solve_step": solved,
            "plateaus": plateaus(steps, values, order, before=solved),
            "last_step": int(steps[-1]) if len(steps) else None,
        })
    return rows, unfinished


def summarise(rows: list[dict]) -> dict:
    """Per group: counts, solve steps, and how many runs sat on each level."""
    out: dict = {}
    for r in rows:
        g = out.setdefault(r["group"], {"runs": 0, "escaped": 0, "solve_steps": [],
                                        "leave_steps": [], "with_plateau": {},
                                        "order": r["order"]})
        g["runs"] += 1
        if r["outcome"] == "escaped":
            g["escaped"] += 1
        if r["solve_step"] is not None:
            g["solve_steps"].append(r["solve_step"])
        if r["leave_step"] is not None:
            g["leave_steps"].append(r["leave_step"])
        for level in sorted({p["level"] for p in r["plateaus"]}):
            g["with_plateau"][level] = g["with_plateau"].get(level, 0) + 1
    for g in out.values():
        g["solve_steps"].sort()
        g["leave_steps"].sort()
        g["latest_leave"] = g["leave_steps"][-1] if g["leave_steps"] else None
        g["latest_solve"] = g["solve_steps"][-1] if g["solve_steps"] else None
    return out


# ---------------------------------------------------------------------------
# The registered timing predictions, D-059
# ---------------------------------------------------------------------------
#
# Written before any confirmation seed had run, on purpose. Each returns
# (verdict, detail) with verdict one of "held", "failed" or "pending".

WINDOW_ARMS = (("D4", "identity"), ("S3", "identity"))
STAIRCASE_ARM = ("D4", "position")
#: How many of the staircase arm's confirmation seeds must show ln 4.
STAIRCASE_MIN = 4
#: How many confirmation seeds each arm has, from D-057.
SEEDS_PER_ARM = 5


def _confirm_rows(rows: list[dict], arm: tuple[str, str]) -> list[dict]:
    return [r for r in rows if r["confirm_arm"] and tuple(r["confirm_arm"]) == arm]


def registered(rows: list[dict]) -> dict[str, tuple[str, str]]:
    """Evaluate P1 to P3 of D-059 on the held-out confirmation seeds only.

    P1  no identity seed, D4 or S3, leaves chance after WINDOW_LIMIT.
    P2  at least STAIRCASE_MIN of the D4 position seeds show an ln 4 plateau.
    P3  no identity seed, D4 or S3, shows a plateau at any quotient level.
    """
    out = {}

    late = []
    finished = 0
    departures = 0
    for arm in WINDOW_ARMS:
        for r in _confirm_rows(rows, arm):
            finished += 1
            if r["leave_step"] is not None:
                departures += 1
            if r["leave_step"] is not None and r["leave_step"] > WINDOW_LIMIT:
                late.append(f"{r['run_id']} left chance at {r['leave_step']}")
    if late:
        out["P1"] = ("failed", "; ".join(late))
    elif finished < SEEDS_PER_ARM * len(WINDOW_ARMS):
        out["P1"] = ("pending", f"{finished} of {SEEDS_PER_ARM * len(WINDOW_ARMS)} finished")
    else:
        # Held with no departures at all is vacuous, and the detail says so
        # in numbers rather than leaving a reader to assume there were some.
        out["P1"] = ("held", f"no identity seed left chance after {WINDOW_LIMIT}; "
                             f"{departures} of {finished} seeds left chance at all")

    stair = _confirm_rows(rows, STAIRCASE_ARM)
    with_ln4 = [r for r in stair if any(p["level"] == "ln4" for p in r["plateaus"])]
    without = len(stair) - len(with_ln4)
    if without > SEEDS_PER_ARM - STAIRCASE_MIN:
        out["P2"] = ("failed", f"{len(with_ln4)} of {len(stair)} finished seeds show ln4")
    elif len(stair) < SEEDS_PER_ARM:
        out["P2"] = ("pending", f"{len(with_ln4)} of {len(stair)} finished seeds show ln4")
    else:
        out["P2"] = ("held", f"{len(with_ln4)} of {len(stair)} seeds show ln4")

    colour = [r for arm in WINDOW_ARMS for r in _confirm_rows(rows, arm)]
    stepped = [r["run_id"] for r in colour if r["plateaus"]]
    if stepped:
        out["P3"] = ("failed", "plateau in " + ", ".join(stepped))
    elif len(colour) < SEEDS_PER_ARM * len(WINDOW_ARMS):
        out["P3"] = ("pending", f"{len(colour)} of {SEEDS_PER_ARM * len(WINDOW_ARMS)} finished")
    else:
        out["P3"] = ("held", f"no plateau in {len(colour)} seeds")
    return out


def _fmt_k(steps: list[int]) -> str:
    return "[" + ", ".join(str(round(s / 1000)) for s in steps) + "]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Escape timing and quotient plateaus")
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    ap.add_argument("--json", default=None, type=Path, help="write rows and summary here")
    ap.add_argument("--registered", action="store_true",
                    help="evaluate D-059 on the confirmation seeds; exit 1 if any "
                         "failed, 2 if any is pending, 0 if all held")
    args = ap.parse_args(argv)

    if not args.runs.is_dir():
        print(f"no runs directory at {args.runs}")
        return 2
    rows, unfinished = scan(args.runs)
    summary = summarise(rows)

    if args.registered:
        verdicts = registered(rows)
        for k, (v, detail) in verdicts.items():
            print(f"{k}  {v:8s} {detail}")
        states = {v for v, _ in verdicts.values()}
        return 1 if "failed" in states else 2 if "pending" in states else 0

    print(f"leave: trailing {PLATEAU_WINDOW} step mean below ln C - {PLATEAU_BAND} for "
          f"at least {PLATEAU_MIN_STEPS} steps, or to the end of the log.")
    print(f"solve: trailing {SOLVE_WINDOW} step mean below {SOLVE_FRACTION} ln C. "
          f"plateau: trailing {PLATEAU_WINDOW} step mean within {PLATEAU_BAND} of "
          f"ln(C/q) for at least {PLATEAU_MIN_STEPS} steps, before the solve.")
    print(f"{unfinished} in-scope runs skipped as unfinished.\n")
    for name, g in sorted(summary.items()):
        levels = " ".join(f"{k}:{v}" for k, v in sorted(g["with_plateau"].items())) or "-"
        print(f"{name:48s} C={g['order']}  escaped {g['escaped']:2d} of {g['runs']:2d}  "
              f"plateaus {levels}")
        print(f"{'':48s} leave k-steps {_fmt_k(g['leave_steps'])}")
        print(f"{'':48s} solve k-steps {_fmt_k(g['solve_steps'])}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"rows": rows, "summary": summary,
                                         "unfinished": unfinished}, indent=1),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
