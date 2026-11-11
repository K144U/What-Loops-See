"""M5 analysis. The two registered H2 tests, per criterion.

Preregistration Section 3 fixes both tests and neither may move:

  signed error  one sample test that the median of k_hat_c - k*_model is
                nonzero, sign reported alongside magnitude because halting
                early and halting late have opposite engineering fixes
  slope         regress k_hat_c - k*_model on k*_task, test slope nonzero

Four criteria, so eight of the fourteen registered tests live here. H2 is
supported if, for at least one criterion, BOTH are significant after Holm.
It is falsified only if, for every criterion, both are indistinguishable
from zero. Anything between those is neither, and this module says so
rather than rounding toward one.

SEEDS ARE THE UNIT, NOT ITEMS. One run contributes a few hundred items that
share a set of weights and, worse, share thresholds fitted on that run's own
training split. Treating them as independent would turn one model into a few
hundred observations and shrink every interval to nothing. So every statistic
is computed per seed first and the bootstrap resamples whole seeds, exactly as
M1 does and for the same reason.

WHICH MEDIAN IS REPORTED, because two different numbers both answer to that
name. The preregistration asks whether the median of k_hat_c - k*_model is
nonzero. Pooling every item and taking one median would weight a run that
contributed six hundred items six times as heavily as one that contributed a
hundred, and those items are not independent anyway. What is reported instead
is the median across seeds of each seed's own median, which weights runs
equally and is the estimator the seed level bootstrap is an interval for. The
two coincide when every run contributes the same number of items and diverge
when they do not, so the choice is stated rather than left to be inferred from
the code. Item counts per criterion are printed for the same reason.

WHAT THIS REFUSES TO DO. It does not print an H2 verdict from the eight tests
alone. Holm runs over the registered family of fourteen and six of those are
H1 and H3. Correcting eight as though they were the family would be a smaller
correction than the preregistration demands. Two conclusions survive without
the missing six and both are reported, because they hold whatever those six
turn out to be:

  p <= alpha / 14  is rejected by Holm no matter what the other p-values are
  p >  alpha       is rejected by nothing

Between those the answer genuinely depends on tests that have not run.

WHAT THE DESIGN MAY NOT BE ABLE TO REACH, said before any number exists. Holm
needs p <= 0.05/14 = 0.00357 for a first rejection. A two sided sign test over
n seeds cannot return anything below 2 * 0.5^n, whatever the effect size:

    n = 5   0.0625     above alpha before any correction at all
    n = 8   0.0078     below alpha, still above alpha/14
    n = 9   0.0039     still above alpha/14, and only just
    n = 10  0.00195    the first n that clears Holm

**Ten seeds is not a round number here, it is the smallest n at which the
distribution free test can clear the correction.** The preregistration already
requires ten for any number reaching an abstract, a headline or a figure
caption, and paper.md calls H2 the highest practical impact result in the
paper, so H2 will reach one. Those two rules coincide on ten by arithmetic
rather than by agreement, which is worth knowing because it means five seed H2
cells cannot be rescued by a better test, only by more seeds.

The percentile bootstrap can go below these floors, but at five seeds it does
so largely because it is optimistic there rather than because the evidence is
strong. Both are reported side by side for exactly that reason: when the
bootstrap says 0.0001 and the sign test says 0.0625, the disagreement is the
seed count and not the data.

CENSORING IS NOT NEUTRAL AND IS NEVER SILENT. k*_model is censored when the
model never answers an item correctly at any k, and k_hat_c is censored when
the criterion never fires. The difference is undefined in both cases, so those
items leave the test. They are not a random subset: items the model never
solves are the hard ones, and a criterion that never fires is a badly
calibrated one. Dropping them flatters both the model and the criterion. Every
count is reported per criterion so the reader can see what was removed.

CLAIM 3, THE BLIND AXIS, IS HERE AND IS NOT A REGISTERED TEST. paper.md
Section 4 states it: the error is sensitive to depth and insensitive to
breadth, or the reverse, and that is what ties H2 to H1 because it says which
axis of task structure the diagnostics can actually see. The same paragraph
says "Claim 3 is unfalsifiable on its own and is reported descriptively".
The preregistration's Section 3 registers two tests per criterion and both
regress on k*_task; no breadth test appears anywhere in the family of
fourteen. Section 1 then says any statistic not defined in that file is
post-hoc, is labelled post-hoc in the paper, and gets a confirmation run at
milestone 8. So the breadth slope below carries no Holm column, is never
counted toward supporting or falsifying H2, and is printed under a heading
that says all of this. Adding it to the family would also be a change to a
frozen document, which is not a thing this project does.

NO SINGLE TASK CAN ANSWER CLAIM 3, WHICH IS WHY IT IS ASSEMBLED ACROSS THEM.
Family B varies depth 1 to 6 inside one run and holds breadth fixed. Family C
varies breadth 4 to 8 and holds depth at 1. A slope needs two levels of its
own axis, so family B can give a depth slope and no breadth slope, family C
the reverse, and the contrast only exists once both have been passed. That is
why this is a cross-task report rather than another column in the per-task
tables, and why it says which task each half came from.

BREADTH HAS TO HAVE BEEN RECORDED AT PASS TIME. `b_task` was added to
instruments/m5_halting.py on 2026-09-10. Before that `_pass` discarded the
fifth value `make_batch` returns, so every parquet written earlier has no
breadth column and no way to recover one. Those runs are named rather than
dropped, because a missing axis and a flat one are different results and only
one of them is evidence.

    python -m loopvision.analysis.m5_h2
    python -m loopvision.analysis.m5_h2 --runs runs --alpha 0.05
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from loopvision.instruments.m5_halting import CENSORED, CRITERIA

#: Fixed by the preregistration Section 1, on every test in the family.
ALPHA = 0.05

#: Bootstrap resamples, percentile method, preregistration Section 1.
N_BOOT = 10000

#: H1 depth, H1 breadth, H2 signed error times four, H2 slope times four,
#: and the two H3 contrasts. Eight of these are computed here.
REGISTERED_FAMILY_SIZE = 14

#: The preregistration's floor on seeds per cell, and its higher floor for
#: any number that reaches an abstract, headline or figure caption.
MIN_SEEDS = 5
MIN_SEEDS_HEADLINE = 10


def _identity(cfg: dict) -> str:
    """The task a run was trained on, derived from its own config.

    M1 pooled three task definitions into one fit because it grouped by
    family, and every breadth 6 to 9 cell in it came from runs a later
    decision had invalidated. Grouping on what the run was actually trained
    on cannot fall behind the way a list of run ids does.
    """
    return "%s d%s b%s" % (
        cfg.get("family"),
        tuple(cfg.get("depths") or ()),
        tuple(cfg.get("breadths") or ()),
    )


def collect(runs_root: Path):
    """Every run carrying an m5_halting.parquet, with its seed and task."""
    import pandas as pd

    frames = []
    for p in sorted(glob.glob(str(runs_root / "*" / "m5_halting.parquet"))):
        run_dir = Path(p).parent
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.is_file():
            continue
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        df = pd.read_parquet(p)
        if "b_task" not in df.columns:
            # Written before 2026-09-10, when _pass discarded breadth. NaN
            # rather than a sentinel: an item whose breadth is unknown and an
            # item whose breadth is undefined are not the same thing, and only
            # NaN keeps them apart when the slope is fitted.
            df["b_task"] = np.nan
        df["run_id"] = run_dir.name
        df["seed"] = cfg.get("seed", 0)
        # How many breadth levels this run was configured to sweep. A run
        # missing b_task only needs a re-pass if breadth actually varies in
        # it; family B fixes breadth, so re-passing it would spend a slot to
        # learn a constant. The config knows this and the parquet does not.
        df["cfg_breadth_levels"] = len(set(cfg.get("breadths") or ()))
        df["task_key"] = _identity(cfg)
        frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=["run_id", "seed", "task_key", "k_task", "b_task",
                     "cfg_breadth_levels", "k_model"]
            + [f"k_hat_{c}" for c in CRITERIA]
        )
    return pd.concat(frames, ignore_index=True)


def usable(df, criterion: str):
    """Rows where the signed error is defined, plus what was removed.

    Returns the usable frame and a dict of counts, because the counts are
    part of the result rather than diagnostics. A criterion that fires on a
    third of items and looks perfectly calibrated on those is not a well
    calibrated criterion, and only the counts show it.
    """
    col = f"k_hat_{criterion}"
    n_total = len(df)
    model_censored = int((df.k_model == CENSORED).sum())
    crit_censored = int((df[col] == CENSORED).sum())
    keep = (df.k_model != CENSORED) & (df[col] != CENSORED)
    return df[keep], {
        "n_total": n_total,
        "n_used": int(keep.sum()),
        "model_never_correct": model_censored,
        "criterion_never_fired": crit_censored,
    }


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least squares slope of y on x, or NaN when x has no leverage.

    Two distinct finite x values are the minimum, and NaN rows are dropped
    pairwise. Returning 0.0 for a single-level x would be a fitted flat line
    where there is no line at all, which is exactly the confusion Claim 3
    turns on: a criterion blind to breadth and a run that never varied
    breadth produce the same zero and mean opposite things.

    THIS IS A MARGINAL FIT, one axis at a time, which is what the
    preregistration asks for on k*_task and what Claim 3 needs on breadth.
    It is unconfounded only where the other axis is held fixed or crossed
    evenly against this one. Families B and C each fix the axis they are
    not sweeping, so today every slope is clean. A family varying both
    unevenly would have each slope absorb part of the other, and
    print_blind_axis says so when it sees one.
    """
    fin = np.isfinite(x) & np.isfinite(y)
    if fin.sum() < 2:
        return float("nan")
    xf, yf = x[fin], y[fin]
    if len(np.unique(xf)) < 2:
        return float("nan")
    return float(np.polyfit(xf, yf, 1)[0])


def per_seed(df, criterion: str):
    """One median signed error and two slopes per seed.

    The registered slope needs at least two distinct k*_task values inside
    that seed. A seed trained on a single depth contributes a median and no
    slope, and is counted rather than dropped silently.

    `slope_breadth` is the same fit against b_task and is NOT registered. It
    exists for Claim 3, the blind axis, which paper.md Section 4 reports
    descriptively. It is NaN for a seed whose breadth never varies and for
    every run passed before b_task was recorded, and those two cases are
    told apart upstream in `axes_present` rather than here.
    """
    import pandas as pd

    col = f"k_hat_{criterion}"
    has_breadth = "b_task" in df.columns
    rows = []
    for s, g in df.groupby("seed"):
        err = (g[col] - g.k_model).to_numpy(dtype=float)
        if len(err) == 0:
            continue
        kt = g.k_task.to_numpy(dtype=float)
        bt = (g.b_task.to_numpy(dtype=float) if has_breadth
              else np.full(len(err), np.nan))
        rows.append(
            {
                "seed": s,
                "median_error": float(np.median(err)),
                "slope": _slope(kt, err),
                "slope_breadth": _slope(bt, err),
                "n_items": int(len(err)),
            }
        )
    return pd.DataFrame(rows)


def axes_present(df) -> dict:
    """Which levels of each axis this frame actually holds.

    Three states have to stay distinct and only this function can separate
    them: breadth varies, breadth is fixed at one level, and breadth was
    never recorded. The last is a property of when the pass ran rather than
    of the task, and it is the one that would otherwise be misread as a flat
    axis. Runs missing the column are named so the remedy is obvious.
    """
    kt = df.k_task.to_numpy(dtype=float)
    if "b_task" in df.columns:
        bt = df.b_task.to_numpy(dtype=float)
    else:
        bt = np.full(len(df), np.nan)
    stored = np.isfinite(bt)
    gone = df[~stored] if len(df) else df
    missing = sorted(gone.run_id.unique().tolist()) if len(gone) else []

    # Unknown configs are treated as sweeping, which errs toward offering a
    # re-pass rather than quietly withholding one.
    if len(gone) and "cfg_breadth_levels" in gone.columns:
        sweeps = gone.groupby("run_id").cfg_breadth_levels.max()
        helps = sorted(sweeps[sweeps >= 2].index.tolist())
        moot = sorted(sweeps[sweeps < 2].index.tolist())
    else:
        helps, moot = missing, []

    return {
        "depth_levels": sorted({int(x) for x in kt[np.isfinite(kt)]}),
        "breadth_levels": sorted({int(x) for x in bt[stored]}),
        "runs_without_breadth": missing,
        "repass_would_add_an_axis": helps,
        "repass_would_add_nothing": moot,
        "n_items": int(len(df)),
    }


def bootstrap_stat(values: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Point estimate, percentile interval and two sided p, over seeds.

    The p is the usual percentile bootstrap two sided form, twice the
    smaller tail mass on either side of zero, floored at 1/n_boot because
    no resampling scheme can report a probability below its own resolution.
    Reporting 0.0 there would be a claim the procedure cannot make.
    """
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = len(values)
    if n == 0:
        return {"point": None, "ci": None, "p": None, "n_seeds": 0}
    point = float(np.median(values))
    if n < 2:
        return {"point": point, "ci": None, "p": None, "n_seeds": n}
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, n), replace=True)
    stats = np.median(draws, axis=1)
    lo, hi = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)
    below = float(np.mean(stats <= 0))
    above = float(np.mean(stats >= 0))
    p = min(1.0, 2 * min(below, above))
    return {
        "point": point,
        "ci": (float(np.percentile(stats, lo)), float(np.percentile(stats, hi))),
        "p": max(p, 1.0 / n_boot),
        "n_seeds": n,
    }


def sign_test_p(values: np.ndarray) -> float | None:
    """Two sided sign test that the median is nonzero.

    Distribution free, and reported next to the bootstrap because it is the
    honest floor at small n. Its smallest possible value is 2 * 0.5^n, so at
    five seeds it cannot go below 0.0625 no matter how large the effect. When
    the bootstrap says 0.0001 and this says 0.0625, the disagreement is the
    seed count and not the evidence.
    """
    from math import comb

    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    nz = values[values != 0]
    n = len(nz)
    if n == 0:
        return None
    k = int((nz > 0).sum())
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def sign_test_floor(n_seeds: int) -> float:
    """The smallest p a two sided sign test can return at this many seeds."""
    if n_seeds < 1:
        return 1.0
    return float(min(1.0, 2 * 0.5 ** n_seeds))


def holm_bound(p: float | None, family_size: int = REGISTERED_FAMILY_SIZE,
               alpha: float = ALPHA) -> str:
    """What Holm says about one p when the rest of the family is missing.

    Two conclusions do not depend on the other p-values.

    A test with p <= alpha/m is rejected whatever the others are. Holm
    compares the i-th smallest p against alpha/(m-i+1), which is never
    smaller than alpha/m, and every test ordered ahead of this one also has
    p <= alpha/m and so clears its own threshold, which is what the step
    down procedure needs before it reaches this one.

    A test with p > alpha is rejected by nothing, since alpha/(m-i+1) <=
    alpha at every rank.

    Between the two, the answer depends on tests that have not run, and the
    honest report is that it is undetermined rather than a guess.
    """
    if p is None:
        return "no test"
    if p <= alpha / family_size:
        return "rejected under Holm"
    if p > alpha:
        return "not rejected"
    return "undetermined until the rest of the family runs"


_EMPTY = {"point": None, "ci": None, "p": None, "n_seeds": 0}


def analyse(df, n_boot: int = N_BOOT, seed: int = 0) -> list[dict]:
    """The eight registered tests, two per criterion, and one that is not.

    `breadth_slope` rides along because it needs the same usable rows and the
    same per-seed reduction as the registered slope, and computing it in a
    second pass would risk the two disagreeing about which items were
    censored. It is not one of the eight. Nothing may put it through
    `holm_bound`, and `blind_axis` is where it gets reported.

    Each statistic gets its own bootstrap stream, offset from `seed`, so the
    three are not read off identical resamples.
    """
    out = []
    for c in CRITERIA:
        use, counts = usable(df, c)
        ps = per_seed(use, c)
        med = (bootstrap_stat(ps.median_error.to_numpy(), n_boot, seed)
               if len(ps) else dict(_EMPTY))
        slp = (bootstrap_stat(ps.slope.to_numpy(), n_boot, seed + 1)
               if len(ps) else dict(_EMPTY))
        brd = (bootstrap_stat(ps.slope_breadth.to_numpy(), n_boot, seed + 2)
               if len(ps) else dict(_EMPTY))
        out.append(
            {
                "criterion": c,
                "counts": counts,
                "signed_error": med,
                "slope": slp,
                "breadth_slope": brd,
                "sign_p": sign_test_p(ps.median_error.to_numpy()) if len(ps) else None,
                "sign_floor": sign_test_floor(med["n_seeds"]),
                "n_seeds_with_slope": int(np.isfinite(ps.slope).sum()) if len(ps) else 0,
                "n_seeds_with_breadth_slope": (
                    int(np.isfinite(ps.slope_breadth).sum()) if len(ps) else 0),
                "axes": axes_present(use),
            }
        )
    return out


def blind_axis(per_task: list) -> dict:
    """Claim 3, assembled across tasks. Descriptive, and outside the family.

    Takes the (task_key, rows) pairs `main` has already computed and, per
    criterion, collects the depth slope from every task whose depth varies
    and the breadth slope from every task whose breadth varies. Each entry
    names the task it came from, because the two halves come from different
    families by construction and a reader who cannot see that would take the
    pair for one measurement.

    THE TWO SLOPES ARE PER UNIT OF THEIR OWN AXIS and are not rescaled here.
    Depth runs 1 to 6 and breadth 4 to 8, so the axes differ in range and in
    what one step of them means. Dividing by the range, or standardising,
    would make them look commensurable at the cost of a transformation that
    is unregistered and chosen after seeing the numbers. The levels are
    reported next to each slope instead and the comparison is left explicit.

    No Holm column, no threshold, no verdict. See the module docstring.
    """
    report: dict = {"criteria": {}, "tasks": {}, "runs_without_breadth": []}
    missing: set = set()
    helps: set = set()
    moot: set = set()

    for task_key, rows in per_task:
        axes = rows[0]["axes"] if rows else {}
        report["tasks"][str(task_key)] = {
            "depth_levels": axes.get("depth_levels", []),
            "breadth_levels": axes.get("breadth_levels", []),
        }
        missing.update(axes.get("runs_without_breadth", []))
        helps.update(axes.get("repass_would_add_an_axis", []))
        moot.update(axes.get("repass_would_add_nothing", []))

    keys = ("point", "ci", "p", "n_seeds")
    for c in CRITERIA:
        depth, breadth = [], []
        for task_key, rows in per_task:
            r = next((x for x in rows if x["criterion"] == c), None)
            if r is None:
                continue
            axes = r["axes"]
            if len(axes["depth_levels"]) >= 2 and r["slope"]["point"] is not None:
                entry = {k: r["slope"][k] for k in keys}
                entry.update(task=str(task_key), levels=axes["depth_levels"])
                depth.append(entry)
            if len(axes["breadth_levels"]) >= 2 and r["breadth_slope"]["point"] is not None:
                entry = {k: r["breadth_slope"][k] for k in keys}
                entry.update(task=str(task_key), levels=axes["breadth_levels"])
                breadth.append(entry)
        report["criteria"][c] = {"depth": depth, "breadth": breadth}

    report["runs_without_breadth"] = sorted(missing)
    report["repass_would_add_an_axis"] = sorted(helps)
    report["repass_would_add_nothing"] = sorted(moot)
    report["has_depth"] = any(v["depth"] for v in report["criteria"].values())
    report["has_breadth"] = any(v["breadth"] for v in report["criteria"].values())
    return report


def accuracy_curve(runs_root: Path, run_ids) -> list[dict]:
    """Held-out accuracy at each swept k, averaged over cells and seeds.

    Part A of H2 in findings.md. The halting distributions say where each
    criterion stops; this says what accuracy was available there, and F6
    needs both on one axis or the gap has no consequence attached to it.

    Read from each run's own metrics.parquet rather than recomputed, and
    averaged here rather than in the figure, because a plot script may not
    take a statistic.
    """
    import glob

    import pandas as pd

    frames = []
    for rid in sorted(set(run_ids)):
        p = runs_root / rid / "metrics.parquet"
        if not p.is_file():
            continue
        d = pd.read_parquet(p, columns=["metric_name", "k", "depth", "value"])
        sw = d[(d.metric_name == "sweep_accuracy") & (d.depth > 0)]
        if len(sw):
            frames.append(sw[["k", "value"]])
    if not frames:
        return []
    allf = pd.concat(frames, ignore_index=True)
    return [{"k": int(k), "accuracy": float(g.value.mean()), "cells": int(len(g))}
            for k, g in allf.groupby("k")]


def halting_summary(df, runs_root: Path, rows: list[dict]) -> dict:
    """Everything F6 draws, taken here where it is tested.

    Per criterion the distribution of where it fires, both oracles on the
    same loop-count axis, and the accuracy available at each k. Quantiles
    are over items pooled across seeds: this is a picture of where the
    criteria land, not a test, and the tests that carry the inference are
    in `analyse` where seeds are the unit.
    """
    out: dict = {
        "n_seeds": int(df.seed.nunique()),
        "n_items": int(len(df)),
        "runs": sorted(df.run_id.unique().tolist()),
        "criteria": {},
        "oracles": {},
    }

    def quant(series):
        s = series[series != CENSORED].astype(float)
        if not len(s):
            return None
        return {"median": float(s.median()), "q25": float(s.quantile(0.25)),
                "q75": float(s.quantile(0.75)), "n": int(len(s))}

    for c in CRITERIA:
        col = f"k_hat_{c}"
        entry = quant(df[col]) or {}
        entry["never_fired"] = int((df[col] == CENSORED).sum())
        row = next((r for r in rows if r["criterion"] == c), None)
        if row is not None:
            entry["signed_median"] = row["signed_error"]["point"]
            entry["signed_ci"] = row["signed_error"]["ci"]
            entry["signed_p"] = row["signed_error"]["p"]
        out["criteria"][c] = entry

    out["oracles"]["k_model"] = quant(df.k_model)
    out["oracles"]["k_task"] = quant(df.k_task)
    out["accuracy_curve"] = accuracy_curve(runs_root, out["runs"])
    return out


def k_model_vs_k_task(df) -> dict:
    """Descriptive, and explicitly outside the corrected family.

    The preregistration reports this and does not test it. It is the one
    comparison the text literature cannot make, since k*_task needs a
    generating program, and it separates whether a criterion tracks the
    model from whether the model tracks the task.
    """
    ok = df[df.k_model != CENSORED]
    if len(ok) < 2:
        return {"n": int(len(ok)), "pearson": None}
    kt, km = ok.k_task.to_numpy(float), ok.k_model.to_numpy(float)
    if np.std(kt) == 0 or np.std(km) == 0:
        return {"n": int(len(ok)), "pearson": None}
    return {"n": int(len(ok)), "pearson": float(np.corrcoef(kt, km)[0, 1])}


def _fmt(v, spec=".3f"):
    return "n/a" if v is None else format(v, spec)


def print_blind_axis(report: dict) -> None:
    """Claim 3 at the terminal, with its status stated before its numbers.

    The heading carries the caveat rather than a footnote below the table,
    because a table of slopes with p-values next to eight registered tests
    is exactly the thing that gets quoted as a ninth.
    """
    print("\n=== Claim 3, the blind axis. DESCRIPTIVE, not one of the "
          f"{REGISTERED_FAMILY_SIZE} ===")
    print("    paper.md Section 4 reports this claim descriptively and calls it")
    print("    unfalsifiable on its own. The preregistration registers two tests")
    print("    per criterion and both regress on k*_task, so a breadth slope is")
    print("    post-hoc under its Section 1 and owes a confirmation run at")
    print("    milestone 8. No Holm column below, and no verdict.")

    print("\n    what each task can answer:")
    crossed = []
    for task_key, ax in sorted(report["tasks"].items()):
        d, b = ax["depth_levels"], ax["breadth_levels"]
        d_s = f"depth {d}" if len(d) >= 2 else f"depth fixed at {d or 'none'}"
        b_s = (f"breadth {b}" if len(b) >= 2 else
               f"breadth fixed at {b}" if b else "breadth NOT RECORDED")
        print(f"    {task_key:<28} {d_s},  {b_s}")
        if len(d) >= 2 and len(b) >= 2:
            crossed.append(task_key)

    if crossed:
        print("\n    BOTH AXES VARY IN: " + ", ".join(crossed))
        print("    Each slope below is a marginal fit, one axis at a time,")
        print("    so unless the two are crossed evenly each absorbs part")
        print("    of the other. Families B and C fix the axis they do not")
        print("    sweep and are unaffected. Read a slope from a task in")
        print("    this list only after checking the design is balanced.")

    if report["repass_would_add_an_axis"]:
        n = len(report["repass_would_add_an_axis"])
        print(f"\n    {n} run(s) sweep breadth but carry no b_task column, so")
        print("    Claim 3 is missing an axis it could have. Breadth is a")
        print("    property of the item and is gone once the pass ends, so these")
        print("    need passing again. scripts/m5_halting.pbs skips a run that")
        print("    already has m5_halting.parquet, so submit it with FORCE=1 or")
        print("    delete the parquet first.")
        for rid in report["repass_would_add_an_axis"]:
            print(f"      {rid}")

    if report["repass_would_add_nothing"]:
        n = len(report["repass_would_add_nothing"])
        print(f"\n    {n} further run(s) lack b_task and hold breadth fixed by")
        print("    config, so a re-pass would record a constant and add no axis.")
        print("    Not worth a slot. Named so the absence is not mistaken for an")
        print("    oversight:")
        for rid in report["repass_would_add_nothing"]:
            print(f"      {rid}")

    if not report["has_breadth"]:
        print("\n    NO BREADTH AXIS IS PRESENT, so Claim 3 is not reported.")
        print("    That is an absent result and not a flat one. A criterion")
        print("    blind to breadth and a set of runs that never varied breadth")
        print("    both produce no slope, and only one of them is evidence.")
        return

    print("\n    criterion            axis      slope         95% CI      "
          "boot p  seeds  levels")
    for c in CRITERIA:
        entry = report["criteria"][c]
        for axis in ("depth", "breadth"):
            for e in entry[axis]:
                ci = e["ci"]
                cis = "n/a" if ci is None else f"[{ci[0]:+.2f},{ci[1]:+.2f}]"
                print("    %-20s %-8s %+7s  %14s %7s %6d  %s" % (
                    c, axis, _fmt(e["point"], "+.2f"), cis,
                    _fmt(e["p"], ".4f"), e["n_seeds"], e["levels"]))

    print("\n    The two slopes are per unit of their own axis and are not")
    print("    rescaled. Depth and breadth differ in range and in what one step")
    print("    of them means, so read the levels column before comparing the")
    print("    magnitudes. Normalising here would be an unregistered choice")
    print("    made after seeing the numbers.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M5 registered H2 tests")
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--summary", default=None,
                    help="write what F6 draws here, keyed by task")
    args = ap.parse_args(argv)

    if args.alpha != ALPHA:
        print(f"alpha is fixed at {ALPHA} by the preregistration and this "
              f"run asked for {args.alpha}. Refusing.")
        return 2

    summaries: dict = {}
    per_task: list = []
    df = collect(args.runs)
    if len(df) == 0:
        print("no run carries an m5_halting.parquet yet.")
        print("M5 needs a GPU pass per run: python -m loopvision.instruments.m5_halting")
        return 1

    for task_key, g in df.groupby("task_key"):
        runs = sorted(g.run_id.unique())
        print(f"\n=== {task_key}   {len(runs)} run(s), "
              f"{g.seed.nunique()} seed(s), {len(g)} item rows")
        if g.seed.nunique() < MIN_SEEDS:
            print(f"    below the preregistration's floor of {MIN_SEEDS} seeds "
                  f"per cell. Estimates print; the tests do not bind.")

        d = k_model_vs_k_task(g)
        print(f"    k*_model against k*_task, descriptive and uncorrected: "
              f"r = {_fmt(d['pearson'])} on n = {d['n']}")

        rows = analyse(g, n_boot=args.n_boot, seed=args.seed)
        per_task.append((task_key, rows))
        if args.summary:
            summaries[str(task_key)] = halting_summary(g, args.runs, rows)
        print("\n    criterion            median err        95% CI      "
              "boot p   sign p   Holm")
        for r in rows:
            se, ci = r["signed_error"], r["signed_error"]["ci"]
            cis = "n/a" if ci is None else f"[{ci[0]:+.2f},{ci[1]:+.2f}]"
            print("    %-20s %+8s  %14s %8s %8s   %s" % (
                r["criterion"], _fmt(se["point"], "+.2f"), cis,
                _fmt(se["p"], ".4f"), _fmt(r["sign_p"], ".4f"),
                holm_bound(se["p"], REGISTERED_FAMILY_SIZE, ALPHA)))

        print("\n    criterion            slope/k*_task     95% CI      "
              "boot p            Holm")
        for r in rows:
            sl, ci = r["slope"], r["slope"]["ci"]
            cis = "n/a" if ci is None else f"[{ci[0]:+.2f},{ci[1]:+.2f}]"
            print("    %-20s %+8s  %14s %8s   %s" % (
                r["criterion"], _fmt(sl["point"], "+.2f"), cis,
                _fmt(sl["p"], ".4f"),
                holm_bound(sl["p"], REGISTERED_FAMILY_SIZE, ALPHA)))

        print("\n    what left each test, and why it is not a random subset:")
        for r in rows:
            c = r["counts"]
            print("    %-20s used %5d of %5d   model never correct %5d   "
                  "never fired %5d" % (
                      r["criterion"], c["n_used"], c["n_total"],
                      c["model_never_correct"], c["criterion_never_fired"]))

        n_seeds = max((r["signed_error"]["n_seeds"] for r in rows), default=0)
        floor = sign_test_floor(n_seeds)
        holm_edge = ALPHA / REGISTERED_FAMILY_SIZE
        # The threshold that matters for a registered rejection is alpha
        # over the family, not alpha. An earlier version warned only when
        # the floor cleared alpha, which is silent at eight seeds: there
        # the sign test can reach 0.0078, under alpha and over alpha/14, so
        # every "rejected under Holm" in the table above rests on the
        # bootstrap alone while the distribution free test cannot get there
        # however large the effect. That is exactly the case this data is
        # in, and it was the case the warning missed.
        if floor > holm_edge:
            print(f"\n    SEEDS. At {n_seeds} seeds a two sided sign test "
                  f"cannot return below {floor:.4f}. Holm needs "
                  f"{holm_edge:.5f}, so the distribution free test cannot "
                  f"clear the correction at this n whatever the effect size, "
                  f"and every rejection above rests on the bootstrap alone. "
                  f"{MIN_SEEDS_HEADLINE} seeds reach {sign_test_floor(MIN_SEEDS_HEADLINE):.5f} "
                  f"and are required by the preregistration for a headline "
                  f"number, which H2 is intended to be.")
            if floor <= ALPHA:
                print(f"    The floor is under alpha {ALPHA}, so this is not "
                      f"visible from the uncorrected p alone.")

    axis_report = blind_axis(per_task)
    print_blind_axis(axis_report)

    if args.summary:
        Path(args.summary).write_text(
            json.dumps({"alpha": ALPHA,
                        "family_size": REGISTERED_FAMILY_SIZE,
                        "criteria_order": list(CRITERIA),
                        "tasks": summaries,
                        # Outside "tasks" and under a name that says what it
                        # is, so a figure reading this file cannot pick the
                        # breadth slope up as a ninth registered test.
                        "blind_axis_descriptive": axis_report},
                       indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nwrote {args.summary}")

    print(f"\nNo H2 verdict is printed. Holm runs over the registered family of "
          f"{REGISTERED_FAMILY_SIZE} tests; eight are here and the remaining six "
          f"are H1 and H3. The Holm column states only what holds whatever those "
          f"six turn out to be.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
