"""M1. Loop-count requirement curves, and the H1 depth against breadth test.

The contribution is the dissociation, not the depth trend. Depth raising the
loop count is close to known in text; that loops track depth and are flat in
breadth, in a setting where the two are controlled independently, is not.

Preregistration Section 2 fixes every choice here and none of them may move:

  k_min     smallest k at which held-out accuracy reaches tau = 0.90
  censoring a cell that never reaches tau is recorded at 33, and reported as
            censored rather than dropped
  model     k_min ~ a * depth + b * breadth + c, with a per-seed random
            intercept, plus the saturating alternative a * log(depth)
  verdict   supported if the CI for a excludes 0 from above and the CI for b
            contains 0, after Holm correction

WHY THIS REFUSES TO DECLARE A VERDICT. Holm runs across the registered
family of fourteen tests: two here, eight in H2, two in H3, two more H2
slopes. Ten of those do not exist yet. Correcting two tests as though they
were the family would be a smaller correction than the preregistration
demands and would make support easier to reach, so this module reports
estimates and intervals and says the verdict is not available. It becomes
available when H2 and H3 have run, not when someone wants it.

WHAT THE STORED DATA CANNOT GIVE. The preregistration asks for k = 1..32.
Runs swept k in {1,2,3,4,6,8,12,16,24,32,48,64}, so k_min resolves only to
those points: a cell whose true k_min is 5 is recorded as 6. This is a
resolution limit, not a bias, and it is reported with every fit. Closing it
needs a re-sweep at every integer k, which is inference over stored
checkpoints and costs no training.

A CURVE POOLS ONE SETUP. Runs of one task trained differently are different
experiments, so they are kept apart by `setup_key` as runs of different tasks
are kept apart by `task_key`. Until 2026-10-22 only the task was keyed, and
family B's curve pooled ten arms that share its task: feedforward and untied
baselines, three model widths and three jitter widths. F2 drew that pool and
its input was never stored, so the drawn figure stopped matching the runs as
arms were added. See D-074. The figure's summary now names the registered
configs, and a summary that would pool two setups is refused.

    python -m loopvision.analysis.m1_curves
    python -m loopvision.analysis.m1_curves --family B --tau 0.90
    python -m loopvision.analysis.m1_curves --no-fit --summary m1_summary.json \\
        --config configs/m3/famB_curve_111.yaml \\
        --config configs/m3/famC_breadth_111.yaml
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

#: Fixed by the preregistration. Not a tunable.
TAU = 0.90

#: Value recorded for a cell that never reaches tau within the swept range.
#: The preregistration names 33, one past its k = 1..32 range.
CENSORED = 33

#: Bootstrap resamples for the CIs, and the nominal two sided alpha.
N_BOOT = 10000
ALPHA = 0.05

#: The registered family Holm corrects across, from prereg Section 1.
REGISTERED_FAMILY_SIZE = 14

#: Config fields that may differ between repeats of one setup. A seed is what
#: a setup is repeated over, and the worker count and checkpoint interval only
#: pace the job.
NOT_SETUP = ("seed", "master_seed", "num_workers", "checkpoint_every_minutes")


def setup_key(cfg: dict) -> str:
    """How a run was trained, apart from which repeat it is, as a short id.

    Missing keys take the trainer's defaults, because a key added after a
    run launched was added with the default that reproduced that run: older
    runs carry no `substrate` and trained on the native one. A recipient's
    donor counts by its arm and not its seed, since every recipient seed has
    a donor seed of its own.
    """
    from loopvision.train.cli import DEFAULTS

    full = {**DEFAULTS, **cfg}
    for k in NOT_SETUP:
        full.pop(k, None)
    if full.get("init_from"):
        full["init_from"] = re.sub(r"_s\d+$", "", str(full["init_from"]))
    blob = json.dumps(full, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def config_setup(path: str | Path) -> str:
    """The setup a config file trains, resolved as the trainer resolves it."""
    from loopvision.train.cli import load_config

    return setup_key(load_config(str(path), {}))


def arms(run_ids) -> list[str]:
    """Run ids without their seed suffix. For reading output, never for
    grouping: grouping comes from the config."""
    return sorted({re.sub(r"_s\d+$", "", str(r)) for r in run_ids})


def mixed_setups(df) -> dict[str, list[str]]:
    """Each task that more than one setup shares, with the arms sharing it."""
    if "setup" not in df:
        return {}
    return {str(t): arms(g.run_id.unique())
            for t, g in df.groupby("task_key") if g.setup.nunique() > 1}


def k_min_from_curve(ks: np.ndarray, acc: np.ndarray, tau: float = TAU) -> int:
    """Smallest swept k whose accuracy reaches tau, else CENSORED.

    Only k <= 32 counts. The sweep runs to 64 but the preregistration's
    range stops at 32, and a cell rescued only at 48 or 64 is censored
    under the registered definition rather than credited.
    """
    order = np.argsort(ks)
    ks, acc = np.asarray(ks)[order], np.asarray(acc)[order]
    ok = (acc >= tau) & (ks <= 32)
    return int(ks[ok][0]) if ok.any() else CENSORED


def collect(runs_root: Path, family: str | None = None) -> "object":
    """One row per (run, depth, breadth) cell with its k_min."""
    import pandas as pd

    rows = []
    for p in sorted(glob.glob(str(runs_root / "*" / "metrics.parquet"))):
        run = Path(p).parent.name
        summ = Path(p).parent / "summary.json"
        if not summ.exists():
            continue
        cfg = json.loads(summ.read_text(encoding="utf-8"))["config"]
        if family and cfg.get("family") != family:
            continue
        try:
            d = pd.read_parquet(
                p, columns=["metric_name", "k", "depth", "breadth", "value"]
            )
        except Exception:
            continue
        sw = d[(d.metric_name == "sweep_accuracy") & (d.depth > 0)]
        if not len(sw):
            continue
        for (dep, br), g in sw.groupby(["depth", "breadth"]):
            rows.append(
                {
                    "run_id": run,
                    "family": cfg.get("family"),
                    "seed": cfg.get("seed"),
                    "d_model": cfg.get("d_model"),
                    "depth": int(dep),
                    "breadth": int(br),
                    # The task a run was trained on, not just its family.
                    # D-028 moved family B's breadths from 6 to 9 up to 12
                    # to 16, so runs either side of it are different tasks
                    # and pooling them into one fit compares k_min across a
                    # task redefinition. Derived from the run's own config
                    # rather than from a list of dates or run ids.
                    "task_key": "%s d%s b%s" % (
                        cfg.get("family"),
                        tuple(cfg.get("depths") or ()),
                        tuple(cfg.get("breadths") or ()),
                    ),
                    # And how it was trained on it, so that a baseline or
                    # another width sharing the task is not pooled into the
                    # curve. D-074.
                    "setup": setup_key(cfg),
                    "k_min": k_min_from_curve(g.k.to_numpy(), g.value.to_numpy()),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["censored"] = out.k_min == CENSORED
    return out


def _fit_once(df, log_depth: bool):
    """Mixed model with a per-seed random intercept.

    Returns (a, b, converged). Convergence is returned rather than warned
    about, because a bootstrap that quietly includes fits whose optimiser
    gave up produces an interval nobody can defend. The caller counts them.
    """
    import warnings

    import statsmodels.formula.api as smf

    d = df.copy()
    d["x_depth"] = np.log(d.depth) if log_depth else d.depth.astype(float)
    d["x_breadth"] = d.breadth.astype(float)
    d["y"] = d.k_min.astype(float)
    if d.seed.nunique() < 2:
        import statsmodels.api as sm

        X = sm.add_constant(d[["x_depth", "x_breadth"]])
        res = sm.OLS(d.y, X).fit()
        return float(res.params["x_depth"]), float(res.params["x_breadth"]), True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = smf.mixedlm("y ~ x_depth + x_breadth", d, groups=d["seed"]).fit()
        converged = not any(
            "converg" in str(w.message).lower() for w in caught
        )
    return (float(res.params["x_depth"]), float(res.params["x_breadth"]),
            bool(converged))


def resample_frame(by_seed: dict, pick) -> "object":
    """Build one bootstrap resample, relabelling every draw as its own group.

    A seed drawn twice has to become two groups. If it keeps its label the
    random intercept merges the copies into one group with twice the rows,
    the resample stops carrying the extra weight it was supposed to, and
    the interval comes out too narrow. The bug is invisible in the output:
    the slope barely moves and only the width is wrong.

    So the invariant is asserted here rather than inferred from a CI: the
    number of distinct groups equals the number of draws, always.
    """
    import pandas as pd

    parts = []
    for j, s in enumerate(pick):
        part = by_seed[s].copy()
        part["seed"] = j
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def fit(df, log_depth: bool = False, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Point estimates plus bootstrap CIs, resampling whole seeds.

    Seeds are the unit of resampling because they are the unit of
    independence: the thirty cells inside one run share a set of weights,
    so resampling cells would treat one model as thirty observations.
    """
    import pandas as pd

    a, b, converged = _fit_once(df, log_depth)
    seeds = df.seed.dropna().unique()
    rng = np.random.default_rng(seed)
    by_seed = {s: df[df.seed == s] for s in seeds}
    boot_a, boot_b, n_failed = [], [], 0
    if len(seeds) >= 2:
        for _ in range(n_boot):
            pick = rng.choice(seeds, size=len(seeds), replace=True)
            sub = resample_frame(by_seed, pick)
            try:
                ba, bb, ok = _fit_once(sub, log_depth)
            except Exception:
                n_failed += 1
                continue
            if not ok:
                n_failed += 1
                continue
            boot_a.append(ba)
            boot_b.append(bb)
    lo, hi = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)
    return {
        "a_depth": a,
        "b_breadth": b,
        "a_ci": (float(np.percentile(boot_a, lo)), float(np.percentile(boot_a, hi)))
        if boot_a else None,
        "b_ci": (float(np.percentile(boot_b, lo)), float(np.percentile(boot_b, hi)))
        if boot_b else None,
        "n_boot_ok": len(boot_a),
        "n_boot_failed": n_failed,
        "converged": converged,
        "n_seeds": int(len(seeds)),
    }


def curve_summary(df) -> dict:
    """Per-axis quantiles of k_min, for the figure to read rather than derive.

    CLAUDE.md: plot scripts do not compute statistics. A median is a
    statistic, so F2 must not take one. It is taken here, where the module
    is tested, and written to json for the figure to read.

    CENSORED CELLS ARE EXCLUDED FROM THE QUANTILES AND COUNTED BESIDE THEM.
    A censored cell is recorded at 33, one past the measurable range, so
    leaving it in would drag a median toward a number that is not a k_min
    at all. Dropping it is not neutral either: the cells that never reach
    tau are the hard ones, so every quantile here is conditional on the
    cell being solvable within the sweep and the count says how often it
    was not.

    REFUSES A TASK THAT TWO SETUPS SHARE. F2 reads one entry per task, so
    an entry pooling two setups would be drawn as one curve. D-074.
    """
    mixed = mixed_setups(df)
    if mixed:
        raise ValueError(
            "one curve would pool different setups: "
            + "; ".join(f"{t}: {', '.join(a)}" for t, a in mixed.items()))
    out: dict = {}
    for task_key, g in df.groupby("task_key"):
        entry: dict = {
            "runs": int(g.run_id.nunique()),
            "seeds": int(g.seed.nunique()),
            "cells": int(len(g)),
            "axes": {},
        }
        for axis in ("depth", "breadth"):
            values = sorted(g[axis].unique().tolist())
            if len(values) < 2:
                continue  # not swept in this task, so it is not a curve
            rows = []
            for v in values:
                cell = g[g[axis] == v]
                ok = cell[~cell.censored]
                rows.append({
                    axis: int(v),
                    "median": float(ok.k_min.median()) if len(ok) else None,
                    "q25": float(ok.k_min.quantile(0.25)) if len(ok) else None,
                    "q75": float(ok.k_min.quantile(0.75)) if len(ok) else None,
                    "n": int(len(ok)),
                    "censored": int(cell.censored.sum()),
                })
            entry["axes"][axis] = rows
        out[str(task_key)] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description="M1 loop-count curves, H1")
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    ap.add_argument("--family", default=None)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None,
                    help="write per-axis quantiles here, for F2 to read")
    ap.add_argument("--no-fit", action="store_true",
                    help="collect and summarise only, skipping the bootstrap")
    ap.add_argument("--config", action="append", default=[],
                    help="keep only runs trained as this config file trains; "
                         "repeat for several")
    args = ap.parse_args(argv)

    if args.tau != TAU:
        print(f"refusing: tau is fixed at {TAU} by the preregistration, "
              f"Section 2. Changing it is a deviation and belongs in "
              f"decisions.md, not on a command line.")
        return 2

    df = collect(args.runs, args.family)
    if not len(df):
        print("no run has a stored k sweep with per-cell rows. "
              "M1 reads sweep_accuracy, which cli.py writes at the end of a "
              "run, so nothing here has finished with a sweep yet.")
        return 2

    for path in args.config:
        if not Path(path).is_file():
            print(f"no config at {path}")
            return 2
    if args.config:
        wanted = {config_setup(p): p for p in args.config}
        found = set(df.setup)
        unmatched = [p for s, p in wanted.items() if s not in found]
        if unmatched:
            print("no run was trained as " + ", ".join(unmatched)
                  + ". A config that matches nothing selects nothing, and a "
                  "curve drawn without it is missing an arm.")
            return 2
        df = df[df.setup.isin(set(wanted))]
        print(f"kept the runs of {len(wanted)} config(s): "
              + ", ".join(arms(df.run_id.unique())))

    out = Path(args.out) if args.out else Path("m1_kmin.parquet")
    df.to_parquet(out, index=False)
    print(f"wrote {out}  ({len(df)} cells from {df.run_id.nunique()} run(s))")

    if args.summary and mixed_setups(df):
        print("refusing to write a summary: these tasks are shared by more "
              "than one setup, and F2 would draw each as one curve.")
        for t, a in mixed_setups(df).items():
            print(f"    {t}   {', '.join(a)}")
        print("Name the registered arms with --config. D-074.")
        return 2

    if args.summary:
        Path(args.summary).write_text(
            json.dumps({"tau": TAU, "censored_at": CENSORED,
                        "tasks": curve_summary(df)}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.summary}")

    if args.no_fit:
        print("--no-fit: stopping before the bootstrap")
        return 0

    for fam, fg in df.groupby("family"):
        keys = sorted(fg.task_key.unique())
        if len(keys) < 2:
            continue
        print(f"\n=== family {fam}: {len(keys)} DIFFERENT TASK DEFINITIONS ===")
        print("  Fitted separately and never pooled. A run trained before a "
              "task changed and one trained after are different experiments, "
              "so k_min is not comparable across them. D-026 invalidated the "
              "family B runs before 2026-09-03 and D-028 moved family B "
              "breadths from 6 to 9 up to 12 to 16. Which group carries H1 "
              "is a judgement for decisions.md, not for this module.")
        for kk in keys:
            sub = fg[fg.task_key == kk]
            print(f"    {kk}   {sub.run_id.nunique()} run(s), "
                  f"{sub.seed.nunique()} seed(s)")

    for (fam, task_key, setup), g in df.groupby(["family", "task_key", "setup"]):
        print(f"\n=== family {fam}   task {task_key}   setup {setup} ===")
        print(f"  arms {', '.join(arms(g.run_id.unique()))}")
        print(f"  seeds {g.seed.nunique()}   d_model {sorted(g.d_model.unique())}   "
              f"cells {len(g)}")
        print(f"  censored at {CENSORED}: {int(g.censored.sum())} of {len(g)}")
        piv = g.pivot_table(index="depth", columns="breadth", values="k_min",
                            aggfunc="median")
        print("\n  median k_min by depth and breadth")
        print("  " + piv.to_string().replace("\n", "\n  "))

        if g.seed.nunique() < 2:
            print("\n  one seed: no bootstrap over seeds, so no interval.")
            continue
        for label, logd in (("linear in depth", False),
                            ("saturating, log depth", True)):
            try:
                r = fit(g, log_depth=logd)
            except Exception as e:
                print(f"\n  {label}: fit failed, {e}")
                continue
            print(f"\n  {label}   ({r['n_seeds']} seeds, "
                  f"{r['n_boot_ok']} bootstrap fits, "
                  f"{r['n_boot_failed']} discarded for non convergence)")
            if not r["converged"]:
                print("    WARNING the point estimate's own fit did not "
                      "converge. Treat these coefficients as unreliable.")
            if r["n_boot_ok"] and r["n_boot_failed"] > r["n_boot_ok"]:
                print("    WARNING more bootstrap resamples failed than "
                      "succeeded, so this interval rests on a minority.")
            for name, key, ci in (("a, depth", "a_depth", r["a_ci"]),
                                  ("b, breadth", "b_breadth", r["b_ci"])):
                s = f"{r[key]:+.4f}"
                if ci:
                    s += f"   CI [{ci[0]:+.4f}, {ci[1]:+.4f}]"
                print(f"    {name:<12} {s}")

    print(f"\nNo verdict on H1 is printed, and that is deliberate. The "
          f"preregistration corrects across a registered family of "
          f"{REGISTERED_FAMILY_SIZE} tests; two of them are here and the rest "
          f"need H2 and H3. Correcting two tests as if they were the family "
          f"would be a weaker correction than registered and would make "
          f"support easier to reach.")
    print(f"\nResolution limit: k_min resolves only to the swept grid "
          f"{{1,2,3,4,6,8,12,16,24,32}}, so a true k_min of 5 reads as 6. "
          f"A re-sweep at every integer k is inference over stored "
          f"checkpoints and needs no training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
