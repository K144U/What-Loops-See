"""Untied against looped and against feedforward. The gate G2 diagnosis.

**This does not reopen gate G2.** G2 failed on 2026-09-07, is recorded as
failed in the ledger, in `findings.md` and in STATE.md, and nothing here
changes that. D-038 says so and this module says so. It exists because the
gate lost to an arm that changed two properties at once, so it establishes
that looping lost without establishing what it lost to.

    looped       tied core, reused k times   1.000x parameters, loop yes
    untied       k distinct cores            1.938x parameters, loop yes
    feedforward  k distinct blocks           1.886x parameters, loop NO

The untied arm sits between the other two and separates them:

    untied vs feedforward   parameters and compute held, loop varies
    untied vs looped        compute and depth held, weight tying varies

**The three outcomes, quoted from D-038 so they cannot drift:**

- *untied also beats looped:* the win came from untying the weights, not
  from abandoning the loop. That leaves the looped architecture's serial
  story intact, at a cost in parameters.
- *untied does not beat looped, but feedforward still does:* the win came
  from removing the loop structure itself, which is the harder result to
  write around.
- *untied matches feedforward:* the loop structure is inert at this
  operating point and the comparison reduces to a parameter count.

**On thresholds.** D-038 registered the three outcomes but no margin, and
this is a diagnostic rather than a gate, so no pass/fail threshold is
invented here. "Beats" means a bootstrap CI over seeds excluding zero at
the pre-registration's alpha of 0.05. That is stated rather than chosen
after seeing the numbers.

**On incomplete runs.** This refuses to compare a run that has not written
DONE. A mid-training arm read against a finished one is not a comparison,
and the untied arm passed the looped model's final accuracy at 48 percent
of budget, which is exactly the number that would be tempting to quote.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from loopvision.analysis.gate_g2 import final_accuracy

ALPHA = 0.05
RESAMPLES = 10000

LOOPED = [f"famB_curve111_s{s}" for s in range(8)]
FFWD = [f"famB_g2_ffwd_s{s}" for s in (0, 1)]
UNTIED = [f"famB_g2_untied_s{s}" for s in (0, 1)]


def is_done(runs_root: Path, run_id: str) -> bool:
    return (runs_root / run_id / "DONE").exists()


def per_depth(runs_root: Path, run_id: str) -> pd.Series | None:
    """Final accuracy by depth cell, the axis on which the gate failed."""
    f = runs_root / run_id / "metrics.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    fin = d[(d.metric_name == "final_accuracy") & (d.depth > 0)]
    if not len(fin):
        fin = d[(d.metric_name == "accuracy") & (d.split == "iid_val") & (d.depth > 0)]
        if not len(fin):
            return None
        fin = fin[fin.step == fin.step.max()]
    return fin.groupby("depth").value.mean()


def arm_values(runs_root: Path, run_ids: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Per-seed mean final accuracy, plus which runs were used and skipped."""
    vals, used, skipped = [], [], []
    for r in run_ids:
        if not is_done(runs_root, r):
            skipped.append(r)
            continue
        v = final_accuracy(runs_root, r)
        if v is None:
            skipped.append(r)
            continue
        vals.append(v)
        used.append(r)
    return np.array(vals), used, skipped


def contrast(a: np.ndarray, b: np.ndarray, rng) -> dict:
    obs = float(a.mean() - b.mean())
    draws = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        draws[i] = (
            rng.choice(a, size=len(a), replace=True).mean()
            - rng.choice(b, size=len(b), replace=True).mean()
        )
    lo, hi = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {
        "diff": obs,
        "lo": float(lo),
        "hi": float(hi),
        "beats": bool(lo > 0),
        "loses": bool(hi < 0),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    root = Path(a.runs_root)

    arms = {}
    for name, ids in (("looped", LOOPED), ("feedforward", FFWD), ("untied", UNTIED)):
        vals, used, skipped = arm_values(root, ids)
        arms[name] = {"vals": vals, "used": used, "skipped": skipped}
        print(f"{name:12} n={len(used)}  mean={vals.mean():.4f}" if len(vals)
              else f"{name:12} n=0")
        if skipped:
            print(f"             skipped, not DONE: {', '.join(skipped)}")

    if len(arms["untied"]["vals"]) == 0:
        print()
        print(
            "CANNOT COMPARE. No untied run has written DONE. This module "
            "deliberately refuses a mid-training arm: at 48 percent of budget "
            "the untied arm already exceeded the looped model's final "
            "accuracy, and quoting that against a finished run would not be a "
            "comparison."
        )
        return 2
    if len(arms["looped"]["vals"]) < 2 or len(arms["feedforward"]["vals"]) < 2:
        print()
        print("CANNOT COMPARE. Both reference arms need at least two finished seeds.")
        return 2

    print()
    print("Per depth, mean over finished seeds")
    table = {}
    for name, ids in (("looped", LOOPED), ("feedforward", FFWD), ("untied", UNTIED)):
        cols = [per_depth(root, r) for r in arms[name]["used"]]
        cols = [c for c in cols if c is not None]
        if cols:
            table[name] = pd.concat(cols, axis=1).mean(axis=1)
    if table:
        df = pd.DataFrame(table)
        df["untied - looped"] = df["untied"] - df["looped"]
        df["untied - ffwd"] = df["untied"] - df["feedforward"]
        print(df.round(4).to_string())

    rng = np.random.default_rng(a.seed)
    vs_looped = contrast(arms["untied"]["vals"], arms["looped"]["vals"], rng)
    vs_ffwd = contrast(arms["untied"]["vals"], arms["feedforward"]["vals"], rng)

    print()
    print("Contrasts, bootstrap over seeds, alpha 0.05")
    for label, c in (("untied - looped", vs_looped), ("untied - feedforward", vs_ffwd)):
        verdict = "beats" if c["beats"] else ("loses to" if c["loses"] else "indistinguishable from")
        print(
            f"  {label:22} {c['diff']:+.4f}  CI [{c['lo']:+.4f}, {c['hi']:+.4f}]  "
            f"untied {verdict} the other"
        )

    print()
    print("Reading, per D-038")
    if vs_looped["beats"] and not vs_ffwd["loses"]:
        print(
            "  UNTIED ALSO BEATS LOOPED. The win came from untying the weights, "
            "not from abandoning the loop. The looped architecture's serial "
            "story survives, at a cost in parameters. Note the untied arm "
            "carries 1.938x the looped parameters, so this does not separate "
            "untying from simply having more parameters; the matched-parameter "
            "question is a different run."
        )
    elif not vs_looped["beats"] and vs_ffwd["loses"]:
        print(
            "  UNTIED DOES NOT BEAT LOOPED WHILE FEEDFORWARD DOES. The win came "
            "from removing the loop structure itself. This is the harder result "
            "to write around and it is the one D-038 named as such."
        )
    elif not vs_ffwd["beats"] and not vs_ffwd["loses"]:
        print(
            "  UNTIED MATCHES FEEDFORWARD. The loop structure is inert at this "
            "operating point and the comparison reduces to a parameter count."
        )
    else:
        print(
            "  MIXED. The result does not fall cleanly into any of D-038's three "
            "outcomes. Report the contrasts as they are rather than forcing one."
        )
    print()
    print("Gate G2 remains FAILED. This is diagnosis of a reported failure, not an appeal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
