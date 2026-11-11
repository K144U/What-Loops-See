"""E1 and E2 of docs/why-experiments.md, the causal test of H-V. Registered as D-062.

H-V says D4 position learns reliably because its rendering makes a quotient
of the group visible, which gives SGD a staircase. E1 takes the visible
quotient away from D4 position two ways; E2 gives one to D4 identity, beside
a decoy that is equally visible and does not compose.

    python -m loopvision.analysis.why_e1e2

Per arm it reads two things from finished runs: how many escaped
(`escape_rates`), and of the runs that left chance, how many climbed an ln 4
plateau on the way (`escape_timing`). Six registered contrasts follow, each a
two sided Fisher test that must reach p below 0.05 IN THE PREDICTED DIRECTION
to hold. The wrong direction, or a quiet p, fails.

  V1, V2  E1a (square glyph) escapes less often, and climbs less often, than
          the sealed D4 position arm.
  V3, V4  the same for E1b (scrambled tiles).
  V5, V6  E2a (composing cue) escapes more often, and climbs more often,
          than E2b (decoy).

A climbing contrast with no run that left chance on one side cannot be read
and is `undetermined`, never held. Exit 1 if anything failed, 2 while
anything is pending, 3 if anything is undetermined, 0 when all held.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from loopvision.analysis import escape_rates as ER
from loopvision.analysis import escape_timing as T

ARMS = {
    "E1a": "why_e1a_square",
    "E1b": "why_e1b_scrambled",
    "E2a": "why_e2a_cue",
    "E2b": "why_e2b_decoy",
    "sealed": "famA_d2_d4only",
}
#: key: (measure, arm expected LOWER, arm expected HIGHER)
CONTRASTS = {
    "V1": ("escaped", "E1a", "sealed"),
    "V2": ("climbed", "E1a", "sealed"),
    "V3": ("escaped", "E1b", "sealed"),
    "V4": ("climbed", "E1b", "sealed"),
    "V5": ("escaped", "E2b", "E2a"),
    "V6": ("climbed", "E2b", "E2a"),
}
SEEDS = {"E1a": 20, "E1b": 20, "E2a": 20, "E2b": 20, "sealed": 10}
ALPHA = 0.05
LEVEL = "ln4"


def stem(run_id: str) -> str:
    """The stem `escape_rates.stem_rates` keys by."""
    return run_id.rstrip("0123456789").rstrip("_s") or run_id


def scan(runs_root: Path) -> list[dict]:
    """Every run of an arm above, with outcome, departure and plateaus."""
    wanted = set(ARMS.values())
    rows = []
    for r in ER.scan(runs_root):
        if stem(r["run_id"]) not in wanted:
            continue
        row = {"run_id": r["run_id"], "stem": stem(r["run_id"]), "done": r["done"],
               "outcome": r["outcome"], "left": None, "climbed": False}
        mp = runs_root / r["run_id"] / "metrics.parquet"
        if r["done"] and mp.is_file():
            cfg = yaml.safe_load((runs_root / r["run_id"] / "config.yaml").read_text(encoding="utf-8"))
            order = T.group_order(cfg)
            steps, values = T.train_loss(pd.read_parquet(mp))
            solved = T.solve_step(steps, values, order)
            row["left"] = T.leave_step(steps, values, order)
            row["climbed"] = any(p["level"] == LEVEL
                                 for p in T.plateaus(steps, values, order, before=solved))
        rows.append(row)
    return rows


def tally(rows: list[dict]) -> dict:
    out = {k: {"finished": 0, "escaped": 0, "left": 0, "climbed": 0, "unfinished": 0}
           for k in ARMS}
    by_stem = {v: k for k, v in ARMS.items()}
    for r in rows:
        e = out[by_stem[r["stem"]]]
        if not r["done"]:
            e["unfinished"] += 1
            continue
        e["finished"] += 1
        e["escaped"] += r["outcome"] == "escaped"
        if r["left"] is not None:
            e["left"] += 1
            e["climbed"] += bool(r["climbed"])
    return out


def _counts(e: dict, measure: str) -> tuple[int, int]:
    """(successes, trials) for a measure. Climbing is out of runs that left."""
    if measure == "escaped":
        return e["escaped"], e["finished"]
    return e["climbed"], e["left"]


def registered(rows: list[dict]) -> dict[str, tuple[str, str]]:
    t = tally(rows)
    out = {}
    for key, (measure, low, high) in CONTRASTS.items():
        lo, hi = t[low], t[high]
        if any(t[a]["unfinished"] or t[a]["finished"] < SEEDS[a] for a in (low, high)):
            out[key] = ("pending", f"{low} {lo['finished']} of {SEEDS[low]} finished, "
                                   f"{high} {hi['finished']} of {SEEDS[high]}")
            continue
        (a, na), (b, nb) = _counts(lo, measure), _counts(hi, measure)
        what = f"{measure}: {low} {a} of {na}, {high} {b} of {nb}"
        if na == 0 or nb == 0:
            out[key] = ("undetermined", what + ", one side has no run that left chance")
            continue
        p = ER.fisher_two_sided(a, na - a, b, nb - b)
        right_way = a / na < b / nb
        out[key] = ("held" if (p < ALPHA and right_way) else "failed", f"{what}, Fisher p = {p:.3f}")
    return out


EXIT = {"failed": 1, "pending": 2, "undetermined": 3}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D-062, E1 and E2")
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    args = ap.parse_args(argv)
    if not args.runs.is_dir():
        print(f"no runs directory at {args.runs}")
        return 2
    verdicts = registered(scan(args.runs))
    for k, (v, detail) in verdicts.items():
        print(f"{k}  {v:13s} {detail}")
    states = {v for v, _ in verdicts.values()}
    for worst in ("failed", "pending", "undetermined"):
        if worst in states:
            return EXIT[worst]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
