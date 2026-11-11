"""Verdicts for D-066 to D-069, the revision experiments registered on 2026-10-11.

    python -m loopvision.analysis.revision_verdicts

Written and committed before any of their runs finished, so nothing here was
shaped by a result. Every rate comes from `escape_rates` (escaped means final
accuracy at or above 0.95) and every departure from chance from
`escape_timing`, exactly as D-061 and D-062 read them.

D-066, the learning-rate by weight-decay grid.
  G1  pooled over the six cells, position escapes more often than appearance:
      Cochran-Mantel-Haenszel, continuity corrected, two sided, with the
      common odds ratio above 1.
  G2  no cell shows appearance escaping significantly more often than
      position, by Fisher two sided per cell, Holm across the six cells.
  Holm across G1 and G2: G1's p is the CMH p; G2's is its smallest
  Holm-adjusted reversal p, or 1 when no cell leans the reversed way. With
  two tests the smaller p is held to 0.025 and the larger to 0.05, step down.

D-067  T1  position escapes more often than appearance at pixel scale 2.

D-068  T2  E2b escapes more often than E2a, new seeds alone, Fisher.
       T3  of new E2a runs that left chance, at least half finish between
           0.24 and 0.26; undetermined below four that left chance.
  T3 is a threshold rule with no p value, so "Holm across T2 and T3" cannot
  rank it. Read conservatively, as fixed here before any result: T2 is held
  to 0.025, the stricter of the two Holm thresholds.

D-069  K1  same-task donor against wrong-group donor; K2 other-substrate
       against wrong-group; Fisher, Holm across the two.

Pending until every seed of the arms a test reads has finished. Exit 1 if
anything failed, 2 while anything is pending, 3 if anything is undetermined,
0 when all held.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.contingency_tables import StratifiedTable

from loopvision.analysis import escape_rates as ER
from loopvision.analysis import escape_timing as T

ALPHA = 0.05
TRAP_BAND = (0.24, 0.26)
TRAP_MIN_LEFT = 4

#: D-066's cells, in the config names scripts/make_revision_configs.py writes.
C1_CELLS = [f"{lr}_{wd}" for lr in ("lr1e4", "lr3e4", "lr1e3") for wd in ("wd005", "wd05")]
SEEDS = {"c1": 5, "c2": 10, "c3": 10, "c4": 10}


def stem(run_id: str) -> str:
    """The stem `escape_rates.stem_rates` keys by."""
    return run_id.rstrip("0123456789").rstrip("_s") or run_id


def wanted_stems() -> set[str]:
    s = {f"c1_{sub}_{cell}" for sub in ("pos", "app") for cell in C1_CELLS}
    s |= {"c2_pos_x2", "c2_app_x2", "c3_e2a_cue", "c3_e2b_decoy",
          "c4_curr", "c4_currctrl", "c4_s3donor"}
    return s


def scan(runs_root: Path) -> list[dict]:
    """Every revision run, with outcome, final accuracy and departure."""
    wanted = wanted_stems()
    rows = []
    for r in ER.scan(runs_root):
        if stem(r["run_id"]) not in wanted:
            continue
        row = {"run_id": r["run_id"], "stem": stem(r["run_id"]), "done": r["done"],
               "outcome": r["outcome"], "final": None, "left": None}
        d = runs_root / r["run_id"]
        if r["done"]:
            summ = d / "summary.json"
            if summ.is_file():
                row["final"] = float(pd.read_json(summ, typ="series").get("final_accuracy"))
            mp = d / "metrics.parquet"
            if mp.is_file():
                cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
                steps, values = T.train_loss(pd.read_parquet(mp))
                row["left"] = T.leave_step(steps, values, T.group_order(cfg))
        rows.append(row)
    return rows


def tally(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {s: {"finished": 0, "escaped": 0, "unfinished": 0, "rows": []}
                            for s in wanted_stems()}
    for r in rows:
        e = out[r["stem"]]
        if not r["done"]:
            e["unfinished"] += 1
            continue
        e["finished"] += 1
        e["escaped"] += r["outcome"] == "escaped"
        e["rows"].append(r)
    return out


def complete(t: dict, stems: list[str], n: int) -> bool:
    return all(t[s]["unfinished"] == 0 and t[s]["finished"] >= n for s in stems)


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p values, in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def cmh(tables: list[list[list[int]]]) -> tuple[float, float]:
    """Cochran-Mantel-Haenszel p, continuity corrected, and the pooled odds ratio.

    One 2x2 table per stratum, rows the two arms, columns escaped and not.
    statsmodels' implementation; tests/test_revision_verdicts.py checks it
    against R's documented mantelhaen.test example.
    """
    st = StratifiedTable(np.array(tables, dtype=float).transpose(1, 2, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        # Every arm-two run escaping in every cell makes the pooled odds ratio
        # infinite, which correctly reads as above 1.
        return float(st.test_null_odds(correction=True).pvalue), float(st.oddsratio_pooled)


def _fisher(t: dict, high: str, low: str) -> tuple[float, bool, str]:
    a, na = t[high]["escaped"], t[high]["finished"]
    b, nb = t[low]["escaped"], t[low]["finished"]
    p = ER.fisher_two_sided(a, na - a, b, nb - b)
    return p, a / na > b / nb, f"{high} {a} of {na}, {low} {b} of {nb}, Fisher p = {p:.3f}"


def d066(t: dict) -> dict[str, tuple[str, str]]:
    stems = [f"c1_{sub}_{c}" for sub in ("pos", "app") for c in C1_CELLS]
    if not complete(t, stems, SEEDS["c1"]):
        fin = sum(t[s]["finished"] for s in stems)
        return {k: ("pending", f"{fin} of {len(stems) * SEEDS['c1']} finished") for k in ("G1", "G2")}
    tables, rev_p = [], []
    for c in C1_CELLS:
        pos, app = t[f"c1_pos_{c}"], t[f"c1_app_{c}"]
        a, b = pos["escaped"], pos["finished"] - pos["escaped"]
        c_, d = app["escaped"], app["finished"] - app["escaped"]
        tables.append([[a, b], [c_, d]])
        p = ER.fisher_two_sided(a, b, c_, d)
        rev_p.append(p if c_ / (c_ + d) > a / (a + b) else 1.0)
    p1, orr = cmh(tables)
    p2 = min(holm(rev_p))
    adj1, adj2 = holm([p1, p2])
    g1 = "held" if (adj1 < ALPHA and orr > 1) else "failed"
    g2 = "failed" if adj2 < ALPHA else "held"
    pos_tot = sum(t[f"c1_pos_{c}"]["escaped"] for c in C1_CELLS)
    app_tot = sum(t[f"c1_app_{c}"]["escaped"] for c in C1_CELLS)
    n = len(C1_CELLS) * SEEDS["c1"]
    return {
        "G1": (g1, f"position {pos_tot} of {n}, appearance {app_tot} of {n}, CMH p = {p1:.4f}, "
                   f"pooled odds ratio {orr:.2f}, Holm-adjusted p = {adj1:.4f}"),
        "G2": (g2, f"smallest Holm-adjusted reversal p across cells = {p2:.4f}, "
                   f"adjusted with G1 = {adj2:.4f}"),
    }


def d067(t: dict) -> dict[str, tuple[str, str]]:
    if not complete(t, ["c2_pos_x2", "c2_app_x2"], SEEDS["c2"]):
        return {"T1": ("pending", f"position {t['c2_pos_x2']['finished']}, appearance "
                                  f"{t['c2_app_x2']['finished']} of {SEEDS['c2']} finished")}
    p, right, what = _fisher(t, "c2_pos_x2", "c2_app_x2")
    return {"T1": ("held" if (p < ALPHA and right) else "failed", what)}


def d068(t: dict) -> dict[str, tuple[str, str]]:
    arms = ["c3_e2a_cue", "c3_e2b_decoy"]
    if not complete(t, arms, SEEDS["c3"]):
        msg = ", ".join(f"{a} {t[a]['finished']} of {SEEDS['c3']}" for a in arms)
        return {"T2": ("pending", msg), "T3": ("pending", msg)}
    p, right, what = _fisher(t, "c3_e2b_decoy", "c3_e2a_cue")
    t2 = "held" if (p < ALPHA / 2 and right) else "failed"
    left = [r for r in t["c3_e2a_cue"]["rows"] if r["left"] is not None]
    in_band = sum(1 for r in left if r["final"] is not None
                  and TRAP_BAND[0] <= r["final"] <= TRAP_BAND[1])
    if len(left) < TRAP_MIN_LEFT:
        t3 = ("undetermined", f"only {len(left)} E2a runs left chance, fewer than {TRAP_MIN_LEFT}")
    else:
        t3 = ("held" if 2 * in_band >= len(left) else "failed",
              f"{in_band} of {len(left)} E2a runs that left chance finished between "
              f"{TRAP_BAND[0]} and {TRAP_BAND[1]}")
    return {"T2": (t2, what + ", held to 0.025"), "T3": t3}


def d069(t: dict) -> dict[str, tuple[str, str]]:
    arms = ["c4_curr", "c4_currctrl", "c4_s3donor"]
    if not complete(t, arms, SEEDS["c4"]):
        msg = ", ".join(f"{a} {t[a]['finished']} of {SEEDS['c4']}" for a in arms)
        return {"K1": ("pending", msg), "K2": ("pending", msg)}
    p1, r1, w1 = _fisher(t, "c4_curr", "c4_s3donor")
    p2, r2, w2 = _fisher(t, "c4_currctrl", "c4_s3donor")
    a1, a2 = holm([p1, p2])
    return {"K1": ("held" if (a1 < ALPHA and r1) else "failed", f"{w1}, Holm-adjusted {a1:.3f}"),
            "K2": ("held" if (a2 < ALPHA and r2) else "failed", f"{w2}, Holm-adjusted {a2:.3f}")}


def registered(rows: list[dict]) -> dict[str, tuple[str, str]]:
    t = tally(rows)
    return {**d066(t), **d067(t), **d068(t), **d069(t)}


EXIT = {"failed": 1, "pending": 2, "undetermined": 3}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D-066 to D-069, the revision experiments")
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
