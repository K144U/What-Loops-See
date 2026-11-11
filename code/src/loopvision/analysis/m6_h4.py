"""H4 from the M6 attractor rates: the sweep, and the confound that eats it.

`instruments/m6_attractor` computes one wrong-attractor rate per run. This
aggregates them over the four registered jitter widths, and does the one
thing that decides whether the sweep says anything: it checks whether the
rate carries information beyond whether the model trained.

**It does not.** The raw sweep looks like strong support, 0.5684 at zero
jitter against 0.06 to 0.11 at every jittered width. But the rate tracks
final accuracy at r = -0.87, the zero-jitter arm has no run above 0.95, and
among the runs that did train the three jittered widths sit within 0.026 of
each other with no ordering. The apparent effect is the training outcome
wearing the metric's clothes.

STATE.md named this trap before the instrument existed: the training split
is a run-level dichotomy while the registered metric is an item-level
property, so the dichotomy must not be reported as H4. What this module adds
is that the item-level metric, now that it can be computed, carries almost
nothing the dichotomy did not already say.

Exploratory, and corrected separately from the registered family of Section
1, per preregistration Section 5. H4 stays labelled exploratory whatever it
eventually says.

    python -m loopvision.analysis.m6_h4 --rates h4.json
    python -m loopvision.analysis.m6_h4 --rates h4.json --summary h4_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

#: Above this a run has clearly trained, so the arms can be compared without
#: the comparison being decided by whether training succeeded. Not tuned: it
#: is the same 0.95 the escape threshold uses everywhere else in the project.
SOLVED = 0.95

#: The width axis only. famB_curve111_d256 and friends vary WIDTH, which is a
#: different axis and no part of Section 5.
H4_ARM = re.compile(r"famB_curve111(_jit\d{3})?_s\d+")


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Correlation, or None when it is not defined.

    Returned rather than printed so the figure can read it from the summary
    instead of computing a statistic of its own, which the standing rule in
    CLAUDE.md forbids of plot scripts.
    """
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs)
    dy = sum((b - my) ** 2 for b in ys)
    if dx <= 0 or dy <= 0:
        return None
    return num / (dx * dy) ** 0.5


def load(rates_path: Path, runs_root: Path) -> list[dict]:
    """M6 rates joined to each run's final accuracy, jitter axis only."""
    rows = json.loads(rates_path.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        if not H4_ARM.fullmatch(r["run"]):
            continue
        s = runs_root / r["run"] / "summary.json"
        acc = None
        if s.is_file():
            acc = json.loads(s.read_text(encoding="utf-8")).get("final_accuracy")
        out.append({
            "run": r["run"],
            "sigma": r["jitter_sigma"],
            "rate_by_label": r["rate_by_label"],
            "rate_by_prediction": r["rate_by_prediction"],
            "readings_agree": r["readings_agree"],
            "accuracy": acc,
        })
    return out


def summarise(rows: list[dict]) -> dict:
    """Per arm, then again over the runs that clearly trained."""
    def arm_table(subset: list[dict]) -> list[dict]:
        arms: dict = {}
        for r in subset:
            arms.setdefault(r["sigma"], []).append(r)
        table = []
        for sigma in sorted(arms):
            v = arms[sigma]
            accs = [r["accuracy"] for r in v if r["accuracy"] is not None]
            table.append({
                "sigma": sigma,
                "seeds": len(v),
                "rate_mean": sum(r["rate_by_label"] for r in v) / len(v),
                "rate_min": min(r["rate_by_label"] for r in v),
                "rate_max": max(r["rate_by_label"] for r in v),
                "accuracy_mean": sum(accs) / len(accs) if accs else None,
            })
        return table

    scored = [r for r in rows if r["accuracy"] is not None]
    solved = [r for r in scored if r["accuracy"] > SOLVED]

    # Which arms can even enter the matched comparison. An arm with no run
    # above the threshold is not a low score, it is an absent comparison,
    # and saying so is the whole finding.
    by_sigma: dict = {}
    for r in scored:
        by_sigma.setdefault(r["sigma"], []).append(r)
    coverage = [{
        "sigma": s,
        "solved": sum(1 for r in v if r["accuracy"] > SOLVED),
        "seeds": len(v),
    } for s, v in sorted(by_sigma.items())]

    return {
        "solved_threshold": SOLVED,
        "all_arms": arm_table(rows),
        "matched_arms": arm_table(solved),
        "coverage": coverage,
        "pearson_rate_vs_accuracy": pearson(
            [r["rate_by_label"] for r in scored],
            [r["accuracy"] for r in scored]),
        "n_runs": len(rows),
        "n_scored": len(scored),
        "n_solved": len(solved),
        "readings_disagree": sum(1 for r in rows if not r["readings_agree"]),
        "runs": rows,
    }


def report(s: dict) -> int:
    print("H4, wrong-attractor landings by jitter width. Exploratory.\n")
    print("  %-7s %-6s %-9s %-9s %-9s %s" %
          ("sigma", "seeds", "mean", "min", "max", "mean acc"))
    for a in s["all_arms"]:
        print("  %-7s %-6d %-9.4f %-9.4f %-9.4f %s" %
              (a["sigma"], a["seeds"], a["rate_mean"], a["rate_min"],
               a["rate_max"],
               "n/a" if a["accuracy_mean"] is None else "%.4f" % a["accuracy_mean"]))

    r = s["pearson_rate_vs_accuracy"]
    print(f"\n  The rate tracks final accuracy at pearson r = {r:.4f} "
          f"over {s['n_scored']} runs.")
    print("  A metric that moves with accuracy is not yet evidence about "
          "attractors,")
    print("  so the arms have to be compared where training succeeded.\n")

    print(f"  Runs above {s['solved_threshold']:.2f}, by arm:")
    for c in s["coverage"]:
        note = "   <-- no run trained, the comparison is absent" if c["solved"] == 0 else ""
        print("    sigma %-6s %d of %d%s" % (c["sigma"], c["solved"], c["seeds"], note))

    print("\n  Matched on training success:")
    print("  %-7s %-6s %-9s" % ("sigma", "seeds", "mean rate"))
    for a in s["matched_arms"]:
        print("  %-7s %-6d %-9.4f" % (a["sigma"], a["seeds"], a["rate_mean"]))

    spread = ([a["rate_mean"] for a in s["matched_arms"]] or [0.0])
    print(f"\n  Spread across the matched arms: {max(spread) - min(spread):.4f}.")
    print("  H4 CANNOT BE RESOLVED from this sweep. The zero-jitter arm has no")
    print("  run that trained, so the comparison the hypothesis asks for does")
    print("  not exist here, and among the arms that did train the width makes")
    print("  no orderly difference. Reported as the no-effect paragraph of")
    print("  preregistration Section 5, with the qualification that what is")
    print("  absent is a comparison rather than an effect.")

    if s["readings_disagree"]:
        print(f"\n  {s['readings_disagree']} of {s['n_runs']} runs score "
              f"differently under the two readings of 'modal answer', D-053.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="H4 from M6 attractor rates")
    ap.add_argument("--rates", required=True, type=Path,
                    help="json written by m6_attractor --all --out")
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    ap.add_argument("--summary", default=None, type=Path)
    args = ap.parse_args(argv)

    rows = load(args.rates, args.runs)
    if not rows:
        print("no H4 sweep runs in that file. Nothing is computed and "
              "nothing is claimed.")
        return 2
    s = summarise(rows)
    rc = report(s)
    if args.summary:
        args.summary.write_text(json.dumps(s, indent=2), encoding="utf-8")
        print(f"\nwrote {args.summary}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
