"""Per iteration shape of the loop's trajectory, aggregated for F5.

`instruments/m4_geometry` writes one row per item, per iteration, per metric.
This reduces that to a per iteration median and interquartile range so the
figure can read numbers instead of taking a statistic of its own, which the
standing rule in CLAUDE.md forbids of plot scripts.

**WHICH METRICS LEAD, AND WHY IT MATTERS HERE.** `RMSNorm` sits before the
head, so the raw state norm is invisible to the model: two states differing
only in magnitude decode identically. STATE.md carries this as the standing
trap in F5. The scale free quantities therefore lead and the norm follows,
labelled as something the head cannot see. On `famB_curve111_s0` the point is
stark: the unit step falls from 1.42 to 0.019 while the norm sits between 241
and 254 the whole way. A figure that led with the norm would show a flat line
and suggest nothing happens, when in fact the direction is converging
geometrically.

    python -m loopvision.analysis.m4_shape --run-id famB_curve111_s0
    python -m loopvision.analysis.m4_shape --all --summary shape.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Scale free, and the ones the halting argument rests on. Order is the order
#: F5 draws them in, so the leading metric is first here rather than chosen
#: in the plot.
DIRECTIONAL = ("unit_step_norm", "cos_state")

#: Reported, never led with. See the module docstring.
MAGNITUDE = ("norm", "norm_growth")


def summarise_run(path: Path) -> dict | None:
    """Median and IQR per iteration per metric, for one stored geometry file."""
    import pandas as pd

    if not path.is_file():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None

    out: dict = {"run": str(df["run_id"].iloc[0]), "metrics": {}}
    for name in DIRECTIONAL + MAGNITUDE:
        sub = df[df["metric_name"] == name]
        if sub.empty:
            continue
        g = sub.groupby("iteration")["value"]
        med = g.median()
        q1 = g.quantile(0.25)
        q3 = g.quantile(0.75)
        out["metrics"][name] = {
            "iteration": [int(i) for i in med.index],
            # NaN is not JSON, and it is meaningful here: cos_step is
            # undefined at i = 1 and ratio is gated. None keeps the gap
            # visible instead of letting a reader read across it.
            "median": [None if v != v else float(v) for v in med.values],
            "q1": [None if v != v else float(v) for v in q1.values],
            "q3": [None if v != v else float(v) for v in q3.values],
        }
    out["items"] = int(df["item"].nunique())
    out["k"] = int(df["k"].iloc[0])
    return out


def summarise(paths: list[Path]) -> dict:
    runs = [r for r in (summarise_run(p) for p in paths) if r]
    return {
        "runs": runs,
        "directional": list(DIRECTIONAL),
        "magnitude": list(MAGNITUDE),
        "n_runs": len(runs),
    }


def report(s: dict) -> int:
    if not s["runs"]:
        print("no stored geometry found. Nothing is computed and nothing "
              "is claimed.")
        return 2
    for r in s["runs"]:
        print(f"  {r['run']:<34} {r['items']} items, k={r['k']}")
        for name in s["directional"] + s["magnitude"]:
            m = r["metrics"].get(name)
            if not m:
                continue
            vals = [v for v in m["median"] if v is not None]
            if not vals:
                continue
            lead = "  " if name in s["directional"] else "  (inert) "
            print(f"    {lead}{name:<16} {vals[0]:>10.4f} -> {vals[-1]:>10.4f}")
    print("\n  The magnitude rows are marked inert because RMSNorm sits before")
    print("  the head: the model cannot see the raw norm, so a flat norm is")
    print("  not evidence that the trajectory has stopped moving.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M4 trajectory shape for F5")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    ap.add_argument("--summary", default=None, type=Path)
    args = ap.parse_args(argv)

    if args.run_id:
        paths = [args.runs / args.run_id / "m4_geometry.parquet"]
    elif args.all:
        paths = sorted(args.runs.glob("*/m4_geometry.parquet"))
    else:
        print("give --run-id or --all")
        return 2

    s = summarise(paths)
    rc = report(s)
    if args.summary and s["runs"]:
        args.summary.write_text(json.dumps(s, indent=2), encoding="utf-8")
        print(f"\nwrote {args.summary}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
