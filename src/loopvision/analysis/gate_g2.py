"""Gate G2: do loops buy anything a matched-compute baseline cannot?

**Moved from family A depth 3 by docs/decisions.md D-030.** There the gate
compared a looped model at 0.0225 against a feedforward one at 0.0225,
both on the 1/48 chance floor, so it could neither pass nor fail. A gate
whose every outcome is identical is not a gate.

Here it runs on family B at a 1/1/1 core across depths 1 to 6, where a
loop-count curve exists and the comparison can come out either way.

Matched compute is the whole point and is defined structurally: a looped
model at k passes executes `prelude + k*core + coda` blocks, so the
feedforward baseline is built with that same block count at k=4, the
operating point where the looped model clears tau at every depth.

Two checks, both must pass:

  G2.1  the looped model beats matched-compute feedforward
  G2.2  the looped model beats the echo baseline, whose core is the
        identity so injection still refreshes the state but nothing is
        learned by iterating

**This gate can fail.** Gao et al. arXiv 2607.16051 report that parameter
scaling usually beats looping at matched compute and needed a specific
design at 20B to overturn it. A failure is reported, not worked around.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

#: How much the looped model must win by. Not zero: two runs of anything
#: differ, and a gate that trips on noise is worse than no gate.
MARGIN = 0.02

LOOPED_RUNS = [f"famB_curve111_s{s}" for s in range(8)]
FFWD_RUNS = [f"famB_g2_ffwd_s{s}" for s in (0, 1)]
ECHO_RUNS = [f"famB_g2_echo_s{s}" for s in (0, 1)]


def final_accuracy(runs_root: Path, run_id: str) -> float | None:
    """Mean final accuracy over the depth cells, or None if unavailable."""
    p = runs_root / run_id / "metrics.parquet"
    if not p.is_file():
        return None
    df = pd.read_parquet(p)
    fin = df[(df.metric_name == "final_accuracy") & (df.depth > 0)]
    if fin.empty:
        fin = df[(df.metric_name == "accuracy") & (df.split == "iid_val")
                 & (df.depth > 0)]
        if fin.empty:
            return None
        fin = fin[fin.step == fin.step.max()]
    return float(fin.value.mean())


def arm(runs_root: Path, run_ids: list[str]) -> tuple[float | None, int]:
    vals = [v for v in (final_accuracy(runs_root, r) for r in run_ids) if v is not None]
    return (sum(vals) / len(vals) if vals else None), len(vals)


def run(runs_root: Path) -> tuple[list[dict], bool | None]:
    looped, n_loop = arm(runs_root, LOOPED_RUNS)
    ffwd, n_ffwd = arm(runs_root, FFWD_RUNS)
    echo, n_echo = arm(runs_root, ECHO_RUNS)

    checks = []
    for name, other, n in (("G2.1 looped beats matched-compute feedforward", ffwd, n_ffwd),
                           ("G2.2 looped beats the echo baseline", echo, n_echo)):
        if looped is None or other is None:
            checks.append({"name": name, "passed": None, "looped": looped,
                           "baseline": other, "n_baseline": n,
                           "detail": "baseline has not run"})
        else:
            checks.append({"name": name, "passed": bool(looped > other + MARGIN),
                           "looped": looped, "baseline": other, "n_baseline": n,
                           "detail": f"margin {looped - other:+.4f}, need > {MARGIN}"})

    if any(c["passed"] is None for c in checks):
        return checks, None
    return checks, all(c["passed"] for c in checks)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--json")
    args = p.parse_args()

    checks, passed = run(Path(args.runs_root))
    print()
    print("Gate G2: loops against matched-compute baselines (family B, D-030)")
    print("=" * 72)
    for c in checks:
        status = "PASS" if c["passed"] else ("FAIL" if c["passed"] is False else "....")
        lo = "n/a" if c["looped"] is None else f"{c['looped']:.4f}"
        ba = "n/a" if c["baseline"] is None else f"{c['baseline']:.4f}"
        print(f"  {status}  {c['name']}")
        print(f"        looped {lo}   baseline {ba} ({c['n_baseline']} seed(s))")
        print(f"        {c['detail']}")
    print("=" * 72)
    if passed is None:
        print("  GATE G2: NOT YET EVALUABLE, the baselines have not finished")
    else:
        print(f"  GATE G2: {'PASSED' if passed else 'FAILED'}")
    print()

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps({"passed": passed, "checks": checks}, indent=2), encoding="utf-8")
    return 0 if passed else 1
if __name__ == "__main__":
    raise SystemExit(main())
