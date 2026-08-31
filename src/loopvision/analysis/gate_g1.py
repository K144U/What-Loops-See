"""Run gate G1 and report the four checks as a table.

    python -m loopvision.analysis.gate_g1

Exits 0 if every check passes, 1 otherwise. **A failing gate is a hard
stop, not a nuisance.** If G1 fails there is no valid measuring instrument
and nothing downstream is interpretable, so the response is to deepen the
task and re-run the gate with both attempts recorded in docs/findings.md,
never to adjust a threshold.

Thresholds live in loopvision.data.validate and were committed before the
first gate job was submitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loopvision.data import validate as V


def run(runs_root: Path, skip_slow: bool = False, attempt: int = 2) -> tuple[list[V.CheckResult], bool]:
    results: list[V.CheckResult] = []

    runs = V.ATTEMPT_1_RUNS if attempt == 1 else V.GATE_RUNS
    results.extend(
        V.check_g1_1(runs_root / runs["d3_looped"], runs_root / runs["d3_ffwd"])
    )
    results.append(V.check_g1_2(runs_root / runs["d1_looped"]))
    if not skip_slow:
        results.append(V.check_g1_3())
    results.append(V.check_g1_4(runs_root / runs["famC"]))

    return results, all(r.passed for r in results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--skip-slow", action="store_true", help="skip G1.3")
    parser.add_argument("--attempt", type=int, default=2, choices=[1, 2],
                        help="1 reads the 20k step runs, 2 the 200k step reruns")
    parser.add_argument("--json", help="also write the results here")
    args = parser.parse_args()

    try:
        results, passed = run(Path(args.runs_root), skip_slow=args.skip_slow, attempt=args.attempt)
    except FileNotFoundError as exc:
        print(f"gate G1 cannot be evaluated: {exc}")
        return 1

    width = max(len(r.name) for r in results) + 2
    print()
    print(f"Gate G1: task validity (attempt {args.attempt})")
    print("=" * (width + 46))
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {status}  {r.name:<{width}} {r.measured:.4f}  (need {r.threshold:.4f})")
        if r.detail:
            print(f"        {r.detail}")
    print("=" * (width + 46))
    print(f"  GATE G1: {'PASSED' if passed else 'FAILED'}")
    print()

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(
                {
                    "passed": passed,
                    "checks": [
                        {
                            "name": r.name,
                            "passed": r.passed,
                            "measured": r.measured,
                            "threshold": r.threshold,
                            "detail": r.detail,
                        }
                        for r in results
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if not passed:
        print("  A failing gate is a hard stop. Do not proceed to milestone 3.")
        print("  Deepen the task, re-run the gate, and record both attempts")
        print("  in docs/findings.md. Do not adjust a threshold.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
