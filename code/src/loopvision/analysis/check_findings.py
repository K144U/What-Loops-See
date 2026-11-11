"""Every accuracy in findings.md must be readable from the run it names.

The provenance rule at the top of `findings.md` says no number enters that
file without a run id and a stored metrics file it can be read back out of.
On 2026-09-08 an audit found that rule had been broken for every headline
failure accuracy in the document: 0.1249 where the run reports 0.1234,
0.1277 where it reports 0.1254, 0.1645 where it reports 0.1708, and 0.1659
where it reports 0.1787. The solved runs, at 1.0000, were all correct. The
rule was stated in prose and enforced by nothing, so nothing noticed.

This is the enforcement. It reads the tables, finds the rows that name a
run, and compares every accuracy in them against that run's summary.json.

WHAT IT DOES NOT DO, said here rather than discovered later. It checks
table rows that name a run in backticks, and nothing else. It cannot check
a row whose numbers belong to a run named somewhere else, such as the
seed 0 and seed 1 columns of the Section 2 arm tables, where the row is an
arm and the run ids live in a Provenance line below it. Those rows are
counted and reported as unattributed, so the coverage of this checker is
visible rather than assumed. A clean run means the rows it could attribute
are right. It does not mean the file is right.

Only columns whose header names an accuracy are checked. A chance, loss,
parameters or step column holds numbers that are not the run's accuracy and
must not be compared against it: a chance floor of 0.1250 sits within a
whisker of a real 0.1234 and would fire on every row. A table that
deliberately quotes superseded numbers, such as the CORRECTION entry
recording what the wrong values were, opts out with an HTML comment on one
of the three lines above it:

    <!-- check-findings: skip -->

    python -m loopvision.analysis.check_findings
    python -m loopvision.analysis.check_findings --doc docs/findings.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: Column headers whose cells hold a run's own accuracy, lowercased and
#: stripped of markdown emphasis. Deliberately a small closed set.
ACCURACY_COLUMNS = {
    "accuracy",
    "final accuracy",
    "final_accuracy",
    "value",
}

SKIP_MARKER = "check-findings: skip"

_NUM = re.compile(r"\d+\.\d+")


def _clean(cell: str) -> str:
    return cell.replace("*", "").replace("`", "").strip()


def _ticked(cell: str) -> list[str]:
    """Backticked tokens in one cell."""
    out = []
    parts = cell.split("`")
    for i in range(1, len(parts), 2):
        out.append(parts[i].strip())
    return out


def run_values(run_dir: Path) -> set[float] | None:
    """Every accuracy this run actually recorded.

    The final accuracy and each per-cell accuracy, because a table row may
    legitimately quote either. Returns None when the run has no summary,
    which is not a failure: it means the run has not finished.
    """
    summary = run_dir / "summary.json"
    if not summary.exists():
        return None
    data = json.loads(summary.read_text(encoding="utf-8"))
    values = {float(data["final_accuracy"])}
    values.update(float(v) for v in data.get("by_cell", {}).values())
    return values


def parse_tables(text: str) -> list[dict]:
    """Markdown tables, with header cells, rows and 1-based line numbers."""
    tables: list[dict] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        is_header = (
            line.strip().startswith("|")
            and i + 1 < len(lines)
            and lines[i + 1].strip().startswith("|")
            and set(lines[i + 1].strip()) <= set("|-: ")
        )
        if not is_header:
            i += 1
            continue
        header = [_clean(c).lower() for c in line.strip().strip("|").split("|")]
        skip = any(
            SKIP_MARKER in lines[j] for j in range(max(0, i - 3), i)
        )
        rows = []
        j = i + 2
        while j < len(lines) and lines[j].strip().startswith("|"):
            rows.append((j + 1, lines[j]))
            j += 1
        tables.append({"line": i + 1, "header": header, "rows": rows, "skip": skip})
        i = j
    return tables


def check(doc: Path, runs_root: Path) -> dict:
    text = doc.read_text(encoding="utf-8")
    mismatches: list[dict] = []
    checked = 0
    unattributed = 0
    unfinished = 0
    skipped_tables = 0

    for table in parse_tables(text):
        if table["skip"]:
            skipped_tables += 1
            continue
        cols = [n for n, h in enumerate(table["header"]) if h in ACCURACY_COLUMNS]
        if not cols:
            continue
        for lineno, row in table["rows"]:
            cells = row.strip().strip("|").split("|")
            ids = [
                t
                for c in cells
                for t in _ticked(c)
                if (runs_root / t).is_dir()
            ]
            if not ids:
                unattributed += 1
                continue
            values = run_values(runs_root / ids[0])
            if values is None:
                unfinished += 1
                continue
            for n in cols:
                if n >= len(cells):
                    continue
                for quoted in _NUM.findall(cells[n]):
                    # Compare at the precision the document chose, so a
                    # table quoting 0.125 is not failed for a run that
                    # recorded 0.1254.
                    places = len(quoted.split(".")[1])
                    got = {f"{v:.{places}f}" for v in values}
                    checked += 1
                    if quoted not in got:
                        mismatches.append(
                            {
                                "line": lineno,
                                "run": ids[0],
                                "quoted": quoted,
                                "recorded": sorted(f"{v:.4f}" for v in values),
                                "column": table["header"][n],
                            }
                        )
    return {
        "checked": checked,
        "mismatches": mismatches,
        "unattributed": unattributed,
        "unfinished": unfinished,
        "skipped_tables": skipped_tables,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check findings.md against runs/")
    ap.add_argument("--doc", default=Path("docs/findings.md"), type=Path)
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    args = ap.parse_args(argv)

    if not args.doc.exists():
        print(f"no such document: {args.doc}", file=sys.stderr)
        return 2
    if not args.runs.is_dir():
        print(f"no such runs directory: {args.runs}", file=sys.stderr)
        return 2

    r = check(args.doc, args.runs)
    print(f"checked {r['checked']} number(s) against summary.json")
    print(f"  rows naming no run, not checkable here : {r['unattributed']}")
    print(f"  rows whose run has not finished        : {r['unfinished']}")
    print(f"  tables opted out with the skip marker  : {r['skipped_tables']}")
    if not r["mismatches"]:
        print()
        print("No mismatch in the rows this can attribute. That is not the")
        print("same as the file being right: see the unattributed count,")
        print("which is rows nothing checks.")
        return 0
    print()
    print(f"{len(r['mismatches'])} MISMATCH(ES)")
    for m in r["mismatches"]:
        print(f"  {args.doc}:{m['line']}  {m['run']}")
        print(f"      column   {m['column']}")
        print(f"      quoted   {m['quoted']}")
        print(f"      recorded {', '.join(m['recorded'][:8])}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
