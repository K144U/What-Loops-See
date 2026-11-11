"""The milestone 8 confirmation of claims.md rows 4 to 6. Registered as D-061.

Rows 4, 5 and 6 are the D4 identity curriculum arms: a depth 1 donor of the
same task, a same-group donor on the other substrate, and a wrong-group
donor. Each gets five held-out seeds, and each seed gets its OWN fresh depth 1
donor, `famA_d1_*_s10` to `s14`, so no held-out seed shares a donor with the
arm it tests.

    python -m loopvision.analysis.curriculum_confirm

Two kinds of check, and they are not the same kind of evidence.

  R4 to R6  each held-out arm against its sealed arm, failing when the two
            sided Fisher p is below 0.05. The same rule `escape_rates
            --confirm` applies to rows 1 to 3 and 7, and like that rule it
            can only find a disagreement, never an agreement.
  C1, C2    the claims themselves, on held-out seeds alone. Row 6 says a
            wrong-group donor does not rescue, so the rescue is not generic
            warm starting. That holds on fresh seeds only if the rescue arm
            escapes more often than the wrong-group arm with p below 0.05.
            At five seeds a side that needs near perfect separation, which
            is what makes it a test.

Exit 1 if anything failed, 2 while anything is pending, 0 otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loopvision.analysis import escape_rates as ER

#: Held-out stem, sealed stem.
ROWS = {
    "R4": ("famA_d2_d4colour_curr_confirm", "famA_d2_d4colour_curr"),
    "R5": ("famA_d2_d4colour_currctrl_confirm", "famA_d2_d4colour_currctrl"),
    "R6": ("famA_d2_d4colour_curr_s3donor_confirm", "famA_d2_d4colour_curr_s3donor"),
}
#: Held-out stem expected higher, held-out stem expected lower.
CLAIMS = {
    "C1": ("famA_d2_d4colour_curr_confirm", "famA_d2_d4colour_curr_s3donor_confirm"),
    "C2": ("famA_d2_d4colour_currctrl_confirm", "famA_d2_d4colour_curr_s3donor_confirm"),
}
SEEDS = 5
ALPHA = 0.05


def stem(run_id: str) -> str:
    """The same stem `escape_rates.stem_rates` keys by, so both agree."""
    return run_id.rstrip("0123456789").rstrip("_s") or run_id


def tally(rows: list[dict]) -> dict:
    """Escaped, finished and unfinished per stem. `neither` is finished."""
    out: dict = {}
    for r in rows:
        e = out.setdefault(stem(r["run_id"]), {"escaped": 0, "finished": 0, "unfinished": 0})
        if not r["done"]:
            e["unfinished"] += 1
        elif r["outcome"] == "escaped":
            e["escaped"] += 1
            e["finished"] += 1
        elif r["outcome"] in ("collapsed", "neither"):
            e["finished"] += 1
    return out


def _complete(e: dict | None) -> bool:
    return e is not None and e["unfinished"] == 0 and e["finished"] >= SEEDS


def _p(a: dict, b: dict) -> float:
    return ER.fisher_two_sided(a["escaped"], a["finished"] - a["escaped"],
                               b["escaped"], b["finished"] - b["escaped"])


def _rate(e: dict | None) -> str:
    return "none yet" if e is None else f"{e['escaped']} of {e['finished']}"


def registered(rows: list[dict]) -> dict[str, tuple[str, str]]:
    t = tally(rows)
    out = {}
    for key, (held, sealed) in ROWS.items():
        h, s = t.get(held), t.get(sealed)
        if not _complete(h) or s is None:
            out[key] = ("pending", f"{held} {_rate(h)}")
            continue
        p = _p(h, s)
        detail = f"held out {_rate(h)} against sealed {_rate(s)}, Fisher p = {p:.3f}"
        # Not failing is not confirming; the verdict word says only that.
        out[key] = ("failed" if p < ALPHA else "not contradicted", detail)
    for key, (hi, lo) in CLAIMS.items():
        a, b = t.get(hi), t.get(lo)
        if not (_complete(a) and _complete(b)):
            out[key] = ("pending", f"{hi} {_rate(a)}, {lo} {_rate(b)}")
            continue
        p = _p(a, b)
        higher = a["escaped"] / a["finished"] > b["escaped"] / b["finished"]
        detail = f"{_rate(a)} against {_rate(b)}, Fisher p = {p:.3f}"
        out[key] = ("held" if (p < ALPHA and higher) else "failed", detail)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D-061, claims.md rows 4 to 6")
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    args = ap.parse_args(argv)
    if not args.runs.is_dir():
        print(f"no runs directory at {args.runs}")
        return 2
    verdicts = registered(ER.scan(args.runs))
    for k, (v, detail) in verdicts.items():
        print(f"{k}  {v:17s} {detail}")
    states = {v for v, _ in verdicts.values()}
    return 1 if "failed" in states else 2 if "pending" in states else 0


if __name__ == "__main__":
    raise SystemExit(main())
