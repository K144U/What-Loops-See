"""Escape rate per cell, generated from the runs rather than typed by hand.

CLAUDE.md makes this a non-negotiable: counts and totals in the documents
come from a generator, because a hand-typed total is correct on the day it
is written and silently wrong afterwards. The substrate 2x2 in paper.md
section 3b was retyped three times on 9 September and drifted every time,
which is what this module exists to stop.

    python -m loopvision.analysis.escape_rates
    python -m loopvision.analysis.escape_rates --json out.json

THREE OUTCOMES, NOT TWO, AND THE THIRD IS THE POINT. A finished run is
sorted into escaped, collapsed, or neither. The paper claims escape is all
or nothing at the level of a run: 50 finished family A runs, 19 above 0.95
and 31 on a floor, none in between. That claim is an empirical fact about
runs that have finished so far and it can stop being true at any moment. A
classifier with two buckets would have to put a counterexample in one of
them and the claim would quietly survive its own refutation. So `neither`
is a bucket, it is printed first when it is non-empty, and every run in it
is named. It is empty today.

HOW A RUN IS SORTED, and which part is measured.

  escaped    final accuracy >= 0.95
  collapsed  n_distinct_predictions == 1, measured directly
  collapsed  no such field, and accuracy < 0.20, inferred and marked
  neither    anything else

The 0.95 is inherited from findings.md and paper.md, which have used it
since before this module existed. It is not chosen here and must not be
moved here: CLAUDE.md forbids tuning a threshold to improve a result, and
a threshold that lives in a generator is exactly where such a move would
be invisible. The two thresholds sit far outside the observed gap, where
every finished family A run is either above 0.95 or below 0.20, so no run
currently sits near either edge.

Runs finishing before 2026-09-08 predate `n_distinct_predictions` and are
classified by accuracy alone. That is an inference, it is marked
`inferred` in the output, and it is supported by a separate fact: runs
that collapse onto the same class report the same accuracy to the last
stored digit, so the pre-instrumentation floors match the measured ones
exactly rather than approximately.

UNFINISHED RUNS ARE NEVER COUNTED, in either direction. A rate is over
finished seeds and the unfinished ones are listed with their current
reading, because escape can arrive late: famA_d2_d4only_s2 read 0.5039 at
step 210000 and finished at 0.9999. Counting it early would have been
wrong in the direction that helped.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from loopvision.analysis.check_configs import _identity as cell_identity

#: Inherited from findings.md and paper.md. Never moved to fit a result.
ESCAPE_TAU = 0.95

#: Only ever applied to runs that predate n_distinct_predictions.
COLLAPSE_ACC = 0.20

#: The registered 2x2, keyed on the config fields that actually separate its
#: four arms. From experiment-substrate.md, "The four arms, concretely":
#: d4_position and s3_identity were already built and use the native
#: substrate with a single factor; s3_position and d4_identity are the new
#: arms and move each group onto the other medium.
SUBSTRATE_ARMS = {
    ("d4", "native"): ("D4", "position"),
    ("s3", "native"): ("S3", "identity"),
    ("full", "d4_colour"): ("D4", "identity"),
    ("full", "s3_spatial"): ("S3", "position"),
}

#: An arm is a BASELINE cell, not every run that shares a factor.
#: experiment-substrate.md: "Every arm keeps depth 2, the same architecture,
#: the same step budget", so a run differing in width, block count,
#: architecture, budget, depth or warm start is a different experiment and
#: does not belong in the table.
#:
#: The first version of this module omitted this and reported D4 acting on
#: identity as 6 of 19, having swept in the wide, big, feedforward,
#: matched-parameter, curriculum, depth 1 and long-budget runs. The number
#: was obviously wrong, which is the only reason it was caught; a subtler
#: version of the same error would have read like a result.
#:
#: `famA_d2_d4colour_deep` matches on every key here because it is 2/2/2 at
#: 384, identical to the base arm. That is correct and D-043 already says
#: so: the config was meant to be a depth arm and duplicates the baseline
#: instead, so its run is a sixth seed of that cell rather than a control.
#: The parameter-matched depth arm is `depth4` at 4/4/4, still queued.
BASELINE = {
    "arch": "looped",
    "d_model": 384,
    "depths": [2],
    "prelude_blocks": 2,
    "core_blocks": 2,
    "coda_blocks": 2,
    "k_schedule": "sampled",
    "k_train": 1,
    "init_from": None,
    "steps": 300000,
    # Added 2026-10-11 for D-066 and D-067, whose runs reuse the 2x2's own
    # substrates and differ only in these. Without them a learning-rate grid
    # seed at 1e-3 would be counted as a seed of the sealed cell it varies.
    "lr": 3e-4,
    "weight_decay": 0.05,
    "warmup": 2000,
    "early_stop_tau": None,
    "early_stop_window": None,
    "pixel_scale": 1,
}

#: What a key the sealed runs predate means when it is absent from a stored
#: config. Every 2x2 seed was written before these existed, so absence has to
#: read as the default, or the whole grid would drop out at once.
ABSENT_MEANS = {
    "early_stop_tau": None,
    "early_stop_window": None,
    "pixel_scale": 1,
}

GROUPS = ("D4", "S3")
MEDIA = ("position", "identity")

#: A run carrying this key in its own stored config is a milestone 8
#: confirmation seed and is HELD OUT of the registered 2x2.
#:
#: **Why a hold-out is the only design that works here.** The four confirmation
#: configs differ from their baselines in nothing but the seed, which is
#: exactly what makes them confirmations, and also exactly what makes
#: `arm_of` put them straight into the sealed grid. Without this key, five
#: fresh seeds of `famA_d2_d4colour` would take D4/identity from 40 seeds to
#: 45, move identity off 15 of 50, and change the Fisher p of 0.0011 that
#: `paper.md` Section 3b was revised around on 2026-09-21 under D-052. The
#: grid closed at 70 of 70 and reopening it silently is the failure this
#: prevents.
#:
#: **And pooling them would not confirm anything anyway.** A confirmation run
#: has to be independent of the data the claim was found in. A seed that
#: joins the cell it is meant to test stops being a test of that number and
#: becomes part of it, so the design defeats itself in both directions at
#: once. Held out, the comparison is the real one: a rate measured on data
#: nobody had seen against a rate that was.
#:
#: It lives in the run's own `config.yaml` rather than in a list here, so a
#: run carries its own status forever and no later edit to this file can
#: reclassify a run that has already finished.
CONFIRMATION_KEY = "confirmation"


def is_baseline(cfg: dict) -> bool:
    """True when a config differs from the registered arm only in seed."""
    for key, want in BASELINE.items():
        got = cfg.get(key, ABSENT_MEANS.get(key))
        if isinstance(want, list):
            got = list(got) if got is not None else None
        if got != want:
            return False
    return True


def classify(summary: dict) -> tuple[str, str]:
    """Sort one finished run, and say how the decision was made."""
    acc = summary.get("final_accuracy")
    if acc is None:
        return "neither", "no final accuracy recorded"
    if acc >= ESCAPE_TAU:
        return "escaped", "measured"
    n = summary.get("n_distinct_predictions")
    if n is not None:
        if n == 1:
            return "collapsed", "measured"
        return "neither", f"{n} distinct classes at {acc:.4f}"
    if acc < COLLAPSE_ACC:
        return "collapsed", "inferred from accuracy, predates the field"
    return "neither", f"{acc:.4f} with no class count recorded"


def is_confirmation(cfg: dict) -> bool:
    """True when a config is a held-out milestone 8 confirmation seed.

    **The two mistakes here are not symmetric, so neither is this test.** A
    confirmation seed wrongly admitted to the grid changes a sealed headline
    number and nothing announces it. An ordinary seed wrongly held out is
    missing from a count that `cell_closure` already checks against a
    registered minimum, so it fails loudly at the next `--closed`.

    So absent, `false` and `null` are ordinary data, and **anything else at
    all is held out**. A config that says `confirmation: "yes"` or
    `confirmation: 1` was written by someone who meant to hold it out, and
    reading either as ordinary data is the failure that cannot be seen.
    """
    val = cfg.get(CONFIRMATION_KEY, False)
    if val is False or val is None:
        return False
    return True


def confirmation_arm_of(cfg: dict) -> tuple[str, str] | None:
    """The cell a confirmation seed is testing, or None if it is not one.

    This is what `arm_of` would have returned had the run not been held out.
    It exists so the hold-out can be compared against the thing it holds out
    from: a confirmation seed that nothing can attribute to a cell would be
    unusable.
    """
    if not is_confirmation(cfg):
        return None
    if not is_baseline(cfg):
        return None
    return SUBSTRATE_ARMS.get(
        (cfg.get("factor", "full"), cfg.get("substrate") or "native")
    )


def arm_of(cfg: dict) -> tuple[str, str] | None:
    """The 2x2 cell a config belongs to, or None if it is not in the 2x2.

    Three conditions have to hold. The factor and substrate say which arm a
    config would be, the baseline check says whether it is that arm or a
    variant of it that varies something the design held fixed, and the
    confirmation check holds out the milestone 8 seeds.

    **The confirmation test comes first and is not an optimisation.** Those
    configs differ from their baselines in nothing but the seed, so every
    other test here passes for them. This line is the only thing standing
    between a confirmation seed and the sealed grid. See CONFIRMATION_KEY.
    """
    if is_confirmation(cfg):
        return None
    if not is_baseline(cfg):
        return None
    return SUBSTRATE_ARMS.get(
        (cfg.get("factor", "full"), cfg.get("substrate") or "native")
    )


def _read(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        return None
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    done = (run_dir / "DONE").is_file()
    summary = {}
    sp = run_dir / "summary.json"
    if sp.is_file():
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    return {"run_id": run_dir.name, "cfg": cfg, "done": done, "summary": summary}


def scan(runs_root: Path) -> list[dict]:
    """Every run, with its cell, its outcome and how that was decided."""
    out = []
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        r = _read(d)
        if r is None:
            continue
        if r["done"]:
            outcome, why = classify(r["summary"])
        else:
            outcome, why = "unfinished", ""
        out.append(
            {
                "run_id": r["run_id"],
                "cell": cell_identity(r["cfg"]),
                "arm": arm_of(r["cfg"]),
                # The cell this run is a held-out confirmation OF, and None
                # for every ordinary run. Exactly one of `arm` and
                # `confirm_arm` is ever set, which is what keeps the two
                # populations from being added together by accident.
                "confirm_arm": confirmation_arm_of(r["cfg"]),
                "seed": r["cfg"].get("seed"),
                "done": r["done"],
                "outcome": outcome,
                "why": why,
                "accuracy": r["summary"].get("final_accuracy"),
                "n_distinct": r["summary"].get("n_distinct_predictions"),
                "modal": r["summary"].get("modal_prediction"),
            }
        )
    return out


def rates(rows: list[dict], key: str = "arm") -> dict:
    """Escaped over finished, per group, never counting the unfinished."""
    acc: dict = {}
    for r in rows:
        k = r[key]
        if k is None:
            continue
        k = tuple(k) if isinstance(k, list) else k
        e = acc.setdefault(
            k, {"escaped": 0, "collapsed": 0, "neither": 0, "unfinished": 0,
                "runs": [], "floors": []}
        )
        e[r["outcome"]] += 1
        e["runs"].append(r)
        if r["outcome"] == "collapsed" and r["accuracy"] is not None:
            e["floors"].append(r["accuracy"])
    for e in acc.values():
        e["finished"] = e["escaped"] + e["collapsed"] + e["neither"]
    return acc


#: Marks the region of a document whose escape counts come from here.
#: A region rather than a table, because the same counts appear in the
#: sentence under the table and go stale in exactly the same way.
#: Below this many finished seeds, an arm's rate is reported with a warning
#: that any contrast using it needs its p computed. D-050.
SMALL_N = 5


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two sided Fisher exact on [[a, b], [c, d]], no scipy.

    Sums the probability of every table at least as extreme as the observed
    one, which is the conventional two sided form for a test with no natural
    ordering of the alternatives.

    WHY THIS IS IN THE GENERATOR AND NOT IN A NOTEBOOK. D-050 retired a claim
    that had stood for days: "escape needs capacity, not the removal of the
    loop", resting on 1 of 5 against 0 of 2. That contrast has a p of 1.000.
    Not weak evidence, none. It became a claim because two consecutive
    failures look like a pattern and because 0 of 2 reads as a zero rather
    than as two draws.

    The fix is not more care, it is putting the p where the rate is read. A
    rate printed alone invites a contrast; a rate printed next to its own
    small-n warning does not.
    """
    from math import comb

    n = a + b + c + d
    row1, col1 = a + b, a + c
    if n == 0 or row1 == 0 or col1 == 0 or row1 == n or col1 == n:
        return 1.0

    def prob(x: int) -> float:
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)

    observed = prob(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                        if prob(x) <= observed + 1e-12))


def stem_rates(rows: list[dict]) -> dict:
    """Escape counts keyed by the run-id stem the listing prints.

    `rates(rows, "arm")` keys by 2x2 cell, which cannot name the comparison
    D-050 is actually about: the looped d4colour arm against the two
    feedforward ones. Those are three different configs inside one cell. This
    keys by the stem a reader sees in the listing, which is what they will
    type.
    """
    named: dict = {}
    for r in rows:
        stem = r["run_id"].rstrip("0123456789").rstrip("_s") or r["run_id"]
        e = named.setdefault(stem, {"escaped": 0, "finished": 0})
        if r["outcome"] == "escaped":
            e["escaped"] += 1
            e["finished"] += 1
        elif r["outcome"] in ("collapsed", "neither"):
            e["finished"] += 1
    return named


def contrast(by_arm: dict, arm_a: str, arm_b: str) -> dict | None:
    """The escape rates of two arms and the Fisher p between them."""
    ea, eb = by_arm.get(arm_a), by_arm.get(arm_b)
    if ea is None or eb is None:
        return None
    a, na = ea["escaped"], ea["finished"]
    b, nb = eb["escaped"], eb["finished"]
    return {
        "a": arm_a, "b": arm_b,
        "rate_a": f"{a} of {na}", "rate_b": f"{b} of {nb}",
        "p": fisher_two_sided(a, na - a, b, nb - b),
        "small": min(na, nb) < SMALL_N,
    }


BEGIN_MARKER = "escape-rates: begin"
END_MARKER = "escape-rates: end"

_RATE = re.compile(r"(\d+)\s+of\s+(\d+)")
_FLOOR = re.compile(r"floor\s+(\d+\.\d+)(?:\s+to\s+(\d+\.\d+))?")


#: An inline annotation declaring where a prose count came from. A count
#: carrying one is exempt from the outside-the-markers check below.
COUNT_OK_MARKER = "count-ok"


def unannotated_counts(text: str) -> list[dict]:
    """Counts in prose outside the marked region that declare no source.

    CLAUDE.md already says it: "Counts outside them are unchecked, so prose
    outside the markers should not quote a number that moves as seeds land."
    That rule was stated and enforced by nothing, and on 2026-09-11 it cost a
    real error. `paper.md` Section 3b carried "Every finished failure carrying
    the field, 7 of 7, emits exactly one class", hand typed when there were
    seven such runs. There were eighteen by the time anyone looked, and the
    claim had also become false. The marked region twenty lines above it was
    checked on every commit and never drifted once.

    So the difference between the count that went stale and the counts that
    did not was whether something read them, which is [[l-013]] again.

    WHY THIS DOES NOT SIMPLY BAN THEM. Plenty of counts in prose are fixed
    forever: an analytic ceiling, a count from a cited paper, a superseded
    number kept deliberately so the correction can be read. Failing on those
    would make the check noise, and a noisy check gets disabled. So a count
    may stay if it says where it comes from:

        <!-- count-ok: from the 2026-09-11 survey of finished failures -->

    on the same line or within the two lines above it. The annotation is not
    verified, and cannot be. What it does is force the writer to name a source
    at the moment of writing, which is the moment they still know it.
    """
    lines = text.splitlines()
    inside: set[int] = set()
    depth = 0
    for i, line in enumerate(lines):
        if BEGIN_MARKER in line:
            depth += 1
        if depth:
            inside.add(i)
        if END_MARKER in line:
            depth -= 1

    out = []
    for i, line in enumerate(lines):
        if i in inside:
            continue
        near = lines[max(0, i - 2):i + 1]
        if any(COUNT_OK_MARKER in n for n in near):
            continue
        for m in _RATE.finditer(line):
            out.append({
                "line": i + 1,
                "quoted": m.group(0),
                "context": line.strip()[:88],
            })
    return out


def marked_region(text: str) -> str | None:
    """The text between the markers, or None if both are not present."""
    lines = text.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if BEGIN_MARKER in line and start is None:
            start = i
        elif END_MARKER in line and start is not None:
            end = i
            break
    if start is None or end is None:
        return None
    return "\n".join(lines[start + 1:end])


def expected_claims(by_arm: dict) -> dict:
    """Every count the marked region is allowed to state, and what it means.

    Four cells, one total per medium, and the finished-over-total for the
    whole design. A count in the document that is none of these is stale.
    """
    claims: dict[tuple[int, int], list[str]] = {}

    def add(escaped, finished, why):
        claims.setdefault((escaped, finished), []).append(why)

    for g in GROUPS:
        for m in MEDIA:
            e = by_arm.get((g, m))
            if e is not None:
                add(e["escaped"], e["finished"], f"{g} acting on {m}")
    for m in MEDIA:
        esc = sum(v["escaped"] for (g, mm), v in by_arm.items() if mm == m)
        fin = sum(v["finished"] for (g, mm), v in by_arm.items() if mm == m)
        add(esc, fin, f"both groups acting on {m}")
    fin = sum(v["finished"] for v in by_arm.values())
    unf = sum(v["unfinished"] for v in by_arm.values())
    add(fin, fin + unf, "seeds finished in the 2x2")
    return claims


def check_document(doc: Path, by_arm: dict) -> dict:
    """Every count in the marked region, against what the runs now say.

    This enforces CLAUDE.md's rule that counts in prose come from a
    generator. A generator nobody reruns is a hand-typed number one day
    later, which is what happened to this table three times on 2026-09-09,
    twice after the generator existed.

    The test is that every "N of M" inside the region is a count the runs
    currently support. That catches the failure that actually happens, a
    number going stale, without pinning the wording around it.
    """
    text = Path(doc).read_text(encoding="utf-8")
    region = marked_region(text)
    if region is None:
        return {"ok": False, "reason": "markers", "stale": [],
                "bad_floors": [], "found": 0}

    allowed = expected_claims(by_arm)
    stale = []
    for m in _RATE.finditer(region):
        pair = (int(m.group(1)), int(m.group(2)))
        if pair not in allowed:
            stale.append({
                "quoted": f"{pair[0]} of {pair[1]}",
                "context": region[max(0, m.start() - 50):m.end() + 12]
                .replace("\n", " ").strip(),
            })

    loose = unannotated_counts(text)

    real = [(min(e["floors"]), max(e["floors"]))
            for e in by_arm.values() if e["floors"]]
    bad_floors = []
    for m in _FLOOR.finditer(region):
        places = len(m.group(1).split(".")[1])
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else lo
        if not any(round(a, places) == round(lo, places)
                   and round(b, places) == round(hi, places)
                   for a, b in real):
            bad_floors.append(m.group(0))

    return {
        "ok": not stale and not bad_floors and not loose,
        "reason": None,
        "stale": stale,
        "bad_floors": bad_floors,
        "unannotated": loose,
        "found": len(_RATE.findall(region)),
        "allowed": allowed,
    }


#: D-042: ten seeds for any cell whose number reaches an abstract, a headline
#: sentence or a figure caption. Every cell of the 2x2 does, so this is the
#: registered minimum and not a number chosen here.
SEED_MIN = 10


def cell_closure(rows: list[dict]) -> dict:
    """Per cell: finished, unfinished, and whether the 2x2 is closed.

    "Closed" is NOT "nothing unfinished". An empty runs directory, an
    unreadable one, or a scan that silently returned nothing all have zero
    unfinished seeds and would announce closure of an experiment that has not
    run. That exact shape cost a false all-clear on 2026-09-17, where an ssh
    failure and an empty GPU were indistinguishable.

    So closure has to be earned by a positive fact: every one of the four
    registered cells carries at least its D-042 minimum of finished seeds AND
    none of them has a seed still running.
    """
    cells: dict = {}
    for g in GROUPS:
        for m in MEDIA:
            cells[(g, m)] = {"finished": 0, "unfinished": 0}
    for r in rows:
        if r["arm"] is None:
            continue
        key = "unfinished" if not r["done"] else "finished"
        cells[r["arm"]][key] += 1

    short = [k for k, v in cells.items() if v["finished"] < SEED_MIN]
    busy = [k for k, v in cells.items() if v["unfinished"] > 0]
    return {
        "cells": cells,
        "short": short,
        "busy": busy,
        "closed": not short and not busy,
        "finished": sum(v["finished"] for v in cells.values()),
        "unfinished": sum(v["unfinished"] for v in cells.values()),
    }


def confirmation_tally(rows: list[dict]) -> dict:
    """Per cell, the held-out confirmation seeds beside the sealed rate.

    Two populations that must never be added together, built in one pass so
    it is visible here that they are not. A run reaches `held` through
    `confirm_arm` and `sealed` through `arm`, and `scan` sets at most one of
    those, so no run can reach both.
    """
    held: dict = {}
    sealed: dict = {}
    for g in GROUPS:
        for m in MEDIA:
            held[(g, m)] = {"finished": 0, "escaped": 0, "unfinished": 0}
            sealed[(g, m)] = {"finished": 0, "escaped": 0}
    for r in rows:
        if r["confirm_arm"] is not None:
            b = held[r["confirm_arm"]]
            if not r["done"]:
                b["unfinished"] += 1
                continue
            b["finished"] += 1
            if r["outcome"] == "escaped":
                b["escaped"] += 1
        elif r["arm"] is not None and r["done"]:
            s = sealed[r["arm"]]
            s["finished"] += 1
            if r["outcome"] == "escaped":
                s["escaped"] += 1
    return {"held": held, "sealed": sealed}


def report_confirmation(rows: list[dict]) -> int:
    """The milestone 8 confirmation register. Non-zero when one has failed.

    **Failing is a result and is reported as one.** Preregistration Section 1:
    "A confirmation run that fails is reported as a failure", and nothing
    here may be tuned to make a row confirm.

    **Not failing is not the same as confirming, and the output says so.**
    Five seeds against a rate of three in ten cannot produce a significant
    Fisher p in the agreeing direction, so a cell that comes back quiet has
    shown only that it is not loudly inconsistent. That is worth having and
    it is not what the word confirmed usually implies, so the word is not
    used for it.
    """
    t = confirmation_tally(rows)
    held, sealed = t["held"], t["sealed"]
    pending = sum(v["unfinished"] for v in held.values())
    done = sum(v["finished"] for v in held.values())

    if done == 0 and pending == 0:
        print("No confirmation seed exists yet, held out or otherwise. The "
              "milestone 8 register is empty and nothing is claimed.")
        return 2

    print("Milestone 8 confirmation, held out of the registered 2x2:")
    print(f"  {'cell':<14} {'held out':>12}   {'sealed':>12}   {'Fisher p':>9}")
    failed, thin = [], []
    for g in GROUPS:
        for m in MEDIA:
            h, s = held[(g, m)], sealed[(g, m)]
            if h["finished"] == 0 and h["unfinished"] == 0:
                continue
            name = f"{g}/{m}"
            if h["finished"] == 0:
                print(f"  {name:<14} {'0 finished':>12}   "
                      f"{s['escaped']:>3} of {s['finished']:<6}   "
                      f"{'--':>9}   {h['unfinished']} still running")
                continue
            p = fisher_two_sided(
                h["escaped"], h["finished"] - h["escaped"],
                s["escaped"], s["finished"] - s["escaped"])
            flag = ""
            if p < 0.05:
                failed.append(name)
                flag = "  <-- DISAGREES with the sealed rate"
            if h["finished"] < SMALL_N:
                thin.append(name)
            print(f"  {name:<14} {h['escaped']:>3} of {h['finished']:<6}   "
                  f"{s['escaped']:>3} of {s['finished']:<6}   {p:>9.4f}{flag}")

    if pending:
        print(f"\n{pending} confirmation seed(s) still running. Every number "
              f"above will move.")
    if thin:
        print(f"\nUnder {SMALL_N} finished confirmation seeds: "
              f"{', '.join(thin)}. A quiet Fisher p on a handful of draws is "
              f"not evidence of agreement, it is an absence of evidence of "
              f"disagreement. D-050 retired a claim that stood for days on a "
              f"contrast whose p was 1.000.")
    if failed:
        print(f"\nA CONFIRMATION HAS FAILED: {', '.join(failed)}. The "
              f"preregistration requires this to be reported as a failure "
              f"and forbids tuning anything to make it confirm.")
        return 1
    print("\nNo held-out cell contradicts its sealed rate. That is weaker "
          "than confirmation and is not the same sentence.")
    return 0


def queued_2x2(configs_root: Path) -> tuple[list[str], str | None]:
    """2x2 runs sitting in the queue, and why the queue could not be read.

    A queued run has NO directory under runs/, so `scan` cannot see it and
    `cell_closure` cannot count it. That is a real window and not a corner
    case: the nine `famA_d2_d4colour_deep` seeds are D4 identity members by
    D-043 and had not started on 2026-09-17, so had the five d4colour seeds
    ahead of them landed first, every cell would have read finished and idle
    while nine members of one sat waiting for a slot.

    An unreadable queue returns a REASON rather than an empty list, because
    "nothing queued" and "could not ask" must never be the same answer. That
    conflation is what made an ssh failure look like an idle GPU.
    """
    from loopvision.analysis import check_queue as Q

    text = Q.queued_text()
    if not text.strip():
        return [], "the queue could not be read, so nothing is claimed about it"
    try:
        return [r[0] for r in Q.headline_backlog(text, configs_root)], None
    except Exception as e:
        return [], f"the queue could not be parsed: {type(e).__name__}: {e}"


def report_closure(rows: list[dict], configs_root: Path = Path("configs")) -> int:
    """Print the closure state. 0 when closed, 1 when not. See cell_closure."""
    st = cell_closure(rows)
    queued, why = queued_2x2(configs_root)
    print("The registered 2x2, by cell:")
    for g in GROUPS:
        for m in MEDIA:
            v = st["cells"][(g, m)]
            note = ""
            if v["finished"] < SEED_MIN:
                note = f"  <-- under the registered minimum of {SEED_MIN}"
            elif v["unfinished"]:
                note = "  <-- still running"
            print(f"  {g}/{m:<9} {v['finished']:>3} finished, "
                  f"{v['unfinished']:>2} unfinished{note}")
    print(f"\n{st['finished']} finished and {st['unfinished']} unfinished "
          f"across the four cells.")

    if st["closed"] and why is not None:
        print(f"\nEvery cell looks finished and idle, but {why}. "
              f"**A queued seed has no run directory at all**, so it is "
              f"invisible here and cannot be ruled out. NOT declaring "
              f"closure. See D-043 for the nine deep seeds this exists for.")
        return 1

    if st["closed"] and queued:
        print(f"\nEvery cell is finished and idle, but {len(queued)} "
              f"seed(s) of the 2x2 are still QUEUED with no run directory "
              f"yet: {chr(44).join(sorted(queued))}. NOT closed.")
        return 1

    if st["closed"]:
        print("\nTHE 2x2 HAS CLOSED. Every cell carries at least its "
              f"registered {SEED_MIN} seeds, nothing is still running, and "
              f"nothing is left in the queue.")
        return 0

    if st["short"]:
        names = ", ".join(f"{g}/{m}" for g, m in sorted(st["short"]))
        print(f"\nNOT closed: {names} below the registered {SEED_MIN} "
              f"finished seeds. **A cell can be short and have nothing "
              f"unfinished at the same time**, which is what an empty or "
              f"unreadable runs directory looks like, so this is checked "
              f"rather than inferred from the unfinished count alone.")
    if st["busy"]:
        names = ", ".join(f"{g}/{m}" for g, m in sorted(st["busy"]))
        print(f"\nNOT closed: {names} still has seeds running.")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Escape rates, generated")
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    ap.add_argument("--json", default=None, help="write the numbers here too")
    ap.add_argument("--check", default=None, type=Path,
                    help="a document whose marked region must match the runs")
    ap.add_argument("--closed", action="store_true",
                    help="exit 0 only when every 2x2 cell has its registered "
                         "minimum finished and none is still running")
    ap.add_argument("--contrast", nargs=2, metavar=("ARM_A", "ARM_B"),
                    default=None,
                    help="Fisher exact between two arms, before you write the "
                         "contrast as a claim. See D-050.")
    ap.add_argument("--confirm", action="store_true",
                    help="the milestone 8 confirmation register: held-out "
                         "seeds against the sealed 2x2 rate. Non-zero when a "
                         "confirmation has failed.")
    args = ap.parse_args(argv)

    if not args.runs.is_dir():
        print(f"no such runs directory: {args.runs}")
        return 2

    rows = scan(args.runs)

    if args.closed:
        return report_closure(rows, Path("configs"))

    if args.confirm:
        return report_confirmation(rows)

    if args.contrast is not None:
        by_arm = stem_rates(rows)
        c = contrast(by_arm, *args.contrast)
        if c is None:
            print("no such arm. Known arms:")
            for k in sorted(by_arm):
                print(f"  {k}")
            return 2
        print(f"  {c['a']}: {c['rate_a']}")
        print(f"  {c['b']}: {c['rate_b']}")
        print(f"  Fisher exact, two sided: p = {c['p']:.3f}")
        if c["p"] > 0.05:
            print("  These rates are NOT distinguishable. Do not write this "
                  "contrast as a claim.")
        if c["small"]:
            print(f"  At least one arm has under {SMALL_N} finished seeds, "
                  f"which is the case D-050 exists for.")
        return 0

    if args.check is not None:
        if not args.check.is_file():
            print(f"no such document: {args.check}")
            return 2
        by_arm = rates(rows, "arm")
        r = check_document(args.check, by_arm)
        if r["reason"] == "markers":
            print(f"{args.check} has no marked region. Wrap the generated "
                  f"counts in a pair of HTML comments:\n"
                  f"    <!-- {BEGIN_MARKER} -->\n    ...\n"
                  f"    <!-- {END_MARKER} -->")
            return 2
        loose = r.get("unannotated") or []
        print(f"checked {r['found']} count(s) in {args.check}, and "
              f"{len(loose)} unannotated count(s) outside the markers")
        if r["ok"]:
            print("Every count in the marked region matches the runs, and "
                  "every count outside them names a source.")
            return 0
        for s in r["stale"]:
            print(f"\nSTALE  {s['quoted']}\n  ...{s['context']}...")
        for f in r["bad_floors"]:
            print(f"\nSTALE  {f}")
        for u in loose:
            print(f"\nUNANNOTATED  {u['quoted']}  at line {u['line']}")
            print(f"  {u['context']}")
        if loose:
            print(
                f"\nA count in prose outside the markers is read by nothing, so it "
                f"goes stale silently. That is how paper.md carried 7 of 7 until "
                f"it was eighteen. Either move it inside the markers, or say "
                f"where it came from:")
            print(f"    <!-- {COUNT_OK_MARKER}: from <source>, <date> -->")
            print("  on the same line or within the two lines above it.")

        print("\nWhat the runs say now:")
        for (e, f), whys in sorted(r["allowed"].items()):
            print(f"  {e} of {f:<3} {'; '.join(whys)}")
        print(f"\nRegenerate with: python -m loopvision.analysis.escape_rates")
        return 1

    # Printed first when non-empty, because it is the bucket that would
    # refute the claim the other two support.
    odd = [r for r in rows if r["outcome"] == "neither"]
    if odd:
        print(f"{len(odd)} finished run(s) are NEITHER escaped nor collapsed. "
              f"The 'none in between' claim in paper.md 3b has exceptions and "
              f"they are named here:")
        for r in odd:
            print(f"  {r['run_id']:38s} {r['accuracy']}  {r['why']}")
        print()
    else:
        print("No finished run is between escaped and collapsed. The "
              "dichotomy holds on every run scanned.\n")

    by_arm = rates(rows, "arm")
    print("The registered substrate 2x2, escape rate over FINISHED seeds.")
    print("From experiment-substrate.md. Unfinished seeds are excluded and "
          "listed below.\n")
    head = "%-14s %-22s %-22s" % ("", "acts on position", "acts on identity")
    print("    " + head)
    for g in GROUPS:
        cells = []
        for m in MEDIA:
            e = by_arm.get((g, m))
            if e is None:
                cells.append("no runs")
                continue
            txt = f"{e['escaped']} of {e['finished']}"
            if e["collapsed"] and e["floors"]:
                txt += f", floor {min(e['floors']):.4f}"
                if max(e["floors"]) != min(e["floors"]):
                    txt += f" to {max(e['floors']):.4f}"
            cells.append(txt)
        print("    %-14s %-22s %-22s" % (g, cells[0], cells[1]))

    tot = {m: [0, 0] for m in MEDIA}
    for (g, m), e in by_arm.items():
        tot[m][0] += e["escaped"]
        tot[m][1] += e["finished"]
    print()
    for m in MEDIA:
        print(f"    acts on {m:9s} {tot[m][0]} of {tot[m][1]} finished "
              f"across both groups")
    n_fin = sum(e["finished"] for e in by_arm.values())
    n_unf = sum(e["unfinished"] for e in by_arm.values())
    print(f"    {n_fin} of {n_fin + n_unf} seeds in the 2x2 have finished")

    unfinished = [r for r in rows if r["outcome"] == "unfinished" and r["arm"]]
    if unfinished:
        print(f"\n    the {len(unfinished)} unfinished seed(s), counted in "
              f"neither direction:")
        for r in sorted(unfinished, key=lambda x: x["run_id"]):
            print(f"      {r['run_id']:36s} {r['arm'][0]}/{r['arm'][1]}")

    inferred = [r for r in rows
                if r["outcome"] == "collapsed" and "inferred" in r["why"]]
    print(f"\n{len(inferred)} collapsed run(s) classified by accuracy rather "
          f"than by a recorded class count, being older than the field.")

    print("\nEvery family A cell with a finished run:")
    by_cell = rates(rows, "cell")
    named = {}
    for r in rows:
        named.setdefault(tuple(r["cell"]), []).append(r["run_id"])
    for k, e in sorted(by_cell.items(), key=lambda kv: named[kv[0]][0]):
        if e["finished"] == 0:
            continue
        ids = named[k]
        stem = ids[0].rstrip("0123456789").rstrip("_s") or ids[0]
        extra = f", {e['unfinished']} unfinished" if e["unfinished"] else ""
        # D-050: a rate from few draws invites a contrast it cannot support.
        # The warning goes next to the number rather than in a document,
        # because the number is what gets copied into a claim.
        thin = "   <-- under %d finished, any contrast needs its p" % SMALL_N             if e["finished"] < SMALL_N else ""
        print(f"  {stem:40s} {e['escaped']} of {e['finished']} escaped{extra}{thin}")

    if args.json:
        # Per-run rows as well as the counts, because the figures plot one
        # point per seed and a plot script is not allowed to work out which
        # run belongs to which arm. That decision is made here, where it is
        # tested, and the figure reads the answer.
        Path(args.json).write_text(
            json.dumps(
                {
                    "escape_tau": ESCAPE_TAU,
                    "collapse_acc": COLLAPSE_ACC,
                    "arms": {f"{g}/{m}": {kk: vv for kk, vv in
                                          by_arm[(g, m)].items()
                                          if kk != "runs"}
                             for g in GROUPS for m in MEDIA
                             if (g, m) in by_arm},
                    "runs": [
                        {
                            "run_id": r["run_id"],
                            "arm": f"{r['arm'][0]}/{r['arm'][1]}",
                            "seed": r["seed"],
                            "outcome": r["outcome"],
                            "accuracy": r["accuracy"],
                            "n_distinct": r["n_distinct"],
                            "modal": r["modal"],
                        }
                        for r in rows if r["arm"] is not None
                    ],
                    "neither": [r["run_id"] for r in odd],
                },
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
