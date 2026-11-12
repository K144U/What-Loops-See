"""F15. The rescue knows its group: every seed of every donor arm, three times over.

Added 2026-11-12 (D-080). Section 5.3's three tables as one picture. Each mark
is one run, by its stored outcome: solved, stuck at chance, or left chance
without solving. Rows are what the learner started from; columns are the
first runs and the two fresh sets whose tests were registered before the runs.

Everything comes from one stored file, `runs/_revision_verdicts_20261021/
timing.json`, written by the tested `escape_timing` module: each run's outcome
label from its `rows`, and each count printed beside the marks from its
`summary`. A plot script does not compute outcomes, so this draws the stored
labels and refuses if they disagree with the stored count, or if an arm is
missing. The module skips unfinished runs and stores only how many it
skipped, so every mark is a finished run.

    python -m loopvision.analysis.plots.f15_rescue \\
        --timing runs/_revision_verdicts_20261021/timing.json \\
        --out figures/f15_rescue
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NL = "\n"

#: What the learner started from: label, and the donor's tile as (substrate,
#: element) drawn by the renderer, or None for a cold start.
STARTS = (
    ("cold", "no donor:" + NL + "a cold start", None),
    ("same", "donor: the same task" + NL + "(D4 on colour)", ("d4_colour", 6)),
    ("other", "donor: the same group," + NL + "other picture (D4 on position)",
     ("native", 6)),
    ("wrong", "donor: a different group" + NL + "(S3 on colour)", ("native", 1)),
)

#: The three sets, and the stored arm of each start in each. The cold start
#: was not part of the registered donor tests, so it is in the first set only.
SETS = (
    ("first runs", {
        "cold": "famA_d2_d4colour",
        "same": "famA_d2_d4colour_curr [init_from]",
        "other": "famA_d2_d4colour_currctrl [init_from]",
        "wrong": "famA_d2_d4colour_curr_s3donor [init_from]"}),
    ("fresh seeds, first" + NL + "registered test", {
        "same": "famA_d2_d4colour_curr_confirm [init_from]",
        "other": "famA_d2_d4colour_currctrl_confirm [init_from]",
        "wrong": "famA_d2_d4colour_curr_s3donor_confirm [init_from]"}),
    ("fresh seeds, second" + NL + "registered test", {
        "same": "c4_curr [init_from]",
        "other": "c4_currctrl [init_from]",
        "wrong": "c4_s3donor [init_from]"}),
)

PER_LINE = 10


def _seed(run_id: str) -> int:
    m = re.search(r"_s(\d+)$", run_id)
    if not m:
        raise ValueError(f"{run_id}: no seed suffix")
    return int(m.group(1))


def arm(timing: dict, key: str) -> tuple[list[str], int, int]:
    """(outcomes in seed order, escaped, runs) for one stored arm."""
    from loopvision.analysis.plots import style

    rows = sorted((r for r in timing["rows"] if r["group"] == key),
                  key=lambda r: _seed(r["run_id"]))
    if key not in timing["summary"] or not rows:
        raise KeyError(f"no stored arm {key!r}")
    s = timing["summary"][key]
    outcomes = [r["outcome"] for r in rows]
    unknown = sorted(set(outcomes) - set(style.OUTCOME_MARKS))
    if unknown:
        raise ValueError(f"{key}: outcomes with no mark: {unknown}")
    if len(rows) != s["runs"] or outcomes.count("escaped") != s["escaped"]:
        raise ValueError(f"{key}: {outcomes.count('escaped')} of {len(rows)} drawn, "
                         f"{s['escaped']} of {s['runs']} stored")
    return outcomes, s["escaped"], s["runs"]


def draw(timing_path: Path, out: str):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from loopvision.analysis.plots import pictures as P
    from loopvision.analysis.plots import style

    style.use()
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    dx = 0.6
    block = PER_LINE * dx + 2.2
    x_sets = [10.6 + i * (block + 0.9) for i in range(len(SETS))]
    line_h = 0.62

    heights = {}
    for key, _, _ in STARTS:
        n = max(len(arm(timing, keys[key])[0]) for _, keys in SETS if key in keys)
        heights[key] = max(1.5, -(-n // PER_LINE) * line_h + 0.6)
    fig_h_units = sum(heights.values()) + 4.6
    width_units = x_sets[-1] + block + 0.3
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL,
                                    style.WIDTH_FULL * fig_h_units / width_units))
    P.blank_axes(ax, (0, width_units), (-fig_h_units + 3.0, 1.2))

    ax.text(0, 0.1, "Every learner is the two-step" + NL + "D4 colour task. It starts from:",
            fontsize=7, weight="bold", va="bottom", linespacing=1.05)
    for (title, _), x in zip(SETS, x_sets):
        ax.text(x, 0.1, title, fontsize=7, weight="bold", va="bottom", linespacing=1.05)

    y = -0.4
    for key, label, donor in STARTS:
        h = heights[key]
        top = y
        yc = top - h / 2
        if donor is None:
            ax.add_patch(Rectangle((0, yc - 0.5), 1.0, 1.0, fill=False, ec="#888888",
                                   lw=0.7, ls=(0, (2, 1.5))))
        else:
            P.draw_pixels(ax, P.tile(donor[1], donor[0]), 0, yc - 0.5, 1.0, frame=None)
        ax.text(1.5, yc, label, fontsize=6.6, va="center", linespacing=1.1)
        for (_, keys), x in zip(SETS, x_sets):
            if key not in keys:
                ax.text(x, yc, "not part of this test", fontsize=6, color="#888888",
                        va="center", style="italic")
                continue
            outcomes, k, n = arm(timing, keys[key])
            lines = -(-len(outcomes) // PER_LINE)
            y0 = yc + (lines - 1) * line_h / 2
            for i, o in enumerate(outcomes):
                P.outcome_mark(ax, x + 0.25 + (i % PER_LINE) * dx,
                               y0 - (i // PER_LINE) * line_h, o, size=5.2)
            ax.text(x + min(len(outcomes), PER_LINE) * dx + 0.25, yc, f"{k} of {n}",
                    fontsize=6.8, va="center", weight="bold")
        ax.plot([0, width_units - 0.3], [top - h, top - h], color="#e6e6e6", lw=0.5)
        y = top - h

    for i, (o, m) in enumerate(style.OUTCOME_MARKS.items()):
        xk = x_sets[0] + i * 6.2
        P.outcome_mark(ax, xk, y - 0.75, o, size=5.2)
        ax.text(xk + 0.4, y - 0.75, m["label"], fontsize=6.4, va="center")
    ax.text(0, y - 0.75, "one mark per run", fontsize=6.4, va="center", color="#555555")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return style.save(fig, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timing", type=Path,
                    default=Path("runs/_revision_verdicts_20261021/timing.json"))
    ap.add_argument("--out", default="figures/f15_rescue")
    args = ap.parse_args(argv)
    for p in draw(args.timing, args.out):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
