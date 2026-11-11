"""F6. Where each criterion stops, against both oracles and the cost of it.

paper.md section 6 calls this the practical figure: extrapolation curves
with both oracles marked and every criterion's firing point overlaid, so
the reader sees the gap without being told it is there.

The panel is two rows on one loop-count axis:

  top     held-out accuracy at each swept k, which is what stopping at a
          given loop actually buys
  bottom  where each criterion fires, median and interquartile range, with
          both oracles as vertical rules through both rows

WHY THE ACCURACY CURVE IS NOT A BACKDROP. It peaks at k=8, where these
models were trained, and falls away to 0.823 by k=64. Looping too long is
not merely wasted compute here, it costs accuracy, so a criterion that
fires late is not conservative. That is the consequence the gap has, and
without the curve the bottom row is a set of offsets with nothing at
stake.

THE TWO ROWS MUST NOT BE READ ACROSS, AND THE FIGURE SAYS SO ON ITS FACE.
The curve is accuracy when every item runs the same fixed k. The rules
below stop per item, at a different k for each one. So a rule whose median
firing point is 1 does NOT score the 0.722 the curve shows at k=1: it
stops at 1 on the items that are already right and later on the ones that
are not. Reading one row off the other confuses a fixed-k policy with an
adaptive one, and the adaptive one is the entire point of having a halting
criterion. That misreading was made once while drawing this figure, which
is why the caution is drawn rather than left to the caption.

What the stored data cannot give is the accuracy each rule actually
achieves, because `k*_model` is the smallest k at which an item is correct
and correctness is not monotone in k here: the curve falls after 8, so
being right at `k*_model` does not imply being right at a later `k_hat`.
Scoring the rules end to end needs a pass that evaluates each item at its
own firing point, which M5 does not currently store.

BOTH ORACLES, BECAUSE THEY DISAGREE. `k*_model` sits at 1 and `k*_task` at
4: this model is typically right after one pass on items whose generating
program is four deep. Prior work has only `k*_model`, and scoring a
criterion against it alone would call a rule that fires at 1 perfectly
calibrated while it stops four times short of the computation the task
describes. Which of those matters depends on the question being asked, so
the figure draws both and takes no side.

THE X AXIS IS LOGARITHMIC because the sweep is, {1,2,3,4,6,8,12,16,24,32,
48,64}. On a linear axis the eight points below k=8, where every criterion
fires and where the whole comparison happens, would occupy a tenth of the
width.

THIS SCRIPT COMPUTES NOTHING. Medians, quartiles and the averaged accuracy
curve are taken in `m5_h2 --summary`, where they are tested.

    python -m loopvision.analysis.m5_h2 --summary m5_summary.json
    python -m loopvision.analysis.plots.f6_halting --summary m5_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loopvision.analysis.plots import style

#: Printed names, in the preregistration's order. The second difference is
#: named for its source because it is the current best published rule and
#: the figure's headline is that it is the one that misses.
LABELS = {
    "update_norm": "update norm",
    "kl": "Kullback-Leibler",
    "entropy": "predictive entropy",
    "second_difference": "second difference\n(Pappone et al.)",
}

#: The two oracles, and how each is drawn. Neither is the reference: they
#: disagree, and the figure says so rather than choosing.
ORACLES = (
    ("k_model", "k*_model\nthis model is right"),
    ("k_task", "k*_task\nthe program's depth"),
)


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pick_task(summary: dict, task_key: str | None) -> tuple[str, dict]:
    tasks = summary.get("tasks") or {}
    if not tasks:
        raise KeyError("the summary carries no task")
    if task_key is not None:
        if task_key not in tasks:
            raise KeyError(f"no task {task_key!r}; have {sorted(tasks)}")
        return task_key, tasks[task_key]
    if len(tasks) > 1:
        # Pooling task definitions is the fault M1 shipped with once. The
        # figure will not choose one silently.
        raise KeyError(f"{len(tasks)} task definitions in the summary. Name "
                       f"one with --task: {sorted(tasks)}")
    key = next(iter(tasks))
    return key, tasks[key]


def order_for(summary: dict, task: dict) -> list[str]:
    """Criteria top to bottom, in the order the preregistration names them.

    Pulled out of main so it can be tested. It was inline, and the test
    that was supposed to cover it checked the constant in the summary and
    the label table instead of what the figure actually received, so
    sorting the criteria alphabetically passed every test. Alphabetical
    would put the best published rule in the middle of the panel for no
    reason, and the reader would have no way to know the order meant
    nothing.
    """
    order = summary.get("criteria_order")
    if order:
        return list(order)
    return sorted(task.get("criteria") or {})


def draw(task: dict, order: list[str]):
    import matplotlib.pyplot as plt

    style.use()
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(style.WIDTH_COLUMN * 1.5, 3.2), sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.16},
    )

    curve = task["accuracy_curve"]
    ks = [p["k"] for p in curve]
    acc = [p["accuracy"] for p in curve]
    s = style.series(0)
    ax.plot(ks, acc, **s)
    best = max(acc)
    ax.axhline(best, color=style.REFERENCE, linewidth=0.7,
               linestyle=(0, (2, 2)), zorder=0)
    ax.annotate(f"peak {best:.3f} at k={ks[acc.index(best)]}, "
                f"falling to {acc[-1]:.3f} by k={ks[-1]}",
                xy=(0.30, 0.05), xycoords="axes fraction", fontsize=5.5,
                color="#444444")
    ax.set_ylabel("held-out accuracy")
    ax.set_ylim(min(acc) - 0.06, 1.03)

    # Both oracles, through both rows, so the offsets are readable against
    # them without the reader tracking a legend entry across panels.
    for i, (key, label) in enumerate(ORACLES):
        o = task["oracles"].get(key)
        if not o:
            continue
        for a in (ax, ax2):
            a.axvline(o["median"], color=style.SERIES_STYLES[4 + i]["color"],
                      linewidth=1.0, linestyle=(0, (4, 2)), zorder=0)
        ax.annotate(label, xy=(o["median"], 1.02), xycoords=("data", "axes fraction"),
                    fontsize=5.5, ha="left", va="bottom",
                    color=style.SERIES_STYLES[4 + i]["color"],
                    annotation_clip=False)

    for j, c in enumerate(order):
        e = task["criteria"].get(c) or {}
        if e.get("median") is None:
            continue
        y = len(order) - 1 - j
        sc = style.series(j)
        ax2.plot([e["q25"], e["q75"]], [y, y], color=sc["color"],
                 linewidth=1.6, solid_capstyle="butt", zorder=2)
        ax2.plot([e["median"]], [y], marker=sc["marker"], color=sc["color"],
                 markersize=5, linestyle="none", zorder=3)
        note = ""
        if e.get("never_fired"):
            note = f", never fired on {e['never_fired']} items"
        # To the right of the bar, not above it. Above collided with the
        # next criterion's bar as soon as a label ran to two lines.
        ax2.annotate(LABELS.get(c, c) + note, xy=(e["q75"] * 1.18, y),
                     xycoords="data", fontsize=6, va="center", ha="left",
                     color=sc["color"])

    # Drawn, not left to the caption. A reader who takes a rule's median
    # off the curve above has confused a fixed-k policy with an adaptive
    # one, and that is the whole subject of the figure.
    ax2.annotate("rules stop per item, so a median here is not a point on "
                 "the curve above",
                 xy=(0.5, -0.62), xycoords=("axes fraction", "data"),
                 ha="center", va="center", fontsize=5.5,
                 color=style.REFERENCE)

    ax2.set_ylim(-0.95, len(order) - 0.15)
    ax2.set_yticks([])
    ax2.set_xscale("log")
    ax2.set_xticks(ks)
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_xlabel("loops")
    ax2.set_ylabel("where it stops")
    ax2.grid(True, axis="x")
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F6, halting against the oracles")
    ap.add_argument("--summary", required=True, type=Path,
                    help="json from m5_h2 --summary")
    ap.add_argument("--task", default=None, help="which task definition")
    ap.add_argument("--out", default="figures/f6_halting")
    args = ap.parse_args(argv)

    if not args.summary.is_file():
        print(f"no summary: {args.summary}")
        print("make one: python -m loopvision.analysis.m5_h2 "
              "--summary m5_summary.json")
        return 2

    summary = load(args.summary)
    try:
        key, task = pick_task(summary, args.task)
    except KeyError as e:
        print(f"{e}")
        return 2

    if not task.get("accuracy_curve"):
        print(f"{key} has no accuracy curve, so the bottom row would have "
              f"nothing at stake. Refusing to draw half the figure.")
        return 2

    order = order_for(summary, task)
    fig = draw(task, order)
    for f in style.save(fig, args.out):
        print(f"wrote {f}")

    print(f"task {key}: {task['n_seeds']} seeds, {task['n_items']} items")
    for k, _ in ORACLES:
        o = task["oracles"].get(k)
        if o:
            print(f"  {k:8s} median {o['median']:.1f}  IQR {o['q25']:.1f} "
                  f"to {o['q75']:.1f}")
    for c in order:
        e = task["criteria"].get(c) or {}
        if e.get("median") is None:
            continue
        print(f"  {c:20s} median {e['median']:.1f}  IQR {e['q25']:.1f} to "
              f"{e['q75']:.1f}  never fired on {e.get('never_fired', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
