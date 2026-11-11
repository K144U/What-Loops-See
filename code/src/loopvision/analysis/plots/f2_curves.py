"""F2. Loops needed against depth, and against breadth, on a shared y.

paper.md section 6 asks for both axes on the same panel, because the flat
breadth line is the contribution and it only reads as flat next to the
rising depth line.

IT IS TWO PANELS SHARING ONE Y, AND HERE IS WHY THE SPEC WAS NOT FOLLOWED.
The same-panel version was drawn first, with depth on the bottom axis and
breadth on a twin axis along the top. It misleads. Family C sweeps breadth
4 to 8, and those land at the same horizontal positions as depth 1 to 2.7
on the bottom axis, so the flat green line appears to run across the rising
blue one and a reader has no way to tell which axis it belongs to. The
overlapping interquartile bands make it worse.

Two panels on a shared y remove the ambiguity and still satisfy the reason
section 6 gave: the flat lines sit immediately beside the rising one at the
same scale, so flatness still reads as flatness. The requirement was one
panel and the reason was legibility; where they conflict the reason wins,
and this note records that so nobody has to rediscover the conflict.

THE STRONGEST LINE IS THE ONE INSIDE FAMILY B. That task sweeps depth 1 to
6 AND breadth 12 to 16 in the same runs, so its depth curve and its breadth
curve come from one set of weights, one task definition and one set of
seeds. Everything that could differ between two families is held fixed, and
depth still rises while breadth is flat. Family C is the second breadth
line: a task built to vary breadth alone, which shows the flatness is not
an artefact of family B's particular breadth range.

CENSORED CELLS ARE NOT PLOTTED AS A VALUE. A cell that never reaches tau is
recorded at 33, one past the measurable range, and drawing that as a k_min
would put a spike on the curve where there is no measurement. The quantiles
exclude them and the count is annotated where they occur, which is entirely
at depth 6. Excluding them is not neutral, since the cells that never reach
tau are the hard ones, so the depth curve is conditional on the cell being
solvable inside the sweep and the annotation says how often it was not.

THIS SCRIPT COMPUTES NOTHING. A median is a statistic and plot scripts here
do not take them. `m1_curves --summary` writes the quantiles and this reads
them.

    python -m loopvision.analysis.m1_curves --summary m1_summary.json --no-fit
    python -m loopvision.analysis.plots.f2_curves --summary m1_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loopvision.analysis.plots import style

#: The two task definitions that carry H1, and nothing else. Every other
#: task_key in the store is a family A run at one depth, or a family B run
#: from before D-028 moved the breadth range, and pooling any of them into
#: this figure would put a task redefinition on one axis.
DEPTH_TASK = "B d(1, 2, 3, 4, 5, 6) b(12, 13, 14, 15, 16)"
BREADTH_TASKS = (
    (DEPTH_TASK, "family B, breadth 12 to 16"),
    ("C d(1,) b(4, 5, 6, 7, 8)", "family C, breadth 4 to 8"),
)


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def series_for(summary: dict, task_key: str, axis: str):
    """The stored quantiles for one task on one axis, as plain lists."""
    task = summary["tasks"].get(task_key)
    if task is None:
        raise KeyError(f"no task {task_key!r} in the summary")
    rows = task["axes"].get(axis)
    if not rows:
        raise KeyError(f"task {task_key!r} does not sweep {axis}")
    return {
        "x": [r[axis] for r in rows],
        "median": [r["median"] for r in rows],
        "q25": [r["q25"] for r in rows],
        "q75": [r["q75"] for r in rows],
        "censored": [r["censored"] for r in rows],
        "n": [r["n"] for r in rows],
        "seeds": task["seeds"],
    }


def draw(depth, breadths, tau: float):
    import matplotlib.pyplot as plt

    style.use()
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(style.WIDTH_COLUMN * 1.55, 2.3), sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    s = style.series(0)
    ax.plot(depth["x"], depth["median"], label="family B", **s)
    ax.fill_between(depth["x"], depth["q25"], depth["q75"],
                    color=s["color"], alpha=0.15, linewidth=0)
    ax.set_xlabel("composition depth")
    ax.set_ylabel(f"loops to reach tau = {tau:.2f}")
    ax.set_xticks(depth["x"])
    ax.set_title("depth is swept", loc="left")
    ax.legend(loc="upper left", handlelength=2.2)

    for i, (b, label) in enumerate(breadths, start=1):
        sb = style.series(i)
        ax2.plot(b["x"], b["median"], label=label, **sb)
        ax2.fill_between(b["x"], b["q25"], b["q75"],
                         color=sb["color"], alpha=0.15, linewidth=0)
    ax2.set_xlabel("scene breadth")
    ax2.set_title("breadth is swept", loc="left")
    ax2.legend(loc="upper left", handlelength=2.2)

    ax.set_ylim(0, max(max(depth["q75"]), 4) + 0.6)

    # Every depth that has censored cells is named. This said "all at depth
    # {the worst}" until 2026-10-22, which was true of the pooled curve and
    # false of the registered arm alone, whose censored cells sit at depths
    # 5 and 6. D-074.
    total_censored = sum(depth["censored"])
    if total_censored:
        where = ", ".join(f"{c} at depth {x}"
                          for x, c in zip(depth["x"], depth["censored"]) if c)
        ax.annotate(
            f"{total_censored} of {sum(depth['n']) + total_censored} cells "
            f"never reach tau,\n{where}, excluded here",
            xy=(0.97, 0.03), xycoords="axes fraction", va="bottom",
            ha="right", fontsize=5.5, color=style.REFERENCE,
        )
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F2, loop-count curves")
    ap.add_argument("--summary", required=True, type=Path,
                    help="json from m1_curves --summary")
    ap.add_argument("--out", default="figures/f2_curves")
    args = ap.parse_args(argv)

    if not args.summary.is_file():
        print(f"no summary: {args.summary}")
        print("make one: python -m loopvision.analysis.m1_curves "
              "--summary m1_summary.json --no-fit")
        return 2

    summary = load(args.summary)
    try:
        depth = series_for(summary, DEPTH_TASK, "depth")
        breadths = [(series_for(summary, key, "breadth"), label)
                    for key, label in BREADTH_TASKS]
    except KeyError as e:
        print(f"{e}")
        return 2

    fig = draw(depth, breadths, summary["tau"])
    for f in style.save(fig, args.out):
        print(f"wrote {f}")

    print(f"depth  {DEPTH_TASK}: {depth['seeds']} seeds, "
          f"median {depth['median'][0]:.1f} at depth {depth['x'][0]} to "
          f"{depth['median'][-1]:.1f} at depth {depth['x'][-1]}, "
          f"{sum(depth['censored'])} censored")
    for b, label in breadths:
        flat = max(b["median"]) - min(b["median"])
        print(f"breadth {label}: {b['seeds']} seeds, median "
              f"{min(b['median']):.1f} to {max(b['median']):.1f}, "
              f"range {flat:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
