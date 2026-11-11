"""F4. Coda-lens trajectories: the answer sharpens, the intermediates never do.

One line per hop, accuracy against loop iteration, decoded through the real
output head. For a depth 6 task the last hop IS the answer, and hops 1 to 5
are the intermediate composites the task passes through by construction.

**The shape is the result, which is why this plots trajectories rather than a
scalar.** Section 6 of `paper.md` asks for exactly that: report whether the
curve is smooth or discontinuous, do not summarise it away. A single
"hop rate" number would say 0.0 here and hide that the final answer rises
from the floor to near 1.0 while every intermediate stays flat. Those are
different claims and only the picture separates them.

This replicates Lu et al. arXiv 2507.02199 in a different modality, and the
draft says so. What is ours is that the intermediates are known by
construction, so a flat line here is a measurement of absence rather than an
inability to decode. `f8_absent` carries the probe that makes that
distinction; this figure carries the shape.

Reads the json written by `analysis/coda_lens` and computes nothing.

    python -m loopvision.analysis.plots.f4_coda --lens runs/<id>/coda_lens.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loopvision.analysis.plots import style


def draw(d: dict, out: Path | None = None):
    import matplotlib.pyplot as plt

    grid = d["grid"]
    n_iter = len(grid)
    n_hop = len(grid[0])
    iters = list(range(n_iter))

    style.use()
    fig, ax = plt.subplots(figsize=(style.WIDTH_COLUMN, 2.6))

    for h in range(n_hop):
        series = [grid[i][h] for i in iters]
        final = h == n_hop - 1
        st = style.series(0 if final else 1)
        ax.plot(iters, series,
                color=st["color"] if final else style.REFERENCE,
                marker=st["marker"] if final else None,
                markersize=3.2 if final else 0,
                linestyle=st["linestyle"] if final else "-",
                linewidth=1.6 if final else 0.9,
                alpha=1.0 if final else 0.75,
                label=(f"hop {h + 1}, the answer" if final
                       else ("hops 1 to %d, the intermediates" % (n_hop - 1)
                             if h == 0 else None)),
                zorder=3 if final else 2)

    # Chance for this task, from the stored file rather than assumed.
    chance = d.get("chance")
    if chance is None and grid:
        # The first iteration decodes nothing, so its mean is the floor the
        # intermediates sit on. Stated, not silently used as a statistic:
        # it is a reference line, and the caption says where it came from.
        chance = sum(grid[0]) / len(grid[0])
    style.chance_line(ax, chance, "floor at iteration 0")

    ax.set_xlabel("loop iteration")
    ax.set_ylabel("decode accuracy")
    ax.set_xlim(-0.2, n_iter - 0.8)
    ax.set_ylim(-0.04, 1.06)
    ax.set_title(
        f"depth {d['depth']}: the answer sharpens, the intermediates do not",
        loc="left", pad=8)
    # Lower right: the only quadrant both curves leave empty. Centre left
    # put it on top of the annotation below.
    ax.legend(loc="lower right", fontsize=6, frameon=False,
              handletextpad=0.5, borderpad=0.2)

    last = grid[-1]
    ax.annotate(
        f"at the last iteration: answer {last[-1]:.3f},\n"
        f"deepest intermediate {max(last[:-1]):.3f}",
        xy=(0.97, 0.62), xycoords="axes fraction", ha="right", va="top",
        fontsize=6, color="#444444")

    if out is not None:
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    else:
        for p in style.save(fig, "f4_coda"):
            print(f"wrote {p}")
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F4 coda-lens trajectories")
    ap.add_argument("--lens", required=True, type=Path,
                    help="a coda_lens.json written by analysis/coda_lens")
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args(argv)

    d = json.loads(args.lens.read_text(encoding="utf-8"))
    if not d.get("grid"):
        print("no grid in that file. Nothing is drawn and nothing is claimed.")
        return 2
    if len(d["grid"][0]) < 2:
        print("a depth 1 task has no intermediate hops, so this figure has "
              "nothing to show. Use a deeper run.")
        return 2

    draw(d, args.out)
    last = d["grid"][-1]
    print(f"{d['run']}: depth {d['depth']}, k {d['k']}, {d['n']} items. "
          f"Answer {last[-1]:.4f}, deepest intermediate {max(last[:-1]):.4f}, "
          f"hop_rate {d.get('hop_rate')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
