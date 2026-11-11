"""F3. Patching over (loop, patch position), one column per family.

paper.md section 6: the lead figure and the one a reviewer remembers. It
carries both halves of H3 in one picture, which is why it is two rows and
not one:

  grid    the (iteration, position) heatmap, where the spatial spread
          lives, on one colour scale so the families can be compared
  margin  full-state recovery against the same iteration axis, sharing it
          with the grid, which is where the loop localisation lives. Family
          A is flat until the last pass and then jumps to 1.0; the breadth
          family rises through every pass

WHAT THE SCALE HAD TO BE, and why the obvious choice is wrong. Recovery
runs to 1.0, so a heatmap drawn on 0 to 1 is the natural first guess. It
produces three near-blank panels, because no single position recovers more
than 0.081 in any family while the full state recovers exactly 1.0. That
gap is not a plotting nuisance, it is the result: the computation is spread
across positions and no one patch site carries it. So the grid is scaled to
the positions and the full state is drawn separately, in the margin, on
its own axis. Putting them on one scale would hide the finding inside the control.

POSITION -1 IS NOT A POSITION. It is the full-state patch, the positive
control that says the instrument landed at all, and h3_contrast keys its
instrument-failure check on it. Drawing it in the grid would add a
saturated column that no reader could tell from a real patch site.

EXPLORATORY AND CONFIRMATORY ARE NEVER MIXED. Section 0 of the frozen
preregistration excludes the three `m2_patching` grids from the H3
confirmation set, and `h3_contrast` enforces that by being unable to load
them. This figure can draw either set and refuses to combine them, and the
panel says which it drew. On 2026-09-09 the confirmation set holds family B
only, at five seeds, so the between-family contrast H3 registers cannot be
drawn from it at all and `--source confirm` will say so rather than
producing a one column figure that looks like the result.

    python -m loopvision.analysis.plots.f3_patching --out figures/f3_patching
    python -m loopvision.analysis.plots.f3_patching --source confirm
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from loopvision.analysis.plots import style

#: Gamma on the shared colour scale. A DISPLAY TRANSFORM AND NOT A STATISTIC.
#:
#: The three families differ by roughly six times in single position recovery,
#: family A peaking near 0.12 and family C near 0.02. On a linear shared scale
#: family C renders almost black, and family C being the spatially uniform case
#: is the surviving half of H3, so the linear scale draws the finding in the way
#: that makes it hardest to see. Uniform and small are different things and a
#: linear scale collapses them.
#:
#: Below 1 this lifts the low end: family C's peak goes from 18 percent of the
#: colour range to about 42. The scale stays shared, so the families remain
#: comparable, which is what the shared scale exists for.
#:
#: THE COST, WHICH THE CAPTION MUST CARRY. A compressed scale makes small values
#: look larger than they are. The colourbar therefore keeps its ticks in data
#: units, where uneven spacing is the visible signal that the mapping is not
#: linear, and the caption names the gamma. Pass --gamma 1.0 for the linear
#: scale. The array itself is never transformed; only the mapping to colour is.
GAMMA = 0.5

#: The file each source reads. Never both.
SOURCES = {
    "exploratory": "m2_patching.parquet",
    "confirm": "m2_confirm.parquet",
}

#: Column order, and what each family is for. Depth families first, because
#: the prediction is that they are the localised ones and the breadth family
#: is the contrast.
FAMILY_ORDER = ("A", "B", "C")
FAMILY_ROLE = {
    "A": "depth, composition",
    "B": "depth, chain",
    "C": "breadth",
}

#: The full-state patch, which is the positive control and not a position.
CONTROL_POSITION = -1

#: Sequential, perceptually uniform and monotonic in luminance, so the grid
#: survives a greyscale print the way the series colours do.
CMAP = "magma"


def load(runs_root: Path, source: str) -> list[dict]:
    """One entry per family, from the chosen source only."""
    import pandas as pd

    filename = SOURCES[source]
    frames = []
    for p in sorted(glob.glob(str(runs_root / "*" / filename))):
        df = pd.read_parquet(p)
        if len(df) == 0:
            continue
        frames.append(df)
    if not frames:
        return []

    out = []
    allf = pd.concat(frames, ignore_index=True)
    for fam, g in allf.groupby("family"):
        grid = g[g.position >= CONTROL_POSITION + 1]
        ctl = g[g.position == CONTROL_POSITION]
        # Seeds are averaged only when the source is the confirmation set,
        # which is what it is for. The exploratory grids are one run each
        # and a mean over one run is that run.
        pivot = grid.pivot_table(index="iteration", columns="position",
                                 values="recovery_mean", aggfunc="mean")
        ctl_by_iter = ctl.groupby("iteration").recovery_mean.mean()
        query_positions = sorted(
            g.loc[g.is_query_token, "position"].unique().tolist()
        )
        out.append({
            "family": str(fam),
            "runs": sorted(g.run_id.unique().tolist()),
            "n_seeds": int(g.run_id.nunique()),
            "iterations": [int(i) for i in pivot.index],
            "positions": [int(c) for c in pivot.columns],
            "grid": pivot.to_numpy(dtype=float),
            "control": [float(v) for v in ctl_by_iter.values],
            "control_iterations": [int(i) for i in ctl_by_iter.index],
            "query_positions": query_positions,
            "n_items": int(g.n_items.max()),
        })
    # Depth families first, because the prediction is that they are the
    # localised ones and the breadth family is the contrast.
    #
    # This sort is a no-op today and is kept deliberately. groupby returns
    # families alphabetically and FAMILY_ORDER is alphabetical, so deleting
    # it changes nothing for A, B and C: mutation testing confirmed that
    # removing it entirely leaves every test green. What it does carry is
    # the "else 99", which keeps a family nobody added to FAMILY_ORDER at
    # the end rather than between the registered ones, where it would read
    # as part of the contrast. That half is tested.
    out.sort(key=lambda d: FAMILY_ORDER.index(d["family"])
             if d["family"] in FAMILY_ORDER else 99)
    return out


def draw(panels: list[dict], source: str, gamma: float = GAMMA):
    import matplotlib.pyplot as plt

    style.use()
    n = len(panels)
    fig = plt.figure(figsize=(style.WIDTH_FULL, 2.5))
    # Each family is a heatmap with a marginal on its right, and the two
    # SHARE the iteration axis. An earlier version stacked the marginal
    # above the grid, which put iteration on x in one panel and on y in the
    # panel directly below it. Sharing the axis is what lets a reader read
    # the jump in the margin straight across into the row it belongs to.
    # Grid and margin per family, a spacer between families, and one
    # column reserved for the colourbar. Letting matplotlib steal space
    # from the axes list put the bar between the last grid and its own
    # margin, which read as a fourth panel.
    ratios = []
    for j in range(n):
        ratios += [5, 1.5]
        if j < n - 1:
            ratios += [0.9]
    ratios += [0.5]
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.12)
    cbar_col = len(ratios) - 1

    # One scale across every family, or the panels cannot be compared, and
    # comparing them is the entire point of the figure.
    vmax = max(float(np.nanmax(p["grid"])) for p in panels)

    # Display transform only. p["grid"] is handed to imshow untouched, so
    # every number in this figure is still the number in the parquet.
    from matplotlib.colors import PowerNorm

    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)

    im = None
    grid_axes = []
    for j, p in enumerate(panels):
        ax_g = fig.add_subplot(gs[0, 3 * j])
        ax_c = fig.add_subplot(gs[0, 3 * j + 1], sharey=ax_g)
        grid_axes.append(ax_g)

        first_query = min(p["query_positions"]) if p["query_positions"] else None
        im = ax_g.imshow(p["grid"], aspect="auto", origin="lower",
                         cmap=CMAP, norm=norm,
                         extent=(min(p["positions"]) - 0.5,
                                 max(p["positions"]) + 0.5,
                                 min(p["iterations"]) - 0.5,
                                 max(p["iterations"]) + 0.5))
        if first_query is not None:
            # The divider marks where image tokens end and query tokens
            # begin. An in-panel word for it was clipped by the panel edge
            # at this width, so the axis label carries it instead.
            ax_g.axvline(first_query - 0.5, color="white", linewidth=0.8)
        ax_g.set_xlabel("patch position")
        ax_g.set_yticks(p["iterations"][::2])
        ax_g.set_title(f"family {p['family']}, {FAMILY_ROLE.get(p['family'], '')}",
                       loc="left")
        if j == 0:
            ax_g.set_ylabel("loop iteration")
        else:
            ax_g.set_yticklabels([])
        ax_g.grid(False)

        ax_c.plot(p["control"], p["control_iterations"], **style.series(j))
        ax_c.set_xlim(-0.08, 1.12)
        ax_c.set_xticks([0, 1])
        ax_c.set_xlabel("full\nstate")
        ax_c.tick_params(labelleft=False)
        ax_c.grid(True, axis="x")

    cax = fig.add_subplot(gs[0, cbar_col])
    cbar = fig.colorbar(im, cax=cax)
    # Ticks stay in data units. Under a gamma they come out unevenly spaced,
    # and that unevenness is the reader's cue that the mapping is compressed.
    cbar.set_ticks([t for t in np.linspace(0.0, vmax, 5)])
    cbar.set_label("recovery at one position", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Wrapped by hand rather than left to run on. The caption is drawn
    # outside the axes and the figure is saved with a tight bounding box, so
    # one long line stretches the saved image to fit it: lengthening this
    # text to declare the gamma took the PNG from 4139 to 6802 pixels wide
    # and shrank the panels to compensate.
    caption = []
    if gamma != 1.0:
        caption.append(
            f"{source} grids. Colour is gamma {gamma:g} on a shared scale, so "
            f"low recovery is lifted")
        caption.append(
            "and the colourbar ticks are unevenly spaced. Magnitudes read off "
            "the bar, not off the brightness.")
    else:
        caption.append(f"{source} grids.")
    caption.append(
        f"Single position recovery peaks at {vmax:.3f}; the full state "
        f"recovers 1.000, so no one position")
    caption.append(
        "carries the computation. The grid and the margin are on different "
        "scales for that reason.")
    caption.append("The white rule separates image tokens from query tokens.")

    fig.text(0.005, -0.10, "\n".join(caption),
             fontsize=6, color="#444444", va="top")
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F3, patching heatmaps")
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    ap.add_argument("--source", choices=sorted(SOURCES), default="exploratory")
    ap.add_argument("--out", default="figures/f3_patching")
    ap.add_argument("--gamma", type=float, default=GAMMA,
                    help="display gamma on the shared colour scale. "
                         "1.0 is linear. Never changes the data.")
    args = ap.parse_args(argv)

    panels = load(args.runs, args.source)
    if not panels:
        print(f"no {SOURCES[args.source]} under {args.runs}")
        return 2

    families = [p["family"] for p in panels]
    if len(families) < 2:
        # H3 is a contrast between families. One column is not a contrast,
        # and a figure showing one would read as the result.
        print(f"the {args.source} set holds family {families[0]} only "
              f"({panels[0]['n_seeds']} seed(s)). H3 is a contrast between a "
              f"depth family and a breadth family, so one column cannot show "
              f"it. Refusing to draw a figure that would look like the "
              f"result.")
        return 1

    fig = draw(panels, args.source, gamma=args.gamma)
    for f in style.save(fig, args.out):
        print(f"wrote {f}")

    print(f"source: {args.source}")
    for p in panels:
        peak_it = p["control_iterations"][int(np.argmax(p["control"]))]
        print(f"  family {p['family']:1s}  {p['n_seeds']} seed(s), "
              f"{p['n_items']} items, {len(p['positions'])} positions, "
              f"grid max {np.nanmax(p['grid']):.4f}, "
              f"full state peaks at iteration {peak_it}")
    if args.source == "exploratory":
        print("These grids are excluded from the H3 confirmation set by "
              "Section 0 of the preregistration. The figure is exploratory "
              "and the caption must say so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
