"""F13. One problem, as the model sees it, and its answer worked out.

Added 2026-11-12 (D-080) so that a reader who looks only at the figures still
learns what the task is. Everything drawn is real: the canvas is a sample from
`data.dataset.generate`, the worked steps are the group's own products from
`data.groups`, and every tile is the renderer's.

    (a) a depth 2, breadth 4 problem on its 8 by 8 grid of patches, with the
        row that is asked about outlined
    (b) that row worked out: each operator applied in turn, ending at the
        answer, once with the group moving the glyph and once with the same
        problem drawn with the group moving the colours

WHICH SAMPLE. Not chosen by eye. `pick` takes the first validation index
whose asked-about row has no "do nothing" anywhere, in its start, its
operators or its answer, since a step that changes nothing teaches nothing.
At the fixed master seed that is index 1. The rule is in code and tested, so
the sample cannot drift to a prettier one.

    python -m loopvision.analysis.plots.f13_anatomy --out figures/f13_anatomy
"""

from __future__ import annotations

import argparse

#: The task drawn: family A at depth 2 and breadth 4, D4 only.
DEPTH, BREADTH, SPLIT = 2, 4, "iid_val"

NL = "\n"


def config(substrate: str = "native"):
    from loopvision.data import dataset as D

    return D.TaskConfig(depths=(DEPTH,), breadths=(BREADTH,), factor="d4",
                        substrate=substrate)


def worked(sample) -> tuple[int, list[int], list[int], int]:
    """(start, operators, state after each operator, answer) of the asked row.

    The states are the running products, and the last one must be the label
    the generator stored, or this refuses.
    """
    from loopvision.data import groups as G
    from loopvision.data.family_a import parse_program

    initial, strips, queried, _, _ = parse_program(sample.program)
    start, ops = initial[queried], strips[queried]
    states = [G.multiply(p, start) for p in G.prefix_products(ops)]
    if states[-1] != sample.label:
        raise ValueError(f"index {sample.idx}: the worked answer is not the label")
    return start, ops, states, sample.label


def pick(limit: int = 1000) -> int:
    """The first index whose asked row has no identity in it."""
    from loopvision.data import dataset as D
    from loopvision.data import groups as G

    for i in range(limit):
        start, ops, _, answer = worked(D.generate("A", i, SPLIT, config()))
        if G.IDENTITY not in (start, *ops, answer):
            return i
    raise ValueError(f"no index below {limit} without an identity step")


def _wrap(words: str) -> str:
    """Break a long move name after its comma, so it fits under a tile."""
    return words.replace(", then ", "," + NL + "then ")


def panel_canvas(ax, sample, queried: int) -> None:
    from matplotlib.patches import Rectangle

    from loopvision.analysis.plots import pictures as P
    from loopvision.data import render

    size = 9.6
    x0, y0 = 0.4, 0.9
    step = size / render.GRID
    P.draw_pixels(ax, P.canvas_of(sample), x0, y0, size, grid=render.PATCH,
                  frame="#444444")
    top = y0 + size
    row_y = y0 + (render.GRID - 1 - queried) * step
    ax.add_patch(Rectangle((x0, row_y), size, step, fill=False, ec="#ffffff",
                           lw=1.1, ls=(0, (3, 2)), zorder=5))

    def bracket(c0: int, c1: int, text: str) -> None:
        xa, xb = x0 + c0 * step + 0.08, x0 + (c1 + 1) * step - 0.08
        ax.plot([xa, xa, xb, xb], [top + 0.15, top + 0.32, top + 0.32, top + 0.15],
                color="#333333", lw=0.7)
        ax.text((xa + xb) / 2, top + 0.42, text, ha="center", va="bottom", fontsize=6.4)

    bracket(render.STATE_COL, render.STATE_COL, "start")
    bracket(render.FIRST_OP_COL, render.FIRST_OP_COL + DEPTH - 1,
            "operators," + NL + "in order")
    bracket(render.MARKER_COL, render.MARKER_COL, "marker")
    ax.text(x0, y0 - 0.25,
            NL.join(("Each row is one object. The marker picks the",
                     "row that is asked about (outlined); the other",
                     "rows are distractors with their own operators.")),
            ha="left", va="top", fontsize=6.2, color="#333333")
    ax.text(x0, 12.7, "(a) One problem, as the model sees it", ha="left",
            va="bottom", fontsize=8)


def panel_steps(ax, start: int, ops: list[int], states: list[int], answer: int) -> None:
    from matplotlib.patches import Rectangle

    from loopvision.analysis.plots import pictures as P

    big, small = 2.0, 1.15
    xs = [13.2, 18.4, 23.6]
    lines = (("native", 8.1, "the group moves the glyph"),
             ("d4_colour", 1.7, "the same problem, the group moving the colours"))
    ax.text(12.4, 12.7, "(b) The asked-about row, worked out", ha="left", va="bottom",
            fontsize=8)
    for substrate, y, title in lines:
        ax.text(12.4, y + big + 1.6, title, ha="left", va="bottom", fontsize=6.8,
                style="italic", color="#333333")
        tiles = [start] + states
        names = (["start"] + ["after one move"] * (len(states) - 1) + ["answer"])
        for x, g, name in zip(xs, tiles, names):
            P.draw_pixels(ax, P.tile(g, substrate), x, y, big)
            if name == "answer":
                ax.add_patch(Rectangle((x - 0.12, y - 0.12), big + 0.24, big + 0.24,
                                       fill=False, ec="#c99a00", lw=1.4, zorder=6))
            ax.text(x + big / 2, y - 0.2, name, ha="center", va="top", fontsize=6.2,
                    weight="bold" if name == "answer" else "normal")
            ax.text(x + big / 2, y - 0.75, _wrap(P.d4_words(g)), ha="center", va="top",
                    fontsize=5.8, color="#444444", linespacing=1.1)
        for k, op in enumerate(ops):
            xa, xb = xs[k] + big + 0.3, xs[k + 1] - 0.3
            ax.annotate("", xy=(xb, y + big / 2), xytext=(xa, y + big / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333333",
                                        shrinkA=0, shrinkB=0))
            xm = (xa + xb) / 2
            P.draw_pixels(ax, P.tile(op, substrate), xm - small / 2, y + big / 2 + 0.25,
                          small)
            ax.text(xm, y + big / 2 + 0.25 + small + 0.12, _wrap(P.d4_words(op)),
                    ha="center", va="bottom", fontsize=5.6, color="#444444",
                    linespacing=1.1)
    ax.text(12.4, -0.75,
            "An operator is drawn as the plain glyph after that move. The answer is"
            + NL + "one of the eight symmetries of a square, so a guess scores 1 in 8.",
            ha="left", va="top", fontsize=6.2, color="#333333")


def draw(out: str):
    import matplotlib.pyplot as plt

    from loopvision.analysis.plots import pictures as P
    from loopvision.analysis.plots import style
    from loopvision.data import dataset as D
    from loopvision.data.family_a import parse_program

    style.use()
    index = pick()
    sample = D.generate("A", index, SPLIT, config())
    start, ops, states, answer = worked(sample)
    queried = parse_program(sample.program)[2]
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 3.4))
    P.blank_axes(ax, (0, 27.0), (-2.3, 13.5))
    panel_canvas(ax, sample, queried)
    panel_steps(ax, start, ops, states, answer)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return style.save(fig, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="figures/f13_anatomy")
    args = ap.parse_args(argv)
    for p in draw(args.out):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
