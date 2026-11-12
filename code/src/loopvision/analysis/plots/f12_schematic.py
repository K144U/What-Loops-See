"""F12. The study in one picture: two pictures of one group, the model, the rescue.

A schematic, added 2026-11-12 so a reader meets the setup as a picture
before the prose. It plots no measured number, so it reads no file. Since
D-080 the tiles of (a) are the renderer's own, drawn by `pictures.tile`: the
first version drew a glyph the renderer never draws, which beside the real
tiles of the other figures would have shown a reader two different glyphs.

    (a) the same group element acting on position and on colour
    (b) the looped model and its feedforward baselines
    (c) the rescue experiment: which donor starts which learner

    python -m loopvision.analysis.plots.f12_schematic --out figures/f12_schematic
"""

from __future__ import annotations

import argparse

#: The two elements panel (a) shows: doing nothing and a quarter turn, as
#: indices of the full group with the colour part left alone.
IDENTITY, QUARTER_TURN = 0, 12

NL = "\n"


def outline(g: int, substrate: str) -> frozenset:
    """The cells a tile fills, colour ignored: what a reader calls its outline."""
    from loopvision.analysis.plots import pictures as P

    t = P.tile(g, substrate)
    return frozenset((r, c) for r in range(t.shape[0]) for c in range(t.shape[1])
                     if tuple(int(v) for v in t[r, c]) != tuple(P.BACKGROUND))


def _tile(ax, x0, y0, size, g, substrate):
    from loopvision.analysis.plots import pictures as P

    P.draw_pixels(ax, P.tile(g, substrate), x0, y0, size)


def _arrow(ax, x0, y0, x1, y1, text=None, below=False):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333333",
                                shrinkA=0, shrinkB=0))
    if text:
        dy = -0.04 if below else 0.04
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center",
                va="top" if below else "bottom", fontsize=6.3, color="#333333")


def _box(ax, x, y, w, h, text, fc="#eef2f8", fs=6.6):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.02",
                                fc=fc, ec="#555555", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def panel_substrates(ax):
    ax.set_title("(a) One group, two kinds of picture", fontsize=8, loc="left")
    ax.text(0.0, 0.93, "On position, the element turns the glyph," + NL
            + "so its outline changes.", fontsize=6.3, va="top")
    _tile(ax, 0.08, 0.45, 0.24, IDENTITY, "native")
    _arrow(ax, 0.38, 0.57, 0.60, 0.57, "quarter turn", below=True)
    _tile(ax, 0.66, 0.45, 0.24, QUARTER_TURN, "native")
    ax.text(0.0, 0.34, "On colour, the same element moves the" + NL
            + "colours; the outline never changes.", fontsize=6.3, va="top")
    _tile(ax, 0.08, -0.12, 0.24, IDENTITY, "d4_colour")
    _arrow(ax, 0.38, 0.0, 0.60, 0.0, "quarter turn", below=True)
    _tile(ax, 0.66, -0.12, 0.24, QUARTER_TURN, "d4_colour")


def panel_model(ax):
    ax.set_title("(b) The models", fontsize=8, loc="left")
    x, w, h = 0.04, 0.40, 0.13
    stack = (("image patches", "#ffffff"), ("prelude", "#eef2f8"), ("core", "#dfe7f3"),
             ("coda", "#eef2f8"), ("answer", "#ffffff"))
    ys = [0.80 - i * 0.215 for i in range(len(stack))]
    for (label, fc), y in zip(stack, ys):
        _box(ax, x, y, w, h, label, fc=fc)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        _arrow(ax, x + w / 2, y0, x + w / 2, y1 + h)
    yc = ys[2]
    ax.annotate("", xy=(x + w, yc + h * 0.85), xytext=(x + w, yc + h * 0.15),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#c1442a",
                                connectionstyle="arc3,rad=-1.4", shrinkA=0, shrinkB=0))
    ax.text(x + w + 0.13, yc + h / 2, NL.join(("looped:", "the same core", "run k times")),
            fontsize=6.3, va="center", color="#c1442a")
    ax.text(x + w + 0.13, ys[4] + 0.18,
            NL.join(("feedforward:", "k different cores", "stacked, matched",
                     "in compute or in", "parameter count")),
            fontsize=6.0, va="center", color="#333333")


def panel_rescue(ax):
    ax.set_title("(c) The rescue experiment", fontsize=8, loc="left")
    _box(ax, 0.05, 0.74, 0.90, 0.15, "donor: a model trained" + NL + "on the one-step task")
    _arrow(ax, 0.50, 0.74, 0.50, 0.62)
    ax.text(0.53, 0.68, "copy its weights", fontsize=6.3, va="center", color="#333333")
    _box(ax, 0.05, 0.47, 0.90, 0.15, "learner: the two-step" + NL + "D4 colour task")
    ax.text(0.05, 0.37, "donor", fontsize=6.4, weight="bold", va="top")
    ax.text(0.95, 0.37, "the learner is", fontsize=6.4, weight="bold", va="top", ha="right")
    rows = (("same task" + NL + "(D4 on colour)", "rescued"),
            ("same group, other picture" + NL + "(D4 on position)", "rescued"),
            ("different group" + NL + "(S3 on colour)", "no better than" + NL + "a cold start"))
    for i, (donor, result) in enumerate(rows):
        yy = 0.20 - i * 0.165
        ax.text(0.05, yy, donor, fontsize=6.2, va="center")
        ax.text(0.95, yy, result, fontsize=6.2, va="center", ha="right")


def draw(out: str):
    import matplotlib.pyplot as plt

    from loopvision.analysis.plots import style

    style.use()
    fig, axes = plt.subplots(1, 3, figsize=(style.WIDTH_FULL, 2.9),
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.16, 1.0)
        ax.axis("off")
    # Square pixels: the renderer's tiles must not be stretched. "datalim"
    # widens the limits instead of shrinking the panel, so the layout holds.
    axes[0].set_aspect("equal", adjustable="datalim")
    panel_substrates(axes[0])
    panel_model(axes[1])
    panel_rescue(axes[2])
    fig.tight_layout(w_pad=1.5)
    return style.save(fig, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="figures/f12_schematic")
    args = ap.parse_args(argv)
    for p in draw(args.out):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
