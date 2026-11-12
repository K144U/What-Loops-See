"""The renderer's own tiles and canvases, drawn as squares in a figure.

Added 2026-11-12 (D-080) for the illustrations. Every picture of a stimulus in
the paper is drawn from `data.render`, never redrawn by hand, so a reader who
sees a tile in a figure is seeing the pixels a model was trained on. A
hand-drawn stand-in had already crept into Figure 12, a glyph the renderer
never draws.

Pixels are drawn as filled squares rather than with `imshow`, because a 4 by 4
image shown large is smoothed by some viewers and the point of these figures
is that each pixel can be counted.

Nothing here reads a run. It draws pictures and computes no statistic.
"""

from __future__ import annotations

import numpy as np

#: The canvas colour, from the renderer, so empty space reads as the model's.
from loopvision.data.render import BACKGROUND, PATCH  # noqa: F401


def tile(index: int, substrate: str = "native") -> np.ndarray:
    """One group element as the renderer draws it, (4, 4, 3) uint8."""
    from loopvision.data import render

    return render.to_hwc(render.render_element(index, substrate=substrate))


def canvas_of(sample) -> np.ndarray:
    """A generated sample's image, (H, W, 3) uint8."""
    from loopvision.data import render

    return render.to_hwc(np.asarray(sample.image))


def draw_pixels(ax, rgb: np.ndarray, x0: float, y0: float, size: float,
                grid: int | None = None, grid_colour: str = "#3a3a3a",
                frame: str | None = "#555555", alpha: float = 1.0,
                zorder: float = 2) -> None:
    """Draw an image with its lower left corner at (x0, y0), `size` wide.

    `grid` draws a line every `grid` pixels, for the patch boundaries of a
    canvas. Row 0 of the image is drawn at the top, as it is displayed.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    h, w = rgb.shape[:2]
    s = size / max(h, w)
    squares, colours = [], []
    for r in range(h):
        for c in range(w):
            squares.append(Rectangle((x0 + c * s, y0 + (h - 1 - r) * s), s, s))
            colours.append(tuple(int(v) / 255 for v in rgb[r, c]))
    ax.add_collection(PatchCollection(squares, facecolors=colours, edgecolors="none",
                                      linewidths=0, alpha=alpha, zorder=zorder))
    if grid:
        for k in range(grid, max(h, w), grid):
            ax.plot([x0 + k * s] * 2, [y0, y0 + h * s], color=grid_colour, lw=0.4,
                    zorder=zorder + 0.1, solid_capstyle="butt")
            ax.plot([x0, x0 + w * s], [y0 + k * s] * 2, color=grid_colour, lw=0.4,
                    zorder=zorder + 0.1, solid_capstyle="butt")
    if frame:
        ax.add_patch(Rectangle((x0, y0), w * s, h * s, fill=False, ec=frame, lw=0.6,
                               zorder=zorder + 0.2))


def blank_axes(ax, xlim, ylim) -> None:
    """An axes used as a drawing board: equal scale, no ticks, no frame."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def outcome_mark(ax, x: float, y: float, outcome: str, size: float = 5.5,
                 zorder: float = 3) -> None:
    """One run's outcome, by shape and fill first and colour second."""
    from loopvision.analysis.plots import style

    m = style.OUTCOME_MARKS[outcome]
    ax.plot([x], [y], marker=m["marker"], markersize=size, linestyle="none",
            markerfacecolor=m["color"] if m["filled"] else "white",
            markeredgecolor=m["color"], markeredgewidth=1.0, zorder=zorder)


def d4_words(index: int) -> str:
    """A D4 element in words, from its (rotations, flip) code."""
    from loopvision.data import groups as G

    rot, flip = G.d4_of(index)
    turns = {0: "", 1: "a quarter turn", 2: "a half turn", 3: "three quarter turns"}[rot]
    if not flip:
        return turns or "do nothing"
    return "a flip" + (f", then {turns}" if turns else "")
