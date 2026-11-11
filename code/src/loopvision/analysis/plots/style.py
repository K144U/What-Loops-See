"""Shared figure style, so seven figures look like one paper.

CLAUDE.md: every plotted number must be readable from a stored metrics file,
and plot scripts do not compute statistics. This module holds no numbers at
all. It sets sizes, colours and saving, and nothing here reads a run.

GREYSCALE IS NOT A NICETY. Section 6 of paper.md requires F1 to survive
greyscale, and reviewers print. A figure that separates its series only by
hue is unreadable the moment it is printed or seen by the roughly one in
twelve men with a red green deficiency, and neither reader will write to
tell us. So SERIES_STYLES pairs every colour with a marker and a dash
pattern, and the colours are ordered by luminance as well as hue, which
means a series stays distinguishable when the colour is thrown away. Use
`series(i)` rather than picking a colour, and the redundancy comes free.

NeurIPS is a two column format. A figure drawn at an arbitrary size and
scaled into the column has fonts that no longer match the body text, so the
widths here are the real ones and figures are drawn at final size.

    from loopvision.analysis.plots import style
    style.use()
    fig, ax = plt.subplots(figsize=(style.WIDTH_COLUMN, 2.2))
    ...
    style.save(fig, "figures/f9_binding")
"""

from __future__ import annotations

from pathlib import Path

#: Real widths in inches, so a figure is drawn at the size it is printed.
WIDTH_COLUMN = 3.25
WIDTH_FULL = 6.75

#: Colour, marker and dash together. Any one of the three identifies a
#: series on its own, so the figure survives greyscale, a colourblind
#: reader, and a low resolution screenshot.
#:
#: Luminance is deliberately spread: the ordering below is roughly dark to
#: light under a greyscale conversion, so adjacent series never collapse to
#: the same grey.
SERIES_STYLES = (
    {"color": "#1b3a6b", "marker": "o", "linestyle": "-", "name": "deep blue"},
    {"color": "#c1442a", "marker": "s", "linestyle": "--", "name": "brick"},
    {"color": "#2e7d5b", "marker": "^", "linestyle": "-.", "name": "green"},
    {"color": "#8a6d1f", "marker": "D", "linestyle": ":", "name": "ochre"},
    {"color": "#6b4c9a", "marker": "v", "linestyle": (0, (5, 1)), "name": "purple"},
    {"color": "#767676", "marker": "X", "linestyle": (0, (1, 1)), "name": "grey"},
)

#: For a reference line that is not a series: a chance floor, a threshold,
#: an oracle. Never used for data.
REFERENCE = "#9a9a9a"

#: The one place an outcome gets a colour, so escaped and collapsed read
#: the same way in every figure that shows both.
OUTCOME = {
    "escaped": SERIES_STYLES[2]["color"],
    "collapsed": SERIES_STYLES[1]["color"],
    "neither": SERIES_STYLES[3]["color"],
    "unfinished": REFERENCE,
}


def series(i: int) -> dict:
    """Colour, marker and dash for series i, cycling if there are many."""
    s = dict(SERIES_STYLES[i % len(SERIES_STYLES)])
    s.pop("name")
    return s


def use() -> None:
    """Set the rcParams. Called by every figure script before it draws."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        # Type 3 fonts are rejected by several camera ready checkers, and
        # the failure arrives at the deadline rather than now.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def chance_line(ax, y: float, label: str = "chance") -> None:
    """A reference floor, drawn so it cannot be mistaken for a series."""
    ax.axhline(y, color=REFERENCE, linewidth=0.8, linestyle=(0, (2, 2)),
               zorder=0)
    ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"),
                xytext=(2, 0), textcoords="offset points",
                va="center", ha="left", fontsize=6, color=REFERENCE)


def save(fig, stem: str) -> list[Path]:
    """Write the figure as PDF for the paper and PNG for looking at.

    PDF because the paper needs vector, PNG because nobody opens a PDF to
    check whether a figure came out right.
    """
    out = []
    p = Path(stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        f = p.with_suffix(f".{ext}")
        fig.savefig(f)
        out.append(f)
    return out
