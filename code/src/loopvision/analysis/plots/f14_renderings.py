"""F14. One group drawn seven ways: how far apart the tiles are, and how often each is learned.

Added 2026-11-12 (D-080). Section 5.5's table, as a picture a reader can check
by eye. Each row is one rendering, with every one of its tiles drawn by the
renderer, then its distance between tiles and its escape count, both read from
the stored salience output `runs/_tile_distinguishability_20261022_d076/d070.json`.

Nothing is computed here. The rows are sorted by the stored distance, which
is presentation, and the rank correlation printed under the figure is the one
stored beside the measures. E2a, the composing cue, is drawn below a rule and
greyed, because the registration left it out of the test (Section 5.4 already
shows salience does not account for it).

    python -m loopvision.analysis.plots.f14_renderings \\
        --salience runs/_tile_distinguishability_20261022_d076/d070.json \\
        --out figures/f14_renderings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NL = "\n"

#: Plain names, and whether the rendering draws the group by shape or by colour.
NAMES = {
    "D4 position": ("D4 turns and flips" + NL + "the glyph", "shape"),
    "E1a": ("D4 turns and flips a pentomino" + NL + "whose box is square", "shape"),
    "E1b": ("the glyph tiles, each given" + NL + "to another D4 element", "shape"),
    "S3 position": ("S3 moves three blobs" + NL + "between three places", "shape"),
    "E2b": ("D4 moves four colours, plus a" + NL + "pixel that does not compose", "colour"),
    "D4 appearance": ("D4 moves four colours", "colour"),
    "S3 appearance": ("S3 swaps the glyph's" + NL + "three colours", "colour"),
    "E2a": ("D4 moves four colours, plus" + NL + "a cue pixel that composes", "colour"),
}

#: Category colour and marker. Both pass the dataviz palette validator on white.
KIND = {"shape": ("#2a78d6", "o", "drawn by shape"),
        "colour": ("#eb6834", "s", "drawn by colour")}


def load(path: Path) -> dict:
    """The stored measures, rates and verdict, with every name accounted for."""
    d = json.loads(path.read_text(encoding="utf-8"))
    missing = set(d["measures"]) ^ set(NAMES)
    if missing:
        raise ValueError(f"renderings without a plain name, or names without one: {sorted(missing)}")
    if set(d["rates"]) != set(d["measures"]):
        raise ValueError("a rendering has a measure and no rate, or the reverse")
    return d


def order(d: dict) -> tuple[list[str], list[str]]:
    """The tested renderings by stored distance, largest first, then the rest."""
    tested = sorted(d["tested"], key=lambda k: -d["measures"][k]["m_pix"])
    rest = sorted(set(d["measures"]) - set(tested))
    return tested, rest


def tiles(entry: dict):
    from loopvision.analysis.plots import pictures as P
    from loopvision.data import groups as G

    members = G.subgroup_members(entry["subgroup"])
    if len(members) != entry["n_elements"]:
        raise ValueError(f"{entry['substrate']}: {len(members)} tiles, stored {entry['n_elements']}")
    return [P.tile(g, entry["substrate"]) for g in members]


def draw(salience: Path, out: str):
    import matplotlib.pyplot as plt

    from loopvision.analysis.plots import pictures as P
    from loopvision.analysis.plots import style

    style.use()
    d = load(salience)
    tested, rest = order(d)
    rows = tested + rest
    pitch, size = 1.5, 1.0
    gap = 0.9
    ys = {}
    y = 0.0
    for name in rows:
        if name in rest and not any(r in ys for r in rest):
            y -= gap
        ys[name] = y
        y -= pitch
    bottom = y + pitch

    x_tiles, x_dist, x_rate = 8.6, 20.0, 27.4
    w_col = 5.6
    lo, hi = 1.0, 2.6

    def xd(v):
        return x_dist + (v - lo) / (hi - lo) * w_col

    def xr(v):
        return x_rate + v * w_col

    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 3.55))
    P.blank_axes(ax, (-0.2, 35.6), (bottom - 3.3, 3.5))

    top = 0.0 + size + 0.35
    ax.text(0, top + 0.95, "rendering", fontsize=7, weight="bold", va="bottom")
    ax.text(x_tiles, top + 0.95, "its tiles, one per group element", fontsize=7,
            weight="bold", va="bottom")
    ax.text(x_dist, top + 0.95, "how far apart" + NL + "its tiles are", fontsize=7,
            weight="bold", va="bottom", linespacing=1.05)
    ax.text(x_rate, top + 0.95, "how often" + NL + "it is learned", fontsize=7,
            weight="bold", va="bottom", linespacing=1.05)
    ax.text(x_dist, top + 0.3, "mean pixel distance", fontsize=5.8, color="#555555",
            va="bottom")
    ax.text(x_rate, top + 0.3, "seeds that escaped", fontsize=5.8, color="#555555",
            va="bottom")

    axis_y = bottom - 0.55
    for x0, ticks, fmt, f in ((x_dist, (1.0, 1.4, 1.8, 2.2, 2.6), "{:.1f}", xd),
                              (x_rate, (0, 0.25, 0.5, 0.75, 1.0), "{:g}", xr)):
        ax.plot([x0, x0 + w_col], [axis_y, axis_y], color="#333333", lw=0.6)
        for t in ticks:
            ax.plot([f(t)] * 2, [axis_y, axis_y - 0.18], color="#333333", lw=0.6)
            ax.plot([f(t)] * 2, [axis_y, top], color="#e4e4e4", lw=0.5, zorder=0)
            ax.text(f(t), axis_y - 0.3, fmt.format(t), ha="center", va="top", fontsize=5.8)

    for name in rows:
        y = ys[name]
        entry = d["measures"][name]
        label, kind = NAMES[name]
        colour, marker, _ = KIND[kind]
        faded = name in rest
        alpha = 0.45 if faded else 1.0
        ax.text(0, y + size / 2, label, fontsize=6.4, va="center", linespacing=1.05,
                color="#777777" if faded else "#111111")
        for i, t in enumerate(tiles(entry)):
            P.draw_pixels(ax, t, x_tiles + i * (size + 0.3), y, size, frame=None)
        yc = y + size / 2
        ax.plot([xd(lo), xd(hi)], [yc, yc], color="#f0f0f0", lw=0.5, zorder=0)
        ax.plot([xd(entry["m_pix"])], [yc], marker=marker, color=colour, markersize=5,
                linestyle="none", alpha=alpha, zorder=3)
        k, n = d["rates"][name]
        ax.plot([xr(0), xr(k / n)], [yc, yc], color=colour, lw=1.6, alpha=alpha,
                solid_capstyle="round", zorder=2)
        ax.plot([xr(k / n)], [yc], marker=marker, color=colour, markersize=5,
                linestyle="none", alpha=alpha, zorder=3)
        ax.text(x_rate + w_col + 0.5, yc, f"{k} of {n}", fontsize=6.2, va="center",
                color="#777777" if faded else "#111111")
    first_rest = ys[rest[0]] if rest else None
    if first_rest is not None:
        ax.plot([0, 35.0], [first_rest + pitch - 0.25] * 2, color="#bbbbbb", lw=0.6,
                ls=(0, (2, 2)))
        ax.text(0, first_rest - 0.45,
                "left out of the test: Section 5.4 shows salience does not account for it",
                fontsize=5.8, color="#777777", va="top", style="italic")
    for i, (kind, (colour, marker, text)) in enumerate(KIND.items()):
        xk = x_dist + i * 6.4
        ax.plot([xk], [bottom - 2.1], marker=marker, color=colour, markersize=5,
                linestyle="none")
        ax.text(xk + 0.4, bottom - 2.1, text, fontsize=6.2, va="center")
    ax.text(0, bottom - 2.1,
            f"Rank correlation over the {len(tested)} tested renderings: {d['rho_pix']:.3f}."
            + NL + f"The registered prediction {d['S1']}; with {len(tested)} points, "
            f"p = {d['p_pix']:.3f}.",
            fontsize=6.2, va="center", color="#333333")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return style.save(fig, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--salience", type=Path,
                    default=Path("runs/_tile_distinguishability_20261022_d076/d070.json"))
    ap.add_argument("--out", default="figures/f14_renderings")
    args = ap.parse_args(argv)
    for p in draw(args.salience, args.out):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
