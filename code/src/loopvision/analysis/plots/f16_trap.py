"""F16. How a true bit becomes a trap: the composing cue of Section 5.4, worked out.

Added 2026-11-12 (D-080). A schematic built from real tiles and the group's
own arithmetic, with no measured number in it. Every tile is the renderer's,
every bit is `groups.D4_CHARACTERS["rotflip"]` or `groups.d4_decoy_bit`, and
every answer is a product from `groups.multiply`.

    (a) E2a's cue pixel splits D4's eight tiles into two sets of four
    (b) in a real problem the three cue bits add up, in twos, to the bit of
        the answer, because the bit is a homomorphism onto two elements
    (c) so a model that reads the bit and nothing else is left with four
        candidates and is right one time in four
    (d) E2b's pixel marks a split that is not a homomorphism: the same three
        pixels come with either answer pixel, and they carry nothing about any
        bit of the answer that composes. They do shift the odds of the exact
        answer, to 10 in 64 at best against 8 for a guess, so the panel says
        what is exact (no composing bit) and not that they say nothing.

WHICH EXAMPLES. Not chosen by eye. (b) is the first validation index of the
E2a task whose asked-about row has no "do nothing" in it and at least one lit
cue, so the sum is not trivially zero. (d) is the first pair of D4 triples, in
index order, with no identity anywhere, the same three E2b pixels and answers
whose pixels differ. Both rules are tested.

    python -m loopvision.analysis.plots.f16_trap --out figures/f16_trap
"""

from __future__ import annotations

import argparse
import itertools

NL = "\n"
DEPTH, BREADTH, SPLIT = 2, 4, "iid_val"
CUE, DECOY = "d4_colour_cue", "d4_colour_decoy"


def cue_bit(g: int) -> int:
    from loopvision.data import groups as G

    return G.D4_CHARACTERS["rotflip"](g)


def decoy_bit(g: int) -> int:
    from loopvision.data import groups as G

    return G.d4_decoy_bit(g)


def cue_cell(g: int, substrate: str = CUE) -> tuple[int, int]:
    """(row, column) of the cue pixel's place in g's rendered tile."""
    from loopvision.data import groups as G
    from loopvision.data import render

    cells = G.glyph_action(substrate)(g)
    w = max(x for x, _, _ in cells) + 1
    h = max(y for _, y, _ in cells) + 1
    ox, oy = (render.PATCH - w) // 2, (render.PATCH - h) // 2
    return oy + G.CUE_PIXEL[1], ox + G.CUE_PIXEL[0]


def cue_problem(limit: int = 1000):
    """(start, operators, answer) of the first clear E2a problem; see the docstring."""
    from loopvision.analysis.plots.f13_anatomy import worked
    from loopvision.data import dataset as D
    from loopvision.data import groups as G

    cfg = D.TaskConfig(depths=(DEPTH,), breadths=(BREADTH,), factor="d4", substrate=CUE)
    for i in range(limit):
        start, ops, _, answer = worked(D.generate("A", i, SPLIT, cfg))
        steps = (start, *ops)
        if G.IDENTITY in (*steps, answer) or not any(cue_bit(g) for g in steps):
            continue
        if sum(cue_bit(g) for g in steps) % 2 != cue_bit(answer):
            raise ValueError(f"index {i}: the cue bits do not add up to the answer's")
        return i, start, list(ops), answer
    raise ValueError(f"no clear E2a problem below index {limit}")


def decoy_pair():
    """Two D4 triples with the same E2b pixels and answers whose pixels differ."""
    from loopvision.data import groups as G

    members = G.subgroup_members("d4")
    first: dict[tuple[int, ...], tuple] = {}
    for s, a, b in itertools.product(members, repeat=3):
        answer = G.multiply(b, G.multiply(a, s))
        if G.IDENTITY in (s, a, b, answer):
            continue
        key = (decoy_bit(s), decoy_bit(a), decoy_bit(b))
        if key in first and decoy_bit(first[key][3]) != decoy_bit(answer):
            return first[key], (s, a, b, answer)
        first.setdefault(key, (s, a, b, answer))
    raise ValueError("every E2b pixel triple decides its answer's pixel")


def _ring(ax, x0, y0, size, cell, colour="#c99a00"):
    from matplotlib.patches import Rectangle

    s = size / 4
    r, c = cell
    ax.add_patch(Rectangle((x0 + c * s, y0 + (3 - r) * s), s, s, fill=False, ec=colour,
                           lw=1.1, zorder=6))


def _title(ax, x, y, text):
    ax.text(x, y, text, fontsize=8, ha="left", va="bottom")


def panel_split(ax, x0, y0):
    from loopvision.analysis.plots import pictures as P
    from loopvision.data import groups as G

    size, pitch = 1.35, 1.75
    _title(ax, x0, y0 + 5.0, "(a) The cue pixel shows one bit of every tile")
    for row, bit in enumerate((0, 1)):
        y = y0 + 2.6 - row * 2.3
        ax.text(x0, y + size / 2, ("no cue pixel" if bit == 0 else "cue pixel lit")
                + NL + f"bit {bit}", fontsize=6.4, va="center", linespacing=1.1)
        group = [g for g in G.subgroup_members("d4") if cue_bit(g) == bit]
        for k, g in enumerate(group):
            x = x0 + 4.6 + k * pitch
            P.draw_pixels(ax, P.tile(g, CUE), x, y, size, frame=None)
            _ring(ax, x, y, size, cue_cell(g))
    ax.text(x0, y0 - 0.55,
            "Four symmetries light the pixel and four do not. The ring marks"
            + NL + "its place; the four colours are the appearance task's.",
            fontsize=6.1, va="top", color="#333333", linespacing=1.15)


def panel_sum(ax, x0, y0, start, ops, answer):
    from loopvision.analysis.plots import pictures as P

    size = 1.5
    _title(ax, x0, y0 + 5.0, "(b) In a problem, the bits add up to the answer's bit")
    steps = [("start", start)] + [(f"operator {i + 1}", g) for i, g in enumerate(ops)]
    x = x0
    y = y0 + 1.6
    for k, (name, g) in enumerate(steps):
        P.draw_pixels(ax, P.tile(g, CUE), x, y, size, frame=None)
        _ring(ax, x, y, size, cue_cell(g))
        ax.text(x + size / 2, y + size + 0.25, name, fontsize=6.2, ha="center", va="bottom")
        ax.text(x + size / 2, y - 0.3, f"bit {cue_bit(g)}", fontsize=6.6, ha="center",
                va="top", weight="bold")
        sign = "+" if k < len(steps) - 1 else "="
        ax.text(x + size + 0.55, y + size / 2, sign, fontsize=9, ha="center", va="center")
        x += size + 1.1
    P.draw_pixels(ax, P.tile(answer, CUE), x, y, size, frame=None)
    _ring(ax, x, y, size, cue_cell(answer))
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x - 0.1, y - 0.1), size + 0.2, size + 0.2, fill=False,
                           ec="#c99a00", lw=1.3, zorder=6))
    ax.text(x + size / 2, y + size + 0.25, "answer", fontsize=6.2, ha="center",
            va="bottom", weight="bold")
    ax.text(x + size / 2, y - 0.3, f"bit {cue_bit(answer)}", fontsize=6.6, ha="center",
            va="top", weight="bold")
    bits = [cue_bit(g) for _, g in steps]
    ax.text(x0, y0 - 0.55,
            " + ".join(str(b) for b in bits) + f" is {'odd' if sum(bits) % 2 else 'even'},"
            f" and so is the answer's bit. This holds for every" + NL
            + "problem: the bit is a homomorphism, so it composes. The three"
            + NL + "cue pixels give away one bit of the answer, always correctly.",
            fontsize=6.1, va="top", color="#333333", linespacing=1.15)


def panel_four(ax, x0, y0, bit):
    from matplotlib.patches import Rectangle

    from loopvision.analysis.plots import pictures as P
    from loopvision.data import groups as G

    size, pitch = 1.2, 1.6
    _title(ax, x0, y0 + 5.0, "(c) So the bit leaves four answers, and 1 in 4 is a trap")
    members = G.subgroup_members("d4")
    ax.text(x0, y0 + 3.85, f"the eight possible answers; this problem's bit is {bit}",
            fontsize=6.2, va="center", color="#333333")
    for k, g in enumerate(members):
        x = x0 + k * pitch
        keep = cue_bit(g) == bit
        P.draw_pixels(ax, P.tile(g, "d4_colour"), x, y0 + 1.9, size, frame=None,
                      alpha=1.0 if keep else 0.3)
        if not keep:
            ax.plot([x, x + size], [y0 + 1.9 + size, y0 + 1.9], color="#777777", lw=0.9,
                    zorder=7)
        if keep:
            ax.add_patch(Rectangle((x - 0.08, y0 + 1.9 - 0.08), size + 0.16, size + 0.16,
                                   fill=False, ec="#c99a00", lw=1.1, zorder=6))
    ax.text(x0, y0 + 1.25,
            "A model that reads the bit and nothing else picks among the four"
            + NL + "framed answers, the others ruled out. It is right 1 time in 4: a score"
            + NL + "of 0.25 and a loss of ln 4. It is partly right on every problem, so"
            + NL + "training can stop there; most E2a runs that left chance did.",
            fontsize=6.1, va="top", color="#333333", linespacing=1.15)


def panel_decoy(ax, x0, y0, pair):
    from loopvision.analysis.plots import pictures as P

    size = 1.2
    _title(ax, x0, y0 + 5.0, "(d) A pixel that does not compose gives no bit away")
    for row, (s, a, b, answer) in enumerate(pair):
        y = y0 + 2.9 - row * 1.95
        x = x0
        for g in (s, a, b):
            P.draw_pixels(ax, P.tile(g, DECOY), x, y, size, frame=None)
            _ring(ax, x, y, size, cue_cell(g, DECOY))
            ax.text(x + size / 2, y - 0.15, f"{decoy_bit(g)}", fontsize=6.4,
                    ha="center", va="top", weight="bold")
            x += size + 0.45
        ax.annotate("", xy=(x + 1.2, y + size / 2), xytext=(x + 0.1, y + size / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333",
                                    shrinkA=0, shrinkB=0))
        x += 1.5
        P.draw_pixels(ax, P.tile(answer, DECOY), x, y, size, frame=None)
        _ring(ax, x, y, size, cue_cell(answer, DECOY))
        ax.text(x + size + 0.35, y + size / 2, f"answer's pixel: {decoy_bit(answer)}",
                fontsize=6.4, va="center", weight="bold")
    ax.text(x0, y0 - 0.55,
            "E2b's pixel marks a split that is not a homomorphism. The same three"
            + NL + "pixels come with either answer pixel, and they say nothing about any"
            + NL + "bit of the answer that composes: there is no always-right part of the"
            + NL + "answer to stop at.",
            fontsize=6.1, va="top", color="#333333", linespacing=1.15)


def draw(out: str):
    import matplotlib.pyplot as plt

    from loopvision.analysis.plots import pictures as P
    from loopvision.analysis.plots import style

    style.use()
    _, start, ops, answer = cue_problem()
    pair = decoy_pair()
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 4.0))
    P.blank_axes(ax, (-0.2, 33.0), (-3.0, 16.6))
    panel_split(ax, 0.0, 10.4)
    panel_sum(ax, 17.6, 10.4, start, ops, answer)
    panel_four(ax, 0.0, 0.6, cue_bit(answer))
    panel_decoy(ax, 17.6, 0.6, pair)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return style.save(fig, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="figures/f16_trap")
    args = ap.parse_args(argv)
    for p in draw(args.out):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
