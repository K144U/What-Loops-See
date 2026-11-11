"""F10. H4's jitter sweep, and the confound that eats it.

Two panels, and the second is the point of the figure.

  left    wrong-attractor rate by jitter width, the raw sweep. Zero jitter
          sits far above every jittered arm and this looks like strong
          support for H4.
  right   the same runs plotted against their final accuracy. The rate
          follows accuracy down a line. The zero-jitter arm is high because
          those models are bad, not because their states land in the wrong
          basins for any reason jitter addresses.

**A figure that showed only the left panel would be a false claim in
pictures.** The whole finding is that the apparent effect is the training
outcome wearing the metric's clothes, so both panels have to be on the page
together and the right one has to be as prominent as the left.

Every number here is read from the summary written by `analysis/m6_h4`. This
script computes no statistic of its own, including the correlation, per the
standing rule in CLAUDE.md.

    python -m loopvision.analysis.m6_h4 --rates h4.json --summary h4s.json
    python -m loopvision.analysis.plots.f10_attractor --summary h4s.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loopvision.analysis.plots import style

#: Shared by BOTH panels, and that is the point rather than tidiness. The two
#: panels plot the same rates against different x axes, and the whole figure
#: is the comparison between them, so a reader who has to rescale one eye to
#: read the other is being asked to do the work the figure exists to do.
#:
#: The top is 1.0 rather than 0.85 because the rate is a fraction and the
#: extra headroom is where the "comparison is absent" note goes. At 0.85 the
#: highest run, 0.7539 at sigma 0.0, sat at 89 percent of the panel height
#: and the note landed on top of it.
RATE_YLIM = (-0.04, 1.0)


def draw(s: dict, out: Path | None = None):
    import matplotlib.pyplot as plt

    style.use()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(style.WIDTH_FULL, 2.9), gridspec_kw={"wspace": 0.34})

    # ---- left: the raw sweep, which is what looks like support -----------
    arms = s["all_arms"]
    xs = list(range(len(arms)))
    ax_a.plot(xs, [a["rate_mean"] for a in arms], **style.series(0))
    for x, a in zip(xs, arms):
        ax_a.plot([x, x], [a["rate_min"], a["rate_max"]],
                  color=style.series(0)["color"], linewidth=1.0, alpha=0.5,
                  solid_capstyle="butt")
        ax_a.annotate(f"n={a['seeds']}", xy=(x, a["rate_max"]),
                      xytext=(0, 4), textcoords="offset points",
                      ha="center", fontsize=6, color="#444444")
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels([str(a["sigma"]) for a in arms])
    ax_a.set_xlabel("jitter width sigma")
    ax_a.set_ylabel("wrong-attractor rate")
    ax_a.set_ylim(*RATE_YLIM)
    ax_a.set_title("Looks like support", loc="left", pad=8)

    # ---- right: the same runs against accuracy, which is the finding ----
    scored = [r for r in s["runs"] if r["accuracy"] is not None]
    sigmas = sorted({r["sigma"] for r in scored})
    for i, sig in enumerate(sigmas):
        v = [r for r in scored if r["sigma"] == sig]
        st = style.series(i)
        ax_b.plot([r["accuracy"] for r in v], [r["rate_by_label"] for r in v],
                  marker=st["marker"], color=st["color"], linestyle="none",
                  markersize=4, markerfacecolor="none", markeredgewidth=0.9,
                  label=f"sigma {sig}")
    ax_b.set_xlabel("final accuracy")
    ax_b.set_ylabel("wrong-attractor rate")
    ax_b.set_ylim(*RATE_YLIM)

    r = s["pearson_rate_vs_accuracy"]
    ax_b.set_title(
        "and is accuracy in disguise" if r is None
        else f"and is accuracy in disguise, r = {r:.2f}",
        loc="left", pad=8)

    # The absent comparison, named on the figure rather than in the caption
    # alone. An arm with no trained run is not a low score, and a reader who
    # takes the left panel at face value has to meet that fact here.
    absent = [c for c in s["coverage"] if c["solved"] == 0]
    if absent:
        names = ", ".join(str(c["sigma"]) for c in absent)
        ax_b.annotate(
            f"sigma {names}: 0 runs above {s['solved_threshold']:.2f},\n"
            f"so the comparison H4 asks for is absent",
            xy=(0.03, 0.98), xycoords="axes fraction",
            ha="left", va="top", fontsize=6, color=style.REFERENCE)

    ax_b.legend(loc="lower left", bbox_to_anchor=(0.0, -0.02), fontsize=6,
                frameon=False, handletextpad=0.4, borderpad=0.2)

    fig.suptitle(
        "H4 cannot be resolved from this sweep: the metric moves with training "
        "success", x=0.008, ha="left", fontsize=8)
    fig.subplots_adjust(top=0.80)

    if out is not None:
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    else:
        for p in style.save(fig, "f10_attractor"):
            print(f"wrote {p}")
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F10 H4 attractor sweep")
    ap.add_argument("--summary", required=True, type=Path,
                    help="json written by m6_h4 --summary")
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args(argv)

    s = json.loads(args.summary.read_text(encoding="utf-8"))
    if not s.get("all_arms"):
        print("no arms in that summary. Nothing is drawn and nothing is claimed.")
        return 2
    draw(s, args.out)
    matched = s.get("matched_arms") or []
    spread = [a["rate_mean"] for a in matched]
    print(f"{s['n_runs']} run(s) over {len(s['all_arms'])} width(s). "
          f"Matched arms span {max(spread) - min(spread):.4f}."
          if spread else f"{s['n_runs']} run(s), no matched arms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
