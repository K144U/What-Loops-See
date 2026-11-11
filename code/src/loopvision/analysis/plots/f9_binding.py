"""F9. The binding asymmetry as a rate, and what a probe reads off each side.

paper.md section 6 makes this the figure that must be excellent, ahead of
F8 and F3, now that the substrate experiment supports the binding claim.
Its job is to make one argument visible in a glance and to close the
obvious objection in the same picture:

  top     both media are solved by some seeds and not others, one point per
          seed so the reader counts rather than trusts a mean. The contrast
          is a RATE: position escapes far more often than identity, and
          neither column is uniform. The per column counts and the panel
          title are computed from the plotted rows, never typed, because
          this title read "never on identity" until 2026-09-21 while the
          panel under it drew fifteen identity seeds at 1.0000
  bottom  a trained probe reads the composite off the states of the model
          that solved it and reads chance off the model that did not, so
          the failure is that the factor is absent rather than present and
          undecodable

WHY THE TOP ROW IS THE 2x2 AND NOT TWO FACTORS. The figure was specified
before the crossed design ran, when the comparison available was D4 against
S3 inside one model. That comparison confounds the group with the medium:
D4 acts on position and S3 on identity in the native task, so a difference
between them could be either. The 2x2 separates them and section 3b now
rests on the substrate row, so the figure shows the row it claims. Each
group appears in both media, which is what makes the group order matched in
the sense section 6 asks for.

THIS SCRIPT COMPUTES NOTHING. Every number is read from a stored file:

  escape_rates --json   which run is in which arm, and its final accuracy
  state_probe.json      probe accuracy per pass, and the chance floor

Deciding which run belongs to which arm is exactly the kind of derivation
that must not live in a plot script, so it lives in `escape_rates`, where
it is tested and mutation checked, and this reads the answer. This file
contains no numbers at all: even the collapse floor drawn under each arm is
that arm's own collapsed accuracies, read from the generator.

    python -m loopvision.analysis.escape_rates --json rates.json
    python -m loopvision.analysis.plots.f9_binding \
        --rates rates.json --out figures/f9_binding
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loopvision.analysis.plots import style

#: The two arms the top row contrasts, and the order they are drawn in.
#: Position first because the claim is that position is solved.
ARMS = (
    ("D4/position", "D4\non position"),
    ("S3/position", "S3\non position"),
    ("D4/identity", "D4\non identity"),
    ("S3/identity", "S3\non identity"),
)

#: The probe series in the bottom row. Each names the run whose stored
#: probe file is read and the file within it, so the figure cannot silently
#: change which model it is showing.
PROBE_SERIES = (
    ("famA_d2_d4only_s0", "state_probe_d4.json",
     "D4 on position, the model that solved it"),
    ("famA_d2_d4colour_s0", "state_probe.json",
     "D4 on identity, the model that did not"),
)

#: Which probe readout to plot across passes. `composite` is the whole
#: composition, which is what the behaviour row is about; `query` is the
#: stream the head actually reads.
PROBE_TARGET = "composite"
PROBE_STREAM = "query"


def load_rates(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_probe(runs_root: Path, run_id: str, filename: str) -> dict:
    """Probe accuracy per pass, and the chance floor, as stored."""
    d = json.loads((runs_root / run_id / filename).read_text(encoding="utf-8"))
    passes, values = [], []
    for key, v in d["probes"].items():
        p, stream, target = key.split("|")
        if stream != PROBE_STREAM or target != PROBE_TARGET:
            continue
        passes.append(int(p.replace("pass", "")))
        values.append(float(v))
    order = sorted(range(len(passes)), key=lambda i: passes[i])
    return {
        "run": run_id,
        "passes": [passes[i] for i in order],
        "values": [values[i] for i in order],
        "chance": float(d["chance"]),
        "factor": d.get("factor"),
    }


def draw(rates: dict, probes: list[dict]):
    import matplotlib.pyplot as plt

    style.use()
    fig, (ax_b, ax_p) = plt.subplots(
        2, 1, figsize=(style.WIDTH_COLUMN, 4.0),
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.55},
    )

    # Top: one point per seed, jittered only along x so no accuracy moves.
    # Floors come from the generator, per arm, never from this file.
    arm_floors = {k: v.get("floors") or []
                  for k, v in (rates.get("arms") or {}).items()}

    by_arm: dict[str, list] = {}
    for r in rates["runs"]:
        if r["outcome"] == "unfinished" or r["accuracy"] is None:
            continue
        by_arm.setdefault(r["arm"], []).append(r)

    labels = []
    for i, (arm, label) in enumerate(ARMS):
        labels.append(label)
        rows = sorted(by_arm.get(arm, []), key=lambda r: (r["seed"] or 0))
        n = len(rows)
        for j, r in enumerate(rows):
            # Deterministic spread, not random, so the figure is identical
            # on every regeneration and a reader comparing two drafts sees
            # only real changes.
            dx = 0.0 if n == 1 else (j / (n - 1) - 0.5) * 0.42
            ax_b.plot(i + dx, r["accuracy"],
                      marker=style.SERIES_STYLES[0]["marker"]
                      if r["outcome"] == "escaped"
                      else style.SERIES_STYLES[1]["marker"],
                      color=style.OUTCOME[r["outcome"]],
                      markersize=4, linestyle="none",
                      markeredgewidth=0.6, markerfacecolor="none")
        ax_b.annotate(f"{sum(1 for r in rows if r['outcome'] == 'escaped')}/{n}",
                      xy=(i, 1.10), ha="center", va="bottom", fontsize=6.5,
                      color="#444444", annotation_clip=False)

        # The floor is drawn PER ARM and never across the panel. D4 has
        # order 8 and S3 order 6, so their floors sit at different heights,
        # and one line spanning both would put the S3 points above a floor
        # that is not theirs: the arm would read as beating a constant when
        # it is sitting on one. Same reason the bottom row refuses to draw
        # one chance line under two probes. The values are the collapsed
        # runs' own accuracies, read from the generator, not a constant
        # this file knows.
        floors = arm_floors.get(arm) or []
        if floors:
            lo, hi = min(floors), max(floors)
            ax_b.plot([i - 0.30, i + 0.30], [lo, lo], color=style.REFERENCE,
                      linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
            if hi != lo:
                ax_b.plot([i - 0.30, i + 0.30], [hi, hi],
                          color=style.REFERENCE, linewidth=0.8,
                          linestyle=(0, (2, 2)), zorder=0)

    if any(arm_floors.get(a) for a, _ in ARMS):
        # Inside the panel, in the empty band between the solved seeds at
        # 1.0 and the collapsed ones near 0.15. Below the axes it collided
        # with the lower panel's title.
        ax_b.annotate("dashes: the constant each\ncollapsed run emits",
                      xy=(0.03, 0.42), xycoords="axes fraction",
                      fontsize=6, color=style.REFERENCE, va="top")

    ax_b.set_xticks(range(len(ARMS)))
    ax_b.set_xticklabels(labels)
    ax_b.set_xlim(-0.6, len(ARMS) - 0.4)
    ax_b.set_ylim(-0.04, 1.12)
    ax_b.set_ylabel("final accuracy")
    # The title is COMPUTED, never typed. It read "Solved on position, never
    # on identity" until 2026-09-21, while the panel beneath it plotted
    # fifteen identity seeds sitting at 1.0000. A figure whose title argues
    # with its own data is worse than one with no title, and a hand-typed
    # title is a hand-typed count wearing a different hat. See L-029.
    def tally(medium: str) -> tuple[int, int]:
        rows = [r for arm, _ in ARMS if arm.endswith(medium)
                for r in by_arm.get(arm, [])]
        return sum(1 for r in rows if r["outcome"] == "escaped"), len(rows)

    pos_e, pos_n = tally("position")
    idt_e, idt_n = tally("identity")
    ax_b.set_title(
        f"Position {pos_e} of {pos_n}, identity {idt_e} of {idt_n}: "
        f"a rate, not a wall",
        loc="left", pad=14)
    ax_b.axvline(1.5, color="#dddddd", linewidth=0.8, zorder=0)

    # Bottom: the probe, which decides absent against undecodable.
    for i, pr in enumerate(probes):
        s = style.series(i)
        ax_p.plot(pr["passes"], pr["values"], label=pr["label"], **s)
    chance = probes[0]["chance"] if probes else None
    if chance is not None:
        style.chance_line(ax_p, chance)
    ax_p.set_xlabel("loop pass")
    ax_p.set_ylabel("probe accuracy")
    ax_p.set_ylim(0, 1.0)
    ax_p.set_xticks(probes[0]["passes"] if probes else [])
    ax_p.set_title("A probe reads the composition off one and not the other",
                   loc="left")
    ax_p.legend(loc="upper left", bbox_to_anchor=(0, -0.42), handlelength=2.4)
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F9, the binding asymmetry")
    ap.add_argument("--rates", required=True, type=Path,
                    help="json written by analysis.escape_rates --json")
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    ap.add_argument("--out", default="figures/f9_binding")
    args = ap.parse_args(argv)

    if not args.rates.is_file():
        print(f"no rates file: {args.rates}")
        print("generate it first: python -m loopvision.analysis.escape_rates "
              "--json rates.json")
        return 2

    rates = load_rates(args.rates)
    probes = []
    for run_id, filename, label in PROBE_SERIES:
        p = args.runs / run_id / filename
        if not p.is_file():
            print(f"missing probe file, cannot draw the bottom row: {p}")
            return 2
        pr = load_probe(args.runs, run_id, filename)
        pr["label"] = label
        probes.append(pr)

    chances = {pr["chance"] for pr in probes}
    if len(chances) > 1:
        # Two probes on different chance floors share one axis and one
        # dashed line, and the line would be right for one of them. That is
        # a figure that lies quietly, so it is refused.
        print(f"probe series disagree about chance: {sorted(chances)}. "
              f"Refusing to draw one chance line for both.")
        return 2

    fig = draw(rates, probes)
    written = style.save(fig, args.out)
    for f in written:
        print(f"wrote {f}")

    drawn = sum(1 for r in rates["runs"]
                if r["outcome"] != "unfinished" and r["accuracy"] is not None)
    skipped = len(rates["runs"]) - drawn
    print(f"{drawn} finished seed(s) plotted, {skipped} unfinished not shown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
