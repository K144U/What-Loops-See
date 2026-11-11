"""F8. Absent, or merely undecodable.

paper.md section 6: the figure exists to show that the null is a
measurement and not a failure to measure, so the control must be as
prominent as the result. It is drawn as a two by two of equal panels for
exactly that reason: the control is not an inset.

           the model that failed        the model that solved it
  lens     what the head reads          what the head reads
  probe    what is linearly present     what is linearly present

Read across a row, the pair separates failures a lens alone cannot tell
apart, which is the ambiguity Lu et al. arXiv 2507.02199 leave open and
section 3a claims we can close:

    lens reads it,  probe reads it     computed, and routed to the head
    lens at chance, probe reads it     computed, and NOT routed
    lens at chance, probe at chance    never computed at all

All three appear in this one figure, which was not expected when it was
specified. The solved model shows the first on `composite`, where the lens
reads 1.000 and the probe 0.896, and the second on every intermediate,
where the probe reads well above chance and the head reads nothing at all.
The failed model shows the third everywhere. So the same model both routes
an answer and withholds the steps that produced it, and the figure says so
without anyone having to assert it.

Read down the right column, the same two instruments on a model that
solved the task read the answer loudly. That is what licenses the left
column as evidence of absence rather than evidence of a blunt instrument.

THE PROBE IS READ AT THE QUERY SITE, NOT THE PATCH SITE. `state_probe`
trains at both because a probe can be trained anywhere. `decode` pools over
the query tokens, so the query site is the one the head could in principle
read. Putting the head's reading beside a patch-pooled probe would compare
what the head sees against somewhere it cannot see, and the row would stop
being a comparison.

THE TWO INSTRUMENTS MUST BE ON THE SAME STATES OR THE FIGURE MEANS NOTHING.
The run, the factor, the loop count and the item count all have to match,
and this refuses to draw when they do not rather than producing a panel
that looks like a comparison and is not.

    python -m loopvision.analysis.plots.f8_absent --out figures/f8_absent
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from loopvision.analysis.plots import style

#: Left is the result, right is the control, and they are the same size.
#: Ordered so the eye meets the null first and the control immediately
#: after, which is the order the argument is made in.
COLUMNS = (
    ("famA_d2_d4colour_s0", "D4 on identity\nthe model that failed"),
    ("famA_d2_d4only_s0", "D4 on position\nthe model that solved it"),
)

#: The factor both instruments must agree on. D4 because that is the group
#: the substrate experiment holds fixed across the two columns.
FACTOR = "d4"

#: The probe site the head could read. See the module docstring.
SITE = "query"


def load_probe(run_dir: Path, factor: str) -> dict:
    """The probe file for this factor, found by content not by filename.

    The two runs store it under different names, `state_probe.json` and
    `state_probe_d4.json`, so matching on the name would silently pick the
    wrong factor on one of them.
    """
    for p in sorted(glob.glob(str(run_dir / "state_probe*.json"))):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if d.get("factor") == factor:
            d["path"] = p
            return d
    raise FileNotFoundError(f"no {factor} probe under {run_dir}")


def load_lens(run_dir: Path, factor: str) -> dict:
    p = run_dir / f"factor_lens_{factor}.json"
    if not p.is_file():
        raise FileNotFoundError(f"no {factor} lens under {run_dir}")
    d = json.loads(p.read_text(encoding="utf-8"))
    d["path"] = str(p)
    return d


def same_states(lens: dict, probe: dict) -> list[str]:
    """Every way the two could be measuring different things."""
    bad = []
    if lens["run"] != probe["run"]:
        bad.append(f"different runs: {lens['run']} against {probe['run']}")
    if lens["factor"] != probe.get("factor"):
        bad.append(f"different factors: {lens['factor']} against "
                   f"{probe.get('factor')}")
    if lens["k"] != probe["k"]:
        bad.append(f"different loop counts: k={lens['k']} against "
                   f"k={probe['k']}")
    if lens["n"] != probe["n"]:
        bad.append(f"different item counts: {lens['n']} against {probe['n']}")
    if abs(lens["chance"] - probe["chance"]) > 1e-9:
        bad.append(f"different chance floors: {lens['chance']} against "
                   f"{probe['chance']}")
    return bad


def probe_grid(probe: dict, targets: list[str], site: str) -> np.ndarray:
    """The probe as a passes by targets grid, at one pooling site."""
    passes = sorted({int(k.split("|")[0].replace("pass", ""))
                     for k in probe["probes"]})
    return np.array([[probe["probes"][f"pass{i}|{site}|{t}"] for t in targets]
                     for i in passes], dtype=float)


def draw(panels: list[dict], chance: float, targets: list[str]):
    import matplotlib.pyplot as plt

    style.use()
    n = len(panels)
    fig, axes = plt.subplots(
        2, n, figsize=(style.WIDTH_FULL * 0.72, 3.3), squeeze=False,
        gridspec_kw={"hspace": 0.30, "wspace": 0.12},
    )

    im = None
    for j, p in enumerate(panels):
        for i, (name, grid) in enumerate((("lens", p["lens_grid"]),
                                          ("probe", p["probe_grid"]))):
            ax = axes[i][j]
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="magma",
                           vmin=0.0, vmax=1.0)
            ax.set_xticks(range(len(targets)))
            ax.set_xticklabels(targets, rotation=45, ha="right",
                               fontsize=6)
            ax.set_yticks(range(grid.shape[0]))
            ax.grid(False)
            if j == 0:
                ax.set_ylabel(f"{name}\nloop pass")
            else:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(p["label"], loc="left", fontsize=7.5)
            if i == 0:
                ax.set_xticklabels([])

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.022,
                        pad=0.015)
    cbar.set_label("accuracy", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.axhline(chance, color="white", linewidth=1.0)
    cbar.ax.annotate("chance", xy=(1.6, chance), xycoords=("axes fraction",
                                                           "data"),
                     fontsize=5.5, color=style.REFERENCE, va="center")
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F8, absent or undecodable")
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    ap.add_argument("--factor", default=FACTOR)
    ap.add_argument("--site", default=SITE)
    ap.add_argument("--out", default="figures/f8_absent")
    args = ap.parse_args(argv)

    panels, chances, targets = [], set(), None
    for run_id, label in COLUMNS:
        run_dir = args.runs / run_id
        try:
            lens = load_lens(run_dir, args.factor)
            probe = load_probe(run_dir, args.factor)
        except FileNotFoundError as e:
            print(f"{e}")
            print("Both instruments are needed on both runs. The lens comes "
                  "from scripts/factor_lens.pbs; the probe from state_probe.")
            return 2
        bad = same_states(lens, probe)
        if bad:
            print(f"{run_id}: the lens and the probe are not on the same "
                  f"states, so the row would not be a comparison:")
            for b in bad:
                print(f"  {b}")
            return 2
        targets = lens["targets"]
        chances.add(round(lens["chance"], 9))
        panels.append({
            "label": label,
            "lens_grid": np.array(lens["grid"], dtype=float),
            "probe_grid": probe_grid(probe, targets, args.site),
            "run": run_id,
            "n": lens["n"],
            "k": lens["k"],
        })

    if len(chances) > 1:
        # One colourbar carries one chance mark. Two floors under it would
        # be right for at most one column.
        print(f"the columns disagree about chance: {sorted(chances)}. "
              f"Refusing to draw one floor for both.")
        return 2

    fig = draw(panels, chances.pop(), targets)
    for f in style.save(fig, args.out):
        print(f"wrote {f}")

    print(f"factor {args.factor}, probe site {args.site}")
    for p in panels:
        li = int(np.unravel_index(np.argmax(p["lens_grid"]),
                                  p["lens_grid"].shape)[1])
        pi = int(np.unravel_index(np.argmax(p["probe_grid"]),
                                  p["probe_grid"].shape)[1])
        print(f"  {p['run']:22s} n={p['n']} k={p['k']}  "
              f"lens peaks on {targets[li]} at {p['lens_grid'].max():.3f}, "
              f"probe peaks on {targets[pi]} at {p['probe_grid'].max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
