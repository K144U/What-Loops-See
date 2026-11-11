"""F5. The halting argument in one glance: direction converges, norm does not.

Two panels, and the left one leads deliberately.

  left    the scale free step, how far the state moves in direction per
          iteration, on a log axis because the decay is geometric. This is
          the quantity the halting argument rests on.
  right   the raw norm, flat, labelled as something the model cannot see.

**RMSNorm sits before the head, so the raw state norm is invisible to the
model**: two states differing only in magnitude decode identically. STATE.md
carries this as the standing trap in this figure, and it is a trap because
the norm panel looks like a null result. It is not a null result, it is a
quantity with no causal path to the output. On `famB_curve111_s0` the unit
step falls from 1.42 to 0.019 while the norm sits between 241 and 254
throughout. Leading with the norm would suggest nothing happens.

Reads the summary written by `analysis/m4_shape` and computes nothing, not
even the medians it draws.

    python -m loopvision.analysis.m4_shape --run-id <id> --summary shape.json
    python -m loopvision.analysis.plots.f5_geometry --summary shape.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loopvision.analysis.plots import style


def _series(run: dict, name: str):
    """Iterations and medians with the undefined points dropped in pairs."""
    m = run["metrics"].get(name)
    if not m:
        return [], [], [], []
    xs, med, q1, q3 = [], [], [], []
    for i, v, a, b in zip(m["iteration"], m["median"], m["q1"], m["q3"]):
        if v is None:
            continue
        xs.append(i)
        med.append(v)
        q1.append(a if a is not None else v)
        q3.append(b if b is not None else v)
    return xs, med, q1, q3


def draw(s: dict, out: Path | None = None):
    import matplotlib.pyplot as plt

    style.use()
    fig, (ax_d, ax_m) = plt.subplots(
        1, 2, figsize=(style.WIDTH_FULL, 2.8), gridspec_kw={"wspace": 0.30})

    for i, run in enumerate(s["runs"]):
        st = style.series(i)
        xs, med, q1, q3 = _series(run, "unit_step_norm")
        if xs:
            ax_d.plot(xs, med, label=run["run"], **st)
            ax_d.fill_between(xs, q1, q3, color=st["color"], alpha=0.14,
                              linewidth=0)
        xs, med, q1, q3 = _series(run, "norm")
        if xs:
            ax_m.plot(xs, med, **st)
            ax_m.fill_between(xs, q1, q3, color=st["color"], alpha=0.14,
                              linewidth=0)

    ax_d.set_yscale("log")
    ax_d.set_xlabel("loop iteration")
    ax_d.set_ylabel("unit step, direction only")
    ax_d.set_title("The direction converges, geometrically", loc="left", pad=8)
    if len(s["runs"]) > 1:
        ax_d.legend(fontsize=5.5, frameon=False, loc="upper right",
                    handletextpad=0.4, borderpad=0.2)

    ax_m.set_xlabel("loop iteration")
    ax_m.set_ylabel("state norm")
    ax_m.set_title("The norm does not, and the head cannot see it anyway",
                   loc="left", pad=8)
    # Upper right, which the curve vacates after the first few iterations.
    # Lower left put it straight through the line and the IQR band.
    ax_m.annotate(
        "RMSNorm sits before the head, so two states"
        + chr(10) + "differing only in magnitude decode identically."
        + chr(10) + "A flat norm here is not a null result.",
        xy=(0.97, 0.96), xycoords="axes fraction", ha="right", va="top",
        fontsize=6, color="#555555")

    fig.suptitle(
        "Halting has a scale free signal and a misleading one",
        x=0.008, ha="left", fontsize=8)
    fig.subplots_adjust(top=0.80)

    if out is not None:
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    else:
        for p in style.save(fig, "f5_geometry"):
            print(f"wrote {p}")
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F5 trajectory geometry")
    ap.add_argument("--summary", required=True, type=Path,
                    help="json written by m4_shape --summary")
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--max-runs", type=int, default=4,
                    help="more than a few lines and neither panel is readable")
    args = ap.parse_args(argv)

    s = json.loads(args.summary.read_text(encoding="utf-8"))
    if not s.get("runs"):
        print("no runs in that summary. Nothing is drawn and nothing is claimed.")
        return 2
    s = dict(s, runs=s["runs"][: args.max_runs])

    draw(s, args.out)
    for r in s["runs"]:
        xs, med, _, _ = _series(r, "unit_step_norm")
        nx, nmed, _, _ = _series(r, "norm")
        if med and nmed:
            print(f"{r['run']}: unit step {med[0]:.4f} to {med[-1]:.4f}, "
                  f"norm {nmed[0]:.1f} to {nmed[-1]:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
