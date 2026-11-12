"""F11. The deadlock, the rescue and the trap, in the training curves themselves.

The paper's three findings are claims about how runs move through training,
and until 2026-11-12 no figure showed a single training curve. This one shows
every seed of four arms, held-out accuracy against step, one line per seed:

    cold start                     a donor of the same task
    the composing cue, E2a         the matched pixel that does not compose, E2b

Nothing is selected. Every run of the arm that has a metrics file is drawn, in
one colour, so a reader sees the whole arm and not the seeds someone chose.
Outcomes are not coloured, because the stored outcome labels cover only the
2x2 and a plot script does not compute them.

Two reference levels, both read from stored files and neither computed here:
the escape threshold from the stored rates file, and on the cue panels the
band the trap was registered against, read from its registration output.

    python -m loopvision.analysis.plots.f11_training --runs runs \\
        --rates runs/_figures_20261022_d076/rates.json \\
        --registered runs/_revision_verdicts_20261021/registered.txt \\
        --out figures/f11_training
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

#: Arm prefix and panel title, in reading order: the deadlock and its rescue
#: on top, the trap and its control below.
ARMS = (
    ("famA_d2_d4colour", "Cold start"),
    ("famA_d2_d4colour_curr", "Started from a donor of the same task"),
    ("why_e2a_cue", "Composing cue, E2a"),
    ("why_e2b_decoy", "Matched pixel that does not compose, E2b"),
)

#: The panels that get the registered trap band.
CUE_ARMS = ("why_e2a_cue", "why_e2b_decoy")


def seeds(runs: Path, arm: str) -> list[Path]:
    """Every run of exactly this arm, `<arm>_s<digits>`, with metrics."""
    pat = re.compile(re.escape(arm) + r"_s\d+$")
    return sorted((p for p in runs.iterdir()
                   if pat.fullmatch(p.name) and (p / "metrics.parquet").is_file()),
                  key=lambda p: int(p.name.rsplit("_s", 1)[1]))


def curve(run: Path) -> tuple[list[int], list[float]]:
    """Held-out accuracy at each evaluation step, as stored.

    Each evaluation stores one row per depth and breadth cell and one overall
    row, marked depth -1 and breadth -1. The overall row is the one drawn; the
    cells are not averaged here.
    """
    import pandas as pd

    df = pd.read_parquet(run / "metrics.parquet")
    df = df[(df["split"] == "iid_val") & (df["metric_name"] == "accuracy")
            & (df["depth"] == -1) & (df["breadth"] == -1)]
    df = df.sort_values("step")
    if df["step"].duplicated().any():
        raise ValueError(f"{run.name}: more than one accuracy per step")
    return df["step"].tolist(), df["value"].tolist()


def escape_level(rates: Path) -> float:
    return float(json.loads(rates.read_text(encoding="utf-8"))["escape_tau"])


def trap_band(registered: Path) -> tuple[float, float]:
    """The band as the registration output states it."""
    m = re.search(r"finished between (\d+\.\d+) and (\d+\.\d+)",
                  registered.read_text(encoding="utf-8"))
    if not m:
        raise ValueError(f"no registered band in {registered}")
    return float(m.group(1)), float(m.group(2))


def draw(runs: Path, rates: Path, registered: Path, out: str) -> list[Path]:
    import matplotlib.pyplot as plt

    from loopvision.analysis.plots import style

    style.use()
    tau = escape_level(rates)
    lo, hi = trap_band(registered)
    fig, axes = plt.subplots(2, 2, figsize=(style.WIDTH_FULL, 4.0),
                             sharex=True, sharey=True)
    colour = style.series(0)["color"]
    for ax, (arm, title) in zip(axes.flat, ARMS):
        found = seeds(runs, arm)
        if not found:
            raise FileNotFoundError(f"no runs of {arm} under {runs}")
        for run in found:
            x, y = curve(run)
            ax.plot([s / 1000 for s in x], y, color=colour, alpha=0.45, lw=0.8)
        ax.axhline(tau, color=style.REFERENCE, lw=0.8, ls="--")
        ax.text(298, tau - 0.02, f"escape, {tau:.2f}", ha="right", va="top",
                fontsize=7, color=style.REFERENCE)
        if arm in CUE_ARMS:
            ax.axhspan(lo, hi, color=style.REFERENCE, alpha=0.25, lw=0)
            ax.text(298, hi + 0.02, "what the cue's bit alone scores", ha="right",
                    va="bottom", fontsize=7, color="#444444",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
        ax.set_title(f"{title}, {len(found)} seeds", fontsize=8.5)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, 300)
    for ax in axes[1]:
        ax.set_xlabel("training step (thousands)")
    for ax in axes[:, 0]:
        ax.set_ylabel("held-out accuracy")
    fig.tight_layout()
    return style.save(fig, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="runs", type=Path)
    ap.add_argument("--rates", default="runs/_figures_20261022_d076/rates.json", type=Path)
    ap.add_argument("--registered",
                    default="runs/_revision_verdicts_20261021/registered.txt", type=Path)
    ap.add_argument("--out", default="figures/f11_training")
    args = ap.parse_args(argv)
    for p in draw(args.runs, args.rates, args.registered, args.out):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
