"""H3: does patching separate by task family? The registered test.

Implements `docs/preregistration.md` Section 4 exactly. The definitions are
quoted here rather than paraphrased, because this file and that one must
not drift and the prereg is frozen at tag `prereg-v1`:

    Loop dispersion D_loop is the Shannon entropy of the normalised
    distribution sum_p R(i, p) over i, divided by log(k + 1). It is 0 when
    all recovery sits at one iteration and 1 when it is uniform across
    iterations.

    Spatial dispersion D_space is the Shannon entropy of the normalised
    distribution R(i*, p) over p at the peak iteration i*, divided by
    log(P). It is 0 when one position carries everything and 1 when
    recovery is uniform across positions.

Entropy rather than a top-m share, because m would be a free parameter
chosen by whoever writes the plotting code.

**Prediction.** Depth families are loop-localised and spatially
distributed, breadth families are loop-diffuse and spatially local:

    D_loop(depth)  <  D_loop(breadth)
    D_space(depth) >  D_space(breadth)

**Two things this module refuses to do.**

It reads `m2_confirm.parquet` and never `m2_patching.parquet`. The three
exploratory grids are excluded from the confirmation set by Section 0 of
the prereg, and the cheapest way to keep them out is to be unable to load
them.

It discards any cell whose full-state positive control is at or below
0.95, as an instrument failure rather than a null. A cell where the patch
did not land reads as "no recovery anywhere", which is numerically
indistinguishable from a real negative and would drag a dispersion toward
whatever the noise happens to say.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL_FLOOR = 0.95
ALPHA = 0.05
RESAMPLES = 10000

# Which side of the contrast each family sits on. Family A and B are depth
# families, family C is the breadth family. Fixed by the task design, not
# by anything measured.
AXIS = {"A": "depth", "B": "depth", "C": "breadth"}


def dispersions(df: pd.DataFrame) -> tuple[float, float, int]:
    """(D_loop, D_space, peak_iteration) for one cell's grid."""
    g = df[df.position >= 0]
    wide = g.pivot_table(
        index="iteration", columns="position", values="recovery_mean"
    ).to_numpy()
    r = np.clip(np.nan_to_num(wide, nan=0.0), 0.0, None)

    per_iter = r.sum(axis=1)
    if per_iter.sum() <= 0:
        return float("nan"), float("nan"), -1
    D_loop = _norm_entropy(per_iter)

    peak = int(np.argmax(per_iter))
    at_peak = r[peak]
    D_space = _norm_entropy(at_peak) if at_peak.sum() > 0 else float("nan")
    return D_loop, D_space, peak


def _norm_entropy(v: np.ndarray) -> float:
    """Shannon entropy of v normalised to a distribution, over log(len(v))."""
    total = v.sum()
    if total <= 0 or len(v) < 2:
        return float("nan")
    p = v / total
    nz = p[p > 0]
    h = -np.sum(nz * np.log(nz))
    return float(h / np.log(len(v)))


def load_cells(runs_root: Path, family_of) -> pd.DataFrame:
    """Every confirmation cell, with its admission verdict."""
    rows = []
    for f in sorted(runs_root.glob("*/m2_confirm.parquet")):
        run_id = f.parent.name
        d = pd.read_parquet(f)
        ctl_rows = d[(d.position == -1) & (d.iteration == d.iteration.max())]
        control = float(ctl_rows.recovery_mean.iloc[0]) if len(ctl_rows) else float("nan")
        D_loop, D_space, peak = dispersions(d)
        fam = family_of(run_id, d)
        rows.append(
            {
                "run_id": run_id,
                "family": fam,
                "axis": AXIS.get(fam, "?"),
                "control": control,
                "D_loop": D_loop,
                "D_space": D_space,
                "peak_iteration": peak,
                "n_items": int(d[d.position >= 0].n_items.max()),
                "admitted": bool(control > CONTROL_FLOOR),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_difference(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    """Mean(a) - mean(b) with a percentile CI, resampling seeds within group."""
    obs = float(a.mean() - b.mean())
    draws = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        draws[i] = (
            rng.choice(a, size=len(a), replace=True).mean()
            - rng.choice(b, size=len(b), replace=True).mean()
        )
    lo, hi = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return obs, float(lo), float(hi)


def holm(pvalues: dict[str, float], family_size: int) -> dict[str, bool]:
    """Holm at ALPHA across a family of `family_size` registered tests."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    out, blocked = {}, False
    for rank, (name, p) in enumerate(ordered):
        threshold = ALPHA / (family_size - rank)
        if blocked or p > threshold:
            out[name], blocked = False, True
        else:
            out[name] = True
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", default="runs")
    p.add_argument(
        "--family-size",
        type=int,
        default=14,
        help="registered tests in the Holm family, prereg Section 1",
    )
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    def family_of(run_id: str, d: pd.DataFrame) -> str:
        if "family" in d.columns and len(d):
            return str(d.family.iloc[0])
        return run_id[3] if run_id.startswith("fam") else "?"

    cells = load_cells(Path(a.runs_root), family_of)
    if cells.empty:
        print("no m2_confirm.parquet found. The confirmation set has not run.")
        return 1

    print("Cells")
    print(cells.to_string(index=False))
    print()

    dropped = cells[~cells.admitted]
    for _, r in dropped.iterrows():
        print(
            f"DISCARDED {r.run_id}: control {r.control:.4f} at or below "
            f"{CONTROL_FLOOR}, instrument failure not a null"
        )
    ok = cells[cells.admitted]

    per_axis = ok.groupby("axis").run_id.count().to_dict()
    print(f"admitted cells by axis: {per_axis}")
    need = {"depth", "breadth"}
    if not need.issubset(per_axis):
        print()
        print(
            "CANNOT TEST. The contrast needs admitted cells on both axes and "
            f"has {sorted(per_axis)}. Prereg Section 4 requires 5 seeds per "
            "family; this is not a null result, it is an absent one."
        )
        return 2

    rng = np.random.default_rng(a.seed)
    depth = ok[ok.axis == "depth"]
    breadth = ok[ok.axis == "breadth"]

    results = {}
    for metric, predicted in (("D_loop", "depth < breadth"), ("D_space", "depth > breadth")):
        obs, lo, hi = bootstrap_difference(
            depth[metric].to_numpy(), breadth[metric].to_numpy(), rng
        )
        excludes_zero = (lo > 0) or (hi < 0)
        direction = "depth > breadth" if obs > 0 else "depth < breadth"
        results[metric] = {
            "obs": obs,
            "lo": lo,
            "hi": hi,
            "significant": excludes_zero,
            "direction": direction,
            "predicted": predicted,
            "as_predicted": direction == predicted,
        }

    # A CI that excludes zero is the significance criterion here, so the
    # Holm step uses a p-value proxy of alpha when significant. The prereg
    # requires correction across the registered family, and reporting an
    # uncorrected CI as though it were the test would overstate it.
    pv = {k: (ALPHA / 2 if v["significant"] else 1.0) for k, v in results.items()}
    survives = holm(pv, a.family_size)

    print()
    print("H3 contrasts, prereg Section 4")
    print("-" * 70)
    for metric, r in results.items():
        mark = "as predicted" if r["as_predicted"] else "OPPOSITE to prediction"
        print(
            f"  {metric:8} diff {r['obs']:+.4f}  CI [{r['lo']:+.4f}, {r['hi']:+.4f}]  "
            f"{'significant' if r['significant'] else 'contains zero'}"
        )
        print(
            f"           predicted {r['predicted']}, observed {r['direction']}, {mark}"
            f", survives Holm: {survives[metric]}"
        )

    both_pred = all(r["as_predicted"] and r["significant"] and survives[m]
                    for m, r in results.items())
    any_opposite = any((not r["as_predicted"]) and r["significant"] and survives[m]
                       for m, r in results.items())
    both_zero = all(not r["significant"] for r in results.values())

    print()
    if both_pred:
        print("H3 SUPPORTED. Both contrasts significant and in the predicted direction.")
        return 0
    if any_opposite or both_zero:
        print(
            "H3 FALSIFIED. Prereg Section 4: falsified if either contrast is "
            "significant in the opposite direction, or both contain zero."
        )
        return 1
    print(
        "H3 SPLIT. One contrast supported and the other not. Prereg Section 4 "
        "says report it as split and weaken the lead claim, rather than "
        "asserting it on the surviving half."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
