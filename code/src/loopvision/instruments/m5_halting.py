"""M5, H2 half. Where each halting criterion fires, against two oracles.

The paper's one sentence ends on this: we can say by how much and in which
direction the standard halting rules are wrong, because we hold the answer
they are approximating. Prior work scores a criterion against `k*_model`,
an oracle recovered by searching over loop counts. We have that too, and
also `k*_task`, the composition depth of the generating program, which is
known by construction and does not depend on the model at all.

Three quantities per item, exactly as the preregistration fixes them:

  k*_task   composition depth of the program that generated the item
  k*_model  smallest i at which this model answers this item correctly
  k_hat_c   the i at which criterion c says to stop

And a fourth the preregistration does not fix:

  b_task    parallel breadth of that same generating program

BREADTH IS STORED EVEN THOUGH NO REGISTERED TEST READS IT. Both registered
tests per criterion regress against k*_task, and neither mentions breadth.
paper.md Section 4 carries a third claim, the blind axis, asking whether the
error is sensitive to depth and insensitive to breadth. That claim is
descriptive by paper.md's own words, it is post-hoc under Section 1 of the
preregistration, and it is not in the family of fourteen. None of that lets
it be computed after the fact: breadth is a property of the item and is gone
once the pass ends. Recording it costs one array per batch and forecloses
nothing. Every parquet written before 2026-09-10 lacks the column, because
`_pass` discarded the fifth value `make_batch` returns, and analysis/m5_h2.py
reports that absence rather than treating a missing axis as a flat one.

FOUR CRITERIA, FIXED. The preregistration names them so none can be added
or dropped once the numbers exist: update norm below epsilon, Kullback
Leibler below epsilon, predictive entropy below a threshold, and the step
size second difference. The last two are not ours: the entropy rule is the
LoopViT rule from arXiv 2602.02156 and the second difference rule is
Pappone et al. arXiv 2509.23314, included because it is the current best
published exit rule and its absence would make the comparison look chosen.

THE PAPPONE FORM NEEDS CHECKING BEFORE IT ENTERS A DRAFT. The rule is
implemented here as the second difference of the step size sequence,
firing when it first falls below epsilon in absolute value. That is a
defensible reading of a step-size second difference criterion and it is
not a verified transcription of that paper, whose abstract page has not
been fetched. CLAUDE.md forbids a citation entering a draft unverified,
and this is the same rule one level down: the implementation is provisional
until someone reads the paper and confirms or corrects it.

THRESHOLDS ARE FITTED ON TRAIN, NEVER ON THE EVALUATION SPLIT. Section 3
of the preregistration: epsilon and the entropy threshold are each set to
the value minimising median absolute error against `k*_model` on the
training split, chosen before any evaluation split is touched. So this
module fits first, on `train`, prints what it chose, and only then measures
on `iid_val`. Fitting on the split you then report would tune the criteria
to the answer, which is the thing the whole instrument exists to detect in
other people's work.

    python -m loopvision.instruments.m5_halting --run-id famA_d2_d4only_s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from loopvision.instruments import hooks
from loopvision.train import checkpoint as C
from loopvision.train.cli import build_model, make_batch

#: The four criteria, in the order the preregistration lists them.
CRITERIA = ("update_norm", "kl", "entropy", "second_difference")

#: How many candidate thresholds to try per criterion when fitting.
N_CANDIDATES = 60


def candidate_thresholds(values: np.ndarray, n: int = N_CANDIDATES) -> np.ndarray:
    """Candidate epsilons drawn from the signal itself, not a fixed grid.

    A fixed logspace grid was the first version and it failed silently:
    update norms here run to 1e8 on some runs and to 1e2 on others, so any
    hardcoded range is either too coarse or misses the optimum entirely.
    The first real run fitted three of four criteria to exactly 10, which
    was the top of the grid, meaning the search wanted to go higher and
    could not. A threshold pinned to the edge of its own search range is
    not a fitted threshold, it is an artifact of the range.

    Quantiles of the observed magnitudes cannot have that failure: the
    optimum is inside the candidate set by construction, whatever scale the
    run happens to live at. The endpoints are widened slightly so that
    "fire immediately" and "never fire" are both reachable, since both are
    legitimate answers about a criterion.
    """
    v = np.abs(values[np.isfinite(values)])
    if v.size == 0:
        return np.array([1.0])
    qs = np.quantile(v, np.linspace(0.0, 1.0, n))
    lo, hi = v.min(), v.max()
    span = np.array([lo * 0.5 if lo > 0 else 0.0, hi * 2.0 if hi > 0 else 1.0])
    return np.unique(np.concatenate([qs, span]))

#: Value recorded when a quantity never happens within the k that was run.
#: Right censoring, kept explicit rather than folded into a number, so an
#: item that never halts cannot be mistaken for one that halted late.
CENSORED = -1


def _probs(model, states: list[torch.Tensor]) -> torch.Tensor:
    """Decode every state through the real coda and head.

    `model.decode` is the same path the model itself uses, exposed for
    exactly this reason. An instrument with its own copy of the head can
    drift from the trained one and the drift is invisible.
    """
    out = []
    for s in states:
        out.append(torch.softmax(model.decode(s).float(), dim=-1))
    return torch.stack(out, dim=0)  # [k+1, B, classes]


def trajectory_signals(states: list[torch.Tensor], probs: torch.Tensor) -> dict:
    """The per-iteration quantities every criterion reads.

    Index j of each array is iteration i = j + 1, matching M4 and the hook
    index convention: none of these is defined at i = 0.
    """
    k = len(states) - 1
    flat = [s.reshape(s.shape[0], -1).float() for s in states]
    n = flat[0].shape[0]

    step = np.zeros((n, k))
    kl = np.zeros((n, k))
    ent = np.zeros((n, k))
    for i in range(1, k + 1):
        step[:, i - 1] = (flat[i] - flat[i - 1]).norm(dim=1).numpy()
        p, q = probs[i], probs[i - 1]
        kl[:, i - 1] = (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(-1).numpy()
        ent[:, i - 1] = (-(p * p.clamp_min(1e-12).log()).sum(-1)).numpy()

    # Second difference of the step size sequence, defined from i = 3.
    second = np.full((n, k), np.nan)
    if k >= 3:
        second[:, 2:] = step[:, 2:] - 2 * step[:, 1:-1] + step[:, :-2]
    return {"step": step, "kl": kl, "entropy": ent, "second": second}


def _first_below(values: np.ndarray, threshold: float) -> np.ndarray:
    """First iteration index (1 based) where |value| drops below threshold."""
    hit = np.abs(values) < threshold
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1) + 1, CENSORED)
    return first.astype(np.int64)


def fire_points(signals: dict, thresholds: dict) -> dict:
    """`k_hat_c` for each criterion, given fitted thresholds."""
    return {
        "update_norm": _first_below(signals["step"], thresholds["update_norm"]),
        "kl": _first_below(signals["kl"], thresholds["kl"]),
        "entropy": _first_below(signals["entropy"], thresholds["entropy"]),
        "second_difference": _first_below(
            signals["second"], thresholds["second_difference"]
        ),
    }


def k_star_model(probs: torch.Tensor, label: torch.Tensor) -> np.ndarray:
    """Smallest i at which the decoded state is already correct.

    Censored when the model is never right at any i, which is not the same
    as being right late and must not be averaged in as though it were.
    """
    preds = probs.argmax(-1)          # [k+1, B]
    correct = (preds == label.unsqueeze(0)).numpy()[1:]  # drop i = 0
    any_hit = correct.any(axis=0)
    first = np.where(any_hit, correct.argmax(axis=0) + 1, CENSORED)
    return first.astype(np.int64)


def median_abs_error(k_hat: np.ndarray, k_model: np.ndarray) -> float:
    """Median |k_hat - k*_model| over items where both are defined."""
    ok = (k_hat != CENSORED) & (k_model != CENSORED)
    if not ok.any():
        return float("inf")
    return float(np.median(np.abs(k_hat[ok] - k_model[ok])))


def fit_thresholds(signals: dict, k_model: np.ndarray) -> dict:
    """Choose epsilon per criterion by minimising median absolute error.

    Run on the training split and nowhere else. Ties go to the smaller
    threshold, which is the more conservative exit rule, so the choice is
    deterministic rather than dependent on grid order.
    """
    chosen = {}
    for name, key in (
        ("update_norm", "step"),
        ("kl", "kl"),
        ("entropy", "entropy"),
        ("second_difference", "second"),
    ):
        grid = candidate_thresholds(signals[key])
        best, best_err = None, float("inf")
        for t in grid:
            err = median_abs_error(_first_below(signals[key], t), k_model)
            if err < best_err:
                best, best_err = float(t), err
        chosen[name] = best if best is not None else float(grid[0])
    return chosen


@torch.no_grad()
def _pass(model, cfg, split: str, items: int, k: int, device) -> dict:
    batch = cfg["batch_size"]
    batches = max(1, (items + batch - 1) // batch)
    sig_parts, kmodel_parts, ktask_parts, btask_parts = [], [], [], []
    for b in range(batches):
        image, query, label, depths, breadths = make_batch(
            cfg, split, b * batch, device)
        _, states = hooks.capture(model, image, query, k)
        states = [s.cpu() for s in states]
        probs = _probs(model, states)
        sig_parts.append(trajectory_signals(states, probs))
        kmodel_parts.append(k_star_model(probs, label.cpu()))
        ktask_parts.append(np.asarray([int(d) for d in depths], dtype=np.int64))
        btask_parts.append(np.asarray([int(x) for x in breadths], dtype=np.int64))
    signals = {
        key: np.concatenate([p[key] for p in sig_parts], axis=0)[:items]
        for key in sig_parts[0]
    }
    return {
        "signals": signals,
        "k_model": np.concatenate(kmodel_parts)[:items],
        "k_task": np.concatenate(ktask_parts)[:items],
        "b_task": np.concatenate(btask_parts)[:items],
    }


@torch.no_grad()
def run(run_dir: Path, items: int, k: int | None, device) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    model = build_model(cfg, device)
    ckpt = C.find_latest_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=False)["model"]
    )
    model.eval()
    k = k or max(cfg.get("eval_k_sweep") or [cfg["k_eval"]])

    # Fit on train. Nothing from iid_val has been touched at this point.
    train = _pass(model, cfg, "train", items, k, device)
    thresholds = fit_thresholds(train["signals"], train["k_model"])

    # Then measure, on the split that gets reported.
    val = _pass(model, cfg, "iid_val", items, k, device)
    fired = fire_points(val["signals"], thresholds)

    return {
        "run": run_dir.name,
        "checkpoint": ckpt.name,
        "k": k,
        "items": len(val["k_model"]),
        "thresholds": thresholds,
        "k_model": val["k_model"],
        "k_task": val["k_task"],
        "b_task": val["b_task"],
        "fired": fired,
        "never_correct": int((val["k_model"] == CENSORED).sum()),
    }


def to_frame(result: dict):
    import pandas as pd

    cols = {
        "run_id": result["run"],
        "item": np.arange(result["items"]),
        "k": result["k"],
        "k_task": result["k_task"],
        "b_task": result["b_task"],
        "k_model": result["k_model"],
    }
    for c in CRITERIA:
        cols[f"k_hat_{c}"] = result["fired"][c]
    return pd.DataFrame(cols)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M5 halting oracle, H2 half")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-root", default=Path("runs"), type=Path)
    ap.add_argument("--items", type=int, default=256)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    run_dir = args.runs_root / args.run_id
    if not run_dir.is_dir():
        print(f"no such run: {run_dir}")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = run(run_dir, args.items, args.k, device)

    out = Path(args.out) if args.out else run_dir / "m5_halting.parquet"
    to_frame(r).to_parquet(out, index=False)

    print(f"{r['run']}  {r['checkpoint']}  k={r['k']}  items={r['items']}")
    print(f"  wrote {out}")
    print("  thresholds fitted on train:")
    for c in CRITERIA:
        print(f"    {c:<20} {r['thresholds'][c]:.6g}")
    print(f"  items never correct at any k: {r['never_correct']} of {r['items']}")
    km, kt, bt = r["k_model"], r["k_task"], r["b_task"]
    # Which axis this run can supply a slope for, printed because a run that
    # holds one depth gives the registered slope no leverage and a run that
    # holds one breadth cannot speak to the blind axis at all. Reading it off
    # the log is cheaper than loading the parquet to find out.
    print("  axes in this run: depth %s, breadth %s" % (
        sorted({int(x) for x in kt}), sorted({int(x) for x in bt})))
    ok = km != CENSORED
    if ok.any():
        print(f"  k*_model median {np.median(km[ok]):.1f}   "
              f"k*_task median {np.median(kt[ok]):.1f}")
        print("  criterion            median k_hat   median signed error   censored")
        for c in CRITERIA:
            kh = r["fired"][c]
            both = ok & (kh != CENSORED)
            if not both.any():
                print(f"    {c:<20} never fires")
                continue
            print("    %-20s %8.1f %18.1f %10d" % (
                c, np.median(kh[both]), np.median(kh[both] - km[both]),
                int((kh == CENSORED).sum())))
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "run": r["run"], "checkpoint": r["checkpoint"], "k": r["k"],
                "items": r["items"], "thresholds": r["thresholds"],
                "never_correct": r["never_correct"],
                "depth_levels": sorted({int(x) for x in r["k_task"]}),
                "breadth_levels": sorted({int(x) for x in r["b_task"]}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
