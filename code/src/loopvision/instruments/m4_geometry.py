"""M4. Trajectory geometry: does the state converge in direction while its norm keeps growing?

IMPLEMENTATION.md Section 7 asks for, per iteration, all of `||s_i||`,
`||s_i - s_{i-1}||`, `cos(s_i, s_{i-1})`, `cos(d_i, d_{i-1})` where
`d_i = s_i - s_{i-1}`, and the ratio of directional convergence to norm
growth, both raw and direction normalised. This is the instrument behind
the rays-not-fixed-points question and behind the halting argument, because
a criterion that watches the update norm is watching the growing quantity
rather than the converging one.

WHY THIS IS A SEPARATE PASS RATHER THAN A LOG LINE. The spec says the
geometry costs nothing during an evaluation pass and should therefore be
logged during every evaluation pass. It never was, and 66 finished runs
have no geometry in their metrics. Recomputing it from a stored checkpoint
gives the same numbers, because the quantities are functions of the state
trajectory alone, so nothing is lost by arriving late. Adding the logging
to training now would change what a run records halfway through a campaign
and would not help the runs already finished.

THE TRAP, HONOURED BEFORE ANY FIT EXISTS. A quantity of the form
`1 / slope` blew up in the prior work: one checkpoint at slope 0.036 read
27.6 and moved its group mean by 48 percent. The ratio this instrument is
asked for has that shape, since norm growth appears in the denominator and
is near zero exactly where the state has settled. So the gate is defined
here, in code, before any fit has been run on any output:

    RATIO_MIN_GROWTH = 0.01

An iteration whose absolute norm growth is below that contributes NaN to
the ratio and is counted in `ratio_gated_out`, which is written to the
output like any other number. The components are always stored ungated, so
the ratio can be recomputed under a different gate without rerunning the
model, and `gate_sensitivity` can report the headline both ways.

    python -m loopvision.instruments.m4_geometry --run-id famA_d2_d4only_s0
    python -m loopvision.instruments.m4_geometry --run-id X --items 512 --k 32
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

#: Minimum absolute norm growth for the convergence-to-growth ratio to be
#: defined. Registered here before any fit was run on any output of this
#: module. See the module docstring and IMPLEMENTATION.md Section 7.
RATIO_MIN_GROWTH = 0.01

#: Quantities stored per item per iteration. Components are ungated so the
#: ratio can be recomputed under a different gate without a rerun.
METRICS = (
    "norm",             # ||s_i||
    "step_norm",        # ||s_i - s_{i-1}||
    "cos_state",        # cos(s_i, s_{i-1})
    "cos_step",         # cos(d_i, d_{i-1}), defined from i = 2
    "norm_growth",      # ||s_i|| / ||s_{i-1}|| - 1
    "unit_step_norm",   # ||s_i/||s_i|| - s_{i-1}/||s_{i-1}||||, pure direction
    "ratio",            # (1 - cos_state) / norm_growth, gated
)


def _flat(s: torch.Tensor) -> torch.Tensor:
    """One vector per item. The geometry is of the whole state, not a token."""
    return s.reshape(s.shape[0], -1).float()


def geometry(states: list[torch.Tensor]) -> dict[str, np.ndarray]:
    """Per item, per iteration geometry of a captured trajectory.

    `states` is `states[0..k]` as `hooks.capture` returns it, so index i
    means the state after i core applications. Every returned array is
    shaped `[items, k]` and column j holds iteration i = j + 1, because
    each quantity is a comparison between consecutive states and none of
    them is defined at i = 0.
    """
    if len(states) < 2:
        raise ValueError("need at least s_0 and s_1 to have any geometry")

    flat = [_flat(s) for s in states]
    n = flat[0].shape[0]
    k = len(flat) - 1
    out = {m: np.full((n, k), np.nan, dtype=np.float64) for m in METRICS}

    prev_delta = None
    for i in range(1, k + 1):
        cur, prev = flat[i], flat[i - 1]
        delta = cur - prev
        norm_cur = cur.norm(dim=1)
        norm_prev = prev.norm(dim=1)

        out["norm"][:, i - 1] = norm_cur.numpy()
        out["step_norm"][:, i - 1] = delta.norm(dim=1).numpy()

        cos_state = torch.nn.functional.cosine_similarity(cur, prev, dim=1)
        out["cos_state"][:, i - 1] = cos_state.numpy()

        # Growth as a fraction, so 0 means the norm did not change. A zero
        # previous norm cannot happen for a real state but is guarded
        # rather than trusted.
        growth = torch.where(
            norm_prev > 0, norm_cur / norm_prev - 1.0, torch.full_like(norm_prev, np.nan)
        )
        out["norm_growth"][:, i - 1] = growth.numpy()

        unit_cur = torch.nn.functional.normalize(cur, dim=1)
        unit_prev = torch.nn.functional.normalize(prev, dim=1)
        out["unit_step_norm"][:, i - 1] = (unit_cur - unit_prev).norm(dim=1).numpy()

        if prev_delta is not None:
            out["cos_step"][:, i - 1] = torch.nn.functional.cosine_similarity(
                delta, prev_delta, dim=1
            ).numpy()
        prev_delta = delta

        # The gated ratio. Directional convergence over norm growth, left
        # undefined where the denominator is too small to divide by.
        g = growth.numpy()
        conv = 1.0 - cos_state.numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(np.abs(g) >= RATIO_MIN_GROWTH, conv / g, np.nan)
        out["ratio"][:, i - 1] = ratio

    return out


@torch.no_grad()
def usable_k(model, requested: int) -> int:
    """The deepest trajectory this architecture can actually produce.

    The pbs sends one K for every run in the batch, and two of the three
    architectures have a depth fixed when they were built. Passing 32 to
    them is not a request they can honour:

      tied looped   recurrent, so any k. The request stands.
      feedforward   prelude + k_train * core + coda blocks. Its depth is
                    structural and `forward` ignores the request already.
      untied        one core per iteration, so asking past `untied_copies`
                    raises by design. That guard is right and is why five
                    geometry jobs died: extrapolating an untied model past
                    its own cores would mean reusing a core, which is the
                    one thing the untied baseline exists to avoid.

    Clamping is not extrapolation. It measures the trajectory the model
    actually has, which is the same thing the feedforward path does, and
    `run` records the length it got rather than the length it asked for.
    """
    from loopvision.model.loopvit import FeedforwardBaseline

    if isinstance(model, FeedforwardBaseline):
        return model.k
    cfg = getattr(model, "cfg", None)
    if cfg is not None and not getattr(cfg, "tied", True):
        return min(requested, cfg.untied_copies)
    return requested


def run(run_dir: Path, items: int, k: int | None, device: torch.device) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    model = build_model(cfg, device)
    ckpt = C.find_latest_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=False)["model"]
    )
    model.eval()

    k = k or cfg["k_eval"]
    # Kept before the clamp, because the point of recording it is to show
    # the two differ. Overwriting it here made both fields agree and the
    # clamp invisible, which a test caught on 2026-09-10.
    k_requested = k
    capped = usable_k(model, k)
    if capped != k:
        print(f"  {run_dir.name}: asked for k={k}, this architecture tops out "
              f"at {capped}. Measuring the trajectory it has.")
        k = capped
    arch = cfg.get("arch", "looped")
    state0 = ("post_prelude_activation" if arch == "feedforward"
              else cfg.get("state_init", "randn"))
    batch = cfg["batch_size"]
    batches = max(1, (items + batch - 1) // batch)

    parts: list[dict[str, np.ndarray]] = []
    traj_k = None
    for b in range(batches):
        image, query, _, _, _ = make_batch(cfg, "iid_val", b * batch, device)
        _, states = hooks.capture(model, image, query, k)
        # The trajectory the model actually produced, not the one asked for.
        # A looped model returns k + 1 states and these agree. A feedforward
        # stack has a depth fixed when it was built, so a request for k = 32
        # against a k_train = 8 stack comes back with 9 states, and recording
        # the request would put two different x axes under one column name.
        if traj_k is None:
            traj_k = len(states) - 1
        parts.append(geometry([s.cpu() for s in states]))

    merged = {m: np.concatenate([p[m] for p in parts], axis=0)[:items] for m in METRICS}
    gated = int(np.isnan(merged["ratio"]).sum() - np.isnan(merged["norm_growth"]).sum())
    return {
        "run": run_dir.name,
        "checkpoint": ckpt.name,
        "k": traj_k,
        "k_requested": k_requested,
        # F5 draws both architectures on one axis, and their first columns
        # are not the same measurement: a looped model initialised with randn
        # compares noise against its first real state at i = 1, while a
        # feedforward stack compares two real activations. Recorded so the
        # figure can say so rather than a reader having to know.
        "arch": arch,
        "state0": state0,
        "items": merged["norm"].shape[0],
        "metrics": merged,
        "ratio_gated_out": max(gated, 0),
        "ratio_min_growth": RATIO_MIN_GROWTH,
    }


def to_frame(result: dict):
    import pandas as pd

    n, k = result["metrics"]["norm"].shape
    item = np.repeat(np.arange(n), k)
    iteration = np.tile(np.arange(1, k + 1), n)
    frames = []
    for m in METRICS:
        frames.append(
            pd.DataFrame(
                {
                    "run_id": result["run"],
                    "item": item,
                    "iteration": iteration,
                    "k": result["k"],
                    "metric_name": m,
                    "value": result["metrics"][m].reshape(-1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M4 trajectory geometry")
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
    result = run(run_dir, args.items, args.k, device)

    out = Path(args.out) if args.out else run_dir / "m4_geometry.parquet"
    to_frame(result).to_parquet(out, index=False)

    g = result["metrics"]
    print(f"{result['run']}  {result['checkpoint']}  k={result['k']}  "
          f"items={result['items']}")
    print(f"  wrote {out}")
    print(f"  ratio gated out at |norm_growth| < {RATIO_MIN_GROWTH}: "
          f"{result['ratio_gated_out']} of {g['ratio'].size}")
    print("  iteration    norm   step_norm   cos_state   unit_step")
    for i in range(g["norm"].shape[1]):
        print("  %9d  %6.2f    %8.4f    %8.5f    %8.5f" % (
            i + 1,
            np.nanmean(g["norm"][:, i]),
            np.nanmean(g["step_norm"][:, i]),
            np.nanmean(g["cos_state"][:, i]),
            np.nanmean(g["unit_step_norm"][:, i]),
        ))
    # Beside the parquet, not in the run directory, so --out puts every
    # artifact of a trial run in one place that can be thrown away.
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "run": result["run"],
                "checkpoint": result["checkpoint"],
                "k": result["k"],
                "items": result["items"],
                "ratio_gated_out": result["ratio_gated_out"],
                "ratio_min_growth": result["ratio_min_growth"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
