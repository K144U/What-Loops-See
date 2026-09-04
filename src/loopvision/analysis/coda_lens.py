"""How far along the chain has the model got after each core pass?

The loop-count curve says deeper questions need more passes. It does not
say why, and three seeds that agree on the direction disagree by a factor
of three on the magnitude: inferred from k_min, one advances about 1.8
hops per pass and another about 0.8. That number is reverse engineered
from a thresholded accuracy, so it could be an artefact of the threshold
rather than a property of the model.

This measures it directly. For a depth d question the answer after j hops
is known by construction (`family_b.hop_labels`), so decoding the state
after core pass i through the model's own output head says which hop the
model currently believes it is on. The frontier of that matrix against i
is the hop rate, measured rather than inferred.

Uses `model.decode`, the same path the real output head takes, so the lens
cannot drift from the model. Uses `instruments.hooks.capture` to run the
loop rather than stepping the core here, per CLAUDE.md.

    python -m loopvision.analysis.coda_lens --run runs/famB_curve111_s0 --depth 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from loopvision.data import dataset as D
from loopvision.data import family_b as FB
from loopvision.instruments import hooks
from loopvision.train import checkpoint as C
from loopvision.train.cli import build_model, make_batch

#: A hop counts as reached when the decoded state predicts its label more
#: often than this. Set well above family B's 0.1833 floor, and reported
#: alongside a sweep so a single choice never carries the conclusion.
REACHED = 0.50


@torch.no_grad()
def lens(
    run_dir: Path, depth: int, k: int = 8, batches: int = 8, split: str = "iid_val"
) -> dict:
    """Accuracy of each intermediate state against each hop along the chain."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    cfg = dict(cfg)
    cfg["depths"] = [depth]  # one depth at a time, or the hops do not line up
    tcfg = D.TaskConfig(
        master_seed=cfg["master_seed"],
        depths=(depth,),
        breadths=tuple(cfg["breadths"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    ckpt = C.find_latest_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=False)["model"]
    )
    model.eval()

    # hits[i][j] collects, for core pass i, whether the decoded prediction
    # equals the label at hop j.
    hits = [[[] for _ in range(depth)] for _ in range(k + 1)]

    for b in range(batches):
        cursor = b * cfg["batch_size"]
        image, query, label, _, _ = make_batch(cfg, split, cursor, device)
        targets = np.array([
            FB.hop_labels(D.global_index(split, cursor + i), split, tcfg)
            for i in range(image.shape[0])
        ])
        if not np.array_equal(targets[:, -1], label.cpu().numpy()):
            raise RuntimeError(
                "the regenerated chain disagrees with the batch's own labels, "
                "so the lens would be scoring against the wrong question"
            )

        s0 = hooks.shared_init_state(model, image, query)
        _, states = hooks.capture(model, image, query, k=k, s0=s0)
        for i, s in enumerate(states):
            pred = model.decode(s).float().argmax(-1).cpu().numpy()
            for j in range(depth):
                hits[i][j].extend((pred == targets[:, j]).tolist())

    grid = np.array([[float(np.mean(hits[i][j])) for j in range(depth)]
                     for i in range(len(states))])

    # The frontier: the deepest hop this state has reached. Hop 0 is the
    # anchor, so a frontier of 0 means the model has parsed the question
    # and gone nowhere.
    frontier = [
        max([j for j in range(depth) if grid[i][j] >= REACHED], default=-1)
        for i in range(grid.shape[0])
    ]
    return {
        "run": run_dir.name,
        "depth": depth,
        "k": k,
        "checkpoint": ckpt.name,
        "n": len(hits[0][0]),
        "grid": grid.tolist(),
        "frontier": frontier,
        "final_accuracy": float(grid[-1][depth - 1]),
    }


def hop_rate(frontier: list[int]) -> float | None:
    """Hops advanced per core pass, from the frontier's slope.

    Fitted only over the passes where the frontier is still moving. Once
    the chain is finished the frontier flattens, and including that tail
    would drag the slope toward zero for exactly the models that solve the
    question fastest.
    """
    pts = [(i, f) for i, f in enumerate(frontier) if f >= 0]
    if len(pts) < 2:
        return None
    top = max(f for _, f in pts)
    if top == pts[0][1]:
        # The frontier never rose. That is a model which parses the
        # question and then goes nowhere, and it is a rate of zero rather
        # than an absence of measurement. Trimming here would leave one
        # point and report None, hiding the most interesting failure.
        return 0.0
    first_top = next(i for i, f in pts if f == top)
    pts = [(i, f) for i, f in pts if i <= first_top]
    if len(pts) < 2:
        return None
    xs = [float(i) for i, _ in pts]
    ys = [float(f) for _, f in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    return None if denom == 0 else sum(
        (x - mx) * (y - my) for x, y in zip(xs, ys)
    ) / denom


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--json")
    args = p.parse_args()

    r = lens(Path(args.run), args.depth, args.k, args.batches)
    print(f"\n{r['run']}  depth {r['depth']}  k {r['k']}  n {r['n']}  ({r['checkpoint']})")
    print("  accuracy of the state after each core pass against each hop")
    print("  rows are core passes, columns are hops along the chain")
    print()
    print("        " + "".join(f"hop{j}".rjust(8) for j in range(r["depth"])))
    for i, row in enumerate(r["grid"]):
        mark = "" if r["frontier"][i] < 0 else f"   frontier hop {r['frontier'][i]}"
        print(f"  pass{i:<2}" + "".join(f"{v:.3f}".rjust(8) for v in row) + mark)
    rate = hop_rate(r["frontier"])
    print()
    print(f"  measured hop rate: {rate if rate is None else round(rate, 2)} hops per core pass")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        r["hop_rate"] = rate
        Path(args.json).write_text(json.dumps(r, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
