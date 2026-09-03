"""The blank-image control: does the query alone give away the answer?

Standing rule from docs/learnings.md L-012. A vision task is only a vision
task if blanking the image destroys performance. Twice now a family looked
solved when it was not: family B depth 1 scored 0.833 with the image
blanked because the query described the anchor by the attribute being
asked about, and family C was a presence detector whose label was zero 53
percent of the time. Both were found by asking whether a perfect score was
possible, not by a failing test.

So the check is a module rather than something to remember. Three numbers,
and the comparison between them is the whole instrument:

  real     accuracy on the actual images
  blank    accuracy with the image zeroed, query untouched
  const    the best single label, measured on the same batches

`blank` should sit at `const`. A model cannot do better than guessing the
most common label when it has no image, unless the query is carrying
information it should not. The gap `blank - const` is the leak, and it is
reported directly so nobody has to eyeball two numbers and decide.

This calls `cli.forward` rather than stepping the core itself, per the rule
in CLAUDE.md that instruments never reimplement the loop.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from loopvision.train import checkpoint as C
from loopvision.train.cli import build_model, forward, make_batch

#: How far above the best-constant baseline a blank-image score may sit
#: before the task is treated as leaking. Not zero, because the constant is
#: itself estimated from a finite sample and a model can learn the label
#: prior slightly better than a single argmax over these batches.
LEAK_MARGIN = 0.05


@torch.no_grad()
def blank_control(
    run_dir: Path, split: str = "iid_val", batches: int = 40, k: int | None = None
) -> dict:
    """Compare real, blank-image, and best-constant accuracy for one run."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    k = k or cfg["k_eval"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, device)
    path = C.find_latest_checkpoint(run_dir)
    if path is None:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()

    real_hits, blank_hits, labels = [], [], []
    by_cell: dict[tuple[int, int], list[int]] = {}
    for b in range(batches):
        image, query, label, depths, breadths = make_batch(
            cfg, split, b * cfg["batch_size"], device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            real = forward(model, image, query, k)
            # Zeroed, not noise: noise is a stimulus the model could learn
            # to read, an empty canvas is the absence of one.
            blank = forward(model, torch.zeros_like(image), query, k)
        rh = (real.float().argmax(-1) == label).cpu().numpy()
        bh = (blank.float().argmax(-1) == label).cpu().numpy()
        real_hits.extend(rh.tolist())
        blank_hits.extend(bh.tolist())
        labels.extend(label.cpu().numpy().tolist())
        for h, d, br in zip(bh, depths, breadths):
            by_cell.setdefault((int(d), int(br)), []).append(int(h))

    counts = collections.Counter(labels)
    const = max(counts.values()) / len(labels)
    blank_acc = float(np.mean(blank_hits))
    leak = blank_acc - const

    return {
        "run": run_dir.name,
        "split": split,
        "k": k,
        "n": len(labels),
        "checkpoint": path.name,
        "real": float(np.mean(real_hits)),
        "blank": blank_acc,
        "best_constant": const,
        "leak": leak,
        "leaks": bool(leak > LEAK_MARGIN),
        "blank_by_cell": {
            f"{d}x{b}": float(np.mean(v)) for (d, b), v in sorted(by_cell.items())
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="run directories")
    p.add_argument("--split", default="iid_val")
    p.add_argument("--batches", type=int, default=40)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--json", help="write results here")
    args = p.parse_args()

    out, any_leak = [], False
    for r in args.runs:
        res = blank_control(Path(r), args.split, args.batches, args.k)
        out.append(res)
        any_leak |= res["leaks"]
        verdict = "LEAKS" if res["leaks"] else "clean"
        print(f"\n{res['run']}  ({res['checkpoint']}, k={res['k']}, n={res['n']})")
        print(f"  real           {res['real']:.4f}")
        print(f"  blank image    {res['blank']:.4f}")
        print(f"  best constant  {res['best_constant']:.4f}")
        print(f"  leak           {res['leak']:+.4f}   -> {verdict}")
        if res["leaks"]:
            worst = sorted(
                res["blank_by_cell"].items(), key=lambda kv: -kv[1]
            )[:4]
            print("  worst cells (blank accuracy): "
                  + ", ".join(f"{c}={v:.3f}" for c, v in worst))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print()
    return 1 if any_leak else 0


if __name__ == "__main__":
    raise SystemExit(main())
