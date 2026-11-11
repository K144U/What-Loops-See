"""Blank one part of the scene and see whether the model notices.

A blank-image control asks whether the model uses the image at all. It
passes as long as *something* in the picture is being read, so it cannot
tell a model that composes its operators from one that found a shortcut
using only part of the scene. This asks the sharper question: is every
input the task claims to require actually required?

For family A the queried row holds an initial state and `depth` operator
glyphs, and the answer is their ordered composite. So:

  state       blank the queried row's initial state
  op1..opN    blank one operator of the queried row
  distractor  blank every row except the queried one
  marker      blank the query marker

A model that genuinely composes must collapse toward chance when any of
the state or operator patches is removed, and must not care at all about
the distractor rows. Anything else is a shortcut, and the pattern of which
ablation hurts says which shortcut.

The counterpart to `blank_control`: that one removes everything, this one
removes exactly one thing at a time.

    python -m loopvision.analysis.ablate_input --run runs/famA_d2_d4only_s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from loopvision.data import dataset as D
from loopvision.data import family_a as FA
from loopvision.data import render
from loopvision.train import checkpoint as C
from loopvision.train.cli import build_model, forward, make_batch


def blank_patch(image: torch.Tensor, row: int, col: int) -> None:
    """Overwrite one grid cell with the background colour, in place."""
    p = render.PATCH
    y, x = row * p, col * p
    for c in range(3):
        image[c, y : y + p, x : x + p] = render.BACKGROUND[c] / 255.0 \
            if image.max() <= 1.0 + 1e-6 else render.BACKGROUND[c]


@torch.no_grad()
def ablate(run_dir: Path, batches: int = 16, split: str = "iid_val") -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    ckpt = C.find_latest_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=False)["model"]
    )
    model.eval()

    tcfg = D.TaskConfig(
        master_seed=cfg["master_seed"],
        depths=tuple(cfg["depths"]) if cfg["depths"] else None,
        breadths=tuple(cfg["breadths"]) if cfg["breadths"] else None,
        factor=cfg.get("factor", "full"),
        substrate=cfg.get("substrate", "native"),
    )
    depth = max(tcfg.depths) if tcfg.depths else 1
    names = ["none", "state"] + [f"op{i+1}" for i in range(depth)] + [
        "distractors", "marker", "all"
    ]
    hits = {n: [] for n in names}

    for b in range(batches):
        cursor = b * cfg["batch_size"]
        image, query, label, _, _ = make_batch(cfg, split, cursor, device)
        progs = [
            FA.parse_program(
                FA.generate(D.global_index(split, cursor + i), split, tcfg).program
            )
            for i in range(image.shape[0])
        ]
        queried = [p[2] for p in progs]
        breadths = [p[4] for p in progs]

        for name in names:
            img = image.clone()
            for i in range(img.shape[0]):
                q = queried[i]
                if name == "state":
                    blank_patch(img[i], q, render.STATE_COL)
                elif name.startswith("op"):
                    j = int(name[2:]) - 1
                    blank_patch(img[i], q, render.FIRST_OP_COL + j)
                elif name == "distractors":
                    for r in range(breadths[i]):
                        if r == q:
                            continue
                        for c in range(render.MARKER_COL + 1):
                            blank_patch(img[i], r, c)
                elif name == "marker":
                    blank_patch(img[i], q, render.MARKER_COL)
                elif name == "all":
                    img[i] = torch.zeros_like(img[i])
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = forward(model, img, query, cfg["k_eval"])
            hits[name].extend(
                (logits.float().argmax(-1) == label).cpu().numpy().tolist()
            )

    return {
        "run": run_dir.name,
        "checkpoint": ckpt.name,
        "n": len(hits["none"]),
        "accuracy": {n: float(np.mean(v)) for n, v in hits.items()},
    }


def verdict(acc: dict, chance: float) -> list[str]:
    """Name what each ablation implies, rather than leaving a table."""
    out = []
    intact = acc["none"]
    required = [k for k in acc if k == "state" or k.startswith("op")]
    for k in required:
        drop = intact - acc[k]
        if acc[k] > chance + 0.10:
            out.append(
                f"  WARNING: blanking {k} leaves accuracy at {acc[k]:.3f}, well "
                f"above chance {chance:.3f}. The task claims to need it and the "
                f"model apparently does not, which is a shortcut."
            )
        else:
            out.append(f"  {k}: required, accuracy falls {intact:.3f} to {acc[k]:.3f}")
    d = acc.get("distractors")
    if d is not None:
        if abs(d - intact) > 0.05:
            out.append(
                f"  WARNING: removing the distractor rows moved accuracy "
                f"{intact:.3f} to {d:.3f}. They carry no information about the "
                f"answer by construction, so the model is using them."
            )
        else:
            out.append(f"  distractors: correctly ignored ({d:.3f})")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--chance", type=float, default=None)
    p.add_argument("--json")
    args = p.parse_args()

    r = ablate(Path(args.run), args.batches)
    cfg = yaml.safe_load((Path(args.run) / "config.yaml").read_text(encoding="utf-8"))
    from loopvision.data import groups as G

    sub = G.substrate_subgroup(cfg.get("substrate", "native"), cfg.get("factor", "full"))
    chance = args.chance if args.chance is not None else 1.0 / len(
        G.subgroup_members(sub)
    )

    print(f"\n{r['run']}  ({r['checkpoint']}, n={r['n']}, chance {chance:.4f})")
    for k, v in r["accuracy"].items():
        print(f"    {k:<12} {v:.4f}")
    print()
    for line in verdict(r["accuracy"], chance):
        print(line)
    print()

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        r["chance"] = chance
        Path(args.json).write_text(json.dumps(r, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
