"""The logit-lens leg of the protocol, on family A.

paper.md section 3a claims a protocol for trustworthy claims about absent
latent computation: a logit lens for what the head reads, a trained probe
for what is linearly present, a positive control, and ground-truth
intermediates. Three of those four exist on family A, which is where the
binding claim lives. The lens leg did not, because `coda_lens` decodes hops
along family B's chain and cannot be pointed at a composition task. So the
protocol's own demonstration was missing a leg on the case it demonstrates.

This is that leg. For each core pass it decodes the state through
`model.decode`, the real output head, and scores the prediction against the
same five ground-truth intermediates `state_probe` uses:

    initial    the queried row's starting element
    op1, op2   each operator in the queried strip
    partial    op1 applied to initial, the halfway composite
    composite  the answer

Read beside the probe on the same run, the pair separates two failures a
lens alone cannot:

    lens at chance, probe reads it     computed and not routed to the head
    lens at chance, probe at chance    never computed at all

WHY THERE IS NO SITE ARGUMENT, unlike `state_probe`. A probe can be trained
on any pooling of the state, so that module offers query and patch. A lens
can only see what the head sees, and `decode` pools over the query tokens
by construction. Offering a patch-pooled lens would mean reaching past the
head and calling the result what the head reads. The asymmetry is the point
of the pair rather than an omission.

WHAT MAKES THIS WELL DEFINED. Family A's label space is the full group, 48
elements, and the S3 and D4 factors are projections of it, `element % 6`
and `element // 6`. So every intermediate is expressible in the head's own
output space, exactly as a family B hop label is, and the lens compares
like with like rather than through a learned mapping.

    python -m loopvision.analysis.factor_lens --run runs/famA_d2_d4colour_s0
    python -m loopvision.analysis.factor_lens --run runs/famA_d2_d4only_s0 \
        --factor d4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from loopvision.analysis.state_probe import FACTORS, TARGETS, targets_for
from loopvision.data import dataset as D
from loopvision.data import family_a as FA
from loopvision.instruments import hooks
from loopvision.train import checkpoint as C
from loopvision.train.cli import build_model, make_batch


def assert_same_question(composite: np.ndarray, label_factor: np.ndarray) -> None:
    """The regenerated program must be the one the batch actually asked.

    The guard `coda_lens` carries, for the same reason. If the program
    regenerated here disagrees with the batch's own label, every number
    below is scored against a different question than the model answered,
    and the result looks exactly like a null: a lens reading chance because
    it is asking the wrong thing is indistinguishable, in the output, from
    a model that computed nothing.
    """
    if not np.array_equal(composite, label_factor):
        n = int((composite != label_factor).sum())
        raise RuntimeError(
            f"the regenerated composite disagrees with the batch's own "
            f"labels on {n} of {len(composite)} items, so the lens would be "
            f"scoring against the wrong question"
        )


def score(preds: list, targets: dict) -> list[list[float]]:
    """Per pass, how often the head's reading equals each target.

    Pure, so the arithmetic that turns predictions into the panel can be
    tested without a checkpoint. `preds[i]` is the head's factor reading of
    the state after pass i, already projected onto the factor.
    """
    return [[float(np.mean(np.asarray(p) == np.asarray(targets[t])))
             for t in TARGETS] for p in preds]


@torch.no_grad()
def lens(run_dir: Path, k: int = 8, batches: int = 12,
         split: str = "iid_val", factor: str = "s3") -> dict:
    """How often the head's own reading of each state equals each target."""
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

    index, classes = FACTORS[factor]
    tcfg = D.TaskConfig(
        master_seed=cfg["master_seed"],
        depths=tuple(cfg["depths"]) if cfg["depths"] else None,
        breadths=tuple(cfg["breadths"]) if cfg["breadths"] else None,
        factor=cfg.get("factor", "full"),
        substrate=cfg.get("substrate", "native"),
    )

    # Predictions and targets are accumulated across batches and scored
    # once, so `score` sees the whole evaluation set rather than a mean of
    # per batch means, which would be a different number whenever the last
    # batch is short.
    preds: dict[int, list] = {}
    targets: dict[str, list] = {t: [] for t in TARGETS}
    n_states = 0

    for b in range(batches):
        cursor = b * cfg["batch_size"]
        image, query, label, _, _ = make_batch(cfg, split, cursor, device)
        rows = []
        for i in range(image.shape[0]):
            s = FA.generate(D.global_index(split, cursor + i), split, tcfg)
            rows.append(targets_for(s.program, factor))
        batch_targets = {t: np.array([r[t] for r in rows]) for t in TARGETS}
        assert_same_question(batch_targets["composite"],
                             index(label.cpu().numpy()))
        for t in TARGETS:
            targets[t].extend(batch_targets[t].tolist())

        s0 = hooks.shared_init_state(model, image, query)
        _, states = hooks.capture(model, image, query, k=k, s0=s0)
        n_states = len(states)
        for it, s in enumerate(states):
            pred = index(model.decode(s).float().argmax(-1).cpu().numpy())
            preds.setdefault(it, []).extend(pred.tolist())

    grid = score([preds[it] for it in range(n_states)], targets)
    return {
        "run": run_dir.name,
        "checkpoint": ckpt.name,
        "k": k,
        "factor": factor,
        "chance": 1.0 / classes,
        "n": len(targets[TARGETS[0]]),
        "targets": list(TARGETS),
        "grid": grid,
        "site": "query, because decode pools there and a lens cannot "
                "read anywhere the head does not",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Family A logit lens")
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--split", default="iid_val")
    ap.add_argument("--factor", choices=sorted(FACTORS), default="s3")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        print(f"no such run: {run_dir}")
        return 2

    r = lens(run_dir, args.k, args.batches, args.split, args.factor)
    out = Path(args.json) if args.json else run_dir / f"factor_lens_{args.factor}.json"
    out.write_text(json.dumps(r, indent=2), encoding="utf-8")

    print(f"{r['run']}  {r['checkpoint']}  factor={r['factor']}  "
          f"k={r['k']}  n={r['n']}  chance={r['chance']:.4f}")
    print(f"  wrote {out}")
    print("        " + "".join(t.rjust(11) for t in TARGETS))
    for it, row in enumerate(r["grid"]):
        print(f"  pass{it:<2d}" + "".join(f"{v:11.3f}" for v in row))
    peak = max(max(row) for row in r["grid"])
    print(f"  best readout {peak:.3f} against a floor of {r['chance']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
