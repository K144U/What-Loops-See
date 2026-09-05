"""Is the S3 information absent from the model, or merely unreadable?

The coda lens says the output head decodes nothing about S3 at any core
pass. A logit lens can only see what the frozen head reads, so that result
cannot distinguish two very different failures:

  never computed   the model does not extract S3 from the image at all
  never routed     it is in the residual stream and the head cannot use it

Those call for different responses. The first is a claim about what the
architecture can learn; the second is a claim about what it learns to
*move*, which is a much narrower and more fixable problem. A probe trained
on the states separates them, because it is not restricted to what the
head happens to read.

Five targets, all 6-way over the S3 factor, decomposing the computation
rather than only asking for the answer:

  initial    the S3 part of the queried row's starting state
  op1, op2   the S3 part of each operator in the queried strip
  partial    op1 applied to initial, the halfway composite
  composite  the answer

If the operators are readable and the composite is not, the model sees the
inputs and fails to combine them. If nothing is readable, it never looked.

Two pooling sites, because where the information sits matters as much as
whether it is there:

  query    mean over the query tokens, exactly where `decode` pools from
  patch    mean over the image patch tokens

Readable at `patch` and not at `query` is the routing failure, stated
precisely: extracted from the image, never delivered to the position the
answer is assembled at.

    python -m loopvision.analysis.state_probe --run runs/famA_d2_s3only_s0
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
from loopvision.data import groups as G
from loopvision.instruments import hooks
from loopvision.train import checkpoint as C
from loopvision.train.cli import build_model, make_batch

TARGETS = ("initial", "op1", "op2", "partial", "composite")
SITES = ("query", "patch")


def s3_index(element: int) -> int:
    """The S3 factor of a group element, as 0..5."""
    return element % G.S3_ORDER


def d4_index(element: int) -> int:
    """The D4 factor of a group element, as 0..7."""
    return element // G.S3_ORDER


#: Which factor to probe for, and how many classes it has. The D4 entry
#: exists to make the S3 null falsifiable: a probe that reads nothing is
#: only evidence of absence if the same probe reads a factor the model is
#: known to have. Running it on the solved d4only model is that control.
FACTORS = {"s3": (s3_index, G.S3_ORDER), "d4": (d4_index, G.D4_ORDER)}


def targets_for(program: str, factor: str = "s3") -> dict[str, int]:
    """The five quantities for one sample, in the chosen factor."""
    index, _ = FACTORS[factor]
    initial, strips, queried, _, _ = FA.parse_program(program)
    ops = strips[queried]
    start = initial[queried]
    partial = G.multiply(ops[0], start)
    composite = G.compose_sequence(ops)
    return {
        "initial": index(start),
        "op1": index(ops[0]),
        "op2": index(ops[1]) if len(ops) > 1 else index(ops[0]),
        "partial": index(partial),
        "composite": index(G.multiply(composite, start)),
    }


def fit_probe(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
    classes: int = G.S3_ORDER, epochs: int = 40, seed: int = 0,
) -> float:
    """Held-out accuracy of a multinomial logistic regression.

    Numpy rather than scikit-learn, matching `validate.bag_of_operators_probe`,
    because one linear model does not justify a dependency entry.

    Features are standardised, since residual-stream scales differ by
    orders of magnitude across core passes and an unnormalised probe would
    report the scale rather than the information.
    """
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    W = rng.normal(0, 0.01, size=(d, classes)).astype(np.float32)
    b = np.zeros(classes, dtype=np.float32)
    onehot = np.eye(classes, dtype=np.float32)[ytr]

    lr, batch = 0.2, 256
    for _ in range(epochs):
        order = rng.permutation(len(Xtr))
        for start in range(0, len(Xtr), batch):
            idx = order[start : start + batch]
            xb, yb = Xtr[idx], onehot[idx]
            logits = xb @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            grad = (p - yb) / len(idx)
            W -= lr * (xb.T @ grad)
            b -= lr * grad.sum(axis=0)

    return float((np.argmax(Xte @ W + b, axis=1) == yte).mean())


@torch.no_grad()
def collect(run_dir: Path, k: int, batches: int, split: str = "iid_val",
            factor: str = "s3"):
    """Pooled states at every core pass, plus the five targets per sample."""
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

    n_patch = model.cfg.num_patches
    feats: dict[tuple[int, str], list[np.ndarray]] = {}
    ys: dict[str, list[int]] = {t: [] for t in TARGETS}
    tcfg = D.TaskConfig(
        master_seed=cfg["master_seed"],
        depths=tuple(cfg["depths"]) if cfg["depths"] else None,
        breadths=tuple(cfg["breadths"]) if cfg["breadths"] else None,
        factor=cfg.get("factor", "full"),
    )

    for b in range(batches):
        cursor = b * cfg["batch_size"]
        image, query, _, _, _ = make_batch(cfg, split, cursor, device)
        for i in range(image.shape[0]):
            s = FA.generate(D.global_index(split, cursor + i), split, tcfg)
            for name, v in targets_for(s.program, factor).items():
                ys[name].append(v)

        s0 = hooks.shared_init_state(model, image, query)
        _, states = hooks.capture(model, image, query, k=k, s0=s0)
        for it, st in enumerate(states):
            q = st[:, n_patch:, :].mean(dim=1).float().cpu().numpy()
            pch = st[:, :n_patch, :].mean(dim=1).float().cpu().numpy()
            feats.setdefault((it, "query"), []).append(q)
            feats.setdefault((it, "patch"), []).append(pch)

    X = {key: np.concatenate(v) for key, v in feats.items()}
    Y = {t: np.array(v) for t, v in ys.items()}
    return X, Y, len(states)


def run(run_dir: Path, k: int = 8, batches: int = 12, split: str = "iid_val",
        factor: str = "s3") -> dict:
    X, Y, n_states = collect(run_dir, k, batches, split, factor)
    classes = FACTORS[factor][1]
    n = len(next(iter(Y.values())))
    cut = int(n * 0.8)

    out: dict = {"run": run_dir.name, "k": k, "n": n, "n_train": cut,
                 "factor": factor, "chance": 1.0 / classes, "probes": {}}
    for it in range(n_states):
        for site in SITES:
            feat = X[(it, site)]
            for target in TARGETS:
                y = Y[target]
                acc = fit_probe(feat[:cut], y[:cut], feat[cut:], y[cut:], classes)
                out["probes"][f"pass{it}|{site}|{target}"] = acc
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--batches", type=int, default=12)
    p.add_argument("--factor", default="s3", choices=sorted(FACTORS))
    p.add_argument("--json")
    args = p.parse_args()

    r = run(Path(args.run), args.k, args.batches, factor=args.factor)
    chance = r["chance"]
    print(f"\n{r['run']}  k={r['k']}  n={r['n']} ({r['n_train']} train)")
    print(f"  linear probe accuracy for the {r['factor'].upper()} factor, "
          f"chance {chance:.4f}")
    for site in SITES:
        print(f"\n  pooled at the {site} tokens")
        print("        " + "".join(t.rjust(11) for t in TARGETS))
        for it in range(args.k + 1):
            row = [r["probes"].get(f"pass{it}|{site}|{t}") for t in TARGETS]
            if any(v is None for v in row):
                continue
            print(f"  pass{it:<2}" + "".join(f"{v:.3f}".rjust(11) for v in row))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(r, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
