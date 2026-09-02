"""What did a family A model actually learn?

Runs that escape the ln(48) = 3.8712 chance plateau settle at about 3.19,
and ln(24) = 3.1781. Halving the effective class count means learning
exactly one bit, and the natural candidates are the order-independent ones.

D4 x S3 has three independent homomorphisms onto Z2:

  flip   the reflection parity of the D4 part
  rot2   the rotation index mod 2
  sign   the parity of the S3 permutation

Each is a product mod 2, so each **commutes** and can be computed from the
multiset of operators without knowing their order. The abelianisation is
Z2 x Z2 x Z2, of order 8, so a model that learned all three would sit at
ln(6) = 1.792.

If a model at 3.19 predicts one of these bits well above chance while the
remaining 24-way choice sits at chance, it has learned the order-blind part
of the group and nothing else. That would explain a flat loop-count curve
directly: loops cannot help with a quantity that does not need them.

    python -m loopvision.analysis.parity_probe --run runs/vark_famA_d3_s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from loopvision.data import dataset as D
from loopvision.data import groups as G


def s3_sign(perm: tuple[int, int, int]) -> int:
    """Parity of a permutation, 0 for even and 1 for odd."""
    inversions = sum(
        1 for i in range(3) for j in range(i + 1, 3) if perm[i] > perm[j]
    )
    return inversions % 2


PARITIES = {
    "flip (D4 reflection)": lambda g: G.d4_of(g)[1],
    "rot mod 2": lambda g: G.d4_of(g)[0] % 2,
    "sign (S3 permutation)": lambda g: s3_sign(G.s3_of(g)),
}


def verify_homomorphisms() -> None:
    """Each parity must be additive mod 2, or it is not order independent.

    Checked here rather than assumed, because the whole interpretation
    rests on these being computable without operator order.
    """
    for name, f in PARITIES.items():
        for a in range(G.GROUP_ORDER):
            for b in range(G.GROUP_ORDER):
                if f(G.multiply(a, b)) != (f(a) + f(b)) % 2:
                    raise AssertionError(f"{name} is not a homomorphism onto Z2")


def load_model(run_dir: Path, device):
    from loopvision.model.loopvit import FeedforwardBaseline, LoopViT, LoopViTConfig
    from loopvision.train.checkpoint import find_latest_checkpoint

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    family = D.get_family(cfg["family"])
    model_cfg = LoopViTConfig(
        d_model=cfg["d_model"],
        num_classes=family.NUM_CLASSES,
        vocab_size=D.VOCAB_SIZE,
        query_len=D.L_MAX[cfg["family"]],
        state_init=cfg["state_init"],
        conditioning=cfg["conditioning"],
        inject=cfg["inject"],
    )
    model = (
        FeedforwardBaseline(model_cfg, k=cfg["k_train"])
        if cfg["arch"] == "feedforward"
        else LoopViT(model_cfg)
    )
    ckpt = find_latest_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def collect_predictions(model, cfg: dict, device, n_batches: int, k: int):
    from loopvision.train.cli import forward, make_batch

    preds, labels = [], []
    for b in range(n_batches):
        image, query, label, _, _ = make_batch(cfg, "iid_val", b * cfg["batch_size"], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = forward(model, image, query, k)
        preds.append(logits.float().argmax(-1).cpu().numpy())
        labels.append(label.cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels)


def analyse(preds: np.ndarray, labels: np.ndarray) -> dict:
    out = {
        "n": int(len(labels)),
        "full_accuracy": float((preds == labels).mean()),
        "chance": 1.0 / G.GROUP_ORDER,
        "parities": {},
    }
    for name, f in PARITIES.items():
        pp = np.array([f(int(p)) for p in preds])
        pl = np.array([f(int(l)) for l in labels])
        acc = float((pp == pl).mean())

        # Given the parity bit is right, is the remaining 24-way choice at
        # chance? If so the model knows the bit and nothing more.
        mask = pp == pl
        residual = float((preds[mask] == labels[mask]).mean()) if mask.any() else 0.0
        out["parities"][name] = {
            "accuracy": acc,
            "chance": 0.5,
            "residual_accuracy_given_correct_parity": residual,
            "residual_chance": 1.0 / (G.GROUP_ORDER // 2),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--k", type=int, default=None, help="defaults to the run's k_eval")
    p.add_argument("--json")
    args = p.parse_args()

    verify_homomorphisms()
    run_dir = Path(args.run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(run_dir, device)
    k = args.k if args.k is not None else cfg["k_eval"]

    preds, labels = collect_predictions(model, cfg, device, args.batches, k)
    result = analyse(preds, labels)
    result["run"] = str(run_dir)
    result["k"] = k

    print(f"\n{run_dir.name}  (k={k}, n={result['n']})")
    print(f"  full 48-way accuracy   {result['full_accuracy']:.4f}   (chance {result['chance']:.4f})")
    print(f"  ln(48) = {np.log(48):.4f}   ln(24) = {np.log(24):.4f}   ln(6) = {np.log(6):.4f}")
    print("\n  order-independent bits, each computable without operator order:")
    for name, r in result["parities"].items():
        flag = "  <-- LEARNED" if r["accuracy"] > 0.60 else ""
        print(f"    {name:24s} {r['accuracy']:.4f}  (chance 0.5000){flag}")
        print(
            f"      residual 24-way given correct bit: "
            f"{r['residual_accuracy_given_correct_parity']:.4f} "
            f"(chance {r['residual_chance']:.4f})"
        )

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
