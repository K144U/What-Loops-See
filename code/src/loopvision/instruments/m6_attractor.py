"""M6. Wrong-attractor landings, the registered H4 metric.

Preregistration Section 5, frozen at `prereg-v1`, fixes the metric:

    Cluster final states across seeds. A wrong-attractor landing is an item
    whose final state joins a cluster whose modal answer is not the correct
    answer. Rate is the fraction of such items.

The jitter sweep finished long before anything existed to read it, so H4 has
had all its compute spent and no way to resolve since milestone 6. This is
that instrument.

WHAT THE REGISTERED SENTENCE DOES NOT SAY, and how each gap is closed. Three
things are underspecified. None is a licence to pick whichever gives a
result, so each is resolved on a stated principle and recorded in D-053.

1.  **"modal answer" is ambiguous** between the modal TRUE LABEL of the items
    in a cluster and the modal MODEL PREDICTION for them. They coincide for a
    model that is right about everything and diverge exactly where H4 is
    interesting, so the choice is not cosmetic.

    **Both are computed and both are reported, and neither is called the
    answer.** House usage favours the label reading, since "answer" in this
    codebase means the correct composite and not the model's output, and that
    reading is also the one that makes "wrong ATTRACTOR" mean a basin rather
    than a mistake. But a reading chosen after seeing the numbers is not a
    registered metric, so the honest move is to publish both and let the
    disagreement between them, if any, be part of the result.

2.  **"across seeds" cannot mean what it literally says.** Two independently
    trained networks put their representations in different and arbitrarily
    rotated bases, so raw final states from different seeds are not in a
    shared space and clustering them jointly would measure basis alignment
    rather than attractor structure. **Clustering is therefore within a seed,
    and rates are aggregated across seeds.** That is the only reading under
    which the sentence computes anything at all.

3.  **The number of clusters is not given.** It is the size of the answer
    space observed in the evaluated items, one attractor per answer the task
    can have. Fixed by the task rather than tuned, which matters because k is
    exactly the knob that could be turned until H4 came out.

WHAT "FINAL STATE" HAS TO MEAN, settled by a positive control rather than by
reading. The raw residual stream is dominated by the image; the answer is a
small part of it. Clustering the raw final state of `famB_curve111_s0`, a run
at 1.0000, gives a wrong-attractor rate of 0.83, which is indistinguishable
from the 0.84 of a run at 0.60 and close to what random clustering would give
at eleven answers. **A metric that cannot tell a perfect model from a failing
one is not measuring attractors.** Clustering `model.readout(s)`, which is the
state as the head actually sees it after the coda, the output norm and the
query-token pool, the same two runs read 0.19 and 0.66 with mean cluster
purities of 0.91 and 0.47. That is the fourth gap in the registered sentence
and the control is why it is closed this way rather than by preference.

WHY COSINE. `RMSNorm` before the head makes the state norm inert, which
STATE.md records as the standing trap in F5. Euclidean distance on raw states
would weight a magnitude the head cannot see, so states are normalised and
clustered by direction.

    python -m loopvision.instruments.m6_attractor --run-id famB_curve111_s0
    python -m loopvision.instruments.m6_attractor --all --items 2048
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from loopvision.instruments import hooks

#: k-means is seeded so two passes over one run agree. L-028 is the standing
#: reminder: an instrument that starts from noise and was never run twice had
#: its variance discovered by accident, months in.
#: The four registered jitter widths. The 0.5 arm carries no jit tag in its
#: run id because 0.5 is the default in cli.py, so the base famB_curve111
#: runs ARE that arm. A pattern that only matched jit... would silently drop
#: a quarter of the sweep.
H4_ARM = re.compile(r"famB_curve111(_jit\d{3})?_s\d+")

KMEANS_SEED = 20260921
KMEANS_ITERS = 100


def _unit(x: np.ndarray) -> np.ndarray:
    """Rows to unit length, so distance is direction. See WHY COSINE above."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def spherical_mean(members: np.ndarray) -> np.ndarray:
    """The direction a cluster points, with every member weighting equally.

    Not the raw mean. A member with a hundred times the norm would otherwise
    drag the centre onto itself, which is magnitude deciding where a basin
    is, and RMSNorm before the head makes magnitude the one thing the model
    cannot see. Assignment is scale invariant whatever this does, since unit
    centroids make a point's own norm cancel out of the argmin, so this is
    the only place where normalising the input actually changes an answer.
    """
    return _unit(_unit(members).mean(axis=0, keepdims=True))[0]


def kmeans(x: np.ndarray, k: int, seed: int = KMEANS_SEED,
           iters: int = KMEANS_ITERS) -> np.ndarray:
    """Cluster assignments for rows of `x`, k-means++ init on the unit sphere.

    Written here rather than imported because the cluster has no sklearn and
    adding a dependency to the environment forty finished runs depend on is a
    worse trade than eighty lines of Lloyd's algorithm.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if x.shape[0] < k:
        raise ValueError(
            f"cannot form {k} clusters from {x.shape[0]} items. The rate would "
            f"be computed over empty basins and mean nothing.")

    xs = _unit(x.astype(np.float64))
    rng = np.random.default_rng(seed)

    # k-means++ so the init does not depend on row order. A fixed first row
    # would make the answer depend on which items the batch happened to hold.
    centres = [xs[rng.integers(xs.shape[0])]]
    for _ in range(1, k):
        d = np.min(
            np.stack([np.sum((xs - c) ** 2, axis=1) for c in centres]), axis=0)
        total = d.sum()
        if total <= 0:
            centres.append(xs[rng.integers(xs.shape[0])])
            continue
        centres.append(xs[rng.choice(xs.shape[0], p=d / total)])
    c = np.stack(centres)

    assign = np.zeros(xs.shape[0], dtype=np.int64)
    for _ in range(iters):
        d = ((xs[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
        for j in range(k):
            members = xs[assign == j]
            if members.shape[0]:
                c[j] = spherical_mean(members)
    return assign


def modal(values: np.ndarray) -> int:
    """The most common value, ties broken by the smaller value.

    Deterministic tie breaking matters: a cluster split evenly between two
    answers would otherwise make the rate depend on dictionary order.
    """
    counts = Counter(int(v) for v in values)
    best = max(counts.values())
    return min(v for v, n in counts.items() if n == best)


def wrong_attractor(assign: np.ndarray, labels: np.ndarray,
                    predictions: np.ndarray) -> dict:
    """Both readings of the registered metric. See gap 1 in the module docstring.

    `by_label` calls a cluster's answer the modal TRUE LABEL of its members.
    `by_prediction` calls it the modal MODEL PREDICTION. In both, an item is a
    wrong-attractor landing when its cluster's answer differs from that item's
    own correct answer.
    """
    n = labels.shape[0]
    wrong_l = np.zeros(n, dtype=bool)
    wrong_p = np.zeros(n, dtype=bool)
    per_cluster = []

    for j in sorted(set(int(a) for a in assign)):
        m = assign == j
        ml = modal(labels[m])
        mp = modal(predictions[m])
        wrong_l[m] = labels[m] != ml
        wrong_p[m] = labels[m] != mp
        per_cluster.append({
            "cluster": j,
            "size": int(m.sum()),
            "modal_label": ml,
            "modal_prediction": mp,
            "purity": float((labels[m] == ml).mean()),
        })

    return {
        "rate_by_label": float(wrong_l.mean()),
        "rate_by_prediction": float(wrong_p.mean()),
        "readings_agree": bool(np.array_equal(wrong_l, wrong_p)),
        "n_items": int(n),
        "clusters": per_cluster,
    }


def run(run_dir: Path, items: int, device: torch.device,
        k: int | None = None) -> dict:
    """Final states for one run, clustered, scored under both readings."""
    from loopvision.train import checkpoint as C
    from loopvision.train.cli import build_model, make_batch

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    model = build_model(cfg, device)
    ckpt = C.find_latest_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=False)["model"])
    model.eval()

    k_eval = cfg["k_eval"]
    batch = cfg["batch_size"]
    batches = max(1, (items + batch - 1) // batch)

    finals, labels, preds = [], [], []
    with torch.no_grad():
        for b in range(batches):
            image, query, label, _, _ = make_batch(cfg, "iid_val", b * batch, device)
            logits, states = hooks.capture(model, image, query, k_eval)
            # The final state AS THE HEAD READS IT. Going through the model's
            # own readout rather than copying the coda and pool here means
            # this cannot drift from the real output path, which is the same
            # reason `decode` was exposed for M3. Clustering the raw stream
            # instead fails the positive control above.
            s = model.readout(states[-1]).float().cpu().numpy()
            finals.append(s)
            labels.append(np.asarray([int(v) for v in label], dtype=np.int64))
            preds.append(logits.argmax(dim=-1).cpu().numpy().astype(np.int64))

    x = np.concatenate(finals, axis=0)[:items]
    y = np.concatenate(labels, axis=0)[:items]
    p = np.concatenate(preds, axis=0)[:items]

    # One attractor per answer the task actually produced in these items.
    # Fixed by the task, never tuned: k is the knob that could be turned
    # until H4 came out, so it is not available to turn.
    k = k or len(set(int(v) for v in y))

    assign = kmeans(x, k)
    out = wrong_attractor(assign, y, p)
    out.update({
        "run": run_dir.name,
        "checkpoint": ckpt.name,
        "jitter_sigma": cfg.get("jitter_sigma"),
        "k_clusters": k,
        "k_eval": k_eval,
        "metric": "cosine, states normalised to unit length",
        "representation": "model.readout(final_state), the state as the head sees it",
        "kmeans_seed": KMEANS_SEED,
    })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M6 wrong-attractor landings, H4")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--all", action="store_true",
                    help="the registered H4 jitter sweep, and only that")
    ap.add_argument("--items", type=int, default=2048)
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args(argv)

    if args.run_id:
        targets = [args.runs / args.run_id]
    elif args.all:
        # The H4 sweep is the JITTER axis at one architecture. A glob over
        # famB_curve111* also catches d256, d512 and d768, which vary width,
        # a different axis entirely and no part of Section 5. The first run
        # of this swept in nine of them and they sat in the output looking
        # like sweep members. Excluded by name rather than by hoping.
        targets = sorted(
            d for d in args.runs.iterdir()
            if d.is_dir() and (d / "DONE").is_file()
            and H4_ARM.fullmatch(d.name))
    else:
        print("give --run-id or --all")
        return 2

    if not targets:
        print("no finished runs matched. Nothing is computed and nothing is claimed.")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for d in targets:
        if not d.is_dir():
            print(f"no such run: {d}")
            return 2
        r = run(d, args.items, device)
        results.append(r)
        flag = "" if r["readings_agree"] else "   READINGS DISAGREE"
        print(f"  {r['run']:<34} sigma={str(r['jitter_sigma']):<5} "
              f"by_label={r['rate_by_label']:.4f} "
              f"by_prediction={r['rate_by_prediction']:.4f}{flag}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    disagree = [r for r in results if not r["readings_agree"]]
    if disagree:
        print(f"\n{len(disagree)} of {len(results)} run(s) score differently under "
              f"the two readings of 'modal answer'. That disagreement is a result "
              f"about the metric and belongs in the write-up, not a bug. See D-053.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
