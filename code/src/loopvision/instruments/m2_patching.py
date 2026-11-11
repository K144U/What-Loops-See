"""M2. Activation patching over (loop, patch position). Milestone 6, H3.

IMPLEMENTATION.md Section 7, M2. For one item:

1. Generate a clean sample and a corrupted twin differing in exactly one
   program element.
2. Run clean, store every `s_i`.
3. Run corrupted while overwriting `s_i[:, p, :]` with the clean value, for
   every `(i, p)` in the grid.
4. Record the recovered logit difference, normalised so 0 is the corrupted
   run and 1 is the clean run.

The per-item grid is `instruments/hooks.recovery_grid`, which already
exists and is mutation checked. This module is the driver: it decides
which items are admissible, aggregates over them, and writes the result
where a plot script can read it without computing anything.

**Two ways an item is inadmissible, and both are dropped rather than
averaged.** A twin that does not change the label leaves the readout
`logit(clean) - logit(corrupt)` identically zero, so recovery is a ratio
with a zero denominator. A twin that changes the label but moves the
model's readout by almost nothing has the same problem numerically.
`LogitDiff.recovered` returns NaN in the second case by design. Averaging
NaN into a heatmap cell would fill it with a number meaning "no effect
measured", which is exactly the failure the docstring there warns about.
Both counts are recorded, because a cell aggregated over 12 admissible
items is not the same evidence as one aggregated over 256 and the reader
must be able to tell.

**What is stored.** Mean, standard deviation and item count per cell, in
long form. CLAUDE.md requires every plotted number to be readable from a
stored metrics file, so the aggregation happens here and the figure script
only draws.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from loopvision.data import dataset as D
from loopvision.data import family_a as FA
from loopvision.data import family_b as FB
from loopvision.data import family_c as FC
from loopvision.instruments.hooks import (
    LogitDiff,
    capture,
    logit_difference,
    patched,
    recovery_grid,
    shared_init_state,
)
from loopvision.train import checkpoint as C
from loopvision.train.cli import build_model, task_config

# A twin generator per family. Family A edits one operator of the queried
# strip. Families B and C need their own one element edit, a relation hop
# and a scene element respectively, and until those exist this module
# refuses rather than patching something it cannot interpret.
# Signatures differ: family B needs the split, because it regenerates the
# scene from the index rather than parsing it back out of the program.
TWIN = {
    "A": lambda sample, cfg, split: FA.corrupted_twin(sample, cfg),
    "B": lambda sample, cfg, split: FB.corrupted_twin(sample, cfg, split=split),
    "C": lambda sample, cfg, split: FC.corrupted_twin(sample, cfg, split=split),
}


def to_tensor(sample, device) -> tuple[torch.Tensor, torch.Tensor]:
    """One sample as a batch of one, normalised the way make_batch does."""
    image = torch.from_numpy(np.stack([sample.image])).to(device).float().div_(255.0)
    query = torch.from_numpy(np.stack([sample.query])).to(device).long()
    return image, query


def item_grid(model, clean, corrupt, k: int, device) -> torch.Tensor | None:
    """Recovery grid for one counterfactual pair, or None if inadmissible."""
    if clean.label == corrupt.label:
        # The edit did not change the answer, so there is no clean-versus
        # corrupt direction to measure recovery along.
        return None
    clean_image, query = to_tensor(clean, device)
    corrupt_image, corrupt_query = to_tensor(corrupt, device)
    return recovery_grid(
        model,
        clean_image,
        corrupt_image,
        query,
        int(clean.label),
        int(corrupt.label),
        k,
        corrupt_query=corrupt_query,
    )


def full_state_control(model, clean, corrupt, k: int, device) -> list[float]:
    """Positive control: patch EVERY position at iteration i, not just one.

    A single-position heatmap that reads near zero everywhere has two very
    different explanations. Either the information is distributed, so no one
    position carries it, which is what H3 predicts for depth tasks. Or the
    patch is not reaching the computation at all and the instrument measures
    nothing. A heatmap alone cannot separate those, and reporting one without
    this control would be reporting a number that means "no effect measured"
    as though it meant "no effect".

    Patching the whole state at the last iteration hands the coda exactly the
    clean state, so recovery there must approach 1. If it does not, the
    instrument is broken and the heatmap is uninterpretable.
    """
    if clean.label == corrupt.label:
        return []
    clean_image, query = to_tensor(clean, device)
    corrupt_image, corrupt_query = to_tensor(corrupt, device)

    s0 = shared_init_state(model, clean_image, query)
    clean_logits, clean_states = capture(model, clean_image, query, k, s0=s0)
    corrupt_logits, _ = capture(model, corrupt_image, corrupt_query, k, s0=s0)
    scale = LogitDiff(
        clean=float(logit_difference(clean_logits, int(clean.label), int(corrupt.label))[0]),
        corrupted=float(logit_difference(corrupt_logits, int(clean.label), int(corrupt.label))[0]),
    )

    every = list(range(model.cfg.seq_len))
    out = []
    for i in range(k + 1):
        # patched() YIELDS the hook. It does not mutate the model, so the
        # hook must be handed to forward or nothing is patched at all and
        # the control silently reads 0.0000 everywhere.
        with patched(model, clean_states, i, every) as hook:
            with torch.no_grad():
                logits = model(corrupt_image, corrupt_query, k=k, s0=s0, state_hook=hook)
        v = float(logit_difference(logits, int(clean.label), int(corrupt.label))[0])
        out.append(scale.recovered(v))
    return out


def patch_grid(
    run_dir: Path,
    items: int = 256,
    min_admissible: int | None = None,
    progress_every: int = 25,
    k: int | None = None,
    split: str = "iid_val",
    device: str | None = None,
) -> pd.DataFrame:
    """Aggregate the (loop, position) recovery grid over counterfactual pairs.

    `items` caps how many candidates are examined. `min_admissible` is the
    number that must survive, which is what the pre-registration actually
    specifies: 256 admissible items per cell. Admissibility varies sharply by
    family, 98 percent for family C against 51 percent for family B, so a raw
    count delivers whatever sample the family happens to yield and a cell can
    silently carry half the registered size. Set the target and let the cap
    stop a runaway.
    """
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    family = cfg["family"]
    if family not in TWIN:
        raise NotImplementedError(
            f"family {family} has no corrupted_twin yet, so M2 cannot build a "
            f"counterfactual for it. Implement one element editing in "
            f"data/family_{family.lower()}.py first. Supported: {sorted(TWIN)}"
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    ckpt = C.find_latest_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=False)["model"]
    )
    model.eval()

    k = int(k if k is not None else cfg["k_eval"])
    tcfg = task_config(cfg)
    twin = TWIN[family]

    grids: list[torch.Tensor] = []
    controls: list[list[float]] = []
    skipped_same_label = 0
    skipped_no_twin = 0
    target = min_admissible
    for i in range(items):
        if target is not None and len(grids) >= target:
            break
        gi = D.global_index(split, i)
        clean = D.generate(family, gi, split, tcfg)
        corrupt = twin(clean, tcfg, split)
        if corrupt is None:
            # No counterfactual exists for this item: family B depth 1 has no
            # hop to edit, and an edited hop can point at empty space.
            skipped_no_twin += 1
            continue
        g = item_grid(model, clean, corrupt, k, device)
        if g is None:
            skipped_same_label += 1
            continue
        grids.append(g.cpu())
        # A cell is tens of minutes of forward passes and used to print
        # nothing until it finished, so a run that was progressing normally
        # and a run that had wedged looked identical from outside. Cheap to
        # emit, and it is the only handle on a long instrument.
        if progress_every and len(grids) % progress_every == 0:
            target = min_admissible or items
            print(
                f"  {len(grids)}/{target} admissible after {i + 1} candidates",
                flush=True,
            )
        # The control travels with the grid. A single-position heatmap that
        # reads near zero is uninterpretable without it, so it is computed
        # here rather than left to whoever remembers to ask for it.
        controls.append(full_state_control(model, clean, corrupt, k, device))

    if not grids:
        raise RuntimeError(
            f"no admissible counterfactual pairs in {items} items for {run_dir.name}. "
            f"{skipped_same_label} had a twin that did not change the label, "
            f"{skipped_no_twin} had no twin at all."
        )

    stack = torch.stack(grids)  # (n_items, k+1, positions)
    n_valid = (~torch.isnan(stack)).sum(dim=0)
    mean = torch.nanmean(stack, dim=0)
    # nanstd is not in torch, so compute it from the deviations that exist.
    dev = (stack - mean.unsqueeze(0)) ** 2
    var = torch.nansum(dev, dim=0) / torch.clamp(n_valid - 1, min=1)
    std = torch.sqrt(var)

    depth = max(tcfg.depths) if tcfg.depths else 1
    rows = []
    n_iter, n_pos = mean.shape
    for it in range(n_iter):
        for p in range(n_pos):
            rows.append(
                {
                    "run_id": run_dir.name,
                    "family": family,
                    "depth": depth,
                    "k": k,
                    "iteration": it,
                    "position": p,
                    "is_query_token": p >= model.cfg.num_patches,
                    "recovery_mean": float(mean[it, p]),
                    "recovery_std": float(std[it, p]),
                    "n_items": int(n_valid[it, p]),
                }
            )
    # Control rows, position -1, meaning every position patched at once.
    ctl = np.array(controls, dtype=float)
    for it in range(ctl.shape[1]):
        col = ctl[:, it]
        rows.append(
            {
                "run_id": run_dir.name,
                "family": family,
                "depth": depth,
                "k": k,
                "iteration": it,
                "position": -1,
                "is_query_token": False,
                "recovery_mean": float(np.nanmean(col)),
                "recovery_std": float(np.nanstd(col)),
                "n_items": int((~np.isnan(col)).sum()),
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["items_examined"] = i + 1 if items else 0
    df.attrs["items_requested"] = items
    df.attrs["min_admissible"] = min_admissible
    df.attrs["admissible"] = len(grids)
    df.attrs["skipped_same_label"] = skipped_same_label
    df.attrs["skipped_no_twin"] = skipped_no_twin
    return df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--items", type=int, default=256, help="candidates to examine")
    p.add_argument(
        "--min-admissible",
        type=int,
        default=None,
        help="admissible pairs required, the pre-registered quantity",
    )
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--split", default="iid_val")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    run_dir = Path(a.runs_root) / a.run_id
    df = patch_grid(
        run_dir, items=a.items, min_admissible=a.min_admissible, k=a.k, split=a.split
    )
    out = Path(a.out) if a.out else run_dir / "m2_patching.parquet"
    df.to_parquet(out, index=False)

    kept = df.n_items.max()
    if a.min_admissible and df.attrs["admissible"] < a.min_admissible:
        print(
            f"  SHORT {df.attrs['admissible']} admissible of {a.min_admissible} "
            f"required, after examining {df.attrs['items_examined']} candidates. "
            f"Raise --items or this cell is under the registered size."
        )
    print(f"wrote {out}")
    print(
        f"  grid {df.iteration.nunique()} iterations x {df.position.nunique()} positions"
    )
    print(
        f"  {kept} of {a.items} items admissible, "
        f"{df.attrs['skipped_same_label']} twins did not change the label, "
        f"{df.attrs['skipped_no_twin']} had no twin"
    )
    ctl_last = df[(df.position == -1) & (df.iteration == df.iteration.max())]
    v = float(ctl_last.recovery_mean.iloc[0])
    print(f"  positive control, all positions at the last iteration: {v:.4f}")
    if v < 0.9:
        print(
            "  WARNING control did not recover. Patching every position at the "
            "last iteration hands the coda the clean state, so this must "
            "approach 1. The heatmap is not interpretable until it does."
        )
    thin = df[df.n_items < max(1, kept // 2)]
    if len(thin):
        print(f"  WARNING {len(thin)} cells aggregated over fewer than half the items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
