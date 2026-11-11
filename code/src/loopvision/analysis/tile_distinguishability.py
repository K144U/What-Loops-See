"""D-070: how far apart each rendering draws its group elements, against how often it escapes.

    python -m loopvision.analysis.tile_distinguishability
    python -m loopvision.analysis.tile_distinguishability --json out.json

Registered on 2026-10-11 and signed before anything here was computed. The
measure, the renderings, the prediction and its thresholds are D-070's, and
none of them is a parameter of this module.

M_pix   for a rendering, the mean over all pairs of distinct group elements of
        the Euclidean distance between their rendered tiles, pixel values
        scaled to 0 to 1, from `render.render_element`.
M_emb   the same distance after the model's patch-embedding convolution at
        initialisation, averaged over 20 seeded initialisations. The tile goes
        in exactly as the trainer feeds it, divided by 255.
Both are also reported as the minimum over pairs, the hardest pair, as a
secondary.

S1      across the seven non-trap renderings, the Spearman correlation between
        M_pix and the escape rate is at least 0.6. Held at 0.6 or above, failed
        at 0 or below, undetermined in between. Ties take their average rank;
        E1b draws exactly D4 position's tiles, so those two tie.

E2a, the composing cue, is measured and reported but kept out of S1, because
D-062 already established that salience does not account for it.

The escape rates are the existing sealed 2x2 cells and the D-062 arms, read by
`escape_rates` from each run's stored summary. The correlation with M_emb is
reported beside S1 and decides nothing.

THE TIE WAS BROKEN BY ROUNDING ERROR UNTIL 2026-10-22. E1b's M_pix and D4
position's are the same mean over the same distances summed in a different
order, so they differed in the sixteenth digit, and an exact comparison ranked
them apart. That gave 0.703 where the registered rule, average ranks for the
tie, gives 0.673. Values are now compared at TIE_DECIMALS. D-076.

Each correlation is reported with its exact two-sided permutation p over every
ordering of the escape rates. The registration fixed a threshold and not a
test, so the p decides nothing; it says how much seven points can carry.

Exit 0 if S1 held, 1 if it failed, 3 if undetermined, 2 if a rate is missing.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations, permutations
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from loopvision.analysis import escape_rates as ER
from loopvision.data import groups as G
from loopvision.data import render
from loopvision.model.loopvit import LoopViT, LoopViTConfig

#: D-070's thresholds and initialisation count. Fixed by the registration.
HOLD_AT = 0.6
FAIL_AT = 0.0
N_INIT = 20

#: Values equal to this many decimals are a tie. Rounding error sits near the
#: sixteenth digit and the closest real pair of renderings differs in the
#: second, so the tie D-070 names is a tie and nothing else becomes one.
TIE_DECIMALS = 9

#: The largest sample whose orderings are all enumerated for the p.
MAX_EXACT = 9

#: Name, substrate, subgroup, where its escape rate comes from, and whether it
#: is in S1. A sealed 2x2 cell is ("cell", (group, medium)); a D-062 arm is
#: ("stem", run-name stem).
RENDERINGS: list[tuple[str, str, str, tuple, bool]] = [
    ("D4 position", "native", "d4", ("cell", ("D4", "position")), True),
    ("D4 appearance", "d4_colour", "d4", ("cell", ("D4", "identity")), True),
    ("E1a", "d4_square", "d4", ("stem", "why_e1a_square"), True),
    ("E1b", "d4_scrambled", "d4", ("stem", "why_e1b_scrambled"), True),
    ("E2b", "d4_colour_decoy", "d4", ("stem", "why_e2b_decoy"), True),
    ("S3 position", "s3_spatial", "s3", ("cell", ("S3", "position")), True),
    ("S3 appearance", "native", "s3", ("cell", ("S3", "identity")), True),
    ("E2a", "d4_colour_cue", "d4", ("stem", "why_e2a_cue"), False),
]


def tiles(substrate: str, subgroup: str) -> np.ndarray:
    """Every element of the subgroup drawn as a tile, (n, 3, P, P) uint8."""
    return np.stack([render.render_element(i, substrate=substrate)
                     for i in G.subgroup_members(subgroup)])


def pair_distances(vectors: np.ndarray) -> np.ndarray:
    """Euclidean distance for every unordered pair of distinct rows."""
    v = np.asarray(vectors, dtype=np.float64).reshape(len(vectors), -1)
    return np.array([np.linalg.norm(v[i] - v[j]) for i, j in combinations(range(len(v)), 2)])


def m_pix(tile_stack: np.ndarray) -> tuple[float, float]:
    """Mean and minimum pairwise distance, pixels scaled to 0 to 1."""
    d = pair_distances(np.asarray(tile_stack, dtype=np.float64) / 255.0)
    return float(d.mean()), float(d.min())


def model_patchify(seed: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """The patch-embedding convolution of a freshly initialised model.

    The whole model is built so the convolution gets exactly the
    initialisation a run's model gets, including the order in which the
    model's own initialiser consumes the generator.
    """
    torch.manual_seed(seed)
    conv = LoopViT(LoopViTConfig()).patchify.eval()
    return conv


def m_emb(tile_stack: np.ndarray, n_init: int = N_INIT,
          make_conv: Callable[[int], Callable] = model_patchify) -> tuple[float, float]:
    """Mean and minimum pairwise distance after the patch embedding, each
    averaged over `n_init` seeded initialisations."""
    x = torch.from_numpy(np.asarray(tile_stack)).float().div(255.0)
    means, mins = [], []
    with torch.no_grad():
        for seed in range(n_init):
            out = make_conv(seed)(x).reshape(len(x), -1).numpy()
            d = pair_distances(out)
            means.append(d.mean())
            mins.append(d.min())
    return float(np.mean(means)), float(np.mean(mins))


def average_ranks(values) -> np.ndarray:
    """Ranks from 1, tied values sharing the average of the ranks they span.
    Values equal to TIE_DECIMALS places are tied."""
    v = np.round(np.asarray(values, dtype=np.float64), TIE_DECIMALS)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    """Spearman's rho with average ranks for ties; nan if either side is constant."""
    rx, ry = average_ranks(x), average_ranks(y)
    dx, dy = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    return float((dx * dy).sum() / denom) if denom > 0 else float("nan")


def permutation_p(x, y) -> float:
    """Exact two-sided p of Spearman's rho: the share of all orderings of y
    whose rho is at least as far from zero as the observed one."""
    if len(y) > MAX_EXACT:
        raise ValueError(f"{len(y)} points is too many to enumerate")
    observed = abs(spearman(x, y))
    if observed != observed:
        return float("nan")
    y = list(y)
    hits = total = 0
    for order in permutations(range(len(y))):
        total += 1
        hits += abs(spearman(x, [y[i] for i in order])) >= observed - 1e-12
    return hits / total


def verdict(rho: float) -> str:
    if rho != rho:
        return "undetermined"
    if rho >= HOLD_AT:
        return "held"
    if rho <= FAIL_AT:
        return "failed"
    return "undetermined"


def escape_rates(rows: list[dict]) -> dict[str, tuple[int, int] | None]:
    """Escaped and finished for each rendering, None where nothing finished."""
    by_cell = ER.rates(rows, key="arm")
    by_stem = ER.stem_rates(rows)
    out: dict[str, tuple[int, int] | None] = {}
    for name, _, _, (kind, key), _ in RENDERINGS:
        e = (by_cell if kind == "cell" else by_stem).get(key)
        out[name] = (e["escaped"], e["finished"]) if e and e["finished"] else None
    return out


def evaluate(measures: dict[str, dict], rates: dict[str, tuple[int, int] | None]) -> dict:
    """S1 on M_pix, and the same correlation on M_emb reported beside it."""
    tested = [name for name, *_, in_s1 in RENDERINGS if in_s1]
    if any(rates.get(n) is None for n in tested):
        return {"S1": "pending", "rho_pix": None, "rho_emb": None,
                "p_pix": None, "p_emb": None, "tested": tested}
    r = [rates[n][0] / rates[n][1] for n in tested]
    pix = [measures[n]["m_pix"] for n in tested]
    emb = [measures[n]["m_emb"] for n in tested]
    rho_pix = spearman(pix, r)
    return {"S1": verdict(rho_pix), "rho_pix": rho_pix, "rho_emb": spearman(emb, r),
            "p_pix": permutation_p(pix, r), "p_emb": permutation_p(emb, r),
            "tested": tested}


def measure_all(n_init: int = N_INIT) -> dict[str, dict]:
    out = {}
    for name, substrate, subgroup, _, _ in RENDERINGS:
        t = tiles(substrate, subgroup)
        pm, pn = m_pix(t)
        em, en = m_emb(t, n_init)
        out[name] = {"substrate": substrate, "subgroup": subgroup, "n_elements": len(t),
                     "m_pix": pm, "m_pix_min": pn, "m_emb": em, "m_emb_min": en}
    return out


EXIT = {"held": 0, "failed": 1, "pending": 2, "undetermined": 3}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D-070, tile distinguishability against escape rate")
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    ap.add_argument("--json", default=None, type=Path, help="write every number here too")
    args = ap.parse_args(argv)

    measures = measure_all()
    rows = ER.scan(args.runs) if args.runs.is_dir() else []
    rates = escape_rates(rows)
    result = evaluate(measures, rates)

    print("rendering        in S1  M_pix   min     M_emb   min     escaped")
    for name, *_, in_s1 in RENDERINGS:
        m, rt = measures[name], rates.get(name)
        esc = f"{rt[0]} of {rt[1]}" if rt else "none finished"
        print(f"{name:16s} {'yes' if in_s1 else 'no ':5s}  {m['m_pix']:.4f}  {m['m_pix_min']:.4f}  "
              f"{m['m_emb']:.4f}  {m['m_emb_min']:.4f}  {esc}")
    if result["S1"] == "pending":
        print("\nS1  pending       a rate in the test set has no finished run")
    else:
        print(f"\nS1  {result['S1']:13s} Spearman of M_pix with escape rate over "
              f"{len(result['tested'])} renderings = {result['rho_pix']:.3f}; held at "
              f"{HOLD_AT} or above, failed at {FAIL_AT} or below")
        print(f"    exact two-sided permutation p = {result['p_pix']:.3f}, "
              f"which decides nothing: the registration fixed a threshold")
        print(f"    secondary, decides nothing: Spearman of M_emb with escape rate = "
              f"{result['rho_emb']:.3f}, p = {result['p_emb']:.3f}")
    if args.json:
        args.json.write_text(json.dumps({"measures": measures, "rates": rates, **result},
                                        indent=2), encoding="utf-8")
    return EXIT[result["S1"]]


if __name__ == "__main__":
    raise SystemExit(main())
