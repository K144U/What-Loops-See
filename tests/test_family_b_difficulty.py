"""Family B has to be hard for the reason it claims to be hard.

Every earlier family B check verified that samples were well formed: the
chain resolves, the anchor is unique, the label matches the target. All of
those passed while the task was trivially solvable, because none of them
asked whether the difficulty was where the design said it was.

Two faults hid behind that. `size` was askable and has two values, so one
question in three had a chance level of 0.5 and the family's real floor was
0.2897 rather than the 1/13 = 0.0769 quoted everywhere. And at breadth 6 to
9 the reachable set was so small that a model ignoring the relation chain
entirely scored 0.62 at depth 4.

These tests fail if either returns.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from loopvision.data import dataset as D
from loopvision.data import family_b as FB
from loopvision.data import render, scenes

#: A model that ignores the chain must do clearly worse than one that walks
#: it. Measured at 0.44 for the deepest cells under the current design; the
#: bar is set at 0.55 so it fails on the old configuration, which scored
#: 0.62, without being so tight that sampling noise trips it.
MAX_CHAIN_BLIND = 0.55


def _chain_blind_accuracy(depth: int, breadth: int, trials: int = 300) -> float:
    """Best accuracy from the scene, anchor and asked attribute, no chain.

    Enumerates every sprite reachable from the anchor in depth-1 hops over
    all chains, then takes the most common answer among them. That is
    exactly what a model which never reads the relation tokens can do.
    """
    rng = np.random.default_rng(abs(hash((depth, breadth))) % (2**31))
    best = []
    for _ in range(trials):
        (sprites, anchor, _, _, attr), _ = FB.sample_scene(rng, depth, breadth, "train")
        cur = {(anchor, frozenset({(anchor.row, anchor.col)}))}
        for _ in range(depth - 1):
            nxt = set()
            for sp, seen in cur:
                for rel in scenes.RELATIONS:
                    n = scenes.relate(sprites, sp, rel)
                    if n is not None and (n.row, n.col) not in seen:
                        nxt.add((n, seen | {(n.row, n.col)}))
            cur = nxt
        vals = [sp.attribute(attr) for sp in {sp for sp, _ in cur}]
        best.append(max(collections.Counter(vals).values()) / len(vals) if vals else 1.0)
    return float(np.mean(best))


@pytest.mark.parametrize("depth", [3, 4, 5, 6])
def test_the_relation_chain_is_load_bearing(depth: int) -> None:
    """If the answer is recoverable without walking, depth is not difficulty."""
    breadths = FB.SPLIT_RANGES["train"][1]
    got = _chain_blind_accuracy(depth, breadths[len(breadths) // 2])
    assert got < MAX_CHAIN_BLIND, (
        f"at depth {depth} a model ignoring the relation chain scores {got:.3f}. "
        f"The chain is decoration, not difficulty."
    )


def test_deeper_chains_are_not_easier_to_shortcut() -> None:
    """Chain-blind accuracy must not rise with depth.

    If it did, deep cells would be easier to guess than shallow ones and
    any loop-count curve would be reading the shortcut, not the walk.
    """
    b = FB.SPLIT_RANGES["train"][1][2]
    vals = {d: _chain_blind_accuracy(d, b) for d in (3, 4, 5, 6)}
    assert vals[6] <= vals[3] + 0.05, f"shortcut grows with depth: {vals}"


def test_size_is_never_the_attribute_asked_about() -> None:
    """Two values means a coin flip, which drags the whole floor up."""
    assert "size" not in FB.ASKABLE_ATTRIBUTES
    for i in range(600):
        s = FB.generate(D.global_index("train", i), "train")
        attr = dict(p.split("=", 1) for p in s.program.split(";") if "=" in p)["attr"]
        assert attr != "size", "a size question is a coin flip"
        kind, _ = FB.decode_label(s.label)
        assert kind in FB.ASKABLE_ATTRIBUTES


def test_size_still_describes_the_anchor() -> None:
    """Dropping size as a question must not drop it as a descriptor, or the
    anchor loses a third of the information that makes it unique."""
    used = set()
    for a in FB.ASKABLE_ATTRIBUTES:
        used.update(scenes.DESCRIPTOR_FOR[a])
    assert "size" in used


def test_effective_chance_is_not_one_over_num_classes() -> None:
    """The bug this guards is quoting 1/13 as family B's floor."""
    eff = FB.effective_chance()
    assert eff == pytest.approx(0.1833, abs=1e-3), eff
    assert eff > FB.CHANCE * 2, (
        "the real floor is more than double 1/NUM_CLASSES, so quoting "
        "1/NUM_CLASSES makes a model that learned nothing look partly right"
    )


def test_the_measured_label_floor_matches_the_computed_one() -> None:
    """effective_chance() is arithmetic. This checks it against real data."""
    per_attr = collections.defaultdict(collections.Counter)
    for i in range(4000):
        s = FB.generate(D.global_index("train", i), "train")
        kind, _ = FB.decode_label(s.label)
        per_attr[kind][s.label] += 1
    total = sum(sum(c.values()) for c in per_attr.values())
    measured = sum(
        (sum(c.values()) / total) * (1 / len(c)) for c in per_attr.values()
    )
    assert measured == pytest.approx(FB.effective_chance(), abs=0.02)


def test_the_deepest_chain_fits_in_the_query_budget() -> None:
    """A silent cap here is what held family B at depth 4."""
    deepest = max(FB.SPLIT_RANGES["train"][0])
    needed = 3 + 2 + (deepest - 1)
    assert D.L_MAX["B"] >= needed, (
        f"depth {deepest} needs {needed} query tokens, budget is {D.L_MAX['B']}"
    )
    for i in range(400):
        s = FB.generate(D.global_index("train", i), "train")
        assert len(s.query) == D.L_MAX["B"]


def test_the_depth_range_supports_a_multi_point_loop_count_curve() -> None:
    """H1 regresses k_min against depth, so the range must produce several
    distinct k_min values, not just one step.

    A looped model does prelude + k*core + coda sequential blocks, and a
    depth d chain needs about d hops, so k_min = ceil((d - p - c) / core).
    At 1/1/1 the depth range must yield at least three distinct k_min
    values or the regression has two points and no shape.

    This is what capping depth at 4 costs: depths 1 to 4 give only k=1 and
    k=2, which cannot distinguish a linear rise from any other.
    """
    import math

    prelude = coda = core = 1
    depths = FB.SPLIT_RANGES["train"][0]
    kmin = {d: max(1, math.ceil((d - prelude - coda) / core)) for d in depths}
    distinct = sorted(set(kmin.values()))
    assert len(distinct) >= 3, (
        f"depths {tuple(depths)} give k_min values {kmin}, only "
        f"{len(distinct)} distinct. A curve needs at least three points."
    )


def test_the_block_counts_reach_the_built_model() -> None:
    """A config key that is parsed but never used is invisible in the run.

    `prelude_blocks`, `core_blocks` and `coda_blocks` were absent from
    `build_model`, so a config asking for a 1/1/1 core would have trained a
    2/2/2 model and produced another flat sweep, with nothing in the run
    directory to show why. The check is on the built model, not on the
    config dict, because copying a field is not the same as using it.
    """
    import torch

    from loopvision.train.cli import DEFAULTS, build_model

    for key in ("prelude_blocks", "core_blocks", "coda_blocks"):
        assert key in DEFAULTS, f"{key} has no default, so configs cannot set it"

    cfg = dict(DEFAULTS)
    cfg.update({
        "family": "B", "arch": "looped", "d_model": 64,
        "prelude_blocks": 1, "core_blocks": 1, "coda_blocks": 1,
        "k_train": 2, "k_eval": 2, "untied_copies": None,
        "state_init": "randn", "conditioning": "none", "inject": True,
    })
    model = build_model(cfg, torch.device("cpu"))
    assert len(model.prelude) == 1, f"prelude has {len(model.prelude)} blocks, wanted 1"
    assert len(model.core) == 1, f"core has {len(model.core)} blocks, wanted 1"
    assert len(model.coda) == 1, f"coda has {len(model.coda)} blocks, wanted 1"


def test_family_c_floor_is_computed_not_one_over_num_classes() -> None:
    """Family C is the breadth control, so its floor matters as much.

    Counts are not uniform: small counts dominate, so answering the modal
    count beats uniform guessing by a lot. Quoting 1/11 here would repeat
    the family B mistake on the arm that is supposed to show no effect,
    where a spurious rise would be even harder to spot.
    """
    from loopvision.data import family_c as FC

    floor = FC.effective_floor("train", n=2000)
    assert floor > FC.CHANCE * 3, (
        f"floor {floor:.3f} vs 1/NUM_CLASSES {FC.CHANCE:.3f}: quoting the "
        f"latter would make a model that learned nothing look competent"
    )
    assert 0.30 < floor < 0.50, floor
    wide = FC.effective_floor("breadth_ood", n=2000)
    assert wide < floor, (
        "the floor must fall as scenes get busier, since counts spread out. "
        "If it rose, breadth would be making the task easier to guess"
    )
