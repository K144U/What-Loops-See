"""Family B: relational chain queries. The family vision reviewers read fastest.

A scene of sprites, and a question of the form "what is the <attribute> of
the object <rel_k> of the object ... <rel_1> of the <colour> <shape>".
Depth is the chain length: depth 1 is the anchor alone, depth 4 is the
anchor plus three relational hops.

Depth and breadth cross cleanly here, which is the point. Breadth is the
number of sprites in the scene and has nothing to do with how many hops the
question requires, so a model that needs more loops for deeper chains but
not for busier scenes is telling us something about sequential computation
rather than about scene parsing.

Rejection sampling is used to guarantee every hop resolves. The rejection
rate is instrumented: if it exceeds 30 percent at any (depth, breadth)
cell, the sampler is badly parameterised and the scene distribution is
skewed toward whatever happens to be easy to satisfy.
"""

from __future__ import annotations

import numpy as np

from loopvision.data import render, scenes
from loopvision.data.dataset import (
    ATTR_BASE,
    ATTRIBUTES,
    COLOUR_BASE,
    FAM_B,
    QUERY,
    REL_BASE,
    SHAPE_BASE,
    Sample,
    TaskConfig,
    pad_query,
    sample_rng,
)

FAMILY = "B"

# 6 colours, then 5 shapes, then 2 sizes. One flat label space so the head
# is a single softmax regardless of which attribute was asked for.
NUM_CLASSES = render.N_SPRITE_COLOURS + len(render.SHAPES) + len(render.SIZES)  # 13
CHANCE = 1.0 / NUM_CLASSES

SUPPORTED_SPLITS = frozenset(
    {"train", "iid_val", "depth_ood", "breadth_ood", "combo_ood"}
)

#: (depth, breadth) per split. Chains run 1 to 4, not the 1 to 6 of family A.
#
# The breadth floor of 5 is forced, not chosen. A depth d chain visits d
# distinct sprites, so depth 4 is structurally impossible below breadth 4
# and rare at exactly 4. Measured rejection was 56 percent at depth 3
# breadth 3, and depth 4 breadth 3 could not be sampled at all.
#
# Crucially, train and depth_ood share one breadth range. Letting the
# deeper split use broader scenes would confound the two axes and make H1
# unanswerable, which is a worse failure than a high rejection rate.
SPLIT_RANGES = {
    "train": ((1, 2), (6, 7, 8, 9)),
    "iid_val": ((1, 2), (6, 7, 8, 9)),
    "depth_ood": ((3, 4), (6, 7, 8, 9)),
    "breadth_ood": ((1, 2), (10, 11, 12, 13)),
    "combo_ood": ((1, 2), (6, 7, 8, 9)),
}

MAX_ATTEMPTS = 200


def encode_label(attribute: str, value) -> int:
    if attribute == "colour":
        return int(value)
    if attribute == "shape":
        return render.N_SPRITE_COLOURS + render.SHAPES.index(value)
    if attribute == "size":
        return render.N_SPRITE_COLOURS + len(render.SHAPES) + render.SIZES.index(value)
    raise ValueError(f"unknown attribute {attribute!r}")


def decode_label(label: int) -> tuple[str, object]:
    if label < render.N_SPRITE_COLOURS:
        return "colour", label
    label -= render.N_SPRITE_COLOURS
    if label < len(render.SHAPES):
        return "shape", render.SHAPES[label]
    return "size", render.SIZES[label - len(render.SHAPES)]


def resolve(sprites, anchor, chain: list[str]):
    """Walk the relation chain. Returns None at the first dead end."""
    current = anchor
    for relation in chain:
        current = scenes.relate(sprites, current, relation)
        if current is None:
            return None
    return current


def _attempt(rng: np.random.Generator, depth: int, breadth: int, split: str):
    """One rejection-sampling attempt. Returns None if the chain dies."""
    sprites = scenes.place_sprites(rng, breadth, split)
    anchors = scenes.unique_anchors(sprites)
    if not anchors:
        return None

    if split == "combo_ood":
        # The queried chain must actually involve a held-out pairing,
        # otherwise the split is only nominally out of distribution.
        anchors = [s for s in anchors if s.key() in scenes.HELDOUT_PAIRS]
        if not anchors:
            return None

    anchor = anchors[int(rng.integers(0, len(anchors)))]
    walked = scenes.walk_chain(rng, sprites, anchor, depth - 1)
    if walked is None:
        return None
    chain, target = walked

    attribute = ATTRIBUTES[int(rng.integers(0, len(ATTRIBUTES)))]
    return sprites, anchor, chain, target, attribute


def sample_scene(rng: np.random.Generator, depth: int, breadth: int, split: str):
    """Rejection sample until every hop resolves. Returns (result, attempts)."""
    if depth > breadth:
        raise ValueError(
            f"depth {depth} needs at least {depth} sprites but breadth is "
            f"{breadth}. A depth d chain visits d distinct sprites, so this "
            f"cell is structurally empty rather than merely hard to sample."
        )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = _attempt(rng, depth, breadth, split)
        if result is not None:
            return result, attempt
    raise RuntimeError(
        f"could not build a family B scene at depth {depth}, breadth {breadth}, "
        f"split {split} in {MAX_ATTEMPTS} attempts. The scene sampler is badly "
        f"parameterised, see IMPLEMENTATION.md Section 4.3."
    )


def build_program(anchor, chain, target, attribute, sprites, depth, breadth) -> str:
    return (
        f"B;d={depth};b={breadth};attr={attribute};"
        f"anchor={anchor.colour}{anchor.shape};chain={'>'.join(chain) or '-'};"
        f"target={target.token()};scene={scenes.scene_program(sprites)}"
    )


def generate(idx: int, split: str, cfg: TaskConfig | None = None) -> Sample:
    cfg = cfg or TaskConfig()
    from loopvision.data.dataset import split_spec

    spec = split_spec(FAMILY, split)
    rng = sample_rng(FAMILY, split, idx, cfg)

    depth = int(rng.choice(spec.depth))
    breadth = int(rng.choice(spec.breadth))

    (sprites, anchor, chain, target, attribute), _ = sample_scene(
        rng, depth, breadth, split
    )

    tokens = [
        FAM_B,
        QUERY,
        ATTR_BASE + ATTRIBUTES.index(attribute),
        COLOUR_BASE + anchor.colour,
        SHAPE_BASE + render.SHAPES.index(anchor.shape),
    ]
    tokens += [REL_BASE + scenes.RELATIONS.index(r) for r in chain]

    return Sample(
        image=scenes.render_scene(sprites, cfg.canvas),
        query=pad_query(tokens, FAMILY),
        label=encode_label(attribute, target.attribute(attribute)),
        depth=depth,
        breadth=breadth,
        program=build_program(anchor, chain, target, attribute, sprites, depth, breadth),
        idx=idx,
        layout_seed=int(rng.integers(0, 2**31 - 1)),
    )


def rejection_rate(split: str, depth: int, breadth: int, trials: int = 400) -> float:
    """Fraction of attempts rejected at one (depth, breadth) cell.

    Instrumentation required by IMPLEMENTATION.md Section 4.3. Above 0.30
    the scene sampler is skewing the distribution toward whatever is easy
    to satisfy, which would quietly correlate scene structure with depth.
    """
    rng = np.random.default_rng(abs(hash((split, depth, breadth))) % (2**31))
    attempts = 0
    for _ in range(trials):
        _, used = sample_scene(rng, depth, breadth, split)
        attempts += used
    return 1.0 - trials / attempts
