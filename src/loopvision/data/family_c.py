"""Family C: counting and aggregation. The breadth stressor and the H1 null arm.

Count the sprites satisfying a conjunction of two or three attributes. The
label is that count, capped at 10.

Deliberately breadth-heavy and depth-shallow. Depth is nominally 1 for
every sample, because there is no sequential composition to do: every
sprite can be checked against the conjunction independently and in
parallel. Breadth runs from 4 to 16.

**This family is never cut.** It is the null arm of H1: if required loops
grow here, H1 is falsified. More than that, the August 2026 survey found
the depth half of H1 partly anticipated in text, which makes the breadth
axis the genuinely unexplored half of the contribution. Without family C
there is no breadth arm, and H1 collapses from a dissociation into a
replication. See docs/decisions.md D-007 and the family C note in
IMPLEMENTATION.md Section 4.4.
"""

from __future__ import annotations

import numpy as np

from loopvision.data import render, scenes
from loopvision.data.dataset import (
    COLOUR_BASE,
    FAM_C,
    QUERY,
    SHAPE_BASE,
    SIZE_BASE,
    Sample,
    TaskConfig,
    pad_query,
    sample_rng,
)

FAMILY = "C"

COUNT_CAP = 10
NUM_CLASSES = COUNT_CAP + 1  # 0 through 10 inclusive
CHANCE = 1.0 / NUM_CLASSES

# depth_ood is absent: depth is 1 by construction and there is nothing to
# extrapolate along. A family C depth_ood split would be a relabelling of
# the training distribution, which is worse than not having one.
SUPPORTED_SPLITS = frozenset({"train", "iid_val", "breadth_ood", "combo_ood"})

SPLIT_RANGES = {
    "train": ((1,), (4, 5, 6, 7, 8)),
    "iid_val": ((1,), (4, 5, 6, 7, 8)),
    "breadth_ood": ((1,), (9, 10, 11, 12, 13, 14, 15, 16)),
    "combo_ood": ((1,), (4, 5, 6, 7, 8)),
}


def matches(sprite, colour: int, shape: str, size: str | None) -> bool:
    if sprite.colour != colour or sprite.shape != shape:
        return False
    return size is None or sprite.size == size


def count_matching(sprites, colour: int, shape: str, size: str | None) -> int:
    return min(sum(matches(s, colour, shape, size) for s in sprites), COUNT_CAP)


def generate(idx: int, split: str, cfg: TaskConfig | None = None) -> Sample:
    cfg = cfg or TaskConfig()
    from loopvision.data.dataset import split_spec

    spec = split_spec(FAMILY, split, cfg)
    rng = sample_rng(FAMILY, split, idx, cfg)

    depth = int(rng.choice(spec.depth))
    breadth = int(rng.choice(spec.breadth))
    sprites = scenes.place_sprites(rng, breadth, split)

    # Query a conjunction of two or three attributes. Drawn from the pairs
    # actually present about half the time, so the label is not almost
    # always zero, which would make the task a presence detector.
    use_size = bool(rng.integers(0, 2))
    pairs = scenes.allowed_pairs(split)
    if split == "combo_ood":
        pairs = [p for p in pairs if p in scenes.HELDOUT_PAIRS]

    if rng.random() < 0.5 and sprites:
        present = [s.key() for s in sprites if s.key() in pairs]
        colour, shape = (
            present[int(rng.integers(0, len(present)))]
            if present
            else pairs[int(rng.integers(0, len(pairs)))]
        )
    else:
        colour, shape = pairs[int(rng.integers(0, len(pairs)))]

    size = render.SIZES[int(rng.integers(0, len(render.SIZES)))] if use_size else None
    label = count_matching(sprites, colour, shape, size)

    tokens = [FAM_C, QUERY, COLOUR_BASE + colour, SHAPE_BASE + render.SHAPES.index(shape)]
    if size is not None:
        tokens.append(SIZE_BASE + render.SIZES.index(size))

    program = (
        f"C;d={depth};b={breadth};query={colour}{shape}{size or '*'};"
        f"scene={scenes.scene_program(sprites)}"
    )

    return Sample(
        image=scenes.render_scene(sprites, cfg.canvas),
        query=pad_query(tokens, FAMILY),
        label=label,
        depth=depth,
        breadth=breadth,
        program=program,
        idx=idx,
        layout_seed=int(rng.integers(0, 2**31 - 1)),
    )
