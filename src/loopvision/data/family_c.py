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

from dataclasses import replace as dc_replace

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


def effective_floor(split: str = "train", n: int = 5000) -> float:
    """The score a model that has learned nothing actually gets.

    NOT 1/NUM_CLASSES. Counts are not uniform: small counts are far more
    common than large ones, so always answering the modal count beats
    uniform guessing by a wide margin. Measured 0.377 on train and 0.255
    on breadth_ood against a 1/11 = 0.0909 that means nothing here.

    Family B was quoted against 1/NUM_CLASSES for weeks and it hid runs
    that had learned nothing behind numbers that looked like partial
    progress (D-028). The same mistake is available here, so the floor is
    computed rather than recalled.
    """
    import collections

    from loopvision.data import dataset as D

    counts = collections.Counter(
        generate(D.global_index(split, i), split).label for i in range(n)
    )
    return max(counts.values()) / sum(counts.values())

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


#: Attribute subsets a query may specify. **One or two, never three.**
#:
#: IMPLEMENTATION.md Section 4.4 says "a conjunction of two or three
#: attributes". That makes a match rare, and the consequence was measured:
#: 53.5 percent of labels were 0 and 93.4 percent were 0 or 1, with the
#: count never exceeding 4 despite a cap of 10. A model that always answered
#: "0" scored 0.535. The task was a presence detector, not a counting task.
#:
#: Worse for its actual job: family C exists to stress **breadth**, and a
#: rare conjunction barely responds to it. Going from 4 sprites to 16 left
#: the answer almost unchanged, so the axis existed in the data and not in
#: the task. Looser queries make the count scale with scene size, which is
#: the whole point. See docs/decisions.md D-026.
QUERY_SUBSETS = (
    ("colour",),
    ("shape",),
    ("size",),
    ("colour", "size"),
    ("shape", "size"),
    ("colour", "shape"),
)


def matches(sprite, spec: dict) -> bool:
    """Does this sprite satisfy every attribute the query names?"""
    return all(sprite.attribute(a) == v for a, v in spec.items())


def count_matching(sprites, spec: dict) -> int:
    return min(sum(matches(s, spec) for s in sprites), COUNT_CAP)


ATTRIBUTE_POOL = {
    "colour": lambda: list(range(render.N_SPRITE_COLOURS)),
    "shape": lambda: list(render.SHAPES),
    "size": lambda: list(render.SIZES),
}


def _replay(sample: Sample, cfg: TaskConfig, split: str):
    """Rebuild this sample's scene and query from its index.

    Duplicates the opening of `generate`, in the same order, because the
    random stream is consumed in that order and a scene is a pure function
    of the index. `corrupted_twin` renders the result and compares, so the
    duplication cannot drift silently.
    """
    from loopvision.data.dataset import split_spec

    spec = split_spec(FAMILY, split, cfg)
    rng = sample_rng(FAMILY, split, sample.idx, cfg)
    depth = int(rng.choice(spec.depth))
    breadth = int(rng.choice(spec.breadth))
    sprites = scenes.place_sprites(rng, breadth, split)
    attrs = QUERY_SUBSETS[int(rng.integers(0, len(QUERY_SUBSETS)))]
    if sprites is not None and rng.random() < 0.5:
        source = sprites[int(rng.integers(0, len(sprites)))]
        query_spec = {a: source.attribute(a) for a in attrs}
    else:
        query_spec = {
            a: ATTRIBUTE_POOL[a]()[int(rng.integers(0, len(ATTRIBUTE_POOL[a]())))]
            for a in attrs
        }
    return depth, breadth, sprites, query_spec


def corrupted_twin(
    sample: Sample,
    cfg: TaskConfig | None = None,
    which: int = 0,
    split: str = "iid_val",
) -> Sample | None:
    """A twin where one attribute of one sprite flips its match status.

    The M2 counterfactual for the breadth arm, and the one H3 actually
    turns on. H3 predicts breadth tasks are loop-diffuse and spatially
    local, and "spatially local" is only testable if the corruption sits at
    a single place in the image. Editing the query instead would change
    which sprites are relevant everywhere at once and could not distinguish
    the two predictions.

    So exactly one sprite changes, in exactly one attribute, and the count
    moves by one. Position is untouched: `Sprite` keeps row and col
    separate from appearance, so the edited sprite stays where it was and
    the two images differ in one cell and nowhere else.

    Returns None rather than something misleading when:

    - **No sprite can be flipped by a single attribute.** Turning a
      non-matching sprite on needs every queried attribute to agree, so
      only sprites that already disagree in exactly one are eligible.
    - **The label does not move.** The count is capped at
      COUNT_CAP, so a scene already at the cap can lose a match without the
      answer changing, and a recovery ratio needs a clean-corrupt gap.
    """
    cfg = cfg or TaskConfig()
    depth, breadth, sprites, query_spec = _replay(sample, cfg, split)
    if not sprites:
        return None

    if not np.array_equal(scenes.render_scene(sprites, cfg.canvas), sample.image):
        raise ValueError(
            f"replayed scene for index {sample.idx} does not render to the "
            f"original image, so the replay and generate() have drifted. The "
            f"counterfactual would be against a scene the model never saw."
        )

    queried = list(query_spec)
    rng = np.random.default_rng(sample.layout_seed + 1)

    # Prefer colour, then shape, then size, so the edit is the smallest
    # visible change available rather than whichever attribute comes first.
    order = [a for a in ("colour", "shape", "size") if a in query_spec]

    candidates: list[tuple[int, str, object]] = []
    for i, sp in enumerate(sprites):
        if matches(sp, query_spec):
            # Turn it off: any queried attribute set to a different value.
            for a in order:
                pool = [v for v in ATTRIBUTE_POOL[a]() if v != sp.attribute(a)]
                if pool:
                    candidates.append((i, a, pool[int(rng.integers(0, len(pool)))]))
                    break
        else:
            wrong = [a for a in queried if sp.attribute(a) != query_spec[a]]
            if len(wrong) == 1:
                # Turn it on: agree on the single attribute that disagrees.
                candidates.append((i, wrong[0], query_spec[wrong[0]]))

    if not candidates:
        return None

    idx_sp, attr, value = candidates[which % len(candidates)]
    edited = list(sprites)
    edited[idx_sp] = dc_replace(sprites[idx_sp], **{attr: value})

    label = count_matching(edited, query_spec)
    # Flipping one sprite's match status moves the count by one, so this
    # can only fire when COUNT_CAP truncates the difference away. Measured
    # on 2026-09-07 over 800 iid_val items: the largest count produced was
    # 7 against a cap of 10, so no sample reaches it and this branch is
    # currently unreachable. Kept because the breadth range and the query
    # subsets are both free to change, and pinned by
    # test_the_cap_is_currently_out_of_reach so a change says so.
    if label == sample.label:
        return None

    desc = "+".join(f"{a}:{query_spec[a]}" for a in sorted(query_spec))
    return Sample(
        image=scenes.render_scene(edited, cfg.canvas),
        query=sample.query,
        label=label,
        depth=depth,
        breadth=breadth,
        program=f"C;d={depth};b={breadth};query={desc};"
        f"scene={scenes.scene_program(edited)}",
        idx=sample.idx,
        layout_seed=sample.layout_seed,
    )


def generate(idx: int, split: str, cfg: TaskConfig | None = None) -> Sample:
    cfg = cfg or TaskConfig()
    from loopvision.data.dataset import split_spec

    spec = split_spec(FAMILY, split, cfg)
    rng = sample_rng(FAMILY, split, idx, cfg)

    depth = int(rng.choice(spec.depth))
    breadth = int(rng.choice(spec.breadth))
    sprites = scenes.place_sprites(rng, breadth, split)

    attrs = QUERY_SUBSETS[int(rng.integers(0, len(QUERY_SUBSETS)))]

    # Half the time, build the query from a sprite that is actually there,
    # so the answer is not almost always zero. The other half is drawn
    # freely, which keeps genuine zeros in the distribution.
    if sprites is not None and rng.random() < 0.5:
        source = sprites[int(rng.integers(0, len(sprites)))]
        query_spec = {a: source.attribute(a) for a in attrs}
    else:
        pool = {
            "colour": list(range(render.N_SPRITE_COLOURS)),
            "shape": list(render.SHAPES),
            "size": list(render.SIZES),
        }
        query_spec = {a: pool[a][int(rng.integers(0, len(pool[a])))] for a in attrs}

    label = count_matching(sprites, query_spec)

    tokens = [FAM_C, QUERY]
    for a in ("colour", "shape", "size"):
        if a not in query_spec:
            continue
        if a == "colour":
            tokens.append(COLOUR_BASE + query_spec[a])
        elif a == "shape":
            tokens.append(SHAPE_BASE + render.SHAPES.index(query_spec[a]))
        else:
            tokens.append(SIZE_BASE + render.SIZES.index(query_spec[a]))

    desc = "+".join(f"{a}:{query_spec[a]}" for a in sorted(query_spec))
    program = (
        f"C;d={depth};b={breadth};query={desc};"
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
