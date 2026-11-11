"""Family B: relational chain queries. The family vision reviewers read fastest.

A scene of sprites, and a question of the form "what is the <attribute> of
the object <rel_k> of the object ... <rel_1> of the <colour> <shape>".
Depth is the chain length: depth 1 is the anchor alone, depth 6 is the
anchor plus five relational hops.

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
    SIZE_BASE,
    Sample,
    TaskConfig,
    pad_query,
    sample_rng,
)

FAMILY = "B"

#: Which attributes a question may ask ABOUT. Deliberately not all three.
#:
#: `size` has only two values, so when it was askable, one question in
#: three had a chance level of 0.5 and family B's real chance was 0.2897
#: rather than the 1/13 = 0.0769 reported everywhere. Every family B number
#: recorded before 2026-09-04 was compared against a floor 3.8 times too
#: low, which made runs that had learned nothing look like partial
#: successes. See docs/findings.md and D-028.
#:
#: `size` is still a descriptor: it helps name the anchor, it is just never
#: the thing being asked for.
ASKABLE_ATTRIBUTES = ("colour", "shape")

# 6 colours, then 5 shapes, then 2 sizes. One flat label space so the head
# is a single softmax regardless of which attribute was asked for.
NUM_CLASSES = render.N_SPRITE_COLOURS + len(render.SHAPES) + len(render.SIZES)  # 13
CHANCE = 1.0 / NUM_CLASSES


def effective_chance() -> float:
    """The floor a family B model must actually beat.

    NOT 1/NUM_CLASSES. The label space is a union of three attribute
    spaces of different sizes, and only one attribute is asked per sample,
    so the reachable label set is the asked attribute's alone. A uniform
    guess inside it succeeds with probability 1/|that attribute|, and the
    floor is the average over which attribute gets asked.

    With colour and shape askable this is 0.1833. With size askable too it
    was 0.2897, because a size question is a coin flip. Quoting 1/13 =
    0.0769 against either understates the floor by three to four times.
    """
    sizes = {
        "colour": render.N_SPRITE_COLOURS,
        "shape": len(render.SHAPES),
        "size": len(render.SIZES),
    }
    return sum(
        (1.0 / len(ASKABLE_ATTRIBUTES)) * (1.0 / sizes[a])
        for a in ASKABLE_ATTRIBUTES
    )

SUPPORTED_SPLITS = frozenset({"train", "iid_val", "breadth_ood", "combo_ood"})

#: (depth, breadth) per split. Chains run 1 to 6, matching family A.
#
# A depth d chain visits d distinct sprites, so a cell is structurally
# empty when breadth is below depth and merely hard to sample just above
# it. That is the floor. The range sits well clear of it for a separate
# reason recorded below: sparse scenes make the chain skippable.
#
# Crucially, every split shares one breadth range except `breadth_ood`,
# which is the one arm allowed to move breadth, and it holds depth fixed
# while doing so. Letting a split move both axes at once would confound
# them and make H1 unanswerable, which is a worse failure than a high
# rejection rate.
SPLIT_RANGES = {
    "train": ((1, 2, 3, 4, 5, 6), (12, 13, 14, 15, 16)),
    "iid_val": ((1, 2, 3, 4, 5, 6), (12, 13, 14, 15, 16)),
    "breadth_ood": ((1, 2, 3, 4, 5, 6), (18, 19, 20, 21, 22)),
    "combo_ood": ((1, 2, 3, 4, 5, 6), (12, 13, 14, 15, 16)),
}

# Depth now runs to 6, and the breadth range moved up to reach it. Both
# changes are forced by measurement rather than chosen.
#
# Rejection rate against the 0.30 ceiling, at the new breadths:
#
#            b=12    b=14    b=16
#   depth 4  0.000   0.029   0.029
#   depth 5  0.029   0.057   0.010
#   depth 6  0.083   0.074   0.020
#
# At the old breadths of 6 to 9 the same depths breached badly, 0.362 at
# depth 5 and 0.571 at depth 6, which is why depth used to stop at 4. The
# ceiling was never about depth, it was about having enough sprites for a
# chain to have somewhere to go.
#
# Depth stopping at 4 was the reason family B had no loop-count curve to
# show: a model with prelude 2, core 2 and coda 2 has six sequential
# blocks at k=1, and a four hop chain needs about four. Depth 6 puts the
# deepest cells outside a single pass for a small core.
#
# The breadth move is also what makes the chain load bearing. A model
# ignoring the relation chain entirely scored 0.62 at depth 4 and breadth
# 6 to 9, with 19 percent of questions having only one possible answer.
# At breadth 12 to 16 with size unaskable that falls to 0.44 and near zero.
# See D-028.

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
    """One rejection-sampling attempt. Returns None if the chain dies.

    The attribute is chosen **first**, and the anchor is then identified by
    the other two. At depth 1 the target is the anchor, so describing it by
    the attribute being asked about hands over the answer. Before this
    ordering was fixed, two thirds of depth 1 samples were answerable from
    the query alone.
    """
    sprites = scenes.place_sprites(rng, breadth, split)
    attribute = ASKABLE_ATTRIBUTES[int(rng.integers(0, len(ASKABLE_ATTRIBUTES)))]
    anchors = scenes.unique_anchors(sprites, attribute)
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
    desc = "+".join(
        f"{a}:{anchor.attribute(a)}" for a in scenes.DESCRIPTOR_FOR[attribute]
    )
    return (
        f"B;d={depth};b={breadth};attr={attribute};"
        f"anchor={desc};chain={'>'.join(chain) or '-'};"
        f"target={target.token()};scene={scenes.scene_program(sprites)}"
    )


def generate(idx: int, split: str, cfg: TaskConfig | None = None) -> Sample:
    cfg = cfg or TaskConfig()
    from loopvision.data.dataset import split_spec

    spec = split_spec(FAMILY, split, cfg)
    rng = sample_rng(FAMILY, split, idx, cfg)

    depth = int(rng.choice(spec.depth))
    breadth = int(rng.choice(spec.breadth))

    (sprites, anchor, chain, target, attribute), _ = sample_scene(
        rng, depth, breadth, split
    )

    # One token per descriptor attribute. The token ranges differ per
    # attribute type, so the model can tell which is which without a
    # positional convention, and the queried attribute never appears.
    base = {
        "colour": lambda s: COLOUR_BASE + s.colour,
        "shape": lambda s: SHAPE_BASE + render.SHAPES.index(s.shape),
        "size": lambda s: SIZE_BASE + render.SIZES.index(s.size),
    }
    tokens = [FAM_B, QUERY, ATTR_BASE + ATTRIBUTES.index(attribute)]
    tokens += [base[a](anchor) for a in scenes.DESCRIPTOR_FOR[attribute]]
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


def corrupted_twin(
    sample: Sample,
    cfg: TaskConfig | None = None,
    which: int = 0,
    split: str = "iid_val",
) -> Sample | None:
    """A twin differing in exactly one relation hop. The M2 counterfactual.

    **The image is byte identical.** Family A renders its operators into the
    scene, so its twin re-renders. Here the chain lives in the query, so the
    twin changes one relation token and leaves every pixel alone. That makes
    this the cleaner of the two counterfactuals: patching isolates where the
    edited hop is processed with the visual input held exactly constant.

    The scene is regenerated from the sample's index rather than parsed back
    out of the program string, because a scene is a pure function of that
    index and there is no parser for `scenes.scene_program`. The replay
    duplicates the first few lines of `generate`, and
    `tests/test_family_b_twin.py` asserts the regenerated image matches the
    original byte for byte, so the two cannot drift apart silently.

    Returns None when no counterfactual exists, rather than returning
    something misleading:

    - **Depth 1 has no hop.** The chain is empty, there is nothing to edit.
    - **The edited chain may die.** Family B rejection samples precisely
      because a relation can point at empty space. A replacement hop that
      does not resolve is not a harder question, it is not a question, so
      every alternative is tried and the item is dropped if none survive.
    """
    cfg = cfg or TaskConfig()
    from loopvision.data.dataset import split_spec

    spec = split_spec(FAMILY, split, cfg)
    rng = sample_rng(FAMILY, split, sample.idx, cfg)
    depth = int(rng.choice(spec.depth))
    breadth = int(rng.choice(spec.breadth))
    (sprites, anchor, chain, _target, attribute), _ = sample_scene(
        rng, depth, breadth, split
    )

    # The replay must reproduce the ORIGINAL scene, not merely some scene.
    # Checking `twin.image == sample.image` cannot catch drift, because the
    # twin reuses `sample.image` by construction and that comparison is true
    # however wrong the sprites are. Rendering the replayed sprites and
    # comparing is the check that actually binds.
    if not np.array_equal(scenes.render_scene(sprites, cfg.canvas), sample.image):
        raise ValueError(
            f"replayed scene for index {sample.idx} does not render to the "
            f"original image, so the replay and generate() have drifted. The "
            f"counterfactual would be against a scene the model never saw."
        )

    if not chain:
        return None

    position = which % len(chain)
    here = scenes.RELATIONS.index(chain[position])
    order = [(here + 1 + j) % len(scenes.RELATIONS) for j in range(len(scenes.RELATIONS) - 1)]

    for cand in order:
        edited = list(chain)
        edited[position] = scenes.RELATIONS[cand]
        target = resolve(sprites, anchor, edited)
        if target is None:
            continue

        base = {
            "colour": lambda s: COLOUR_BASE + s.colour,
            "shape": lambda s: SHAPE_BASE + render.SHAPES.index(s.shape),
            "size": lambda s: SIZE_BASE + render.SIZES.index(s.size),
        }
        tokens = [FAM_B, QUERY, ATTR_BASE + ATTRIBUTES.index(attribute)]
        tokens += [base[a](anchor) for a in scenes.DESCRIPTOR_FOR[attribute]]
        tokens += [REL_BASE + scenes.RELATIONS.index(r) for r in edited]

        return Sample(
            image=sample.image,
            query=pad_query(tokens, FAMILY),
            label=encode_label(attribute, target.attribute(attribute)),
            depth=depth,
            breadth=breadth,
            program=build_program(
                anchor, edited, target, attribute, sprites, depth, breadth
            ),
            idx=sample.idx,
            layout_seed=sample.layout_seed,
        )

    return None


def hop_labels(idx: int, split: str, cfg: TaskConfig | None = None) -> list[int]:
    """The label at every point along the chain, not just the end.

    Element j is what the answer would be if the question stopped after j
    hops, so element 0 is the anchor and the last element is the real
    answer. A depth d sample returns d labels.

    This is what turns the coda lens from a picture into a measurement.
    Decoding an intermediate state tells you what the model currently
    believes; comparing that against these tells you **how far along the
    chain it has got**, which is the quantity that explains why one seed
    needs more core passes than another.

    Regenerates the sample rather than storing the labels on it, so the
    Sample type stays the same shape for every family and nothing about
    the training path changes to support an instrument.
    """
    cfg = cfg or TaskConfig()
    from loopvision.data.dataset import split_spec

    spec = split_spec(FAMILY, split, cfg)
    rng = sample_rng(FAMILY, split, idx, cfg)
    depth = int(rng.choice(spec.depth))
    breadth = int(rng.choice(spec.breadth))
    (sprites, anchor, chain, target, attribute), _ = sample_scene(
        rng, depth, breadth, split
    )

    labels = [encode_label(attribute, anchor.attribute(attribute))]
    current = anchor
    for relation in chain:
        current = scenes.relate(sprites, current, relation)
        labels.append(encode_label(attribute, current.attribute(attribute)))

    if current.token() != target.token():
        raise RuntimeError(
            "hop_labels walked to a different sprite than generate() did, so "
            "the regeneration has drifted from the real sample"
        )
    return labels


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
