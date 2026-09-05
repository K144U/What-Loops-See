"""Family A: transformational state tracking in pixels. The flagship task.

The pixel analogue of the group word problems from the text-case work,
which is what lets loop-count curves here be plotted against the text
results on the same axes.

Layout, one object per patch-grid row:

    col 0        the object's initial state
    col 1..d     its operator strip, in application order
    col 7        the query marker, on the queried row only

Reading the label means finding the marked row, reading its initial state,
and composing d operators in order. Distractor objects carry their own
independent strips, so the query has to be located before anything else is
useful. Chance is 1/48 = 0.021, and labels are uniform by construction.

Depth is the operator strip length. Breadth is the number of objects. They
vary independently, which is the whole point: H1 says loop requirements
track the first and not the second.
"""

from __future__ import annotations

import numpy as np

from loopvision.data import groups as G
from loopvision.data import render
from loopvision.data.dataset import (
    FAM_A,
    OP_BASE,
    QUERY,
    Sample,
    TaskConfig,
    pad_query,
    sample_rng,
)

FAMILY = "A"
NUM_CLASSES = G.GROUP_ORDER  # 48
CHANCE = 1.0 / NUM_CLASSES

# combo_ood is deliberately absent. It is defined as unseen colour and
# shape pairings, which family A does not have: its only colours are the
# three group-permuted glyph colours, and holding any of them out would
# break the uniform-factorisation property the anti-shortcut sampler
# depends on. See docs/decisions.md D-018.
SUPPORTED_SPLITS = frozenset({"train", "iid_val", "depth_ood", "breadth_ood"})


def _object_program(state: int, ops: list[int]) -> str:
    return f"{G.element_name(state)}:" + ">".join(G.element_name(o) for o in ops)


def build_program(
    initial: list[int], strips: list[list[int]], queried: int, depth: int, breadth: int
) -> str:
    """Canonical string form of the ground truth program.

    Ordered and fully determined, so two samples with the same program are
    the same problem, and an M2 counterfactual is a one-token edit of this
    string followed by a re-render at the same layout seed.
    """
    bodies = "|".join(
        _object_program(s, ops) for s, ops in zip(initial, strips)
    )
    return f"A;d={depth};b={breadth};q={queried};{bodies}"


def render_scene(
    initial: list[int],
    strips: list[list[int]],
    queried: int,
    canvas_size: int = render.CANVAS,
    substrate: str = "native",
) -> np.ndarray:
    """Draw the scene. Pure function of the program, no randomness."""
    canvas = render.blank_canvas(canvas_size)
    for row, (state, ops) in enumerate(zip(initial, strips)):
        render.paste_patch(
            canvas,
            render.render_element(state, substrate=substrate),
            row,
            render.STATE_COL,
        )
        for j, op in enumerate(ops):
            render.paste_patch(
                canvas,
                render.render_element(op, substrate=substrate),
                row,
                render.FIRST_OP_COL + j,
            )
    render.paste_patch(canvas, render.render_marker(), queried, render.MARKER_COL)
    return canvas


def generate(idx: int, split: str, cfg: TaskConfig | None = None) -> Sample:
    cfg = cfg or TaskConfig()
    from loopvision.data.dataset import split_spec

    spec = split_spec(FAMILY, split, cfg)
    rng = sample_rng(FAMILY, split, idx, cfg)

    depth = int(rng.choice(spec.depth))
    breadth = int(rng.choice(spec.breadth))
    if depth > render.MAX_DEPTH:
        raise ValueError(f"depth {depth} exceeds the {render.MAX_DEPTH} operator columns")
    if breadth > render.MAX_BREADTH:
        raise ValueError(f"breadth {breadth} exceeds the {render.MAX_BREADTH} rows")

    members = G.subgroup_members(G.substrate_subgroup(cfg.substrate, cfg.factor))
    initial = [int(members[int(rng.integers(0, len(members)))]) for _ in range(breadth)]
    queried = int(rng.integers(0, breadth))

    # The label is drawn uniformly and the queried object's composite is
    # solved for, rather than the label falling out of a sampled composite.
    # Both give a uniform label here, but solving for it makes the
    # uniformity structural rather than a consequence to be re-derived.
    label = int(members[int(rng.integers(0, len(members)))])

    strips: list[list[int]] = []
    for b in range(breadth):
        if b == queried:
            composite = G.multiply(label, G.inverse(initial[b]))
        else:
            # Distractors are independent, so their strips carry no
            # information about the answer.
            composite = int(members[int(rng.integers(0, len(members)))])
        strips.append(G.random_factorisation_in(rng, composite, depth, members))

    assert G.multiply(G.compose_sequence(strips[queried]), initial[queried]) == label

    image = render_scene(initial, strips, queried, cfg.canvas, cfg.substrate)

    tokens = [FAM_A, QUERY]
    if cfg.presentation == "tokens":
        # Operators move out of the image and into the query stream, which
        # isolates sequential composition from visual parsing.
        tokens += [OP_BASE + o for o in strips[queried]]
        image = render_scene(initial, [[] for _ in strips], queried, cfg.canvas)
    elif cfg.presentation not in ("strip", "frames"):
        raise ValueError(f"unknown presentation {cfg.presentation!r}")

    return Sample(
        image=image,
        query=pad_query(tokens, FAMILY),
        label=label,
        depth=depth,
        breadth=breadth,
        program=build_program(initial, strips, queried, depth, breadth),
        idx=idx,
        layout_seed=int(rng.integers(0, 2**31 - 1)),
    )


def corrupted_twin(sample: Sample, cfg: TaskConfig | None = None, which: int = 0) -> Sample:
    """A twin differing in exactly one operator of the queried strip.

    The M2 counterfactual. Everything else, including every distractor and
    every layout position, is identical, so a patching experiment isolates
    the effect of one program element rather than of a redrawn scene.
    """
    cfg = cfg or TaskConfig()
    initial, strips, queried, depth, breadth = parse_program(sample.program)

    rng = np.random.default_rng(sample.layout_seed + 1)
    ops = list(strips[queried])
    position = which % len(ops)
    # Uniform over the alternatives *within the same subgroup*. Drawing
    # from the full group under a restricted factor would put an operator
    # in the twin that the task never contains, so the patch would measure
    # an out of distribution input rather than a counterfactual.
    members = G.subgroup_members(G.substrate_subgroup(cfg.substrate, cfg.factor))
    here = members.index(ops[position])
    replacement = int(rng.integers(0, len(members) - 1))
    if replacement >= here:
        replacement += 1
    replacement = int(members[replacement])
    ops[position] = replacement
    strips[queried] = ops

    label = G.multiply(G.compose_sequence(ops), initial[queried])
    return Sample(
        image=render_scene(initial, strips, queried, cfg.canvas, cfg.substrate),
        query=sample.query,
        label=label,
        depth=depth,
        breadth=breadth,
        program=build_program(initial, strips, queried, depth, breadth),
        idx=sample.idx,
        layout_seed=sample.layout_seed,
    )


def parse_program(program: str) -> tuple[list[int], list[list[int]], int, int, int]:
    """Inverse of build_program. Round trips exactly, asserted in tests."""
    head, *bodies = program.split(";")
    if head != "A":
        raise ValueError(f"not a family A program: {program!r}")
    depth = int(bodies[0].split("=")[1])
    breadth = int(bodies[1].split("=")[1])
    queried = int(bodies[2].split("=")[1])

    name_to_index = {G.element_name(i): i for i in range(G.GROUP_ORDER)}
    initial: list[int] = []
    strips: list[list[int]] = []
    for body in bodies[3].split("|"):
        state_name, _, ops_part = body.partition(":")
        initial.append(name_to_index[state_name])
        strips.append(
            [name_to_index[o] for o in ops_part.split(">")] if ops_part else []
        )
    return initial, strips, queried, depth, breadth
