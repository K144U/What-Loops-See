"""Sprite scenes shared by families B and C.

Both families place sprites on the same 8x8 patch grid with the same
attribute space, so placement, held-out pairings and rendering live here
once rather than twice.

Relations are defined as **nearest neighbour along a row or column**, not
as CLEVR's directional sets. Given a sprite, ``left_of`` is the nearest
sprite in the same grid row to its left, and so on. Two reasons, and the
second matters more:

  Uniqueness is structural. A directional set relation needs rejection
  sampling to guarantee a single referent, and at breadth 12 on a small
  grid most draws would be rejected. Nearest-neighbour resolves uniquely
  whenever it resolves at all, so the only rejection cause is a dead end.

  The chain stays a chain. With set semantics, a deep chain tends to
  collapse onto the same few extremal objects, so depth 4 is often no
  harder than depth 2. Nearest-neighbour hops keep every step informative,
  which is what the depth axis is supposed to measure.

This is a deviation from IMPLEMENTATION.md Section 4.3. See
docs/decisions.md D-019.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loopvision.data import render

RELATIONS = ("left_of", "right_of", "above", "below")

#: (colour, shape) pairs never shown in the training splits, so that
#: combo_ood can ask about a pairing the model has never seen. Six of the
#: thirty pairs, spread so no colour and no shape is entirely held out.
HELDOUT_PAIRS: frozenset[tuple[int, str]] = frozenset(
    {
        (0, "circle"),
        (1, "bar"),
        (2, "square"),
        (3, "ring"),
        (4, "triangle"),
        (5, "circle"),
    }
)


@dataclass(frozen=True)
class Sprite:
    row: int
    col: int
    colour: int
    shape: str
    size: str

    def key(self) -> tuple[int, str]:
        return (self.colour, self.shape)

    def attribute(self, name: str) -> int | str:
        if name == "colour":
            return self.colour
        if name == "shape":
            return self.shape
        if name == "size":
            return self.size
        raise ValueError(f"unknown attribute {name!r}")

    def token(self) -> str:
        """Canonical token. No commas: the scene joins tokens with commas,
        and an embedded one made the encoding ambiguous and unparseable."""
        return f"{self.colour}{self.shape[0]}{self.size[0]}@{self.row}x{self.col}"


def allowed_pairs(split: str) -> list[tuple[int, str]]:
    """Colour and shape pairings a sprite may take on this split."""
    every = [
        (c, s) for c in range(render.N_SPRITE_COLOURS) for s in render.SHAPES
    ]
    if split == "combo_ood":
        return every
    return [p for p in every if p not in HELDOUT_PAIRS]


def place_sprites(
    rng: np.random.Generator, count: int, split: str, grid: int = render.GRID
) -> list[Sprite]:
    """Scatter ``count`` sprites onto distinct grid cells, compactly.

    Placement is confined to a small rectangle rather than the whole 8x8
    grid, and this is load bearing rather than cosmetic. Relations resolve
    along shared rows and columns, and sprites scattered uniformly over 64
    cells almost never share either: measured rejection rates with uniform
    placement were 88 percent at depth 2 and 97 percent at depth 3, and
    depth 4 was unreachable. A compact rectangle gives most rows and
    columns several sprites, so chains exist to be walked.

    The rectangle is sized just larger than the sprite count and placed at
    a random offset, so absolute position stays uninformative.
    """
    if count > grid * grid:
        raise ValueError(f"cannot place {count} sprites on a {grid}x{grid} grid")

    cols = max(2, int(np.ceil(np.sqrt(count))))
    rows = max(2, int(np.ceil(count / cols)))
    cols = min(cols + int(rng.integers(0, 2)), grid)
    rows = min(rows + int(rng.integers(0, 2)), grid)
    while rows * cols < count:
        if cols < grid:
            cols += 1
        else:
            rows += 1

    row0 = int(rng.integers(0, grid - rows + 1))
    col0 = int(rng.integers(0, grid - cols + 1))
    cells = rng.permutation(rows * cols)[:count]

    pairs = allowed_pairs(split)
    heldout = sorted(HELDOUT_PAIRS)
    sprites = []
    for i, cell in enumerate(cells):
        if split == "combo_ood" and i == 0:
            # Place a held-out pairing deliberately rather than waiting for
            # one to turn up. Hoping for it meant a third of combo_ood
            # draws were rejected for containing nothing out of
            # distribution, which both wasted samples and skewed the
            # surviving scenes toward whatever happened to contain one.
            colour, shape = heldout[int(rng.integers(0, len(heldout)))]
        else:
            colour, shape = pairs[int(rng.integers(0, len(pairs)))]
        size = render.SIZES[int(rng.integers(0, len(render.SIZES)))]
        sprites.append(
            Sprite(
                row0 + int(cell) // cols,
                col0 + int(cell) % cols,
                colour,
                shape,
                size,
            )
        )
    return sprites


def walk_chain(
    rng: np.random.Generator, sprites: list[Sprite], anchor: Sprite, hops: int
):
    """Build a chain constructively by walking the scene.

    Sampling relations blind and rejecting dead ends wastes almost every
    draw at depth 3 or more, and worse, it biases the surviving scenes
    toward whatever layouts happen to satisfy random chains. Walking
    instead picks uniformly among the relations that actually resolve.

    Already-visited sprites are excluded, which matters for the depth axis:
    without it a chain could go left_of then right_of and land back on the
    anchor, making a depth 3 question exactly as hard as a depth 1
    question while still being labelled depth 3. Every hop therefore
    reaches a sprite the chain has not seen.

    Returns (chain, target) or None if the walk dead ends.
    """
    current = anchor
    visited = {(anchor.row, anchor.col)}
    chain: list[str] = []
    for _ in range(hops):
        options = []
        for relation in RELATIONS:
            nxt = relate(sprites, current, relation)
            if nxt is not None and (nxt.row, nxt.col) not in visited:
                options.append((relation, nxt))
        if not options:
            return None
        relation, nxt = options[int(rng.integers(0, len(options)))]
        chain.append(relation)
        visited.add((nxt.row, nxt.col))
        current = nxt
    return chain, current


def relate(sprites: list[Sprite], anchor: Sprite, relation: str) -> Sprite | None:
    """The nearest sprite in the given direction, or None at a dead end.

    Row indices grow downward, so ``above`` means a smaller row index.
    """
    if relation == "left_of":
        candidates = [s for s in sprites if s.row == anchor.row and s.col < anchor.col]
        return max(candidates, key=lambda s: s.col) if candidates else None
    if relation == "right_of":
        candidates = [s for s in sprites if s.row == anchor.row and s.col > anchor.col]
        return min(candidates, key=lambda s: s.col) if candidates else None
    if relation == "above":
        candidates = [s for s in sprites if s.col == anchor.col and s.row < anchor.row]
        return max(candidates, key=lambda s: s.row) if candidates else None
    if relation == "below":
        candidates = [s for s in sprites if s.col == anchor.col and s.row > anchor.row]
        return min(candidates, key=lambda s: s.row) if candidates else None
    raise ValueError(f"unknown relation {relation!r}")


#: Which two attributes describe an anchor when a given attribute is asked
#: about. Never the queried one: at depth 1 the target IS the anchor, so
#: naming the anchor by the attribute under question writes the answer into
#: the question. Measured before this fix: 66.8 percent of depth 1 family B
#: samples were answerable from the query tokens alone, and a trained model
#: scored 0.833 on a blank image. See docs/findings.md.
DESCRIPTOR_FOR = {
    "colour": ("shape", "size"),
    "shape": ("colour", "size"),
    "size": ("colour", "shape"),
}


def descriptor(sprite: Sprite, asked: str) -> tuple:
    """The sprite's identity under the two attributes not being asked about."""
    return tuple(sprite.attribute(a) for a in DESCRIPTOR_FOR[asked])


def unique_anchors(sprites: list[Sprite], asked: str) -> list[Sprite]:
    """Sprites that their descriptor identifies uniquely.

    An anchor matching two sprites makes the whole chain ambiguous, so only
    these can start a query. ``asked`` is required rather than optional:
    forgetting it is exactly the bug this signature exists to prevent.
    """
    counts: dict[tuple, int] = {}
    for s in sprites:
        key = descriptor(s, asked)
        counts[key] = counts.get(key, 0) + 1
    return [s for s in sprites if counts[descriptor(s, asked)] == 1]


def render_scene(sprites: list[Sprite], canvas_size: int = render.CANVAS) -> np.ndarray:
    canvas = render.blank_canvas(canvas_size)
    for s in sprites:
        render.paste_patch(
            canvas, render.render_sprite(s.colour, s.shape, s.size), s.row, s.col
        )
    return canvas


def scene_program(sprites: list[Sprite]) -> str:
    """Canonical, order independent description of a scene."""
    return ",".join(sorted(s.token() for s in sprites))
