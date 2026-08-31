"""D4 x S3 group algebra for family A.

The group is the direct product of D4, the dihedral group of order 8 acting
on the plane by quarter turns and reflections, with S3, the permutation
group on three colours. Order 48, non-abelian in both factors, so operator
order genuinely matters and a model cannot succeed by counting operators.

Everything is built once into explicit 48 by 48 tables. The tables are the
source of truth and the closed-form formulas are only used to build them,
so a mistake in a formula surfaces as a failed associativity test rather
than as a subtly wrong dataset.

Two conventions, fixed here and relied on everywhere downstream:

  Composition is function composition. ``multiply(x, y)`` means apply y
  first, then x. Written x . y.

  An operator sequence (o_1, ..., o_n) is applied in order, o_1 first. Its
  composite is therefore o_n . ... . o_1, which is ``compose_sequence``.
  Read the sequence left to right as time, not as an algebraic product.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Element encoding
# ---------------------------------------------------------------------------
#
# A D4 element is (rot, flip) with rot in 0..3 quarter turns counterclockwise
# and flip in 0..1 reflections about the main diagonal, denoting r^rot f^flip.
# An S3 element is a 3-tuple p where colour c maps to p[c].
#
# The group element index is d4_index * 6 + s3_index, so index // 6 and
# index % 6 recover the factors. This ordering puts the identity at 0.

D4_ORDER = 8
S3_ORDER = 6
GROUP_ORDER = D4_ORDER * S3_ORDER  # 48

D4_ELEMENTS: tuple[tuple[int, int], ...] = tuple(
    (rot, flip) for rot in range(4) for flip in range(2)
)

S3_ELEMENTS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)

_D4_INDEX = {e: i for i, e in enumerate(D4_ELEMENTS)}
_S3_INDEX = {e: i for i, e in enumerate(S3_ELEMENTS)}

IDENTITY = 0  # (rot=0, flip=0) x (0, 1, 2)


def d4_of(index: int) -> tuple[int, int]:
    return D4_ELEMENTS[index // S3_ORDER]


def s3_of(index: int) -> tuple[int, int, int]:
    return S3_ELEMENTS[index % S3_ORDER]


def element_index(d4: tuple[int, int], s3: tuple[int, int, int]) -> int:
    return _D4_INDEX[d4] * S3_ORDER + _S3_INDEX[s3]


def element_name(index: int) -> str:
    """Human readable label, for sample dumps and debugging."""
    rot, flip = d4_of(index)
    perm = "".join(str(c) for c in s3_of(index))
    return f"r{rot}f{flip}.{perm}"


# ---------------------------------------------------------------------------
# Group law
# ---------------------------------------------------------------------------


def _d4_multiply(x: tuple[int, int], y: tuple[int, int]) -> tuple[int, int]:
    """x . y in D4, apply y first.

    From the dihedral relation f r = r^-1 f:

        (r^a f^b)(r^c f^d) = r^(a + c(-1)^b) f^(b + d)
    """
    a, b = x
    c, d = y
    rot = (a + (c if b == 0 else -c)) % 4
    flip = (b + d) % 2
    return rot, flip


def _s3_multiply(
    x: tuple[int, int, int], y: tuple[int, int, int]
) -> tuple[int, int, int]:
    """x . y in S3, apply y first, so (x.y)[c] = x[y[c]]."""
    return (x[y[0]], x[y[1]], x[y[2]])


def _build_tables() -> tuple[np.ndarray, np.ndarray]:
    mul = np.empty((GROUP_ORDER, GROUP_ORDER), dtype=np.int64)
    for i in range(GROUP_ORDER):
        for j in range(GROUP_ORDER):
            mul[i, j] = element_index(
                _d4_multiply(d4_of(i), d4_of(j)),
                _s3_multiply(s3_of(i), s3_of(j)),
            )

    inv = np.empty(GROUP_ORDER, dtype=np.int64)
    for i in range(GROUP_ORDER):
        # Exactly one j satisfies i . j = identity, guaranteed by the group
        # axioms and checked in tests/test_groups.py.
        inv[i] = int(np.flatnonzero(mul[i] == IDENTITY)[0])

    mul.flags.writeable = False
    inv.flags.writeable = False
    return mul, inv


#: MUL[i, j] is the index of i . j, meaning apply j first then i.
#: INV[i] is the index of the inverse of i.
MUL, INV = _build_tables()


def multiply(i: int, j: int) -> int:
    """i . j, apply j first then i."""
    return int(MUL[i, j])


def inverse(i: int) -> int:
    return int(INV[i])


def compose_sequence(ops) -> int:
    """Composite of an operator sequence applied in order, ops[0] first.

    Equals ops[-1] . ... . ops[0]. An empty sequence is the identity.
    """
    acc = IDENTITY
    for op in ops:
        acc = int(MUL[op, acc])
    return acc


def prefix_products(ops) -> list[int]:
    """Running composite after each operator. Length equals len(ops)."""
    out: list[int] = []
    acc = IDENTITY
    for op in ops:
        acc = int(MUL[op, acc])
        out.append(acc)
    return out


# ---------------------------------------------------------------------------
# Anti-shortcut sampling
# ---------------------------------------------------------------------------


def random_factorisation(rng: np.random.Generator, target: int, n: int) -> list[int]:
    """A uniformly random length-n operator sequence composing to ``target``.

    Draws the first n-1 operators uniformly and solves for the last, which
    yields a uniform distribution over the |G|^(n-1) factorisations of
    ``target``. Every proper prefix product is then uniform over the group,
    which is the property that kills prefix-statistics shortcuts: knowing
    the first j operators tells a model nothing about the answer.

    Note what this does and does not buy, because the distinction matters
    for gate G1. Each operator's marginal distribution is uniform and
    independent of the target, so no single-operator statistic helps. The
    multiset of operators is a different matter: see
    ``bag_of_operators_ceiling``.
    """
    if n < 1:
        raise ValueError(f"factorisation length must be at least 1, got {n}")
    if n == 1:
        return [target]

    ops = [int(x) for x in rng.integers(0, GROUP_ORDER, size=n - 1)]
    prefix = compose_sequence(ops)
    ops.append(multiply(target, inverse(prefix)))
    return ops


def sample_operator_sequence(
    rng: np.random.Generator, n: int
) -> tuple[list[int], int]:
    """Draw a target uniformly, then a uniform factorisation of it.

    Returns (ops, target). This is the sampler family A uses: the label is
    uniform over the 48 classes by construction, so chance accuracy is
    exactly 1/48 and no class prior is exploitable.
    """
    target = int(rng.integers(0, GROUP_ORDER))
    return random_factorisation(rng, target, n), target


def bag_of_operators_ceiling(
    rng: np.random.Generator, depth: int, trials: int = 20000
) -> float:
    """Best accuracy achievable by any order-blind predictor at this depth.

    An order-blind model sees only the multiset of operators. For a given
    multiset it can do no better than always answering the composite that
    the most orderings produce, so this is an upper bound on every
    bag-of-operators probe, logistic regression included, with unlimited
    data and unlimited capacity.

    This is an upper bound rather than a measurement of any particular
    model, which makes it the honest thing to gate on: a probe scoring
    below this tells us nothing, and no probe can score above it.

    Computed as the Bayes accuracy of an order-blind predictor,

        E_M [ max_g P(composite = g | multiset = M) ]

    estimated by sampling multisets and, for each, enumerating orderings to
    get the conditional distribution over composites. Returns a probability
    in [0, 1]. Chance is 1/48, about 0.0208.

    Do not reimplement this by picking the modal composite and checking
    whether it equals the sampled target. ``itertools.permutations`` yields
    the input ordering first, so ``Counter.most_common`` breaks ties in
    favour of the true answer and the estimate comes out at 1.0 for depth
    2, which is an artefact of tie ordering rather than a real shortcut.
    Taking the max probability sidesteps tie breaking entirely.
    """
    from collections import Counter
    from itertools import permutations

    if depth < 1:
        raise ValueError("depth must be at least 1")
    if depth == 1:
        return 1.0  # the single operator is the answer

    # Enumerating all orderings is factorial, so cap the exact treatment.
    # Beyond depth 6 the bound is far below anything of concern anyway.
    if depth > 6:
        raise ValueError("exact enumeration is impractical above depth 6")

    total_best = 0.0
    for _ in range(trials):
        ops, _ = sample_operator_sequence(rng, depth)
        counts = Counter(compose_sequence(p) for p in permutations(ops))
        orderings = sum(counts.values())
        total_best += max(counts.values()) / orderings
    return total_best / trials


# ---------------------------------------------------------------------------
# Action on a coloured glyph
# ---------------------------------------------------------------------------
#
# The stimulus must have a trivial stabiliser, otherwise two distinct group
# elements produce the same picture and the 48-way task is not actually
# 48-way. IMPLEMENTATION.md Section 4.2 suggests an L-tromino with a
# coloured tip. That shape does not work: the L-tromino is symmetric under
# reflection about its diagonal, and pairing that reflection with the
# transposition of the two arm colours fixes the glyph, giving a stabiliser
# of order 2 and only 24 distinct states.
#
# GLYPH_CELLS below is a chiral tetromino with three distinctly coloured
# cells and one neutral cell. Chirality kills the reflection symmetry and
# three distinct colours kill the colour permutations. The stabiliser is
# asserted trivial in tests/test_groups.py rather than argued for here.

NEUTRAL_COLOUR = -1

#: (x, y, colour) with colour in {0, 1, 2} or NEUTRAL_COLOUR.
#: A J-tetromino: a vertical bar of three with one cell attached at the foot.
GLYPH_CELLS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (0, 1, 1),
    (0, 2, 2),
    (1, 0, NEUTRAL_COLOUR),
)


def d4_apply_cell(d4: tuple[int, int], x: int, y: int) -> tuple[int, int]:
    """Apply a D4 element to a lattice point.

    Reflection about the main diagonal first, then quarter turns
    counterclockwise, matching the r^rot f^flip encoding.
    """
    rot, flip = d4
    if flip:
        x, y = y, x
    for _ in range(rot):
        x, y = -y, x
    return x, y


def _normalise(cells) -> frozenset[tuple[int, int, int]]:
    """Translate so the bounding box starts at the origin.

    Without this, a rotated glyph compares unequal to itself purely because
    it sits somewhere else on the lattice.
    """
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    dx, dy = min(xs), min(ys)
    return frozenset((x - dx, y - dy, c) for x, y, c in cells)


def act_on_glyph(
    index: int, cells=GLYPH_CELLS
) -> frozenset[tuple[int, int, int]]:
    """Apply a group element to the glyph, returning normalised cells."""
    d4 = d4_of(index)
    perm = s3_of(index)
    moved = []
    for x, y, colour in cells:
        nx, ny = d4_apply_cell(d4, x, y)
        moved.append((nx, ny, colour if colour == NEUTRAL_COLOUR else perm[colour]))
    return _normalise(moved)


def stabiliser(cells=GLYPH_CELLS) -> list[int]:
    """Group elements that leave the glyph unchanged.

    A glyph usable for the 48-way task has a stabiliser of exactly
    [IDENTITY]. Anything larger means the label is not recoverable from the
    picture and the task is silently smaller than it claims to be.
    """
    base = _normalise(cells)
    return [i for i in range(GROUP_ORDER) if act_on_glyph(i, cells) == base]


def distinct_states(cells=GLYPH_CELLS) -> int:
    """How many visually distinct states the glyph actually has."""
    return len({act_on_glyph(i, cells) for i in range(GROUP_ORDER)})
