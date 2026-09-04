"""Shared sample interface, vocabulary, splits, and the torch Dataset.

All data is generated procedurally and deterministically from
``(family, split, index, master_seed)``. Nothing is cached to disk beyond a
small validation dump. Two consequences that everything else leans on:

  Composition depth is *known*, not estimated. This is the single property
  the whole study rests on, per docs/paper.md Section 2.

  Dataloader position collapses to one integer, which is why training can
  resume bit exactly across a killed job. See train/checkpoint.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from loopvision.data import render

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
#
# One shared vocabulary across all three families, so one embedding table
# serves every task and the model does not need per-family heads on the
# input side. Ranges are contiguous and computed, never hardcoded twice.

PAD = 0
FAM_A, FAM_B, FAM_C = 1, 2, 3
QUERY = 4

_NEXT = 5

OP_BASE = _NEXT                      # 48 operator tokens for the tokens variant
_NEXT += 48

COLOUR_BASE = _NEXT                  # 6 sprite colours, families B and C
N_COLOURS = 6
_NEXT += N_COLOURS

SHAPE_BASE = _NEXT                   # 5 sprite shapes
N_SHAPES = 5
_NEXT += N_SHAPES

SIZE_BASE = _NEXT                    # 2 sprite sizes
N_SIZES = 2
_NEXT += N_SIZES

REL_BASE = _NEXT                     # left_of, right_of, above, below
RELATIONS = ("left_of", "right_of", "above", "below")
_NEXT += len(RELATIONS)

ATTR_BASE = _NEXT                    # which attribute a family B query asks for
ATTRIBUTES = ("colour", "shape", "size")
_NEXT += len(ATTRIBUTES)

ANSWER_BASE = _NEXT                  # yes, no. Reserved, see docs/decisions.md D-019
_NEXT += 2

VOCAB_SIZE = _NEXT

L_MAX = {
    # family, presentation variant -> padded query length
    "A": 2 + render.MAX_DEPTH,  # FAM_A, QUERY, then up to 6 operator tokens
    # FAM_B, QUERY, the asked attribute, two descriptor tokens naming the
    # anchor, then one relation token per hop. A depth d chain has d-1
    # hops, so the deepest chain needs 5. Written out rather than left as
    # a literal because it was a literal 8, which silently capped family B
    # at depth 4 and is part of why it had no loop-count curve.
    "B": 3 + 2 + (render.MAX_DEPTH - 1),
    "C": 6,
}


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One generated example. Matches IMPLEMENTATION.md Section 4.1.

    ``program`` is the canonical string form of the ground truth program.
    It is what makes M2 counterfactuals cheap: a corrupted twin is produced
    by editing exactly one program element and re-rendering with the same
    ``layout_seed``, so the two images differ in one place and nowhere else.
    """

    image: np.ndarray  # uint8, (3, H, W)
    query: np.ndarray  # int32, (L,)
    label: int
    depth: int  # sequential composition depth, the H1 x-axis
    breadth: int  # parallel scene breadth, the H1 null axis
    program: str
    idx: int
    layout_seed: int


@dataclass(frozen=True)
class TaskConfig:
    master_seed: int = 20260901
    presentation: str = "strip"  # strip | frames | tokens, see Section 4.2
    canvas: int = render.CANVAS

    # Narrow a split to specific depths or breadths. Gate G1 needs data at
    # exactly one depth ("depth 3 must be unsolvable at k=1"), which the
    # split ranges alone cannot express since depth is drawn per sample.
    depths: tuple[int, ...] | None = None
    breadths: tuple[int, ...] | None = None

    # Restrict family A to one factor of D4 x S3. "full" is the real task
    # and must stay byte identical to before this option existed, so that
    # runs in flight are unaffected. See groups.SUBGROUPS.
    factor: str = "full"


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
#
# Splits own disjoint index ranges off a single master seed. Because the
# layout seed is derived from the global index, two samples from different
# splits can never share a (program, layout_seed) pair: the seeds differ by
# construction rather than by luck. Asserted in tests/test_data.py.

_SPLIT_STRIDE = 10_000_000


@dataclass(frozen=True)
class SplitSpec:
    name: str
    base: int
    depth: tuple[int, ...]
    breadth: tuple[int, ...]


SPLITS: dict[str, SplitSpec] = {
    "train": SplitSpec("train", 0 * _SPLIT_STRIDE, (1, 2, 3), (1, 2, 3, 4)),
    "iid_val": SplitSpec("iid_val", 1 * _SPLIT_STRIDE, (1, 2, 3), (1, 2, 3, 4)),
    "depth_ood": SplitSpec("depth_ood", 2 * _SPLIT_STRIDE, (4, 5, 6), (1, 2, 3, 4)),
    "breadth_ood": SplitSpec("breadth_ood", 3 * _SPLIT_STRIDE, (1, 2, 3), (5, 6)),
    "combo_ood": SplitSpec("combo_ood", 4 * _SPLIT_STRIDE, (1, 2, 3), (1, 2, 3, 4)),
}


def split_spec(family: str, split: str, cfg: TaskConfig | None = None) -> SplitSpec:
    """The split as this family defines it.

    The spec's ranges in Section 4.5 are written "family-appropriate", and
    they genuinely differ: family A composes up to depth 6 while family B
    chains to 4, and family C is depth 1 by construction. A family
    declaring SPLIT_RANGES overrides the defaults below.
    """
    base = SPLITS[split]
    overrides = getattr(get_family(family), "SPLIT_RANGES", {})
    if split in overrides:
        depth, breadth = overrides[split]
        base = SplitSpec(base.name, base.base, tuple(depth), tuple(breadth))
    if cfg is not None and (cfg.depths or cfg.breadths):
        depth = tuple(cfg.depths) if cfg.depths else base.depth
        breadth = tuple(cfg.breadths) if cfg.breadths else base.breadth
        unknown_d = set(depth) - set(base.depth)
        unknown_b = set(breadth) - set(base.breadth)
        if unknown_d or unknown_b:
            raise ValueError(
                f"{family}/{split}: requested depths {sorted(unknown_d)} and "
                f"breadths {sorted(unknown_b)} are outside the split's own "
                f"ranges {base.depth} and {base.breadth}. Narrowing a split is "
                f"allowed, redefining it is not."
            )
        base = SplitSpec(base.name, base.base, depth, breadth)
    return base


def split_seed(family: str, split: str, idx: int, master_seed: int) -> int:
    """A stable 63 bit seed for one sample.

    Hashed rather than arithmetic so that neighbouring indices do not give
    correlated streams, which would put structure into the layout that a
    model could exploit.
    """
    key = f"{family}|{split}|{idx}|{master_seed}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") >> 1


def sample_rng(family: str, split: str, idx: int, cfg: TaskConfig) -> np.random.Generator:
    return np.random.default_rng(split_seed(family, split, idx, cfg.master_seed))


def global_index(split: str, local_idx: int) -> int:
    """Map a per-split index into the global index space."""
    return SPLITS[split].base + local_idx


# ---------------------------------------------------------------------------
# Family registry
# ---------------------------------------------------------------------------


def get_family(family: str):
    """Import a family module lazily, to keep this module import-cycle free."""
    if family == "A":
        from loopvision.data import family_a

        return family_a
    if family == "B":
        from loopvision.data import family_b

        return family_b
    if family == "C":
        from loopvision.data import family_c

        return family_c
    raise ValueError(f"unknown family {family!r}, expected one of A, B, C")


def generate(family: str, idx: int, split: str, cfg: TaskConfig | None = None) -> Sample:
    """Generate one sample. Deterministic given its arguments."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    module = get_family(family)
    if split not in module.SUPPORTED_SPLITS:
        raise ValueError(
            f"family {family} does not define split {split!r}. "
            f"Supported: {sorted(module.SUPPORTED_SPLITS)}. "
            f"See docs/decisions.md D-018."
        )
    return module.generate(idx, split, cfg or TaskConfig())


def pad_query(tokens: list[int], family: str) -> np.ndarray:
    """Right pad to the family's L_max with PAD."""
    limit = L_MAX[family]
    if len(tokens) > limit:
        raise ValueError(f"query of length {len(tokens)} exceeds L_max {limit} for {family}")
    return np.array(tokens + [PAD] * (limit - len(tokens)), dtype=np.int32)


# ---------------------------------------------------------------------------
# torch Dataset
# ---------------------------------------------------------------------------


class LoopVisionDataset:
    """Map-style dataset over a procedurally generated split.

    Deliberately not importing torch at module scope, so the data layer can
    be tested and inspected without a torch import. ``__getitem__`` returns
    a Sample; collation into tensors happens in the training loop.
    """

    def __init__(
        self,
        family: str,
        split: str,
        length: int,
        cfg: TaskConfig | None = None,
        transform: Callable[[Sample], object] | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        self.family = family
        self.split = split
        self.length = length
        self.cfg = cfg or TaskConfig()
        self.transform = transform

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> object:
        if not 0 <= i < self.length:
            raise IndexError(i)
        sample = generate(self.family, global_index(self.split, i), self.split, self.cfg)
        return self.transform(sample) if self.transform else sample
