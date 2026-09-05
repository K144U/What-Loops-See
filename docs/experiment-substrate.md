# Is the D4 versus S3 asymmetry about the groups, or about what they act on?

**Status:** designed 2026-09-04, not built. Written before any run so the
predictions are on record.
**Motivates:** the finding "One factor is learnable alone and the other is
not" in `findings.md`.

## The result this exists to explain

In family A the group is the direct product D4 x S3. A depth 2 model
solves the D4 factor exactly and leaves S3 at chance, and training each
factor alone reproduces it: D4 alone reaches 1.0000, S3 alone sits on
1.7916 against ln(6) = 1.7918 after 270000 steps with nothing competing.

## The confound

D4 and S3 differ in two ways at once, and the finding cannot separate them.

1. **They are different groups.** Orders 8 and 6, different generators,
   different cycle structure.
2. **They act on different things.** D4 rearranges the glyph in space, a
   rigid motion. S3 permutes the glyph's three colours, leaving every cell
   where it was.

So "S3 is harder" may mean *S3 the group is harder*, or it may mean
*permuting identities is harder than moving things*, which would be a
claim about perceptual binding in a vision transformer and a considerably
more interesting one. Nothing run so far distinguishes them.

## The design

Cross the two factors. Group on one axis, substrate on the other.

| | acts on **position** | acts on **identity** |
|---|---|---|
| **D4** (order 8) | `d4_position`, **already run**, solved at 1.0000 | `d4_identity`, **new** |
| **S3** (order 6) | `s3_position`, **new** | `s3_identity`, **already run**, at chance |

Two new arms. The two existing runs are the other two cells, so half the
experiment is already paid for.

### What makes it buildable

`d4_identity` needs D4 to act faithfully on a set of labels. D4's action
on the four corners of a square is faithful: all eight elements give
distinct permutations of the four corners, kernel size 1, verified against
`groups.d4_apply_cell`. Four labels is enough, and the renderer has six
colours, so no new colours are needed.

`s3_position` needs three items that are distinguishable **without using
colour**, so that the thing being permuted is spatial and the thing being
tracked is not a colour. The renderer has five shapes. Three distinct
shapes at three fixed slots, all one colour, gives S3 something spatial to
permute.

### The four arms, concretely

- **`d4_position`** (built): the glyph is rotated and reflected. Position
  changes, identities fixed.
- **`s3_identity`** (built): the glyph's three cells keep their positions
  and their colours are permuted.
- **`s3_position`** (new): three distinct shapes, one colour, at three
  fixed slots. An operator permutes which shape sits in which slot.
  Positions change, identities fixed.
- **`d4_identity`** (new): four cells at fixed positions, each carrying one
  of four colours. An operator applies D4's corner action to the colour
  labels. Identities change, positions fixed.

Every arm keeps depth 2, the same architecture, the same step budget and
two seeds, so the only thing varying is the cell of the table.

## Predictions, registered before building

| if the cause is | `d4_position` | `s3_position` | `d4_identity` | `s3_identity` |
|---|---|---|---|---|
| **the substrate** | solved | **solved** | **at chance** | at chance |
| **the group** | solved | **at chance** | **solved** | at chance |
| **an interaction** | solved | at chance | at chance | at chance |

The two built cells agree with all three rows, which is exactly why they
cannot settle it. **The whole experiment turns on the two new cells, and
they make opposite predictions under the two hypotheses.**

A fourth outcome is possible and would be the most informative: if
`s3_position` is solved **and** `d4_identity` is solved, then neither the
group nor the substrate is sufficient on its own and the difficulty lives
in the pairing, which would point at the *combination* of a hard group
with a hard substrate rather than at either alone.

## What each result would mean

**Substrate wins.** Then the paper's claim is about vision: a looped
transformer composes spatial rearrangements and fails to compose identity
permutations, at matched group order and matched budget. That is a
statement about what these architectures find hard to bind, and it
generalises past our particular groups.

**Group wins.** Then the claim is algebraic, and the next question is which
property of S3 does it. Order is not the answer, since S3 is *smaller*
than D4. Candidates worth checking are the abelianisation (D4 has order 4,
S3 has order 2, so D4 offers more order-independent structure to latch on
to early) and generator cycle structure.

The abelianisation candidate is worth stating now because it has an
independent prediction: it would explain why the D4 factor was learned
first in the joint task, since more of it is reachable without tracking
order.

## Cost

Two arms, two seeds each, 300000 steps at depth 2. At the rate the current
factor runs achieve when four share a node, roughly 15 hours of wall clock
for all four runs in parallel, plus the task implementation.

## A cheaper thing to do first

Before building either arm, train a linear probe on the intermediate
states of the existing `s3only` model to ask whether the S3 element is
recoverable from the residual stream at all.

This separates two failures that look identical from the outside:

- the information is **never computed**, or
- it is computed and the output head cannot read it.

The coda lens already showed that the frozen head reads nothing, but a
logit lens can only see what the head can decode, so it cannot tell these
apart. A trained probe can. It costs one afternoon and no GPU time beyond
inference on a checkpoint we already have, and its answer changes how the
2x2 result should be read: if the information is present but unused, then
"harder to learn" is really "harder to *route*", which is a different
claim.

This is milestone 7 probe work arriving early, and it is the same
instrument the coda lens entry already called for.
