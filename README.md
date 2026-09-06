# What a Loop Sees

Mechanistic interpretability of looped, or depth-recurrent, vision models.
Work in progress, targeting NeurIPS 2027.

A looped model runs the same block over and over: `prelude`, then `core` k
times, then `coda`. Nothing about the weights says how many times to loop.
So the loop count is a rare thing in deep learning, a dial that maps onto
*amount of computation* rather than onto capacity, and it can be turned at
inference time.

**The question is what that dial is measuring.** When the input is pixels
rather than tokens, does the number of iterations a task needs track the
**sequential depth** of the computation it demands, the **parallel breadth**
of the scene it must look at, or neither?

That question has an obvious guess attached, and the obvious guess is
testable, which is the whole reason to ask it this way.

---

## What we have found so far

**1. Loop count tracks depth, and is flat in breadth.** Deeper composition
chains raise the loops needed in eight seeds out of eight. Wider scenes raise
it in zero out of three. Single-pass accuracy falls from 0.998 to 0.492 as
depth goes 1 to 6, and moves from 1.000 to 0.999 as breadth goes 4 to 8. The
per-seed spread in *magnitude* is large, slopes from 0.229 to 0.886, and is
reported rather than averaged away: the paper needs a per-seed random effect,
not a headline mean hiding a factor of four.

**2. What a model can compose depends on the substrate, not the algebra.**
Take one group, D4. Have it act on **position**: the model solves composition
at 1.0000. Have the same group act on **colour**: the model sits on the chance
floor for the entire 300,000 step budget, 0.1277 against a floor of 0.125, in
both seeds. Same group, same order, same generators, same abelianisation, same
analytically computed order-blind ceiling of 0.8125. The only thing that
differs is what the group acts on.

This refuted our own registered prediction. Before the runs, two readings were
written down: an abelian-foothold account said this cell should self-bootstrap,
a substrate account said it should deadlock. It deadlocked. The losing
prediction is still in the repository, with a pointer, not deleted.

**3. The failure is a bootstrapping deadlock, not difficulty.** A depth-2 task
that never leaves chance in 300,000 steps is solved in **10,000** when
initialised from a depth-1 donor of itself. A wrong-donor control, given the
same head start from an unrelated task, stays at chance for the full budget. A
trained probe shows why: the cold-start model never extracts its own inputs,
the curriculum model does, and a positive control reads the same quantity at
0.896 using identical code.

So the task was never hard. The model simply could not get far enough off the
ground to discover that it was learnable.

**4. Three instruments agree these models compute answers, not trajectories.**
A coda lens finds every intermediate hop pinned at the floor while the final
answer sharpens. A state probe finds the halfway composite to be the *least*
readable thing inside a solved model. This replicates
[Lu et al., arXiv 2507.02199](https://arxiv.org/abs/2507.02199) in a different
modality. We say replication, because that is what it is. What is ours is the
ground truth they lacked and a protocol that resolves an ambiguity they left
open.

---

## What is not settled

This is a live campaign, and the parts that could still overturn things are
listed before the parts that look good.

- **Gate G2 is running and can fail.** It asks whether looping beats a
  matched-compute feedforward baseline at all. [Gao et al., arXiv
  2607.16051](https://arxiv.org/abs/2607.16051) report that parameter scaling
  usually wins that comparison. If G2 fails it will be reported, not worked
  around, and it is a hard stop.
- **The feedforward control is queued and is the one that matters most.** If a
  non-looped model of matched depth deadlocks on colour in the same way, then
  finding 3 is about compositional learning in general and not about looping,
  and this becomes a different paper. Nobody had checked until recently, which
  is the sort of gap this repository is designed to make visible.
- **The fourth cell of the 2x2 is still training.** Both seeds read 1.0000
  early, which is the direction predicted in advance, but neither has finished
  and the order-blind ceiling for that group has not been computed.

---

## How this repository is meant to be read

Most of the interesting content is not code.

| file | what it is |
|---|---|
| `STATE.md` | **start here.** Where things stand, what is running, what to do next |
| `docs/paper.md` | what is claimed, and what it says under every outcome |
| `docs/findings.md` | every result, each with the run ids that produced it |
| `docs/decisions.md` | append-only. Every deviation, dependency and fallback |
| `docs/learnings.md` | mistakes, what each cost, and the rule that now prevents it |
| `docs/progress.md` | milestones, gate ledger, risks, session log |
| `docs/gap-analysis.md` | who else is in this space and where the novelty actually sits |
| `docs/IMPLEMENTATION.md` | the build spec, sections 0 to 15 |

---

## The rules this project runs on

These exist because each one was bought with a mistake. They are enforced in
tests and scripts, not in good intentions.

**A test is not evidence until it has failed on purpose.** Every instrument is
mutation checked: break the thing it measures, confirm the test goes red. On
three separate occasions the *first* version of a test passed against
deliberately broken code. A mocked ssh that ignored the command it was handed.
A substrate test that compared images which differ anyway. A hop-rate metric
that returned the same number for opposite mechanisms. Most recently, a
matched-compute test that read its target from the thing it was supposed to be
checking, so halving that thing moved both sides of the comparison at once.

**A perfect score is a bug report until proven otherwise.** Any score at or
near 1.0000 must clear three controls: a blank-image control, a
one-patch-at-a-time input ablation, and an analytically computed order-blind
ceiling. The solved model needs its state and both operators, correctly
ignores the distractors, and lands above a ceiling computed in closed form.
That is why the number is believed.

**Gates are allowed to fail.** Two gates have been moved, and both moves are
recorded with reasons, because each was pointed somewhere it returned no
information whatever the truth. Moving a gate so it can answer is not the same
as moving it so it passes, and the distinction is written down where a reviewer
can check it.

**Nothing is deleted when it turns out to be wrong.** Superseded conclusions
stay where they are with a pointer forward. The abelian-quotient reading, an
early depth wall, a single-seed curve, and a confident claim that one task was
simply hard, are all still here and all still wrong. How the wrong version was
reached is part of the record.

**No number in prose is typed by hand.** Counts come from
`analysis/registry.py`. This rule exists because a previous submission in this
line acquired an invented count and a promised sweep that had never run.

---

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -e ".[dev]"
    pytest

The cu121 index URL is not optional. The target cluster runs driver 525 with
CUDA 12.0 and cannot load a cu124 or later build.

Run `pytest` bare, not `pytest -q`. The project's `addopts` already supplies
one `-q`, and a second one suppresses the pass count, so it prints dots and no
number.

Cluster identifiers are not in this repository. The watcher script reads them
from `~/.loopvision-cluster`:

    LOOPVISION_SSH_HOST=<your ssh alias for the login node>
    LOOPVISION_CLUSTER_USER=<your username on the cluster>

---

## Provenance

The history was rewritten once, on 6 September 2026, to remove tooling
attribution trailers and to take a private address, a cluster username and two
hostnames out of the published record. Every commit SHA changed.

Run directories record the commit they were produced at, and those files hold
pre-rewrite SHAs. `docs/sha-map.txt` maps old to new, so every provenance line
in `findings.md` still resolves to real code.
