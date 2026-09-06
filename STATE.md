# What a Loop Sees: state as of 4 September 2026

**Read this first after clearing context.** Everything here is derivable
from the other documents; this says where to look, what is running, and
what to do next. Written so a session starting cold can continue without
the conversation that produced it.

Local and cluster are at the same commit. `pytest -q` passes. The
scheduled task `loopvision-run-watch` is armed and reports every run as it
starts and finishes.

---

## 1. The three results the paper rests on

**H1 is supported, two milestones early.** Depth raises the loops needed in
eight seeds of eight; breadth raises it in zero of three. Single-pass
accuracy falls 0.998 to 0.492 across depths 1 to 6 and moves 1.000 to 0.999
across breadths 4 to 8. Seed variance in magnitude is large, slopes 0.229 to
0.886, and is reported rather than averaged away: the paper needs a per-seed
random effect. See `findings.md`, "H1 SUPPORTED".

**Substrate, not group.** D4 solves composition at 1.0000 acting on
position and sits on the chance floor acting on colour, at the full 300000
step budget, with every algebraic property held constant including an
order-blind ceiling of 0.8125 in both. The registered abelian-foothold
alternative predicted the opposite and was refuted. See `findings.md`,
"SUBSTRATE, NOT GROUP".

**The failure is a bootstrapping deadlock, not difficulty.** A depth 2 S3
task that never left chance in 300000 steps is solved in 10000 from a depth
1 donor of itself, while a wrong-donor control stays at chance for the full
budget. A trained probe shows the cold-start model never extracts its own
inputs and the curriculum model does, with a positive control reading D4 at
0.896 on the same code. See `findings.md`, "CORRECTION" and "The deadlock
closes".

A fourth thing is convergent rather than headline: **three instruments in
two task families all say these models compute answers rather than
trajectories.** The coda lens finds every intermediate hop at the floor
while the answer sharpens, the state probe finds the halfway composite the
least readable thing in a solved model, and both replicate Lu et al.
arXiv 2507.02199 in a different modality.

---

## 2. Where the paper stands

`paper.md` section 3a holds the pivot. The short version: the coda lens
result is a **replication** of Lu et al. and must cite them, and what is
ours is the ground truth they lacked plus a protocol that resolves the
ambiguity they leave open. The binding claim is written in as the intended
headline and explicitly **not** as a result until the crossed design
finishes.

`gap-analysis.md` section 10 has the three papers that moved the novelty,
all verified from their arXiv abstract pages.

---

## 3. Running now

| job | experiment | what it decides |
|---|---|---|
| 5178 | d4_colour seed 1 | second seed of result 2 |
| 5180, 5181 | s3_spatial | closes the 2x2. Substrate reading predicts it solves cold |
| 5188 | d4_colour depth 1 donor | feeds 5189 |
| 5189 (held) | d4_colour curriculum and control | does the rescue generalise across groups |
| 5190 | scale control, wide, deep, big | the reviewer objection: was the model too small |
| 5191 | depth ladder, d4only depths 3 to 6 | could give family A a usable depth axis, which H1 lacks |
| 5202 | gate G2 baselines, feedforward and echo | the hard-stop gate |

---

## 4. What to do next, ranked

1. **Read gate G2 when 5202 lands.** `python -m loopvision.analysis.gate_g2`.
   It can fail: Gao et al. arXiv 2607.16051 report parameter scaling
   usually beating looping at matched compute. A failure is reported, not
   worked around.
2. **The feedforward control**, not yet built. Does a non-looped model of
   matched depth deadlock on colour too? If yes the finding is about
   compositional learning in general rather than about looping, which is a
   different paper. Currently assumed untested.
3. **Read the depth ladder.** If d4only solves depths 3 to 6, family A
   gains a depth axis and H1 gets a second independent family.

---

## 5. Things that will bite a fresh session

**Every instrument is mutation checked, and the first version of a test has
passed against deliberately broken code three times.** A mocked ssh that
ignored the command it was handed, a substrate test comparing images that
differ anyway, and a hop-rate metric returning the same number for opposite
mechanisms. See L-011, L-015, L-016, L-017. Do not trust a new test until
it has failed on purpose.

**A perfect score gets three controls**, not one: the blank image control,
the input ablation in `analysis/ablate_input.py`, and an analytic
order-blind ceiling. The solved D4 model needs its state and both operators
and correctly ignores the distractors, and its 1.0000 sits above a computed
ceiling of 0.8125. That is why the score is believed.

**Two gates have been moved and both are recorded.** G1.3 demanded an
unachievable chance-level score (D-016) and G2 was undefined where it was
written (D-030). Both were moved because they returned no information
wherever they pointed, not because they failed. That distinction is the one
to preserve.

**Superseded entries are kept in place with pointers**, never deleted: the
abelian quotient reading, the family A depth wall as first stated, the
single-seed curve result, and the "S3 is genuinely hard" conclusion. The
wrong version is how the right one was reached.

---

## 6. Operational

The machine running the VPN has had repeated power cuts. Cluster jobs are
unaffected, the scheduled watcher survives them, in-session monitors do
not. If a long silence happens, check rather than assume.

The GPU node runs at or near zero free cores most of the time, so work is
queue bound rather than compute bound. Shrinking a core request to fit a
gap has started jobs hours earlier on four occasions. GPUs are now claimed
atomically so concurrent jobs stop landing on the same card.

---

## 7. Open decisions, unchanged

D-003 compute beyond the current block, D-004 co-authorship, D-005 whether
to post a preprint before the deadline, D-006 which family to prioritise.
D-006 now has evidence behind it: family B carries H1, family A carries the
mechanism.
