# Pre-registration

**Frozen at git tag `prereg-v1`. Never edited after that tag.** Changes go in
`docs/decisions.md` with a date and a reason, as IMPLEMENTATION.md Section 8
requires.

Written 2026-09-07, at milestone 6, which is later than intended. Section 0
says exactly what had already been observed when it was written, because a
document that claims to be prior to data it has already seen is worse than no
document at all.

---

## 0. Disclosure. What had been seen before this was written

This project began recording results before this file existed. Pretending
otherwise would make every claim below unfalsifiable in the way that matters,
so the state of knowledge at the moment of freezing is written down first, per
hypothesis, and each hypothesis is then labelled with what this document can
and cannot bind.

| Hypothesis | Data seen before freezing | What this file binds |
|---|---|---|
| **H1** | Fully analysed. Eight seeds of family B report depth raising `k_min` in 8 of 8 and breadth in 0 of 3. Reported in `findings.md` as "H1 SUPPORTED" | **Confirmatory only on cells not yet run**: the size ladder, family C breadth cells beyond those already reported, and family A if it ever gains a depth axis. The existing family B result is **exploratory** and is labelled so in the paper |
| **H2** | Nothing. No halting criterion has been computed, `k*_model` has never been extracted, and M5 has not run | **Fully confirmatory.** This is a genuine pre-registration |
| **H3** | Three heatmaps seen, one per family, one seed each, on 2026-09-07. Summary statistics were invented after looking and did not fit the prediction. Recorded in `findings.md` as POST-HOC | **Confirmatory only on the confirmation set in Section 4.** The three existing grids are **exploratory** and stay labelled so, whatever this file's metrics say about them |
| **H4** | Nothing. No jitter sweep has run, including the width-zero arm | **Fully confirmatory** |

**The honest summary.** H2 and H4 are pre-registered in the ordinary sense. H1
and H3 are pre-registered only for data that does not exist yet, and their
existing results are exploratory findings that a confirmation run may or may
not reproduce. IMPLEMENTATION.md Section 8 already required this labelling and
scheduled the confirmation runs at milestone 8. This file is where the
confirmation criteria are fixed.

---

## 1. Statistical rules, binding on every hypothesis below

- **Alpha is 0.05**, two sided, on every test in the registered family.
- **Holm correction across the registered family.** The family is: H1 depth
  coefficient, H1 breadth coefficient, H2 signed error per criterion (four
  tests), H2 slope per criterion (four tests), and the two H3 contrasts in
  Section 4. Fourteen tests. Exploratory tests are corrected separately and
  labelled exploratory in the text.
- **Seeds.** Minimum 5 per cell. Minimum 10 for any cell whose number appears
  in an abstract, a headline sentence, or a figure caption.
- **Intervals** are bootstrap over seeds, 10000 resamples, percentile method.
- **Any statistic not defined in this file is post-hoc**, is labelled post-hoc
  in the paper, and gets a confirmation run at milestone 8. A confirmation run
  that fails is reported as a failure.
- **No threshold, seed count, task difficulty or test in this file is changed
  to improve a result.** If a registered number fails, it fails and is
  reported. The prior paper in this line reports fourteen failed registered
  predictions.

---

## 2. H1. Loop count tracks composition depth and is flat in scene breadth

**Metric.** For each `(family, depth, breadth, d_model, seed)` cell, train a
variable-k model, evaluate at `k = 1..32`, and record `k_min`, the smallest `k`
at which held-out accuracy reaches `tau = 0.90`. If accuracy never reaches tau
at any k in range, the cell is right-censored at 33 and reported as censored
rather than dropped.

**Test.** Fit `k_min ~ a * depth + b * breadth + c` with a per-seed random
intercept. Bootstrap CIs on `a` and `b` over seeds. Also fit the saturating
alternative `k_min ~ a * log(depth) + b * breadth + c` and report both, because
the text-case work found the linear reading breaks in this kind of place.

**H1 is supported if** the CI for `a` excludes 0 from above and the CI for `b`
contains 0, after Holm correction.

**H1 is falsified if** the CI for `b` excludes 0, or the CI for `a` contains 0.
Either alone falsifies it. The contribution is the dissociation, so a world
where both depth and breadth raise `k_min` falsifies H1 just as surely as one
where neither does.

**Why tau is 0.90 and not something else.** It is above every measured
order-blind ceiling by a clear margin: those ceilings are 0.652, 0.477, 0.375,
0.301 and 0.256 at depths 2 to 6. A model reaching 0.90 cannot be doing it with
an order-blind shortcut. The full accuracy-versus-k curve is reported alongside
`k_min` so a reader can see the threshold choice is not driving the result.

**Seeds.** 10 for family B and family C, which carry the headline. 5 elsewhere.

---

## 3. H2. Halting criteria are biased against the oracle, and the bias grows
with depth

Fully pre-registered. No part of this has been computed.

**Three quantities, per item.**

- `k*_task`, the composition depth of the generating program. Known exactly by
  construction, model independent.
- `k*_model`, the smallest loop count at which this model answers this item
  correctly. Oracle over iterations.
- `k_hat_c`, the loop index at which halting criterion `c` fires.

**Four criteria, fixed here so none can be added or dropped later.** Update
norm below epsilon; Kullback-Leibler below epsilon; predictive entropy below a
threshold, the LoopViT rule from arXiv 2602.02156; and step-size second
difference, Pappone et al. arXiv 2509.23314, which is the current best
published exit rule and is included so the comparison cannot be accused of
picking weak opponents. Epsilon and the entropy threshold are each set to the
value that minimises median absolute error against `k*_model` on the
**training** split, chosen before any evaluation split is touched, and are
reported in the paper.

**Test, per criterion.**

1. **Signed error.** One sample test that the median of `k_hat_c - k*_model` is
   nonzero. The sign is reported alongside the magnitude, because halting early
   and halting late have opposite engineering fixes.
2. **Depth-dependent bias.** Regress `k_hat_c - k*_model` on `k*_task`. Test
   that the slope is nonzero.

**H2 is supported if**, for at least one criterion, both the signed error and
the slope are significant after Holm correction.

**H2 is falsified if**, for every criterion, the signed error is
indistinguishable from zero **and** its slope against `k*_task` is
indistinguishable from zero. Both conditions must hold, for all four criteria,
for H2 to be falsified.

**Also reported, not tested.** The correlation of `k*_model` with `k*_task`.
This separates whether a criterion tracks the model from whether the model
tracks the task, and it is available because both oracles exist here. It is
descriptive and is not part of the corrected family.

---

## 4. H3. Patching separates by task family

**This is the section that needed writing, and it is the one this document can
least afford to get wrong.** Three heatmaps were seen on 2026-09-07, before
this text existed, and the summary statistics used to read them were invented
after looking. Those statistics did not fit the prediction. Everything below is
therefore a specification for a confirmation set, and the three existing grids
remain exploratory whatever the confirmation says.

**Metrics, defined here and parameter free.** For a grid of mean recovery
`R(i, p)` over iterations `i` and positions `p`, with values clipped below at
zero:

- **Loop dispersion** `D_loop` is the Shannon entropy of the normalised
  distribution `sum_p R(i, p)` over `i`, divided by `log(k + 1)`. It is 0 when
  all recovery sits at one iteration and 1 when it is uniform across
  iterations.
- **Spatial dispersion** `D_space` is the Shannon entropy of the normalised
  distribution `R(i*, p)` over `p` at the peak iteration `i*`, divided by
  `log(P)`. It is 0 when one position carries everything and 1 when recovery is
  uniform across positions.

Entropy rather than a top-`m` share, because `m` would be a free parameter
chosen by whoever writes the plotting code, and this file exists to remove that
freedom.

**Prediction.** Depth families are loop-localised and spatially distributed.
Breadth families are loop-diffuse and spatially local. In these metrics:

- `D_loop(depth families) < D_loop(breadth families)`
- `D_space(depth families) > D_space(breadth families)`

**Test.** Two contrasts, each a bootstrap over seeds of the difference in
means between the depth families, A and B pooled, and the breadth family, C.
Both enter the Holm family of Section 1.

**H3 is supported if** both contrasts are significant and in the predicted
direction.

**H3 is falsified if** either contrast is significant in the opposite
direction, or both contrasts contain zero. A split result, one contrast
supported and the other not, is reported as a split result and the lead claim
is weakened accordingly rather than being asserted on the surviving half.

**The confirmation set, fixed here.** Grids at 5 seeds per family, on the
depth cells already trained for families A and B and the breadth cells for
family C, at 256 admissible items per cell, with the full-state positive
control at every cell required to exceed 0.95 or the cell is discarded as an
instrument failure rather than reported as a null. None of these grids exist
at the time of freezing except the three exploratory ones, which are excluded
from the confirmation set.

**A mechanistic alternative, registered now so it cannot be adopted later
without notice.** Counting is a global aggregate: every sprite contributes, so
corrupting one and restoring one position may recover a little from many
places, which would make the breadth family spatially **distributed** rather
than local. Composition depends on one queried strip, which would make depth
families spatially **local**. That is the exact reverse of H3 on the spatial
axis. If the confirmation set shows the reverse, this alternative is the
reading, and it was written down before the confirmation ran rather than
constructed afterwards to explain it.

---

## 5. H4. Loop-count jitter reduces wrong-attractor landings

**Exploratory, and labelled exploratory in the paper.** It replicates a text
result, arXiv 2606.29983, in vision. It is corrected separately from the
registered family of Section 1.

**Metric.** Cluster final states across seeds. A wrong-attractor landing is an
item whose final state joins a cluster whose modal answer is not the correct
answer. Rate is the fraction of such items.

**Sweep.** Jitter width in `{0.0, 0.25, 0.5, 1.0}`. The width-zero arm is
included because the prior sweep lacked it and could not distinguish "jitter
helps" from "any width is better than the one we used".

**Reported either way.** No effect is one paragraph. An effect is one
paragraph and a figure.

---

## 6. Gates, already resolved before freezing

Recorded here so this file is not read as though they were pending.

- **G1, task validity: PASSED** on attempt 2, 2026-09-01. Attempt 1 failed
  G1.2 at 20k steps and is reported in full.
- **G2, loops beat matched-compute baselines: FAILED**, 2026-09-07. G2.1
  failed, looped 0.9966 against matched-compute feedforward 0.9998, where
  +0.02 was required. G2.2 passed against the echo baseline. The failure is
  reported, not worked around, and the paper does not claim looping wins at
  matched compute.

---

## 7. What would make the paper unpublishable

Not a falsified hypothesis. Section 4 of `paper.md` has a cell for every
outcome and none of them is empty. The case that kills it is G1 failing in a
way that cannot be fixed, meaning the tasks do not separate depth from
breadth, and G1 passed.
