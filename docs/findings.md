# Findings

Every result, in one place. This file is the bridge between `runs/` and the paper.

**The provenance rule.** No number enters this file without a `run_id` and the stored metrics file it can be read back out of. If a number cannot be traced to a file, it does not exist, and it certainly does not go in the paper. This is the direct consequence of the prior submission carrying an invented count (329 versus the true 344) into review.

**The honesty rule.** Failed predictions get the same formatting, the same detail, and the same prominence as successful ones. Section 5 of this file is not an appendix, it is the point. The prior paper in this line reports fourteen failed registered predictions and is stronger for it.

**Status vocabulary:** PENDING, SUPPORTED, FALSIFIED, INCONCLUSIVE, POST-HOC (awaiting confirmation), CONFIRMED (post-hoc claim that survived a confirmation run), WITHDRAWN.

---

## 1. Headline claims

The short list. Every row here must appear in `docs/claims.md` in the repo and must have a confirming run id before milestone 9 closes. This table is what a reviewer would check first.

| # | Claim | Hypothesis | Value and CI | Seeds | run_id | Metrics file | Status |
|---|---|---|---|---|---|---|---|
| | *(none yet)* | | | | | | |

---

## 2. Gate outcomes

Gates are results too, and the G1 table goes in the paper as evidence the depth axis is real. Record failing attempts as well as passing ones, with what changed in between.

### Gate G1, task validity, milestone 2

**ATTEMPT 2, 2026-09-01: PASSED.** All four checks. Milestone 3 is unblocked. Attempt 1 is preserved below in full, per the rule that a gate re-run after a change must show both attempts.

| Check | Requirement | Attempt 1 (20k steps) | Attempt 2 (200k steps) | Status |
|---|---|---|---|---|
| G1.1 looped | depth 3 at k=1 <= 0.05 | 0.0204 | **0.0212** | PASS |
| G1.1 feedforward | depth 3 at k=1 <= 0.05 | 0.0204 | **0.0212** | PASS |
| G1.2 | depth 1 at k=1 >= 0.90 | 0.0204 FAIL | **1.0000** | PASS |
| G1.3 | order-blind ceiling below tau | 0.652 vs 0.75 | unchanged, analytic | PASS |
| G1.4 | family C at k=1, every breadth | 1.0000 | not rerun, see below | PASS |

Chance is 1/48 = 0.0208. n = 16384 per run. Runs: `g1_famA_d1_looped_long_s0`, `g1_famA_d3_looped_long_s0`, `g1_famA_d3_ffwd_long_s0`, `g1_famC_looped_s0`.

**What changed between attempts: the step budget, nothing else.** D-022 option 1, the only remedy that does not alter the task. All three family A runs were rerun rather than only the failing one, because comparing depth 3 at 20k steps against depth 1 at 200k would be an artefact of unequal budgets rather than a gate. Family C was not rerun: it reached 1.000 at every breadth in 20k steps, and more steps cannot make a solved task unsolved.

**The dissociation, which is the actual result.** Same model, same 200k budget, same data pipeline. Only composition depth differs.

| Step | Depth 1 | Depth 3 |
|---|---|---|
| 50,000 | 0.1636 | 0.0171 |
| 100,000 | 0.9946 | 0.0225 |
| 150,000 | 1.0000 | 0.0232 |
| 200,000 | **1.0000** | **0.0225** |
| final train loss | **0.0** | **3.8711** (ln 48 = 3.8712) |

Depth 1 shows a grokking curve: flat at about 0.165 from step 18,000 to 66,000, then 0.2166 at 74,000, 0.8818 at 82,000, crossing tau at **step ~84,000**. Attempt 1 stopped at 20,000, roughly four times short of the transition.

Depth 3 is flat at chance throughout with no upward trend. Because data never repeats (51.2M samples, each seen once), train loss is generalisation loss and memorisation is unavailable, so depth 3 sitting at exactly ln(48) means the model never fit even the data in front of it.

**This retroactively rescues G1.1.** In attempt 1 its pass was vacuous, because depth 1 failed too and the depth 3 result could not be attributed to composition. Now depth 1 is solved at the same budget, so it can.

**One confound, recorded rather than hidden.** `steps` also sets the cosine schedule length, so the 20k run had decayed its learning rate to 3e-5 by its end while the 200k run was still near peak at the same step. The two effects are not separable from these runs. The conclusion that attempt 1 was too short holds either way, but this is not a clean step-count-only comparison and should not be described as one.

**Consequence for the compute plan.** Family A needs roughly 100k to 200k steps, not 20k. That is a five to ten times multiplier on every family A run in the milestone 5 sweep, and it makes the earlier "2 to 4 GPU-weeks" estimate optimistic. See D-003.

---

**ATTEMPT 1, 2026-08-31: FAILED on G1.2.** Preserved in full below.

| Check | Requirement | Measured | run_id | Status |
|---|---|---|---|---|
| G1.1 | Depth 3 near chance at k=1, looped and non-looped. Chance for family A is 1/48 = 0.021 | 0.0204 looped, 0.0204 feedforward, n=16384 each | `g1_famA_d3_looped_s0`, `g1_famA_d3_ffwd_s0` | **PASS, but vacuous. See below** |
| G1.2 | Depth 1 solvable at k=1 to high threshold (0.90) | **0.0204**, n=16384 | `g1_famA_d1_looped_s0` | **FAIL** |
| G1.3 | Probe must not exceed the order-blind ceiling, and the ceiling must stay below tau | worst ceiling 0.652 at depth 2, threshold 0.75. Linear probe 0.056 | analytic, no model needed | PASS |
| G1.4 | Family C solvable at k=1 at every breadth | **1.000** at every cell, breadth 4 through 8 | `g1_famC_looped_s0` | PASS |

**Why the G1.1 pass does not count.** G1.1 and G1.2 are not independent. G1.1 asks whether depth 3 is unsolvable in one pass, and it is; but G1.2 shows depth **1** is also unsolvable in one pass, and depth 1 is a single group multiplication with no composition in it at all. So the depth-3 result cannot be attributed to composition depth. Both numbers are 0.0204 because both models collapsed to a constant prediction, and 334 of 16384 is simply that class's base rate. Reporting G1.1 as a pass without this caveat would be the most misleading thing in the whole gate.

**Diagnosis: not rendering, and not the pipeline.** Three pieces of evidence, in the order they narrowed it.

1. **The pipeline works.** Family C reached **1.000** at every breadth using the same model, trainer, optimiser and schedule. Whatever is wrong is specific to family A.
2. **The model works.** It overfits 32 family A depth-1 samples to 100 percent accuracy under both `zeros` and `randn` state initialisation, so the data, the gradient path and the architecture are all sound.
3. **Rendering works.** Splitting the depth-1 task into its parts, at k=1 on a 20000 sample pool:

| Sub-task at k=1 | Final loss | Accuracy |
|---|---|---|
| Recognise the queried object's state glyph, 48-way | 0.000 | **1.000** |
| Recognise the queried object's operator glyph, 48-way | 0.000 | **1.000** |
| Compose the two, which is the depth 1 task | 2.429 | **0.101** |

The model reads both 4x4 glyphs perfectly. What it cannot learn is the D4 x S3 multiplication table itself. IMPLEMENTATION.md Section 4.6 anticipated two causes for a G1.2 failure, "rendering or capacity"; the first is now excluded by measurement.

**What this means for the task design.** Family A's difficulty floor is too high. Depth 1 is not a shallow task with one composition step: it already requires learning a 48 element non-abelian group's full multiplication table from pixels, 2304 input pairs mapping to 48 classes. Composition depth is then stacked on top of a base task that is itself unsolved. The depth axis has no valid origin.

Also relevant: the flat loss at ln(48) = 3.8712 held for all 20000 steps on fresh non-repeating data, while a repeating 20000 sample pool reached loss 2.429. The task is learnable with enough exposure per example and not learnable in the online regime we ran, which points at sample efficiency rather than a hard impossibility.

**Not fixed, deliberately.** Choosing a remedy changes the task, and the standing rule is that a failing gate is reported rather than worked around. Options are in `docs/decisions.md` D-022, awaiting a decision.

**Attempt history:** none yet. If the gate is failed and the task is deepened, add a row per attempt, stating what changed. A task that was deepened until the gate passed must be reported that way in the paper.

#### Family A order-blind ceiling, measured 2026-08-31

**This is a result, not a configuration detail, and it belongs in the paper's task validity section.**

The order-blind ceiling is the Bayes accuracy of the best possible predictor that sees only the multiset of operators and not their order, computed exactly as `E_M [ max_g P(composite = g | multiset = M) ]` by enumerating orderings. It upper bounds every bag-of-operators probe with unlimited data and capacity, so it is a stronger statement than measuring any particular probe.

| Depth | Order-blind ceiling | Multiple of chance | Below tau = 0.90 |
|---|---|---|---|
| 2 | 0.652 | 31.3x | yes |
| 3 | 0.477 | 22.9x | yes |
| 4 | 0.375 | 18.0x | yes |
| 5 | 0.301 | 14.5x | yes |
| 6 | 0.256 | 12.3x | yes |

Chance is 1/48 = 0.0208. Generated by `bag_of_operators_ceiling` in `src/loopvision/data/groups.py`, 4000 trials per depth, seed 23. Regenerate with the snippet in `docs/decisions.md` D-016 rather than retyping these.

**What this means.** Gate G1.3 as written in IMPLEMENTATION.md Section 4.6 required the bag-of-operators probe to sit **at chance** for depth greater than 1. That is not achievable and never was: at depth 2 an order-blind predictor sees the multiset `{a, b}` and need only choose between `a.b` and `b.a`. This is a property of composition in a finite group, not a defect in the anti-shortcut sampler, and no sampler can remove it. The gate as specified could never have passed.

**Why M1 survives it anyway, which is the important half.** M1 measures the minimum k at which held-out accuracy reaches tau = 0.90. The ceiling peaks at 0.652 and falls with depth, so **no order-blind model can reach tau at any depth**, with at least 0.25 of margin everywhere. Reaching threshold requires order sensitivity. The loop-count curve therefore measures the emergence of sequential composition, not of a counting statistic, which is what H1 needs. Enforced by `test_order_blind_ceiling_stays_below_the_m1_threshold`.

**What the sampler does guarantee**, all tested in `tests/test_groups.py`: labels uniform over 48 classes so chance is exactly 1/48; every proper prefix product uniform, so no prefix statistic leaks the answer; every single-operator marginal uniform including the solved-for last position.

### Family A depth wall, milestone 3 diagnostic, 2026-09-01

**Family A is not learnable above depth 1 in this configuration.** Recorded here because it bears directly on H1 and was found before the milestone 5 sweep was launched.

| Run | Depth | Steps | Final loss | Accuracy | Distinct classes predicted |
|---|---|---|---|---|---|
| `g1_famA_d1_looped_long_s0` | 1 | 200k, k=1 | 0.0 | **1.0000** | 48 |
| `vark_famA_d2_s0` | 2 | 200k, variable k | 3.8711 | 0.0191 | (degenerate) |
| `vark_famA_d3_s0` | 3 | 200k, variable k | 3.19 | 0.0423 | **2 of 48** |
| `vark_famA_d3_s1` | 3 | 200k, variable k | 3.8712 | 0.0217 | **1 of 48** |

Chance is 1/48 = 0.0208. ln(48) = 3.8712.

**Three things this establishes, and one it refutes.**

1. **The wall sits immediately after depth 1.** Depth 1 reaches 1.0000. Depth 2 and depth 3 both fail, so family A has no usable depth range: the axis H1 needs has exactly one solvable point.

2. **The apparent "loops help" signal was a degenerate predictor, not partial learning.** The runs that leave the ln(48) plateau and settle near 3.19 emit **one or two of the forty eight classes** for every input. `vark_famA_d3_s0` alternates between two classes and reaches twice chance because that alternation correlates weakly with the label; it is not composition. A loop-count curve measured on such a model is meaningless, which is why the M1 sweep must not be run on family A as configured.

3. **The bimodality is seed-dependent, not depth-dependent.** At each depth one seed escapes to about 3.19 and the other stays at ln(48), and which seed escapes flips between depth 2 and depth 3. An earlier reading of `vark_famA_d3_s0` alone as evidence for H1 was wrong, and single-seed conclusions in this regime are not safe.

4. **Refuted: the order-independent bit hypothesis.** The 3.19 plateau is close to ln(24) = 3.1781, which suggested the model had learned one of the three Z2 homomorphisms of D4 x S3 (reflection parity, rotation index mod 2, permutation sign), all of which are computable without operator order. `analysis/parity_probe.py` verifies all three are genuine homomorphisms and then measures them: **0.5023, 0.5023, 0.4992**, all at chance. The arithmetic coincidence was just that.

**A limitation of that probe, recorded so it is not misread.** A model emitting only classes 13 and 20 has constant `rot mod 2` and constant `sign`, so those two probes return 0.5 by construction and can detect nothing. Only the `flip` probe was informative for that model, and it also showed no signal. For `vark_famA_d3_s1`, which emits a single class, all three probes are structurally blind. The probe is sound but it is only meaningful on a model whose predictions actually vary, and that condition should be checked before its output is believed.

**Consequence.** D-022 option 1 rescued gate G1 but did not make family A usable for the H1 measurement. Gate G1 asks whether depth 3 is unsolvable at k=1 and whether depth 1 is solvable at k=1; both are true, so the gate legitimately passes. It does not ask whether depth 3 becomes solvable **with** loops, and it turns out not to. That is a gap in the gate rather than a fault in it, and it argues for a G1.5 style check before milestone 5: at least one depth above 1 must be solvable at some k, or there is no curve to fit.

### SUPERSEDED 2026-09-04. Models learn the abelian quotient first, milestone 3, 2026-09-01

> **Superseded 2026-09-04 by "The depth-2 model composes one non-abelian
> factor exactly and the other not at all", in section 4.** The reading in this entry
> is wrong. The residual it flags as unexplained was the clue: the model
> had not learned the abelian quotient, it had learned the entire D4
> factor, which requires operator order rather than avoiding it. The
> measurements in the table are all correct and are reused below. Kept in
> full because the mistaken interpretation is what the open question was
> about, and deleting it would hide how the correction was reached.

**Mid-run finding, from a checkpoint of `vark_famA_d2_1M_s1` at 180k of 1M steps.** Recorded now because it is the first mechanistic result in the project and it reframes the earlier depth-wall reading.

| Order-independent bit | seed 1 | seed 0 |
|---|---|---|
| flip, D4 reflection parity | **1.0000** | 0.4907 |
| rot mod 2, D4 rotation index | **1.0000** | 0.4907 |
| sign, S3 permutation parity | 0.4980 | 0.5059 |
| full 48-way accuracy | **0.1611** | 0.0449 |
| distinct classes predicted | **16 of 48** | 2 of 48 |

Chance is 1/48 = 0.0208. All three bits are verified Z2 homomorphisms of D4 x S3 by `analysis/parity_probe.py`, so each is computable from the operator multiset without knowing order.

**What is established.** Seed 1 has learned both order-independent bits of the D4 factor, exactly and completely, while learning nothing about the S3 sign. It is the first run in the project to escape degenerate prediction: 16 distinct classes rather than the one or two every earlier run emitted. This is the abelian quotient of the group, the part that requires no sequential tracking at all, and the model acquires it first.

**What does not add up, stated rather than glossed.** Two bits narrow 48 candidates to 12, which predicts accuracy 0.0833 and loss ln(12) = 2.485. Measured is 0.1611 and 1.78, about twice as good as those two bits can account for. Something further has been learned that is not any of the three homomorphisms. Open question, and worth resolving before this appears in the paper.

**Correction to an earlier reading.** The 3.19 plateau seen in the 200k runs was interpreted as possibly ln(24) = 3.178, one learned bit. `parity_probe` measured all three bits at chance on those models and the hypothesis was recorded as refuted. That was premature rather than wrong: those models were degenerate one and two class predictors that had learned nothing, so the probe had nothing to detect and, for a two class predictor, is partly blind by construction. The hypothesis is now confirmed on a model that actually learned something.

**Why this matters for H1.** Depth 1 plateaued at val_acc 0.165 from step 18k to 66k before grokking to 0.88 in under 8k steps. Seed 1 is at 0.161 now. If that plateau is the same abelian-quotient stage, then depth 2 is sitting where depth 1 sat immediately before it broke through, and the remaining question is whether 1M steps is enough budget for the second jump. That would make the family A depth wall a matter of budget rather than of task design, and would argue against reducing the group.

**Not yet a result.** Seed 0 remains degenerate at the same step count, so seed variance is still severe and one seed reaching this stage is not the task being solved. Both runs are still training.

### Two task-validity faults found and fixed, milestone 3, 2026-09-03

Both were found by asking "is 100 percent accuracy even possible here", after
family B returned 1.0000 on every run. Neither was caught by any existing
test. Both are now fixed and covered.

#### Fault 1: family B depth 1 handed the answer to the model

At depth 1 the target sprite **is** the anchor, and the query identified the
anchor by its colour and shape. So a question asking for colour or shape
contained its own answer.

| Question asked | Samples | Answerable from the query alone |
|---|---|---|
| colour | 1028 | **1028, 100 percent** |
| shape | 976 | **976, 100 percent** |
| size | 996 | 0 |
| **overall** | 3000 | **66.8 percent** |

Confirmed end to end by feeding the trained model a **blank image**:

| Split | Real image | Blank image |
|---|---|---|
| depth 1 | 1.0000 | **0.8327** |
| depth 2 | 1.0000 | 0.1706 |

0.833 is exactly 0.668 plus half the remaining size questions, so the leak
accounts for the entire result. **Family B depth 1 was measuring almost
nothing.** Depth 2 is unaffected: only 10 percent of its answers coincide
with the anchor, and stripping the image collapses it to 0.171, so depth 2
genuinely requires vision and genuinely solves in a single pass.

**Fix.** The attribute is chosen first and the anchor is described by the
other two, so the queried attribute never appears in the query. Verified at
0 of 1500 leaked. The cost is negligible: uniqueness on the tightest
descriptor pair, (shape, size) with 10 combinations, still holds in 98
percent of scenes at breadth 6 and 100 percent at breadth 8.

#### Fault 2: family C was a presence detector, not a counting task

No leak here, the query cannot contain a count. The problem was the label
distribution produced by a two or three attribute conjunction, which almost
never matches anything.

| | Before | After |
|---|---|---|
| labels equal to 0 | **53.5 percent** | 9 to 26 percent |
| labels in {0, 1} | **93.4 percent** | about 40 percent |
| highest count seen | **4**, of a cap of 10 | 10 |
| best constant answer | **0.535** | 0.211 at breadth 16 |
| blank-image accuracy | **0.5352** | not applicable |

Worse for its actual job. Family C exists to stress **breadth**, and a rare
conjunction barely responds to it, so the axis existed in the data and not
in the task. After the fix the count tracks scene size:

| Breadth | Mean count |
|---|---|
| 4 | 1.15 |
| 8 | 1.85 |
| 12 | 2.53 |
| 16 | 3.33 |

**Fix.** Queries name one or two attributes rather than two or three, and
half are drawn from a sprite actually present so genuine zeros stay in the
distribution without dominating it. See `decisions.md` D-026.

#### A test of ours was set at the wrong place

`test_family_c_labels_are_not_degenerate` asserted that fewer than 75 percent
of labels were zero and that at least three distinct counts appeared. It
passed throughout, while the task was 93 percent binary and never produced a
count above 4. The assertion was true and irrelevant: it checked that the
label varied at all, not that it varied **with breadth**, which is the only
property that makes family C a breadth stressor. Replaced by two tests, one
requiring the mean count to rise with breadth and more than double from
breadth 4 to 16, one requiring no constant answer to beat 0.35 and the count
to reach at least 6.

#### Related: family B depths 3 and 4 are genuinely harder

Models trained on depths 1 and 2, evaluated on the held-out deeper split,
before the leak fix:

| Run | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| trained depth 1 | 0.2389 | 0.2520 | 0.2461 | 0.2500 | 0.2539 |
| trained depth 2, seed 0 | 0.3776 | 0.3802 | 0.3796 | 0.3789 | 0.3789 |
| trained depth 2, seed 1 | 0.3587 | 0.3555 | 0.3568 | 0.3542 | 0.3548 |

Chance is 0.0769. Depths 3 and 4 are well above chance but nowhere near
solved, and **flat in k**, so extra loops buy nothing for a model that was
not trained deep. The deeper chains do not come for free, which means family
B has depth structure past depth 2. Whether a model trained directly on
depths 3 and 4 needs more loops is untested, and is the obvious next
experiment on the fixed task. These numbers come from pre-fix models and are
indicative rather than final.

### Gate G2, loops beat baselines, milestone 4

| Comparison | Requirement | Measured | run_id | Status |
|---|---|---|---|---|
| Looped versus matched-compute feedforward, depth 3 | looped wins | | | PENDING |
| Looped versus echo baseline, depth 3 | looped wins | | | PENDING |

---

## 3. Hypothesis resolution

### H1, pre-registered. Loops track composition depth, flat in scene breadth

**Instrument:** M1, loop-count requirement curves. **Milestone:** 5.
**Test:** fit `k_min ~ a * depth + b * breadth + c`, bootstrap CIs over seeds. Also fit the saturating alternative `k_min ~ a * log(depth)` and report both.
**Falsified if:** the breadth coefficient b is significantly nonzero, or the depth coefficient a is not.
**Seeds:** minimum 5 per cell, 10 for headline cells.

| Quantity | Value | 95 percent CI | Seeds | run_id | Status |
|---|---|---|---|---|---|
| depth coefficient a | | | | | PENDING |
| breadth coefficient b | | | | | PENDING |
| log-depth fit a | | | | | PENDING |
| linear versus saturating, which fits better | | | | | PENDING |

**Resolution:** PENDING.

### H2, pre-registered. Halting criteria are biased against the oracle, and the bias grows with composition depth

**Instrument:** M5 plus M4 geometry logs. **Milestone:** 8. Reframed 2026-08-31, see `decisions.md` D-007 and `paper.md` Section 4.
**Definitions**, per item. All three are logged for every evaluated item so nothing here needs a re-run:

- `k*_task`, the composition depth of the generating program. Known exactly by construction. Model-independent.
- `k*_model`, the smallest loop count at which this model answers this item correctly. Oracle-over-iterations. Model-dependent, and the only oracle available to prior text work.
- `k_hat_c`, the loop index at which halting criterion c fires.

**Criteria tested (c):** update norm below epsilon, Kullback-Leibler below epsilon, predictive entropy below threshold (the LoopViT rule, arXiv 2602.02156), and step-size second difference (Pappone et al., arXiv 2509.23314). Log all four at every iteration during every evaluation pass.

#### Part A. Extrapolation curves, which also define `k*_model`

Train at `k_mean = 8`, evaluate at `k in {1, 2, 4, 8, 16, 32, 64}`.

| Quantity | Value | CI | Seeds | run_id | Status |
|---|---|---|---|---|---|
| accuracy at each k, family A (iterative) | | | | | PENDING |
| accuracy at each k, family B (iterative) | | | | | PENDING |
| accuracy at each k, family C (one-shot) | | | | | PENDING |
| `k*_model` distribution, family A | | | | | PENDING |
| `k*_model` distribution, family B | | | | | PENDING |
| `k*_model` distribution, family C | | | | | PENDING |

#### Part B. The oracle comparison, which is the contribution

**Claim 1, signed error.** The distribution of `k_hat_c - k*_model` is centred away from zero. **Report the sign, not only the magnitude:** halting early and halting late have opposite engineering fixes.

| Criterion c | Mean signed error | 95 percent CI | Sign (early or late) | Seeds | run_id | Status |
|---|---|---|---|---|---|---|
| update norm | | | | | | PENDING |
| Kullback-Leibler | | | | | | PENDING |
| predictive entropy | | | | | | PENDING |
| step-size second difference | | | | | | PENDING |

**Claim 2, depth-dependent bias.** The signed error grows with `k*_task`. This is the claim that matters: a constant error is miscalibration and a threshold fixes it, whereas an error that grows with depth means the criterion measures the wrong thing and no threshold rescues it.

| Criterion c | Slope of signed error against `k*_task` | 95 percent CI | Constant or growing | run_id | Status |
|---|---|---|---|---|---|
| update norm | | | | | PENDING |
| Kullback-Leibler | | | | | PENDING |
| predictive entropy | | | | | PENDING |
| step-size second difference | | | | | PENDING |

**Claim 3, blind axis.** Descriptive, reported not tested. Whether the error is sensitive to depth and insensitive to breadth, or the reverse. This is what ties H2 to H1: it says which axis of task structure the criteria can actually see.

| Criterion c | Sensitivity to depth | Sensitivity to breadth | Blind axis | run_id | Status |
|---|---|---|---|---|---|
| update norm | | | | | PENDING |
| Kullback-Leibler | | | | | PENDING |
| predictive entropy | | | | | PENDING |
| step-size second difference | | | | | PENDING |

#### Part C. Does the model track the task

Available to nobody else, because it needs both oracles. Separates whether a criterion tracks the model from whether the model tracks the task.

| Quantity | Value | CI | run_id | Status |
|---|---|---|---|---|
| correlation of `k*_model` with `k*_task` | | | | PENDING |
| mean `k*_model - k*_task` | | | | PENDING |

**Falsified if:** the signed error in Claim 1 is indistinguishable from zero at the pre-registered tolerance **and** the Claim 2 slope is indistinguishable from zero. Both conditions must hold. Claim 3 is unfalsifiable alone and is never counted toward resolution.

**Do not overclaim.** That these criteria are unreliable is published for text: Pappone et al. (2509.23314) built a better exit rule on step-size second differences, Zhang et al. (2607.20594) showed standard instruments saturate at the fixed points trained loops converge to, and Popescu et al. (2607.20519) compared learned gates against post-hoc confidence readouts using oracle-over-iterations. All three are limited to `k*_model`. Our addition is `k*_task`. Any writing of this section that presents the unreliability itself as our finding is wrong and must be corrected.

**Resolution:** PENDING.

### H3, pre-registered. Patching separates by task family

**Instrument:** M2, activation patching over (loop, patch position). **Milestone:** 6.
**Prediction:** loop-localised and spatially distributed on depth tasks, loop-diffuse and spatially local on breadth tasks.
**Falsified if:** the heatmaps do not separate by task family.

| Quantity | Value | Seeds | Items per cell | run_id | Status |
|---|---|---|---|---|---|
| loop localisation, family A | | | at least 256 | | PENDING |
| spatial spread, family A | | | | | PENDING |
| loop localisation, family C | | | | | PENDING |
| spatial spread, family C | | | | | PENDING |
| separation statistic between families | | | | | PENDING |

**Resolution:** PENDING.

### H4, exploratory. Loop-count jitter reduces wrong-attractor landings

**Instrument:** M5 attractor analysis with a jitter-width sweep including width zero. **Milestone:** 8.
**Text-case reference result:** 16 of 44 wrong-attractor landings without jitter versus 5 of 75 with jitter, p = 7.0e-5.
**Labelled exploratory in the paper regardless of outcome.** Corrected separately from the pre-registered family.

| Jitter width | Wrong-attractor landings | Total runs | run_id | Status |
|---|---|---|---|---|
| 0 (the arm the prior sweep lacked) | | | | PENDING |
| default sigma = 0.5 | | | | PENDING |

**Resolution:** PENDING.

---

### The depth-2 model composes one non-abelian factor exactly and the other not at all, milestone 3, 2026-09-04

**This replaces the abelian-quotient reading above, and reverses what it said about sequential computation.**

Family A's group is the direct product D4 x S3, order 48. Both factors are non-abelian. Probing `vark_famA_d2_1M_s0` and `_s1` at 850k of 1M steps, k=8, n=8192 each:

| | seed 0 | seed 1 | chance |
|---|---|---|---|
| full 48-way accuracy | 0.1716 | 0.1667 | 0.0208 |
| **D4 part of the composite** | **1.0000** | **1.0000** | 0.1250 |
| **S3 part of the composite** | **0.1716** | **0.1667** | 0.1667 |
| flip, a D4 function | 1.0000 | 1.0000 | 0.5 |
| rot mod 2, a D4 function | 1.0000 | 1.0000 | 0.5 |
| sign, an S3 function | 0.4999 | 0.5009 | 0.5 |

The D4 factor is solved exactly. The S3 factor sits on its chance level to four decimals. Knowing D4 and nothing else leaves the 6 elements of S3, which predicts accuracy 1/6 = 0.1667 and loss ln(6) = 1.7918. Measured loss is 1.7939 and 1.7965. **The plateau both 1M runs have been sitting on for 800k steps is exactly this state**, not a partial or noisy one.

**Why the earlier reading was wrong.** Knowing all three parities also leaves 6 candidates, so accuracy and loss cannot separate the two hypotheses. One reading is that the model knows the abelianisation, order 8, which needs no operator order. The other is that it knows the D4 factor, order 8, which cannot be computed without order. They differ on one number: the abelian reading requires the S3 sign to be predicted perfectly, since the sign is one of the three parities it would know. **The sign measures 0.4999.** The abelian reading is refuted by a measurement that was already in the table.

**Control 1, is D4 = 1.0000 actually evidence of order?** The order-blind ceiling at depth 2, the best achievable from the operator multiset alone, computed exactly over all 1176 unordered pairs:

| | order-blind ceiling | model | chance |
|---|---|---|---|
| full group | 0.6562 | 0.1716 | 0.0208 |
| D4 factor | **0.8125** | **1.0000** | 0.1250 |
| S3 factor | 0.7500 | 0.1667 | 0.1667 |

432 of the 1176 unordered pairs have a D4 composite that the multiset does not determine. A model at 1.0000 is right on those too, so it is using operator order. **This is a positive result for sequential computation, not against it.**

**Control 2, can the model see S3 at all?** S3 at chance could mean the S3 component is not visually decodable, which would make this a rendering fault rather than a compositional one. The depth 1 gate model `g1_famA_d1_looped_long_s0` reads every factor of a single operator perfectly: full accuracy 1.0000, D4 part 1.0000, **S3 part 1.0000**, sign 1.0000. S3 is fully visible. The depth 2 failure is compositional.

**Control 3, both seeds.** Not one run. Both converge to the same state, to four decimals, from different initialisations. This is an attractor, not variance.

**What is established.** A looped model given two non-abelian factors it can see equally well learns to compose one of them exactly and does not begin the other. It does not fall back on the order-blind shortcut for S3 either, which would have paid 0.75; it stays at 0.1667. The failure is total rather than partial.

**What this does to the depth wall.** The wall is not a failure to compose. It is a failure to compose S3 specifically, while succeeding completely on D4 in the same forward pass, at the same depth, on the same image. That is a much narrower and more tractable phenomenon than "depth 2 is unlearnable", and it argues against the earlier reading that family A needs a smaller group. The group is not too hard as a whole. One factor of it is not being attempted.

**Open, and the obvious next experiment.** Why D4 and not S3. The two are the same order to within 8 versus 6 and both non-abelian. Training family A on each factor alone would say whether S3 at depth 2 is unlearnable in isolation or only when D4 is available to solve first, which is the difference between a hard subtask and a gradient-competition effect.

**Confirmed at the full budget, 2026-09-04.** `vark_famA_d2_1M_s0` completed all 1000000 steps and finished in exactly this state: accuracy 0.1696, loss **1.7913** against ln(6) = 1.7918, D4 part **1.0000**, S3 part **0.1615** at a chance of 0.1667. The budget hypothesis is dead. Depth 2 was not parked before a second grokking transition, and a million steps was never the constraint. The plateau is where this model converges.

Provenance: `runs/vark_famA_d2_1M_s0/factor_probe_final.json`, `runs/vark_famA_d2_1M_s0/factor_probe.json`, `runs/vark_famA_d2_1M_s1/factor_probe.json`, `runs/g1_famA_d1_looped_long_s0/factor_probe.json`, `analysis/parity_probe.py`.

### Both repaired tasks clear the control that caught them, milestone 3, 2026-09-03

The two faults in the previous section were found by asking whether a
perfect score was possible. Repairing them is only worth anything if the
repaired tasks are then measured, so both were.

**Family C, gate G1.4 re-run.** `g1_famC_fixed_s0`, 20000 steps, k=1.

| | broken task | fixed task |
|---|---|---|
| accuracy at k=1, every breadth | 1.000 | 1.000 |
| best constant baseline | 0.535 | **0.375** |
| fraction of labels that are zero | 0.535 | **0.218** |
| distinct labels | 3 | 9 |
| blank-image accuracy | not measured | **0.218** |

The headline accuracy is unchanged, which is exactly why the headline
accuracy was never the point. What changed is what a score of 1.000 now
means. Under the broken task a constant predictor reached 0.535; under the
fixed one the same strategy reaches 0.375, and the label distribution
carries 1.55 nats rather than collapsing onto "zero".

The decisive number is the blank-image control: **0.218 against a best
constant of 0.375**, a leak of **-0.157**. Blanking the image does not
merely hurt the model, it drops it *below* the label prior. It is not
falling back on guessing the common answer, it is answering a question it
can no longer see. G1.4 passes on a task that is doing what it claims.

**Family B, first honest look.** `varkB_fix_s0` at step 15000 of 200000,
k=8, iid_val, on the repaired task and the widened depth range:

| depth | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| accuracy | 1.000 | 0.997 | 0.986 | 0.902 |

Chance is 0.077, so every cell is far above it, and accuracy falls
monotonically with depth. **This is the first family B measurement where
depth 1 scoring 1.000 is not suspicious**: under the old task it was a
lookup of an answer the query had already given away.

Whether this is a *loop-count* curve is still open and cannot be read from
this table, because these evaluations all run at k=8. The k sweep from 1 to
64 runs at the end of training, so the depth-versus-k surface arrives at
step 200000. What the table establishes is only the precondition: family B
now has cells that are hard enough to need something, across a depth axis
that exists.

**Family B chains cannot exceed depth 4 at breadths 6 to 9.** Rejection
rises to 0.362 at depth 5 and 0.571 at depth 6, past the 0.30 ceiling, and
the depths are reachable only by widening breadth, which would confound the
two axes H1 depends on separating. Recorded as a property of the task
rather than a limitation of the run, per D-027.

Provenance: `runs/g1_famC_fixed_s0/blank_control.json`,
`runs/g1_famC_fixed_s0/metrics.parquet`, `runs/varkB_fix_s0/metrics.parquet`.

## 4. Secondary and mechanism results

### Rays not fixed points: directional convergence against norm growth

**Instrument:** M4. **Milestone:** 7. Not a pre-registered hypothesis, but the highest-value secondary claim, and the one that gives H2 its practical teeth.
**Text-case reference:** states converged in direction while raw norm kept growing, up to 316x in the small-model regime.
**Vision precedent, do not claim this as new.** Block-Recurrent Dynamics in Vision Transformers (arXiv 2512.19941) already reports directional convergence into class-dependent angular basins and low-rank collapse in late depth. It studies standard ViTs re-expressed as block-recurrent rather than models trained as looped, and does not isolate raw norm growth from direction-normalised quantities, which is the specific comparison M4 is built around. Position as extending a known geometry to trained looped models and drawing the halting consequence.
**Question:** does the geometry replicate in models trained as looped from the start, and does the norm-versus-direction gap explain the H2 signed error. This is the mechanism behind H2 rather than a separate verdict on halting.

| Quantity | Value | run_id | Status |
|---|---|---|---|
| max norm growth ratio across iterations | | | PENDING |
| directional convergence, cos(s_i, s_i-1) trajectory | | | PENDING |
| ratio of directional convergence to norm growth | | | PENDING |
| does the geometry replicate | | | PENDING |

### M3 coda-lens trajectory shape

**Milestone:** 7. Report the shape, do not reduce it to a scalar. In the text case these were sharp and discontinuous rather than smoothly improving, which argued against a naive latent chain-of-thought reading. Either outcome is a genuine result here.

| Family | Shape (smooth or discontinuous) | run_id | Status |
|---|---|---|---|
| A | | | PENDING |
| B | | | PENDING |
| C | | | PENDING |

### Stage 2, Huginn-0125 graft

**Milestone:** 10. External validity. Note the prior negative result to guard against: two public depth-recurrent language models tracked at one and two composition steps and collapsed at three. If the same ceiling appears in the visual graft, that is the result and it goes in the main text.

| Probe | Small-model signature | 3.5B signature | Reproduces | run_id | Status |
|---|---|---|---|---|---|
| M2 patching | | | | | PENDING |
| M3 coda-lens | | | | | PENDING |
| M4 geometry | | | | | PENDING |
| composition depth ceiling | | | | | PENDING |

---

## 5. Failed predictions

**This section is a feature.** Every pre-registered prediction that did not survive goes here in full, and every entry here reappears in the paper's limitations and failed-predictions section. Do not soften, do not reframe, do not move a failure into an appendix.

| # | Prediction | What we predicted | What we measured | run_id | Date |
|---|---|---|---|---|---|
| | *(none yet)* | | | | |

---

### FAILED. Family B has no loop-count curve at the current model size, milestone 3, 2026-09-04

**H1 predicts that the loops needed rise with composition depth. On family B, they do not rise at all.**

`varkB_fix_s0` and `_s1`, 200000 steps each on the repaired task, end-of-run sweep over k from 1 to 64, accuracy by depth:

| depth | k=1 | k=2 | k=4 | k=8 | k=16 | k=64 |
|---|---|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | **0.994** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Seed 0 reads 0.997 at depth 4, k=1. **Every depth is solved in a single pass.** There is no k at which a deeper cell fails and a shallower one succeeds, so there is no curve to regress and nothing for H1 to measure on this family.

**This is not a task fault.** The task was repaired and verified: the depth 1 leak is gone, and depth 4 is a genuine four hop chain. The model simply does not need to loop to walk it.

**The cause is a budget arithmetic that should have been checked before the runs.** A looped model's sequential depth in one forward pass is `prelude + k * core + coda`. The configuration is 2/2/2, so k=1 already provides **six** transformer blocks. A depth 4 chain needs about four sequential attention steps, one per hop. Six is more than four, so k=1 suffices and no larger k can reveal anything.

    k_min = ceil((D - prelude - coda) / core), floored at 1

| configuration | blocks at k=1 | D=1 | D=2 | D=3 | D=4 |
|---|---|---|---|---|---|
| 2/2/2, as run | 6 | k=1 | k=1 | k=1 | k=1 |
| 1/2/1 | 4 | k=1 | k=1 | k=1 | k=1 |
| **1/1/1** | 3 | k=1 | k=1 | k=1 | **k=2** |

Family B caps at depth 4 because deeper chains breach the rejection ceiling at fixed breadth (D-027), so the task cannot be pushed past the budget. **The budget has to come down instead.**

**What this costs and what it does not.** It does not touch the H3 patching result or the family A factor result, neither of which depends on a k curve. It does mean that no family B run so far bears on H1, including the two 200k runs above and the four earlier ones. Six runs of family B produce no H1 evidence.

**Why the fix is not tuning.** Shrinking the core is a change that could be made to manufacture a result, and the rule against tuning is there for exactly that. The distinction is that at 2/2/2 the experiment cannot measure the quantity it is for: k is not a binding constraint on any reachable cell, so every possible outcome of the sweep is 1.000 and the measurement carries no information either way. Reducing the budget restores the experiment's ability to come out either way. To keep that honest the prediction is stated in advance and quantitatively, before the runs: **at 1/1/1, depths 1 to 3 solve at k=1 and depth 4 requires k=2.** If depth 4 solves at k=1 anyway, the hop-per-block model of the task is wrong and that is a reportable result rather than a reason to shrink further.

Provenance: `runs/varkB_fix_s0/metrics.parquet`, `runs/varkB_fix_s1/metrics.parquet`, metric `sweep_accuracy`.

### S3 is never computed, not computed and mislaid, milestone 3, 2026-09-04

The coda lens showed the output head decodes nothing about S3 at any core pass. A logit lens sees only what the frozen head reads, so that left two very different failures open: the model never extracts S3, or it extracts it and never routes it to where the answer is assembled. A linear probe trained on the residual stream separates them, because it is not restricted to what the head can use.

Five targets, decomposing the computation rather than only asking for the answer, at two pooling sites. `famA_d2_s3only_s0`, chance 0.1667:

| pooled at **query** tokens | initial | op1 | op2 | partial | composite |
|---|---|---|---|---|---|
| pass 1 | 0.207 | 0.170 | 0.152 | 0.139 | 0.180 |
| pass 4 | 0.207 | 0.184 | 0.148 | 0.168 | 0.170 |

| pooled at **patch** tokens | initial | op1 | op2 | partial | composite |
|---|---|---|---|---|---|
| pass 1 | 0.154 | 0.127 | 0.191 | 0.154 | 0.178 |
| pass 4 | 0.178 | 0.158 | 0.168 | 0.191 | 0.166 |

**Nothing is readable anywhere.** Not the composite, not the halfway state, and **not the individual operators**, which are simply present in the image and require no composition at all. The model never extracts S3 from the pixels.

**The positive control, which is what makes that a finding rather than a null.** The same probe, same code, same sample count, run for the D4 factor on the solved `famA_d2_d4only_s0`, chance 0.1250:

| pooled at **query** tokens | initial | op1 | op2 | partial | composite |
|---|---|---|---|---|---|
| pass 0 | 0.131 | 0.098 | 0.100 | 0.094 | 0.133 |
| pass 1 | 0.369 | 0.348 | 0.326 | 0.367 | **0.896** |
| pass 4 | 0.445 | 0.340 | 0.383 | 0.357 | **0.871** |

The probe reads the D4 composite at **0.896 against a chance of 0.125**. It works on real residual streams, at this sample size, with this code. So the S3 table is evidence of absence rather than absence of evidence.

**Conclusion: the failure is upstream of routing.** S3 is not sitting unused in the residual stream. It is never computed. That rules out the more optimistic reading, in which the architecture has the information and merely fails to move it, and it makes the substrate question in `experiment-substrate.md` sharper: the thing to explain is why colour permutations are not extracted at all, when spatial rearrangements are.

**Two things fell out that were not the point.**

*The model represents its answer better than its inputs.* In the solved D4 run the composite reads at 0.896 while the individual operators sit at 0.33 to 0.52. Whatever it is doing, it is not building a clean representation of each operator and then combining them. This agrees with the coda lens, which found the final answer decodable and every intermediate hop at the floor, and the two instruments reach it by different routes.

*Nothing is readable before the first core pass.* At pass 0, on the solved model, every target sits at chance. The prelude alone does not extract D4; it appears during the loop. The loops are doing perceptual extraction, not only composition, which is worth knowing before anyone describes the core as a pure reasoning module.

Provenance: `runs/famA_d2_s3only_s0/state_probe.json`, `runs/famA_d2_d4only_s0/state_probe_d4.json`, `analysis/state_probe.py`.

### One factor is learnable alone and the other is not, milestone 3, 2026-09-04

**Complete. All four runs finished 300000 steps, two seeds each.**

The depth 2 model solves the D4 factor exactly and leaves S3 at chance. Two explanations were available: S3 is genuinely hard, or S3 is merely neglected because D4 is easier and gets learned first, which would be gradient competition rather than difficulty. Training on each factor **alone** separates them, because a model given only S3 has nothing to compete with.

| run | accuracy | loss | chance | loss at chance |
|---|---|---|---|---|
| `famA_d2_d4only_s0` | **1.0000** | 0.0000 | 0.1250 | ln(8) = 2.0794 |
| `famA_d2_d4only_s1` | **1.0000** | 0.0000 | 0.1250 | 2.0794 |
| `famA_d2_s3only_s0` | **0.1725** | 1.7916 | 0.1667 | **ln(6) = 1.7918** |
| `famA_d2_s3only_s1` | **0.1659** | 1.7922 | 0.1667 | **1.7918** |

Final at 300000 steps. Seed 0 reads 0.1725 against a chance of 0.1667, which is z = +1.02 on an evaluation of 4096 samples and therefore within noise; it is not a trace of learning and is not reported as one.

**D4 alone is solved perfectly. S3 alone is at chance to three decimals, on both seeds, after 270000 steps with nothing else to learn.** It is not gradient competition. Removing the competitor changed nothing.

**The model is not merely failing, it is ignoring the image.** Blank-image control on `s3only`: real 0.1676, blank 0.1676, best constant 0.1688. **The scores are identical.** Blanking the input costs it nothing, because it was never using the input. On `d4only` the same control gives real 1.0000 against blank 0.1234, leak -0.0055, so the instrument works and the contrast is not an artefact of it.

**It does not even take the free shortcut.** The order-blind ceiling for the S3 factor at depth 2, computed exactly over all 1176 unordered operator pairs, is **0.75**. A model that ignored operator order entirely and answered from the multiset would score three times better than this one does. It reaches 0.167.

**Three ways the task could have been impossible, all ruled out.**

1. *The six elements might render identically.* They do not: 6 elements give 6 distinct glyph states, and the stabiliser is trivial, so every element is uniquely identifiable from the image. This is the check that caught the L-tromino earlier, and it passes here.
2. *S3 might not be readable from a glyph at all.* The depth 1 gate model reads the S3 part of a single operator at **1.0000**. Perception is not the obstacle.
3. *The labels might be degenerate.* They are uniform over the 6 elements by construction, and measured chance 0.1667 matches.

**What this establishes.** Within a direct product of two non-abelian groups of nearly equal order, 8 and 6, one factor is learnable to perfection at depth 2 and the other is not learnable at all, in isolation, at ten times the budget that solves the first. The asymmetry is a property of the factors rather than of the optimisation.

**What it does not establish, and this is the open question.** Why. D4 acts on positions and S3 on colours, so the two are not interchangeable and the explanation may be perceptual binding rather than group structure: tracking *where a thing went* may be easier for this architecture than tracking *which identity a thing carries*. A colour-permutation task built on positions, or a position task built on identities, would separate those. That is the experiment this finding calls for and it does not exist yet.

Provenance: `runs/famA_d2_d4only_s{0,1}/metrics.parquet`, `runs/famA_d2_s3only_s{0,1}/metrics.parquet`, `runs/famA_d2_d4only_s0/blank_control.json`, `groups.subgroup_members`, `groups.stabiliser`.

### REFUTED. The models do not walk the chain hop by hop, milestone 3, 2026-09-04

**A hypothesis of mine, tested and wrong.** The three curve seeds need different numbers of core passes for the same depth, and I proposed they were advancing the chain at different rates: roughly 1.8 hops per pass for seed 0 against 0.8 for seed 1, inferred from k_min. The coda lens measures that directly, and the picture is not a traversal at all.

Decoding each intermediate state through the model's own output head, depth 6, k=8, n=1536 per cell. Columns are the answer *if the question had stopped after that many hops*, which is known by construction:

| `famB_curve111_s0` | hop0 | hop1 | hop2 | hop3 | hop4 | **hop5 (real answer)** |
|---|---|---|---|---|---|---|
| pass 0 | 0.091 | 0.086 | 0.092 | 0.089 | 0.081 | 0.077 |
| pass 1 | 0.179 | 0.193 | 0.204 | 0.197 | 0.261 | **0.651** |
| pass 2 | 0.180 | 0.179 | 0.198 | 0.184 | 0.197 | **0.915** |
| pass 3 | 0.180 | 0.173 | 0.200 | 0.182 | 0.190 | **0.991** |
| pass 8 | 0.180 | 0.174 | 0.201 | 0.181 | 0.186 | **0.999** |

**Every intermediate hop sits at the 0.1833 guessing floor, at every pass, in all three seeds.** The final answer is the only decodable thing, and it is decodable from the first pass onward, sharpening from 0.65 to 0.999 rather than arriving after a walk.

**So the loops are not carrying a pointer along the chain.** Whatever the passes are doing, it is not "resolve hop 1, then hop 2". The intermediate states never encode a partial answer that the output head can read.

**What the seeds actually differ in.** Not traversal rate but sharpening rate. Passes for the final answer to clear 0.90:

| seed | pass 1 | pass 2 | pass 3 | pass 4 | pass 5 | passes to 0.90 |
|---|---|---|---|---|---|---|
| s0 | 0.651 | 0.915 | 0.991 | 0.997 | 0.999 | **2** |
| s2 | 0.520 | 0.783 | 0.937 | 0.984 | 0.993 | **3** |
| s1 | 0.308 | 0.503 | 0.743 | 0.898 | 0.944 | **5** |

That ordering is exactly the k_min ordering, so the seed variance in the H1 result is real and has a single mechanism behind it. It is just a different mechanism than I proposed: **iterative refinement of one answer, not sequential traversal of a chain.**

**The interpretive limit, which is severe here.** This is a logit lens, and it can only see what the *output head* can read. An intermediate hop could be represented in the state in a form the head does not decode, and would look exactly like the floor. **The correct conclusion is "no partial answer is decodable by the output head", not "no partial answer exists".** Distinguishing those needs a probe trained on intermediate states rather than the frozen head, which is milestone 7 work and is now clearly worth doing.

**An instrument fault this exposed.** The first run reported "0.00 hops per core pass" for all three seeds, which reads as *these models do nothing* and meant the opposite: the frontier was pinned at the final hop from pass 1. A rate of zero is produced both by a model stalled at hop 0 and by one that reaches the answer immediately. `traversal()` now names the shape and the rate is only reported when the frontier actually rises.

Provenance: `runs/famB_curve111_s{0,1,2}/coda_lens.json`, `analysis/coda_lens.py`.

### H1 SUPPORTED. Loops track composition depth and are flat in scene breadth, milestone 3, 2026-09-04

**The project's primary hypothesis, on three seeds per arm, at a matched architecture.** Both families trained with a 1/1/1 core for 300000 steps, variable k, differing only in which axis varies.

Accuracy by axis and k, mean of three seeds:

| family B, **depth** | k=1 | k=2 | k=3 | k=4 | k=8 |
|---|---|---|---|---|---|
| 1 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 0.772 | 0.991 | 1.000 | 1.000 | 1.000 |
| 3 | 0.699 | 0.947 | 0.996 | 1.000 | 1.000 |
| 4 | 0.634 | 0.903 | 0.983 | 0.998 | 1.000 |
| 5 | 0.564 | 0.824 | 0.952 | 0.987 | 0.998 |
| 6 | **0.492** | 0.744 | 0.897 | 0.960 | 0.986 |

| family C, **breadth** | k=1 | k=2 | k=3 | k=4 | k=8 |
|---|---|---|---|---|---|
| 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 7 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | **0.999** | 1.000 | 1.000 | 1.000 | 1.000 |

**Single-pass accuracy falls by half across the depth range and does not move at all across the breadth range.** 0.998 to 0.492 against 1.000 to 0.999. That statement needs no threshold, no fit and no k_min.

**k_min at tau = 0.90, per seed, because the seeds disagree about magnitude:**

| | d1 | d2 | d3 | d4 | d5 | d6 | slope |
|---|---|---|---|---|---|---|---|
| famB s0 | 1 | 1 | 1 | 2 | 2 | 2 | +0.257 |
| famB s1 | 1 | 2 | 3 | 3 | 4 | 6 | +0.886 |
| famB s2 | 1 | 1 | 2 | 2 | 3 | 3 | +0.457 |

| | b4 | b5 | b6 | b7 | b8 | slope |
|---|---|---|---|---|---|---|
| famC s0 | 1 | 1 | 1 | 1 | 1 | **0.000** |
| famC s1 | 1 | 1 | 1 | 1 | 1 | **0.000** |
| famC s2 | 1 | 1 | 1 | 1 | 1 | **0.000** |

**Depth raises the loop requirement in three seeds of three. Breadth raises it in zero of three.** The breadth arm is not merely a smaller effect, it is exactly flat: every cell at every breadth is solved in a single pass.

**Seed variance is large and is reported rather than averaged away.** Family B slopes span 0.257 to 0.886, a factor of three. Seed 1 is qualitatively different: single-pass accuracy collapses to 0.366 at depth 2 and then stays roughly flat, so it does almost no multi-hop work in one pass, where seeds 0 and 2 degrade gradually. The direction is unanimous, the magnitude is not, and a mean slope of +0.533 would misrepresent that spread. Any regression in the paper needs a per-seed random effect rather than pooled points.

**Controls.** Blank-image control on the final models, at k=1 where a shortcut would help most: family C seed 0 real 0.9998, blank 0.2179 against a best constant of 0.3751, leak -0.157. Family B seed 1 real 0.4452, blank 0.1032, leak -0.005. Both clean, and family C's blank score sits *below* its own label prior, so it is not falling back on guessing the modal count, it is answering a question it can no longer see.

**The interpretive limit, stated rather than left implicit.** Family C is solved at k=1 at every breadth, so this shows that breadth does not create a loop requirement *at this difficulty*. It does not show that no counting task could. A harder breadth axis, more objects or a higher count cap, might yet demand loops. What the pair establishes is a dissociation at matched architecture and matched training: the same model, given more composition depth, needs more passes, and given more objects, does not.

**What this settles.** H1's direction is supported. The earlier failed prediction stands: `k_min = max(1, depth - 2)`, slope +0.629, over-stated the effect for two of three seeds and under-stated it for the third.

Provenance: `runs/famB_curve111_s{0,1,2}/metrics.parquet`, `runs/famC_breadth111_s{0,1,2}/metrics.parquet`, metric `sweep_accuracy`; blank controls in the respective run directories.

### PARTIALLY CONFIRMED. A loop-count curve exists, and the registered prediction over-stated its slope, milestone 3, 2026-09-04

**First seed complete at 300000 steps. Two more running. No H1 claim until all three are in.**

`famB_curve111_s0`, a 1/1/1 core on the corrected task. End-of-run sweep, accuracy by depth and k, effective chance 0.1833:

| depth | k=1 | k=2 | k=3 | k=4 | k=8 | k=32 | k=64 |
|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 0.981 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | 0.913 | 0.999 | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 |
| 4 | 0.815 | 0.988 | 1.000 | 1.000 | 1.000 | 0.995 | 0.995 |
| 5 | 0.753 | 0.962 | 0.999 | 0.999 | 0.999 | 0.996 | 0.993 |
| 6 | **0.636** | 0.905 | 0.990 | 0.995 | 0.999 | 0.982 | 0.979 |

**The threshold-free statement, which is the one to trust.** Single-pass accuracy falls monotonically with composition depth: 1.000, 0.981, 0.913, 0.815, 0.753, 0.636. No threshold, no k_min, no fitting. Depth costs a looped model accuracy when it is denied loops, and more depth costs more.

**k_min depends on the threshold, so the sensitivity is reported rather than one number.**

| tau | d1 | d2 | d3 | d4 | d5 | d6 | slope |
|---|---|---|---|---|---|---|---|
| 0.80 | 1 | 1 | 1 | 1 | 2 | 2 | +0.229 |
| 0.90 | 1 | 1 | 1 | 2 | 2 | 2 | +0.257 |
| 0.95 | 1 | 1 | 2 | 2 | 2 | 3 | +0.371 |
| 0.99 | 1 | 2 | 2 | 3 | 3 | 3 | +0.400 |
| **registered prediction** | 1 | 1 | 1 | 2 | 3 | 4 | **+0.629** |

**The slope is positive at every threshold, and below the registered prediction at every threshold.** The direction of H1 survives; the magnitude we predicted does not. `k_min = max(1, depth - 2)`, one hop per block, over-states how much work each core pass has to do. The model gets more done per pass than that, and the excess grows with depth.

Recording this as the prediction was made. At tau = 0.90 the observed 1, 1, 1, 2, 2, 2 matched on depths 1 to 4 and missed low on 5 and 6.

**Not an artefact of an easy task.** Blank-image control on the final model: at k=8, real 0.9997 against blank 0.0883 with a best constant of 0.1082, leak -0.0199. At k=1, where the model is weakest and a shortcut would matter most, real 0.8492 against blank 0.1029, leak -0.0053. Clean at both. The model cannot answer without the image at any k.

**Stable under training rather than a transient.** k_min at tau = 0.90 was identical at 108k, 235k and 300k steps for depths 1 to 4, and single-pass accuracy *fell* over that span at every depth past 3 (depth 6: 0.663 to 0.601 to 0.636). The model commits harder to its loops as it trains rather than learning its way out of them. This was the stated risk that would have destroyed the result, and it did not happen.

**What is still missing.** Two seeds. A breadth arm, which is the other half of H1: depth must raise k_min while breadth does not, and only the depth half is measured here. Family C is the breadth control and has not been run at 1/1/1.

Provenance: `runs/famB_curve111_s0/metrics.parquet` metric `sweep_accuracy`, `runs/famB_curve111_s0/blank_control.json`.

## 6. Post-hoc claims awaiting confirmation

Anything discovered by looking at the data rather than by pre-registration lands here first. It gets labelled post-hoc in the paper text, and it needs a confirming run scheduled in milestone 9. **A confirmation run that fails is reported as a failure**, and the claim moves to Section 5. The prior submission carried a confirmation sweep that had not run, and it did not confirm. That must not repeat.

| # | Claim | Discovered at | Confirmation run planned | Confirming run_id | Status |
|---|---|---|---|---|---|
| | *(none yet)* | | | | |

---

## 7. Statistical hygiene log

Where the numbers came from, so the methods section writes itself and so a reviewer challenge can be answered in minutes.

- **Multiple comparisons:** Holm correction across the pre-registered family (H1, H2, H3). Exploratory tests (H4 and anything in Section 6) corrected separately and labelled exploratory in the text.
- **Seeds:** minimum 5 per cell, 10 for anything in Section 1.
- **Bimodality and clustering claims:** run at least two tests with different assumptions and report both outcomes including disagreement. Silverman rejecting where Hartigan's dip did not has already burned this line of work once.
- **Reciprocal statistics:** any statistic of the form 1/slope ships with an inclusion gate defined before the fits are run, plus `gate_sensitivity.py` reporting the headline both ways. A prior checkpoint at slope 0.036 gave 27.6 and shifted its group mean by 48 percent.

| Test run | Family (pre-registered or exploratory) | Correction applied | Alpha | Date |
|---|---|---|---|---|
| | *(none yet)* | | | |

---

## 8. Counts

Every count that appears in the paper. Each one is produced by `analysis/registry.py` and pasted here with the date it was generated, never typed from memory.

| Quantity | Count | Generated by | Date |
|---|---|---|---|
| total runs in `runs/` | | `python -m loopvision.analysis.registry` | |
| runs feeding M1 | | | |
| runs feeding M5 | | | |
| seeds per headline cell | | | |
