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

### Models learn the abelian quotient first, milestone 3, 2026-09-01

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
