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

| Check | Requirement | Measured | run_id | Status |
|---|---|---|---|---|
| G1.1 | Depth 3 near chance at k=1, looped and non-looped. Chance for family A is 1/48 = 0.021 | | | PENDING |
| G1.2 | Depth 1 solvable at k=1 to high threshold | | | PENDING |
| G1.3 | Bag-of-operators probe at chance for depth greater than 1 | | | PENDING |
| G1.4 | Family C solvable at k=1 at every breadth | | | PENDING |

**Attempt history:** none yet. If the gate is failed and the task is deepened, add a row per attempt, stating what changed. A task that was deepened until the gate passed must be reported that way in the paper.

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
