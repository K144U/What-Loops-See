# What a Loop Sees: Paper Objective

**Target:** NeurIPS 2027 main track. Fallbacks: TMLR, then ICLR 2028.
**Companion documents:** `what-a-loop-sees-IMPLEMENTATION.md` (engineering spec), `progress.md`, `findings.md`, `decisions.md`, `learnings.md`.
**Status:** objective fixed 2026-08-31, before any code exists. Revisit only at milestone 9, and log any revision in `decisions.md`.

This file answers the questions the implementation spec does not: what the paper claims, who it is for, what it looks like under every outcome, and what would make it fail as a paper even if all the code succeeds.

---

## 1. The one sentence

Looped vision models spend iterations on the sequential depth of a visual computation, not on the parallel breadth of the scene, and because we generate every image from a program whose depth we know, we can measure for the first time how far the standard halting criteria land from where the computation actually finishes.

If both halves survive contact with the data, that is the paper. If only the second half survives, that is still a paper, and a more useful one to practitioners.

Note the shape of the second claim. It is not that update-norm and Kullback-Leibler halting are unreliable, which is already published for text (Pappone et al. 2509.23314, Zhang et al. 2607.20594). It is that we can say by how much and in which direction they are wrong, because we hold the answer they are trying to approximate. The novelty is the ruler, not the verdict.

## 2. What could nobody do before

This goes on page one, in plain language, before any formalism. Prof. Garg's standing rule.

Depth-recurrent vision models are being built and shipped (ELT, HIVE, RecursiveVLM, SR-ViT), and every one of them has to answer a question at inference time: how many loops is enough. Today that question is answered with a heuristic, usually an update-norm plateau or a Kullback-Leibler threshold, chosen because it is cheap rather than because anyone has shown it measures the right thing.

Nobody has been able to check those heuristics against ground truth, because on natural images nobody knows how much sequential computation the answer actually required. We can measure it here because we generate the images from programs whose composition depth is known exactly rather than estimated. That single property is what makes the rest of the paper possible, and it is the thing to lead with.

## 3. Contributions, in the order they should appear

Each contribution names the instrument that produces it and the milestone that delivers it, so nothing enters the paper without a run behind it.

1. **A pixel-native benchmark where sequential depth and parallel breadth are crossed and both are known exactly.** Three task families, procedurally generated, with an anti-shortcut sampler and a validity gate that must pass before any modelling claim is made. Delivered by milestones 1 and 2. This contribution stands whether or not any hypothesis survives, and it is the paper's floor.
2. **A mechanism account via activation patching over the (loop, patch position) grid.** Where in the loop, and where in the image, the computation actually happens, and whether that pattern differs between depth tasks and breadth tasks. Delivered by M2, milestone 6. **This is the headline result and the visual centrepiece.** The spatial axis has no analogue in the text literature, which is what makes the instrument new rather than ported.
3. **Loop-count requirement curves.** Minimum loops to threshold, as a function of depth and of breadth, across a model size ladder. Delivered by M1, milestone 5. The depth half of this is close to known in text (Kohli et al. 2604.07822, Zhang et al. 2607.20594). **The contribution is the dissociation, not the depth trend:** loops track depth and not breadth, in a setting where the two are independently controlled for the first time.
4. **An oracle comparison for halting criteria.** We measure the signed error between where each standard halting criterion fires and where the computation actually finishes, against an oracle that is known by construction rather than recovered by searching over loop counts. Delivered by M5 plus the M4 geometry logs, milestones 7 and 8. This is the contribution with the shortest path to affecting what other people build, and it is the one that depends most directly on contribution 1.
5. **A scale check at 3.5B.** Whether the small-model signatures survive grafting onto frozen Huginn-0125. Delivered by milestone 10. This is external validity, not benchmark chasing, and it should be framed that way in the text.

**Why this order.** The benchmark stays first because it is the enabling claim: the patching result is not interpretable without knowing that depth and breadth were independently controlled, so it has to be introduced before any finding. Among the findings, the mechanism leads. The August 2026 survey (`gap-analysis.md`) found the loop-count curves partly anticipated in text while the (loop, spatial position) grid is untouched by anyone, so the ordering follows defensibility rather than the order the work happens in.

**What to protect if the schedule slips.** The oracle comparison for halting criteria. Cut depth from the scale check, then ablations from the loop-count curves, before touching it. Cross-references here name contributions rather than numbers, because numbered references rot on the next reorder.

## 4. What the paper says under each outcome

Written now, before the data exists, so that no result can quietly reshape the story later. This table is the paper's insurance policy: there is no cell in it where we have nothing to publish.

| Hypothesis | If supported | If falsified |
|---|---|---|
| H1 (loops track depth, flat in breadth) | Loop count is a depth meter in vision. The pixel case reproduces the text case, and the two sets of curves go on the same axes. | Loops in vision buy parallel breadth rather than sequential depth. This contradicts the text-case reading and is arguably the more interesting paper, because it says the vision and language cases are mechanistically different rather than the same idea in two modalities. Lead with it. |
| H2 (halting criteria are biased against the oracle, and the bias grows with depth) | We can tell practitioners not just that update-norm and Kullback-Leibler exits are wrong, but by how many loops, in which direction, and which axis of task structure they are blind to. Highest practical impact in the paper. | The criteria are well calibrated against the oracle in vision. That is a genuinely useful result, because it licenses their continued use in a setting where nobody had checked, and it is short. Report it cleanly and do not pad it. |
| H3 (patching separates by task family) | Mechanism established: depth tasks are loop-localised and spatially distributed, breadth tasks are the reverse. The lead result. | The lead result is lost and the loop-count curves become the headline instead. The loop-count curves are untouched by this, since M1 does not depend on M2. Report the heatmaps anyway. A null that is well measured is publishable in this venue when the instrument is novel, and the instrument is the novel part either way. |
| H4 (jitter reduces wrong-attractor landings) | Replicates the text-case effect in vision. Cheap, practical, goes in the training-recipe paragraph. | No effect, which is reportable in one paragraph. Labelled exploratory throughout, as pre-registered. |

### H2 stated precisely

H2 carries more machinery than a table cell holds, and the pre-registration will be written from this paragraph, so it is spelled out here. Three quantities, per item:

- **k\*_task**, the composition depth of the program that generated the image. Known exactly by construction. Task-intrinsic and model-independent.
- **k\*_model**, the smallest loop count at which this model answers this item correctly. Model-dependent. This is oracle-over-iterations, and it is the only oracle the text literature can construct.
- **k_hat_c**, the loop index at which halting criterion c fires. One per criterion: update norm below epsilon, Kullback-Leibler below epsilon, predictive entropy below a threshold, and the step-size second-difference rule of Pappone et al.

Three claims follow, and they are what H2 now means:

1. **Signed error.** The distribution of `k_hat_c - k*_model` is centred away from zero. Reporting the sign matters more than reporting the magnitude, because halting early and halting late have opposite engineering fixes.
2. **Depth-dependent bias.** That error grows with `k*_task`. A criterion whose error is constant is merely miscalibrated and can be fixed with a threshold. A criterion whose error grows with the depth of the computation is measuring the wrong thing, and no threshold rescues it.
3. **Blind axis.** The error is sensitive to depth and insensitive to breadth, or the reverse. This is what ties H2 to H1: it says which axis of task structure the diagnostics can actually see.

**Falsified if** the signed error is indistinguishable from zero at the pre-registered tolerance and does not grow with `k*_task`. Claim 3 is unfalsifiable on its own and is reported descriptively.

**What is new here, stated so we do not overclaim.** That these criteria are unreliable is published for text: Pappone et al. (2509.23314) built a better exit rule on step-size second differences, and Zhang et al. (2607.20594) showed standard instruments saturate at the fixed points trained loops converge to. Popescu et al. (2607.20519) compared learned gates against post-hoc confidence readouts using oracle-over-iterations. All three are limited to `k*_model`, an oracle recovered by searching over loop counts on a model that may itself be wrong about how much computation the task needed. **We add `k*_task`.** The comparison between `k*_task` and `k*_model` is available to nobody else, and it separates two things the text literature must leave entangled: whether a criterion tracks the model, and whether the model tracks the task.

**The all-falsified case.** If H1, H2, H3 and H4 all fail, the paper is contribution 1 plus a four-part negative result, submitted with the same instrument suite and the same rigour. That is a weaker paper, not a dead one. The prior paper in this line reports fourteen failed registered predictions and the failures are the credibility, not the damage.

**The case that actually kills the paper** is not a falsified hypothesis. It is gate G1 failing in a way we cannot fix, meaning the tasks do not cleanly separate depth from breadth. Then there is no measuring instrument and nothing downstream is interpretable. That is why G1 is a hard stop in week 2 rather than a check we do later.

## 5. Paper skeleton and what feeds each section

| Section | Content | Fed by |
|---|---|---|
| Abstract | No mathematics. Motivation, the gap, what we can now measure, the two headline findings. | milestones 5, 8 |
| 1. Introduction | "What could nobody do before" (Section 2 above) on page one. Contributions list. No formalism. | this file |
| 2. Related work | Looped architectures, looped vision, mechanism and geometry, weight-tied vision precedent. Position against HIVE explicitly, since it owns the benchmark framing. | spec Section 15 |
| 3. Formal setup | First mathematics in the paper. Depth, breadth, the loop recurrence, the threshold definition. | spec Sections 4 and 5 |
| 4. Tasks and validity | The three families, the anti-shortcut sampler, and the G1 gate outcomes as a table. Publishing the gate is what makes the depth axis believable. | milestones 1, 2 |
| 5. Loop-count curves | H1. Linear and saturating fits both reported. | milestone 5 |
| 6. Mechanism | H3 patching heatmaps, plus M3 coda-lens trajectories reported as shapes rather than scalars. | milestones 6, 7 |
| 7. Geometry and halting | M4 raw versus direction-normalised, then H2. The practical payload. | milestones 7, 8 |
| 8. Scale check | Stage 2 graft. Includes the ceiling result if the graft tracks only one or two composition steps. | milestone 10 |
| 9. Limitations and failed predictions | First-class section, not an appendix. Every pre-registered prediction that failed, listed. | findings.md |
| Appendix | Ablations, baselines, full sweeps, pre-registration diff against decisions.md. | milestone 8 |

## 6. Figures

Seven figures maximum in the main text. Each one is produced by a script in `analysis/plots/` that reads a stored metrics file and computes nothing.

- **F1** Task schematic: one row per family, showing depth and breadth varying independently. Must be legible in greyscale.
- **F2** Loop-count requirement curves, k_min against depth and against breadth. The H1 figure. Both axes on the same panel, because the flat breadth line is the contribution and it only reads as flat next to the rising depth line.
- **F3** Patching heatmaps over (loop, patch position), one panel per family. **The lead figure, and the one a reviewer remembers.** If only one figure is excellent, it is this one.
- **F4** Coda-lens accuracy trajectories. Report the shape, smooth or discontinuous, do not summarise to a scalar.
- **F5** Geometry: norm growth against directional convergence on the same panel, which is the figure that makes the halting argument visible in one glance.
- **F6** Extrapolation curves with both oracles marked, `k*_task` and `k*_model`, and every halting criterion's firing point overlaid against them. The reader should see the gap without being told it is there. The practical figure.
- **F7** Stage 2 comparison, small model signature against 3.5B signature.

F3, F2 and F6 are the three that must be excellent, in that order of priority. The others can be workmanlike.

Figure numbering follows the section order in Section 5, not the contribution order in Section 3. The contributions list is ordered by defensibility, the body is ordered by exposition, and the mechanism section needs the loop-count curves already on the page to be motivated. If that turns out to bury the lead when the draft exists, reordering sections 5 and 6 is a writing-time decision for milestone 12, not one to make now.

## 7. Reviewer objections and where each is answered

Anticipated now so the answers get built rather than written after the fact.

1. *"These are toy tasks and this will not transfer."* Answered by Stage 2 (Section 8) and by stating plainly that exact depth ground truth is unavailable on natural images, which is the whole point. Do not get defensive. Make the tradeoff explicit in the limitations section.
2. *"Loops are just depth. Why not train a deeper feedforward model?"* Answered by gate G2 and the matched-compute baseline, which runs before any claim is made rather than as a rebuttal appendix.
3. *"This is weight tying, not recurrence."* Answered by the untied-same-depth baseline.
4. *"The loop-count curve is an artifact of how you sampled k during training."* Answered by the fixed-k models trained at each k.
5. *"Your state is doing nothing. The injection is doing the work."* Answered by the echo baseline, which is exactly why the spec refuses to let it be skipped.
6. *"Models under 50M parameters tell us nothing."* Answered by the size ladder trend plus Stage 2. If the trend across d in {256, 384, 512, 768} is flat, say so.
7. *"Threshold tau = 0.90 was chosen to produce this result."* Answered by pre-registering tau and publishing full accuracy-versus-k curves so the reader can pick their own threshold.
8. *"That update-norm and Kullback-Leibler halting are unreliable is already known. What is new?"* The most likely objection to H2, and it must be answered in the introduction rather than the rebuttal. Answered by the `k*_task` oracle: prior work measures halting criteria against a model-dependent oracle recovered by search, and we measure against the task's actual composition depth. Cite Pappone et al., Zhang et al. and Popescu et al. by name while making the distinction, because conceding the prior result plainly is what makes the distinction credible.

## 8. Non-goals

Naming these keeps them out of the paper and out of the schedule.

- Not a benchmark paper. We are not claiming state of the art on CLEVR, MMVP, VSR or BLINK. Those numbers appear only as evidence the Stage 2 graft functions.
- Not a new architecture. The model deliberately mirrors Huginn so the scale comparison is not confounded.
- Not a generation result. H5 and the whole generation stack are out of scope and are the identified creep vector.
- Not an efficiency paper. If looping turns out to be compute-efficient, that is a sentence, not a section.

## 9. Deadlines, working backwards

NeurIPS 2027 abstract deadline is historically early to mid May 2027, with the full paper roughly one week later.

| When | What |
|---|---|
| Sep to Dec 2026 | Milestones 0 to 11. All results exist by end of December. |
| Jan to Mar 2027 | Writing and internal review. Milestone 12. |
| Apr 2027 | Anonymous artifact, code link, DOI, AI use statement. All closed at milestone 10, verified here. |
| Early May 2027 | Abstract. |
| Mid May 2027 | Full submission. |

The hard scheduling fact: **no new experiments after milestone 9 (early December 2026)** except confirmation runs. Anything discovered during writing is post-hoc and is either labelled post-hoc in the text or left out.

## 10. Standing rules that bind the paper text

Carried from the implementation spec, repeated here because they are prose rules, not code rules.

1. Abstract and introduction contain no mathematics.
2. No em dashes or en dashes anywhere in the paper.
3. Every count and total in prose comes from a script, never typed by hand.
4. Every number in a figure is readable from a stored metrics file.
5. Post-hoc statistics are labelled post-hoc in the text, not only in the appendix.
6. Nothing is described as a sweep, a run, or a result unless it is in `runs/`.
