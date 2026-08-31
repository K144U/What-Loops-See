# Decisions

Append-only log of every choice that deviates from `what-a-loop-sees-IMPLEMENTATION.md`, every dependency added, every fallback taken, and every open question resolved.

**Append-only means append-only.** Do not edit or delete a past entry. If a decision is reversed, write a new entry that supersedes it and add a `Superseded by D-NNN` line to the old one. The history of what we thought is part of the record.

**This file is canonical.** When the repository skeleton is created at milestone 0, this file moves to `loopvision/docs/decisions.md`. It is moved, not copied. Two decision logs that drift apart is worse than none.

**When to write an entry.** Any of these, no exceptions:

- Deviating from the implementation spec in any way, however small.
- Adding a dependency. The spec requires a line here for every one.
- Taking a documented fallback (the stability time-box, the size-ladder cut, the probe-density reduction).
- Resolving one of the open questions in Section 4 below.
- Changing anything after `preregistration.md` is frozen. The pre-registration is never edited. Changes live here, with a date and a reason.
- Choosing not to do something the spec calls for.

**Entry format:**

```
### D-NNN. Short title
**Date:** YYYY-MM-DD
**Milestone:** N
**Type:** deviation | dependency | fallback | resolution | scope
**Decision:** what we are doing.
**Reason:** why. Include what we tried first if this is a fallback.
**Consequence:** what this changes downstream, including anything the paper must now say.
**Reversible:** yes or no, and by when.
```

---

## 1. Log

### D-001. Create five planning documents alongside the implementation spec
**Date:** 2026-08-31
**Milestone:** pre-0
**Type:** deviation
**Decision:** create `paper.md`, `progress.md`, `findings.md`, `decisions.md` and `learnings.md` in the project root before starting milestone 0. The spec calls only for `docs/decisions.md`.
**Reason:** the spec is a strong engineering document but has no place to record what the paper claims, how far along we are, what we found, or what we learned from mistakes. Without those, the paper-level story gets reconstructed at writing time from memory, which is exactly how the prior submission acquired an invented count and a promised sweep that had not run.
**Consequence:** five documents to keep current. `progress.md` is updated every session, `findings.md` whenever a run produces a number, `decisions.md` and `learnings.md` on the events named in their own headers. If they go stale they are worse than useless, because they will be trusted.
**Reversible:** yes, at any time.

### D-002. This file becomes `docs/decisions.md` at milestone 0
**Date:** 2026-08-31
**Milestone:** pre-0
**Type:** deviation
**Decision:** when the repo skeleton is built, move this file to `loopvision/docs/decisions.md` rather than creating a new empty one there. Same for the other four, which go to `loopvision/docs/` with `paper.md` alongside `proposal.md`.
**Reason:** the spec places `decisions.md` inside the repo. Creating a second one at milestone 0 would split the log, and the entries written before the repo existed are exactly the ones about why the project is shaped the way it is.
**Consequence:** milestone 0 includes a move step, not a create step. The move itself gets a log entry confirming it happened.
**Reversible:** yes.

### D-007. Related work survey run before milestone 0, and what it changes
**Date:** 2026-08-31
**Milestone:** pre-0
**Type:** scope
**Decision:** ran a full related work survey before writing any code. Results in `gap-analysis.md`. Twenty one directly relevant papers exist that the implementation spec does not cite. Four consequences are adopted now: (a) H2 must be reworded before the pre-registration freeze, from "convergence diagnostics do not predict extrapolation" to an oracle comparison against ground-truth required depth. (b) H4 is demoted to a cited replication of Kuo et al. (arXiv 2606.29983), which already reports the effect in text. (c) Family C is protected absolutely, since the breadth axis is the genuinely unexplored one. (d) The geometry secondary claim is repositioned against arXiv 2512.19941, which already reports directional convergence in vision transformers.
**Reason:** the spec's reference list stops well short of the current literature. The looped vision line roughly tripled between February and August 2026. Two hypotheses as written were at risk of being replications of published text results, which would have been discovered at review rather than at week zero.
**Consequence:** `paper.md` Section 4 needs H2 rewritten and the contribution ordering reconsidered, since H3 and the patching instrument are now more defensible than the loop-count curves. Related work grows to three paragraphs. The introduction must position explicitly against arXiv 2607.20594, which uses group word problems in text and is our nearest neighbour. None of this changes the milestone schedule.
**Reversible:** yes, but not after the pre-registration is frozen at milestone 5.

### D-008. Bibliography verification standard
**Date:** 2026-08-31
**Milestone:** pre-0
**Type:** deviation
**Decision:** no citation enters a draft until its arXiv abstract page has been fetched and its title, authors and date confirmed. `gap-analysis.md` Section 9 holds the current verification debt, fourteen papers from a curated list plus five from the spec's own reference list.
**Reason:** the same principle as the count rule that produced 329 versus 344. A citation carried from a list into a draft without verification is a hand-typed number by another name.
**Consequence:** a verification pass is required before the bibliography freezes, roughly nineteen abstracts.
**Reversible:** no. This is cheap and there is no reason to relax it.

### D-009. Mechanism leads the contributions, benchmark stays first
**Date:** 2026-08-31
**Milestone:** pre-0
**Type:** scope
**Decision:** reordered `paper.md` Section 3. The patching mechanism account (H3) moves ahead of the loop-count curves (H1) and becomes the headline result. The benchmark stays at position 1. New order: benchmark, mechanism, loop-count curves, oracle comparison, scale check. F3 replaces F2 as the lead figure. Executes the reconsideration flagged in D-007.
**Reason:** the August 2026 survey found the depth half of H1 partly anticipated in text (arXiv 2604.07822, 2607.20594), while the (loop, spatial position) patching grid is untouched by anyone. The spatial axis has no text analogue. Ordering now follows defensibility rather than the order the work happens in. The benchmark stays first because the patching result is uninterpretable without knowing depth and breadth were independently controlled.
**Consequence:** M2 at milestone 6 now carries the headline, though M1 at milestone 5 still runs first. If milestone 6 slips, the headline slips, which was not true under the old ordering. Paper body section order is unchanged and stays a writing-time decision for milestone 12. All cross-references in `paper.md` now name contributions instead of numbering them.
**Reversible:** yes, until the draft exists.

### D-010. Local development environment differs from the cluster pin
**Date:** 2026-08-31
**Milestone:** 0
**Type:** deviation
**Decision:** development happens on Windows with Python 3.12.10 and torch 2.10.0+cu128, while `pyproject.toml` pins torch 2.4.1 for the cluster, installed from the cu121 index. The two are not reconciled and will not be.
**Reason:** the cluster's driver 525 with CUDA 12.0 cannot load a cu124 or later build, and the local machine is not the cluster. Forcing the dev box down to 2.4.1 would buy nothing and cost the ability to work locally at all.
**Consequence:** **local test passes are not cluster test passes.** Anything version sensitive must be re-run on the cluster before it counts. The bit-exact resume guarantee in `tests/test_resume.py` is a CPU guarantee under the local torch, and it is re-verified on the cluster as part of the milestone 0 sign-off. GPU determinism is untested by that suite and is deferred to milestone 3 where it can be tested against the real model.
**Reversible:** yes, but there is no reason to.

### D-011. The smoke trainer is a miniature looped model, and mutation checking is now standing practice
**Date:** 2026-08-31
**Milestone:** 0
**Type:** deviation
**Decision:** the milestone 0 smoke trainer is a prelude, tied core applied k times, coda, with the loop count drawn per batch from the global torch RNG stream. The spec (Section 14, prompt 1) asked only for "a smoke training script that counts to 1000 steps". Additionally, every new test is now mutation checked: break the implementation deliberately and confirm the test fails.
**Reason:** the first version was a plain MLP, and disabling RNG restoration entirely failed zero tests, because nothing in that model consumed global RNG. See `learnings.md` L-011. A smoke target that does not exercise the mechanism under test cannot test it.
**Consequence:** the smoke model must stay looped. Simplifying it back to an MLP would silently disarm `tests/test_resume.py`, and the `SmokeModel` docstring says so. Mutation checking adds a couple of minutes per test and is now in `CLAUDE.md`.
**Reversible:** no. Reverting either half would restore a known defect.

### D-012. LF line endings pinned via .gitattributes
**Date:** 2026-08-31
**Milestone:** 0
**Type:** deviation
**Decision:** `.gitattributes` forces `eol=lf` for `.sh`, `.pbs`, `.py`, `.yaml`, `.toml` and `.md`.
**Reason:** development is on Windows and the cluster is Linux. Git was converting `pick_gpu.sh` and `submit.pbs` to CRLF on checkout, which fails on the cluster with `bad interpreter: /bin/bash^M`, an error that points nowhere useful and costs a queue wait to diagnose.
**Consequence:** none locally. Verified by inspecting the stored blobs, which are clean ASCII with no CRLF terminators.
**Reversible:** no reason to.

### D-013. GPU selection gates on free memory, not used memory
**Date:** 2026-08-31
**Milestone:** 0
**Type:** deviation
**Decision:** `pick_gpu.sh` selects the card with the most free memory and refuses to run if that card has less than 20480 MiB free, overridable with `LOOPVISION_MIN_FREE_MIB`. IMPLEMENTATION.md Section 3 specified failing if the selected device has more than 2 GB **in use**.
**Reason:** the spec's rule is unworkable on this cluster and would have blocked every job. First contact with `<gpu-node>` showed eight A100-SXM4-80GB cards with roughly forty concurrent jobs across the gpu queue, and **every single card had more than 2 GB in use** while six of the eight had over 70 GB free. Used memory is the wrong quantity on a large shared card: 4 GB used on an 80 GB A100 is five percent, not contention. Free memory is what determines whether our job fits.
**Consequence:** the Python-side re-check at milestone 3 must use the same free-memory criterion, not the spec's 2 GB used rule, or the two will disagree and the disagreement will surface as an intermittent startup failure. `pick_gpu.sh` exports `LOOPVISION_GPU_FREE_MIB` so the Python side can read what the shell decided rather than re-querying and racing.
**Known limitation:** if no card has enough free memory the job exits non-zero and **does not queue a successor**, so a long run can die permanently on a transient cluster-full condition. Dying loudly beats a resubmit spin loop for now. Revisit at milestone 3 when real multi-day runs start, and consider a bounded retry with a delay.
**Reversible:** yes, it is one threshold.

### D-014. Cluster environment as found, 2026-08-31
**Date:** 2026-08-31
**Milestone:** 0
**Type:** resolution
**Decision:** recorded for reference, since several spec assumptions were checked against reality for the first time.
**What was found on `<login-node>` as user `<cluster-user>`:**

| Item | Spec said | Actually found |
|---|---|---|
| Driver and CUDA | 525.147.05, CUDA 12.0 | **525.147.05 confirmed exactly.** The cu121 pin is correct and cu124 or later would genuinely fail |
| GPU node | 8 GPUs, 96 cores | `<gpu-node>`, 8 x A100-SXM4-80GB, 96 ncpus, confirmed |
| Compute nodes | six workq nodes, 384 cores | six nodes `<cpu-node-01>` to `cn06`, 64 ncpus each, confirmed |
| Python | not stated | system python is 3.8.18, too old. **`module load python311` gives 3.11.11** and is required in every job script |
| Disk | check quota before milestone 5 | `/home` is BeeGFS, 466T total with 324T free. The 150 GB probe checkpoint estimate is a non-issue |
| Outbound network | compute nodes may lack it | the **login node** reaches PyPI and the PyTorch index. Installs happen there, not in jobs |
| Queue load | not stated | gpu queue had 39 running, 4 queued, 2 held at first contact. This is a busy shared cluster, not an idle one |

**Consequence:** the disk quota item in the `progress.md` standing checklist is closed. The `python311` module load is now required in `submit.pbs`. D-003, the compute block question, is informed but not resolved: the queue is busy, which bears on whether a sixteen-week window is realistic.

### D-015. The chain submits its successor through the login node
**Date:** 2026-08-31
**Milestone:** 0
**Type:** deviation
**Decision:** `submit.pbs` attempts a direct `qsub` for the successor job and, on failure, retries by sshing to `<login-node>` (overridable via `CHAIN_SUBMIT_HOST`) and submitting from there. A failure to queue a successor is fatal and loud.
**Reason:** this PBS server rejects `qsub` from inside a running job with `qsub: Bad UID for job execution`, exit code 175. IMPLEMENTATION.md Section 3.2 assumed self-chaining from within the job would work, and it does not here. That assumption failing silently broke the whole cluster strategy: the first job ran, stopped at its wall, checkpointed correctly, and then simply had no successor. A probe job confirmed that ssh from a compute node to the login node is passwordless and that `qsub` is accepted there. `PBS_O_WORKDIR` is on BeeGFS shared storage, so the working directory path resolves identically from both hosts, which is what makes the indirection work.
**Consequence:** every chained job log will contain one `qsub: Bad UID for job execution` line followed by `direct qsub failed (rc=175), retrying via the login node`. That is expected on this cluster and is not an error to chase. The direct attempt is kept so the script still works unmodified on a server configured with `flatuid`. If the noise ever obscures a real failure, add a `CHAIN_MODE` variable to skip the direct attempt.
**Risk retired:** a chain that cannot queue its successor now exits non-zero with an explicit message. Before this change it exited 0 and the run silently stopped advancing, which on a real multi-day run would have looked like a job that had simply not been resubmitted yet.
**Reversible:** yes, if the cluster admins set `flatuid` the fallback becomes dead code and can be removed.

### D-016. Gate G1.3 rewritten, the at-chance criterion was unachievable
**Date:** 2026-08-31
**Milestone:** 1
**Type:** deviation
**Status:** **needs human sign-off.** This changes a gate criterion, which the standing rules say must never be done to make a result pass. It is logged here in full so the change is visible and auditable rather than silent.

**Decision:** gate G1.3 changes from "the bag-of-operators shortcut probe must sit at chance for depth greater than 1" to two checks:

1. The measured probe must not exceed the **order-blind ceiling**, the Bayes accuracy of the best possible predictor that sees only the multiset of operators. A probe above the ceiling means the probe is using order information and the probe implementation is wrong.
2. The order-blind ceiling must sit **below the M1 threshold tau = 0.90 by a clear margin** at every depth in use. This is the property that makes the loop-count curve interpretable.

**Reason:** the original criterion is mathematically impossible, not merely hard. At depth 2 an order-blind predictor sees the multiset `{a, b}` and chooses between `a.b` and `b.a`, scoring about 0.65 against a chance level of 0.021. Measured ceilings are 0.652, 0.477, 0.375, 0.301 and 0.256 at depths 2 through 6, which is 31x down to 12x chance. This follows from composition in a finite non-abelian group and cannot be sampled away. The anti-shortcut sampler is working exactly as specified; the specified gate simply asked for something unattainable.

**Why the science is unaffected.** The ceiling never approaches tau = 0.90. An order-blind model cannot reach threshold at any depth, with at least 0.25 of margin, so `k_min` to threshold cannot be produced by a counting statistic. H1 is intact. Had tau been set below 0.65 this would have been a genuine crisis instead of a documentation fix.

**Consequence:** `validate.py` at milestone 2 implements the two-part check. The ceiling table goes in the paper's task validity section, as evidence rather than as an appendix detail, since "the shortcut exists and here is exactly how large it is" is more credible than claiming no shortcut exists. Regenerate the numbers with:

```bash
python -c "
import numpy as np
from loopvision.data import groups as G
rng = np.random.default_rng(23)
for d in range(2, 7):
    print(d, round(G.bag_of_operators_ceiling(rng, d, trials=4000), 4))
"
```

**Reversible:** the alternative is to change the task until an order-blind model really is at chance, which would need a much larger group and a harder rendering problem, and would cost the comparability with the text-case work that family A exists to provide. Not recommended, but it is the option if the sign-off goes the other way.

### D-017. Family A stimulus is a chiral tetromino, not an L-tromino
**Date:** 2026-08-31
**Milestone:** 1
**Type:** deviation
**Decision:** the family A glyph is a J-tetromino with three distinctly coloured cells and one neutral cell, not the "L-tromino with a coloured tip" of IMPLEMENTATION.md Section 4.2.
**Reason:** the suggested shape does not have a trivial stabiliser, which the spec itself requires in the same sentence. An L-tromino is symmetric under reflection about its diagonal; pairing that reflection with the transposition of its two arm colours fixes the glyph. The stabiliser has order 2, so the group acts with 24 orbits and the "48-way classification" would silently have been 24-way with chance at 1/24 rather than the stated 1/48. A coloured tip does not help either: a single coloured cell is fixed by any colour permutation that fixes that colour.
**Consequence:** four cells rather than three, so the glyph needs slightly more canvas. Chance stays at exactly 1/48 = 0.021 as the spec states. Both the rejection of the tromino and the acceptance of the tetromino are asserted computationally in `tests/test_groups.py`, including a regression test that keeps the tromino rejected so nobody reintroduces it.
**Reversible:** any shape with a trivial stabiliser works. The test computes the stabiliser rather than assuming it, so swapping the glyph is safe.

### D-018. Family A has no combo_ood split
**Date:** 2026-08-31
**Milestone:** 1
**Type:** deviation
**Decision:** family A supports train, iid_val, depth_ood and breadth_ood. It does not define `combo_ood`, and requesting it raises rather than improvising a substitute.
**Reason:** IMPLEMENTATION.md Section 4.5 defines `combo_ood` as "unseen colour and shape pairings". Family A has no shapes and no free colours: it has one glyph and three colours that the S3 factor permutes. Holding out any colour or any operator subset would break the uniform-factorisation property the anti-shortcut sampler depends on, which would cost more than the split is worth. Inventing a contrived substitute and calling it `combo_ood` would be worse than having none, because an OOD result would then be reported for a split that does not mean what its name says.
**Consequence:** `combo_ood` is a families B and C split. Any cross-family comparison on that split covers two families rather than three, and the paper must say so rather than leaving readers to infer coverage from a table. Enforced by `SUPPORTED_SPLITS` on each family module and by `test_unsupported_split_raises_rather_than_improvising`.
**Reversible:** yes, if someone designs a compositional held-out split for family A that preserves label uniformity.

### D-019. Family B relation semantics and scene construction
**Date:** 2026-08-31
**Milestone:** 1
**Type:** deviation
**Decision:** three departures from IMPLEMENTATION.md Section 4.3, all forced by measurement rather than preference.

1. **Relations are nearest-neighbour along a shared row or column**, not CLEVR-style directional sets. `left_of(X)` is the nearest sprite in X's row to its left.
2. **Sprites are placed in a compact rectangle** sized just larger than the sprite count, at a random offset, not scattered uniformly over the 8x8 grid.
3. **Chains are walked constructively**, choosing uniformly among relations that actually resolve to an unvisited sprite, rather than sampled blind and rejected.

**Reason, with the numbers.** The spec's set semantics need rejection sampling to guarantee a unique referent. Nearest-neighbour gives uniqueness structurally, so the only rejection cause is a dead end. But nearest-neighbour needs sprites to share rows and columns, and uniform scatter over 64 cells almost never produces that: measured rejection was **0.885 at depth 2 breadth 3, 0.970 at depth 3, and depth 4 could not be sampled at all**. Compact placement fixed most of it, and constructive walking fixed the rest. Worst cell is now **0.135**, against the spec's 0.30 ceiling.

The visited-sprite exclusion in the walk is the part that matters most for the science. Without it a chain could go `left_of` then `right_of` and land back on its anchor, so a depth 3 question would be exactly as hard as a depth 1 question while still being labelled depth 3. The depth axis would then be part signal and part noise, and H1 would be fitting the mixture without anything flagging it.

**Consequence:** the paper must describe the relation semantics as nearest-neighbour rather than implying CLEVR's, since the two give different chain distributions. The compact placement means absolute scene extent correlates weakly with breadth, which is acceptable because breadth is a declared axis, but it should not be described as uniform scatter. Rejection rates per cell are asserted in `tests/test_families_bc.py`.
**Reversible:** yes, but reverting any one of the three reintroduces the rejection rates above.

### D-020. Label spaces, split ranges and sprite gaps for families B and C
**Date:** 2026-08-31
**Milestone:** 1
**Type:** deviation
**Decision:** four smaller choices, grouped because they are all consequences of fitting the spec's task descriptions onto a 32x32 canvas.

1. **Family B answers attribute queries only**, giving 13 classes (6 colours, 5 shapes, 2 sizes). The spec says "yes/no plus attribute values". Yes/no tokens are reserved in the vocabulary but unused. A verify-style question adds a second query type and a second failure mode for no additional purchase on H1, which is about loop counts rather than answer format.
2. **Family B breadth starts at 6**, not the spec's 3. A depth d chain visits d distinct sprites, so depth 4 is structurally impossible below breadth 4 and rare at 5. Critically, **train and depth_ood share one breadth range** (6 to 9): letting the deeper split use broader scenes would confound the two axes and make H1 unanswerable, which is a far worse failure than a high rejection rate. `sample_scene` raises on `depth > breadth` rather than retrying, since that cell is structurally empty rather than hard.
3. **Family C has no depth_ood split.** Depth is 1 by construction and there is nothing to extrapolate along. A family C depth_ood would be a relabelling of the training distribution.
4. **Every sprite mask leaves its last row and column clear**, guaranteeing a one pixel gap between sprites in adjacent patches. Found by eyeballing a sample dump: two adjacent yellow triangles had rendered as one connected blob. Merely ugly for family B, actively wrong for family C, whose task is counting them. Family A glyphs already got this gap from being centred.

**Consequence:** family C contributes to three of the five splits. Any cross-family table must show coverage rather than let readers assume all families appear everywhere. The eyeball step in milestone 1's definition of done earned its place, since item 4 is invisible to every numerical test we had.
**Reversible:** items 1 and 2 yes, items 3 and 4 no, they fix real defects.

### D-021. Four cores per job, not eight
**Date:** 2026-08-31
**Milestone:** 2
**Type:** deviation
**Decision:** `submit.pbs` requests `ncpus=4`, not the `ncpus=8` of IMPLEMENTATION.md Section 3. `OMP_NUM_THREADS` now follows `$NCPUS` rather than being hardcoded, so changing the request cannot leave threads oversubscribed. A job that needs more can override on the qsub line.
**Reason:** the spec chose 8 so that two jobs run concurrently under the 16 concurrent core cap. That reasoning optimises per-job throughput, which is the right target on an idle cluster and the wrong one here. First contact found the gpu queue at 40 running, 8 queued and 14 held, with 24 hour walltimes, so our four gate jobs sat entirely unscheduled. At `ncpus=4` the cap allows four concurrent jobs instead of two, and smaller requests also backfill into gaps sooner.
**Consequence:** each job gets half the CPU. Data generation is procedural and runs in the training process rather than in DataLoader workers, so this is the throughput that suffers, not GPU utilisation. If steps per second drops enough to matter, the fix is DataLoader workers rather than reverting to 8 cores. Worth measuring once the gate runs land: if wall-clock per run more than doubles, this trade was wrong.
**Reversible:** yes, one line, and the override is documented in the script.

### D-022. Gate G1 failed. Remedy options for family A, awaiting a decision
**Date:** 2026-08-31
**Milestone:** 2
**Type:** resolution
**Status:** **OPEN, blocking milestone 3.** Recorded rather than acted on, because every option below changes the task and a failing gate is reported, not worked around.

**What failed.** G1.2: family A depth 1 at k=1 reached 0.0204 against a required 0.90. Chance is 0.0208. Full diagnosis in `findings.md`. Rendering is excluded by measurement: the model recognises both the state glyph and the operator glyph 48-way at 1.000 accuracy. What it cannot learn is the D4 x S3 multiplication itself, 2304 input pairs to 48 classes, from pixels in the online regime.

**Options, roughly cheapest first.**

1. **Train longer.** Loss was flat at ln(48) for 20000 steps on fresh data, but reached 2.429 on a repeating 20000 sample pool, so the mapping is learnable with more exposure per example. Cheapest test: rerun G1.2 at 100k to 200k steps. Risk: if depth 1 needs 200k steps at k=1, the M1 sweep at milestone 5 becomes ten times more expensive, which collides with the queue reality in D-003.
2. **Shrink the group.** D4 alone is order 8, D4 x S3 is 48. A smaller group means a smaller multiplication table and a lower difficulty floor. Cost: chance rises from 0.021 to 0.125, the order-blind ceiling shifts and D-016's numbers must be regenerated, and the comparison with the text-case work weakens, which is the reason family A exists.
3. **Split the task.** Present the operator in the query token stream rather than as a glyph, which is the `tokens` presentation variant already built and tested. This isolates composition from visual parsing. Cost: it is no longer a purely pixel task, which is the paper's central claim.
4. **Add capacity or curriculum.** Larger d, or pre-train on glyph recognition then fine-tune on composition. Cost: a curriculum makes "loops required" depend on training procedure, which contaminates H1.
5. **Redefine the depth origin.** Accept that depth 1 needs k > 1 and measure the curve from there. Cost: G1.2 exists precisely to prevent this, since without a solvable origin there is no way to tell "needs loops for composition" from "cannot do the base task".

**Recommendation.** Try option 1 first: it is the only one that does not change the task, it is a single rerun, and it directly tests the sample-efficiency hypothesis the pool result points to. If depth 1 still fails at 200k steps, option 2 is the honest next move and the group change gets reported in the paper.

**Note for the paper regardless of outcome.** That family C reaches 1.000 while family A depth 1 sits at chance under identical training is itself a finding about the two task families, and the gate table belongs in the task validity section either way.

### D-003. Compute block, sixteen weeks on the cluster
**Date:** open
**Milestone:** blocks the pre-registration freeze at milestone 5
**Type:** resolution
**Decision:** OPEN. From Section 12 of the spec.
**Question:** is a sixteen-week window on the Jaypee cluster realistic alongside the merging pipeline?
**If access is intermittent:** cut the size ladder from four models to two (d=384 and d=768) and say so in the paper.
**Resolve by:** before milestone 5, mid October 2026.
**Evidence, 2026-08-31.** First real submission met a saturated queue: 40 running, 8 queued, 14 held on the gpu queue, with our four gate jobs waiting behind roughly four others. Jobs carry 24 hour walltimes, so a slot can be a day away. This is one observation and not yet a trend, but the M1 sweep at milestone 5 is 200 to 400 runs, and at this queue depth that is the binding constraint rather than GPU speed. Worth measuring queue latency over the next week before committing to the four-model ladder.

### D-004. Co-authorship and affiliation
**Date:** open
**Milestone:** blocks preprint and submission
**Type:** resolution
**Decision:** OPEN. From Section 12 of the spec.
**Question:** Prof. Garg and Dr. Saini both in play. The Indian Institute of Information Technology Allahabad affiliation affects anonymity handling if a preprint goes up first.
**Resolve by:** before any preprint, and in any case before milestone 13.

### D-005. Preprint timing
**Date:** open
**Milestone:** after milestone 8
**Type:** resolution
**Decision:** OPEN. From Section 12 of the spec.
**Question:** a preprint after milestone 8 protects priority but complicates NeurIPS anonymity. The earlier submission pair already flagged cross-paper anonymity risks.
**Resolve by:** end of milestone 8, late November 2026. Do not let this drift into the submission window, because by then the decision makes itself badly.

### D-006. Task family priority
**Date:** open
**Milestone:** blocks the pre-registration freeze at milestone 5
**Type:** resolution
**Decision:** OPEN. From Section 12 of the spec.
**Question:** family A is the strongest scientific transfer from the state-tracking work. Family B is the most legible to vision reviewers. If only one can be done well, which audience is being optimised for?
**Note for whoever resolves this:** family C is not a candidate for cutting under any circumstance. It is the null arm of H1, and without it the depth-versus-breadth contrast is void rather than merely weaker.
**Resolve by:** before milestone 5, mid October 2026.

---

## 2. Dependency additions

Every package not in the spec's pinned list. The spec's list is: `torch`, `numpy`, `einops`, `pyyaml`, `pandas`, `pyarrow`, `scipy`, `statsmodels`, `matplotlib`, `tqdm`, `pytest`, plus Weights and Biases in offline mode.

| Package | Version | Why | Entry | Date |
|---|---|---|---|---|
| | *(none yet)* | | | |

---

## 3. Deviations from the pre-registration

Empty until `preregistration.md` is frozen at git tag `prereg-v1` before milestone 5. After that, **the pre-registration is never edited** and every difference between what was registered and what was done appears here, and in the paper.

| # | Registered | Actually done | Reason | Date |
|---|---|---|---|---|
| | *(pre-registration not yet frozen)* | | | |

---

## 4. Open questions

Live at the top, resolved ones moved to the log above with a resolution entry.

| # | Question | Blocks | Resolve by | Owner |
|---|---|---|---|---|
| D-003 | Sixteen-week compute window realistic? | pre-registration freeze, size ladder | mid Oct 2026 | |
| D-004 | Co-authorship and affiliation | preprint, submission | before preprint | |
| D-005 | Preprint timing versus NeurIPS anonymity | priority protection | end Nov 2026 | |
| D-006 | Family A or family B priority if only one | pre-registration freeze | mid Oct 2026 | |

---

## 5. Inherited defaults

Choices the spec already fixed. Listed so that a deviation is visible against a baseline rather than being invisible. Changing any of these requires a log entry in Section 1.

| Area | Default | Spec section |
|---|---|---|
| PyTorch build | `torch==2.4.1+cu121`, never cu124 or later | 3 |
| Cores per job | `ncpus=8`, `--num-workers 4` | 3 |
| Resolution | 32x32, 64x64 only for the robustness check | 4.1 |
| Group | D4 x S3, order 48 | 4.2 |
| Presentation variant | `strip` by default | 4.2 |
| Size ladder | d in {256, 384, 512, 768} | 5 |
| Core depth | 2 blocks tied, 1 block as ablation | 5 |
| Loop conditioning | three variants: none, embed, adaln | 5 |
| k_max training | 32 | 6.1 |
| k_mean_target | 8 | 6.1 |
| Jitter | on by default, sigma = 0.5 | 6.1 |
| BPTT window | 8, or 4 under memory pressure | 6.2 |
| Optimiser | AdamW, lr 3e-4, betas (0.9, 0.95), wd 0.05, clip 1.0 | 6.3 |
| Batch size | 256 at 32x32 | 6.3 |
| Precision | bf16 autocast, fp32 parameters | 6.3 |
| torch.compile | off | 6.3 |
| Probe checkpoints | 100 log-spaced steps, M1 and M5 configs only | 6.4 |
| M1 threshold | tau = 0.90, pre-registered | 7 |
| Seeds | 5 per cell, 10 for headline cells | 8 |
| Correction | Holm across the pre-registered family | 8 |
| Generation arm | out of scope | 1 |
