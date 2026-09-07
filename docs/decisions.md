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
**Status:** **RESOLVED 2026-09-01. Option 1 worked.** Depth 1 at 200000 steps reached 1.0000 against a required 0.90, while depth 3 stayed at 0.0212 at the same budget. Gate G1 passed on attempt 2 and milestone 3 is unblocked. The task was never broken; the run was roughly four times too short, with depth 1 grokking at step ~84000. Full result in `findings.md`. The options below are preserved because the reasoning behind rejecting options 2 to 5 still applies if a later family needs the same decision.

**What failed.** G1.2: family A depth 1 at k=1 reached 0.0204 against a required 0.90. Chance is 0.0208. Full diagnosis in `findings.md`. Rendering is excluded by measurement: the model recognises both the state glyph and the operator glyph 48-way at 1.000 accuracy. What it cannot learn is the D4 x S3 multiplication itself, 2304 input pairs to 48 classes, from pixels in the online regime.

**Options, roughly cheapest first.**

1. **Train longer.** Loss was flat at ln(48) for 20000 steps on fresh data, but reached 2.429 on a repeating 20000 sample pool, so the mapping is learnable with more exposure per example. Cheapest test: rerun G1.2 at 100k to 200k steps. Risk: if depth 1 needs 200k steps at k=1, the M1 sweep at milestone 5 becomes ten times more expensive, which collides with the queue reality in D-003.
2. **Shrink the group.** D4 alone is order 8, D4 x S3 is 48. A smaller group means a smaller multiplication table and a lower difficulty floor. Cost: chance rises from 0.021 to 0.125, the order-blind ceiling shifts and D-016's numbers must be regenerated, and the comparison with the text-case work weakens, which is the reason family A exists.
3. **Split the task.** Present the operator in the query token stream rather than as a glyph, which is the `tokens` presentation variant already built and tested. This isolates composition from visual parsing. Cost: it is no longer a purely pixel task, which is the paper's central claim.
4. **Add capacity or curriculum.** Larger d, or pre-train on glyph recognition then fine-tune on composition. Cost: a curriculum makes "loops required" depend on training procedure, which contaminates H1.
5. **Redefine the depth origin.** Accept that depth 1 needs k > 1 and measure the curve from there. Cost: G1.2 exists precisely to prevent this, since without a solvable origin there is no way to tell "needs loops for composition" from "cannot do the base task".

**Recommendation.** Try option 1 first: it is the only one that does not change the task, it is a single rerun, and it directly tests the sample-efficiency hypothesis the pool result points to. If depth 1 still fails at 200k steps, option 2 is the honest next move and the group change gets reported in the paper.

**Note for the paper regardless of outcome.** That family C reaches 1.000 while family A depth 1 sits at chance under identical training is itself a finding about the two task families, and the gate table belongs in the task validity section either way.

### D-023. Size ladder cut from four models to two
**Date:** 2026-09-01
**Milestone:** 3
**Type:** fallback
**Decision:** the size ladder is `d in {384, 768}`, roughly 11M and 43M parameters, not the four-model `{256, 384, 512, 768}` of IMPLEMENTATION.md Section 5. Approved by the user.
**Reason:** this is the fallback D-003 pre-authorised, and the conditions it named have arrived. Two measurements forced it. First, gate G1 established that family A needs 100k to 200k steps rather than 20k, a five to ten times multiplier on every family A run. Second, the queue has never given us the four concurrent jobs the 16-core cap allows; one to two is what we actually get, because the node itself is saturated. The M1 sweep at four sizes is roughly 105 runs at 4 to 6 hours each, which is 420 to 630 GPU-hours and three to thirteen weeks of wall clock for milestone 5 alone. Ten weeks remain for milestones 3 through 11.
**Consequence:** M1 drops to roughly 55 training runs and milestone 8 roughly halves. **The paper can no longer claim a smooth trend across four widths**, only a two-point comparison, and it must say so plainly rather than presenting two points as a ladder. If the two points disagree about the loop-count curve's shape, that is a reportable result and not a reason to reinstate the middle sizes without new compute.
**Alternatives rejected.** Dropping below five seeds per cell would weaken exactly the statistics the pre-registration rests on. Cutting a family would mean losing family C, which is fatal to H1's breadth arm, or family B, our most reviewer-legible task. The ladder is the cheapest thing to lose.
**Reversible:** yes, if compute frees up before the milestone 5 sweep launches. After that, adding sizes means re-running the sweep.

### D-024. GPU selection ranks by utilisation, not free memory
**Date:** 2026-09-03
**Milestone:** 3
**Type:** deviation
**Decision:** `pick_gpu.sh` keeps free memory as a hard floor but ranks eligible cards by **lowest utilisation**, breaking near-ties at random.
**Reason:** the first version ranked by most free memory with utilisation only as a tiebreak, and it cost real time. Both 1M-step runs were found sharing GPU 2 at 100 percent utilisation with three other processes, while GPUs 1, 5 and 7 sat at 0 to 8 percent. An 80 GB A100 can have 60 GB free and still be compute saturated, so free memory says almost nothing about contention. Both our own jobs also chose the same card, hence the random tiebreak.
**Consequence:** restarting the two runs onto idle cards gave about **1.3x**, from 21k to 27.6k steps per hour. Less than the 2x to 4x I estimated, because the "idle" cards still have other tenants. Every future run benefits, including the milestone 5 sweep, so the value compounds even though the single measurement is modest.
**Reversible:** yes, one script.

### D-025. Several runs per PBS job, because cores bind before GPUs do
**Date:** 2026-09-03
**Milestone:** 3
**Type:** deviation
**Decision:** added `scripts/multirun.pbs` and `analysis/orchestrator.py`. A single PBS job now runs several training processes, each pinned to its own GPU, sharing the job's core allocation. `submit.pbs` remains for single runs.
**Reason:** measured cluster state on 2026-09-03: **94 of 96 cores allocated on the GPU node while GPUs 2 and 5 sat at 0 percent utilisation.** The idle cards were idle because nothing had cores left to feed them, not because nobody wanted them. Submitting more PBS jobs cannot fix that, they simply queue. Packing runs into a core allocation we already hold converts spare GPU capacity into throughput.

At `ncpus=8` with four runs per job, the 16 core cap allows two such jobs, so **eight runs in flight against the four `submit.pbs` would manage**. Viable because at k around 10 the runs are GPU bound rather than generation bound: one worker supplies roughly 4000 samples per second against a GPU consuming 1500 to 3000.

**Consequence:** each run gets two cores rather than four, so one generation worker rather than three. If a future configuration becomes generation bound again, at low k or a larger canvas, this trade inverts and `runs_per_job` must come down. The orchestrator prints a plan and requires `--submit` to act, because a sweep is tens of GPU-days and a typo in a config should not cost that.
**What this does not do:** it cannot exceed the 16 concurrent core cap. PBS enforces that and no scheduling logic gets past it.
**Reversible:** yes, `submit.pbs` is untouched and still the path for single runs.

### D-026. Family B anchor descriptor and family C query breadth
**Date:** 2026-09-03
**Milestone:** 3
**Type:** deviation
**Decision:** two task-design fixes, both prompted by asking whether family B's 1.0000 accuracy was even possible.

1. **Family B.** The queried attribute is chosen first and the anchor is described by the other two, rather than always by colour and shape. IMPLEMENTATION.md Section 4.3 does not specify the descriptor, so this is a gap being filled rather than a contradiction.
2. **Family C.** Queries name one or two attributes, not the two or three of Section 4.4, and half are drawn from a sprite actually present in the scene.

**Reason.** Measured, not suspected. Family B depth 1 had the answer in the query for 66.8 percent of samples, because at depth 1 the target is the anchor; a trained model scored 0.833 on a blank image. Family C had 53.5 percent zero labels and 93.4 percent in {0, 1}, with the count never exceeding 4 of a cap of 10, so a constant "0" scored 0.535 and the task did not respond to breadth at all. Full numbers in `findings.md`.

**Consequence.** Every family B and family C run before 2026-09-03 is invalid and must not be used for any claim. That is four family B runs and the family C gate run. **Gate G1.4 relied on the family C run**, so it needs re-running on the fixed task before milestone 5; the check itself, "solvable at k=1 at every breadth", is unaffected in principle, but the number behind it came from a task that was not doing what it claimed.

The family C change also alters the breadth axis's meaning: the count now scales with scene size, which is what makes it a stressor rather than a constant.

**Reversible:** no. Both fix real defects.

### D-027. Family B trains across depths 1 to 4, and has no depth_ood arm
**Date:** 2026-09-03
**Milestone:** 3
**Type:** deviation
**Decision:** family B's `train` and `iid_val` span depths 1 to 4 at breadths 6 to 9, rather than depths 1 to 2 with 3 and 4 held out as `depth_ood`. The `depth_ood` split is removed from family B entirely. `breadth_ood` is unchanged in role: same depths, wider scenes.
**Reason:** two measurements. First, the old training range contained no depth signal at all: depth 1 leaked the answer (D-026) and depth 2 solves in a single pass, so a model trained on 1 and 2 had nothing to learn a loop-count curve from. Second, depths 5 and 6 cannot be added at this breadth range without breaching the 30 percent rejection ceiling:

| depth | b=6 | b=7 | b=8 | b=9 |
|---|---|---|---|---|
| 4 | 0.155 | 0.098 | 0.048 | 0.048 |
| 5 | **0.362** | **0.310** | 0.167 | 0.084 |
| 6 | **0.571** | **0.444** | 0.294 | 0.178 |

They are reachable at wide breadth only, so a depth_ood arm using them would have to move the breadth range too. **That confounds depth with breadth and makes H1 unanswerable**, which is a worse failure than having no depth_ood arm. The same reasoning already forced train and depth_ood to share a breadth range when family B was first built.

**Consequence:** family B's depth axis is 1 to 4, read inside a single split with breadth held fixed, rather than across a train and OOD boundary. The paper cannot claim depth extrapolation for family B, only a within-range curve. If depth extrapolation is wanted later it needs a scene design where deep chains are samplable at narrow breadth, which is a task change rather than a parameter change.

Recorded as a property of the task, not a limitation of the run: **family B chains cannot exceed depth 4 at breadths 6 to 9.** That is worth stating in the paper's task section.
**Reversible:** yes, but only by changing the scene sampler.

### D-028. Family B: size is not askable, breadth moves to 12 to 16, depth runs to 6, core shrinks to 1/1/1
**Date:** 2026-09-04
**Milestone:** 3
**Type:** deviation
**Decision:** four changes to family B, made together because they fix one problem.

1. `size` is no longer an attribute a question can ask about, only one that describes an anchor.
2. Train and iid_val breadth moves from 6 to 9 up to 12 to 16. breadth_ood moves to 18 to 22.
3. Depth runs 1 to 6 rather than 1 to 4, and `L_MAX["B"]` is derived from `MAX_DEPTH` rather than being the literal 8 that silently capped it.
4. The core shrinks from prelude 2, core 2, coda 2 to 1/1/1 for the loop-count runs.

**Reason:** the k sweep came back flat at 1.000 for every depth and every k from 1 to 64, so family B produced no H1 evidence at all. Auditing the data found three compounding causes.

*The floor was misreported.* The label space is a union of three attribute spaces of different sizes, and only one attribute is asked per sample, so the floor is the average of 1/|attribute|, not 1/13. With size askable that is **0.2897, not the 0.0769 quoted everywhere**. A size question is a coin flip. Earlier family B models scoring 0.24 to 0.38 were recorded as showing partial difficulty; they were at or below chance and had learned nothing.

*The chain was skippable.* A model ignoring the relation tokens entirely and guessing the most common answer among all reachable sprites scored **0.62 at depth 4**, with 19 percent of questions having only one possible answer regardless of the chain. Breadth 6 to 9 leaves only about 3.2 reachable sprites carrying about 2.2 distinct answers, so the nominal 13-way task was a 2-way discrimination.

*Depth could not exceed the single-pass budget.* Sequential depth in one forward pass is `prelude + k*core + coda`, six blocks at k=1 under 2/2/2. A depth 4 chain needs about four hops. Six exceeds four, so k bound on no reachable cell and every possible sweep outcome was 1.000.

Measured after the change, at breadth 12 to 16 with size unaskable: chain-blind accuracy at depth 4 falls from 0.62 to **0.44**, forced answers from 19 percent to under 1 percent, and the floor from 0.2897 to **0.1833**. Rejection at depth 6 falls from 0.571 to 0.020, because the ceiling was never about depth, it was about having enough sprites for a chain to have somewhere to go.

**On whether this is tuning.** Shrinking a core until a curve appears is exactly what the no-tuning rule forbids, so the distinction has to be explicit. At 2/2/2 with depth capped at 4 the experiment could not measure its own quantity: every outcome was 1.000 whatever the truth. Restoring the ability to come out either way is not the same as selecting an outcome. To hold that line the prediction is registered in advance and quantitatively: **k_min = max(1, depth - 2)**, so depths 1 to 3 at k=1, depth 4 at k=2, depth 5 at k=3, depth 6 at k=4. If depth 6 solves at k=1 the hop-per-block model is wrong, and that is the reportable result rather than grounds for shrinking again.

**Consequence:** every family B run before 2026-09-04 is superseded, six in total. No family B result may be quoted against 1/13. `effective_chance()` exists so the floor is computed rather than recalled.
**Reversible:** yes, but the old configuration is known not to measure anything.

### D-029. The D4 versus S3 asymmetry gets a crossed design before it gets a paper section
**Date:** 2026-09-04
**Milestone:** 3
**Type:** experiment design
**Decision:** the finding that D4 is learnable alone and S3 is not will not be written up as a claim about group structure until a 2x2 crossing group against substrate has been run. Design in `docs/experiment-substrate.md`, predictions registered there before building.
**Reason:** D4 and S3 differ in two ways at once. They are different groups, and D4 rearranges the glyph in space while S3 permutes its colours. The current result is equally consistent with "S3 the group is harder" and with "permuting identities is harder than moving things". The second is a claim about perceptual binding in a vision transformer and is the more interesting of the two, so choosing between them by assertion would be choosing the conclusion.

Two new arms settle it: S3 acting on three spatial slots, and D4 acting on four colour labels. The two existing runs are the other two cells. D4's action on the four corners of a square is faithful, kernel size 1, so four labels suffice and the renderer's six colours are enough.

**Consequence:** the S3 section of the paper waits on two runs, roughly 15 hours of wall clock. A linear probe on the existing s3only checkpoints comes first and costs no training: it separates "the information is never computed" from "it is computed and the head cannot read it", which the coda lens cannot distinguish because a logit lens only sees what the head decodes.
**Reversible:** yes, but writing the section first would mean writing it twice.

### D-030. Gate G2 moves from family A depth 3 to family B, because it is undefined where it was written
**Date:** 2026-09-04
**Milestone:** 4
**Type:** deviation
**Decision:** G2 is evaluated on family B at a 1/1/1 core across depths 1 to 6, against a matched-compute feedforward baseline and the echo baseline, rather than on family A depth 3.
**Reason:** the gate as written compares two models that are both at chance.

| run | accuracy | loss |
|---|---|---|
| `g1_famA_d3_looped_long_s0` | 0.0225 | 3.8711 |
| `g1_famA_d3_ffwd_long_s0` | 0.0225 | 3.8711 |

Chance is 1/48 = 0.0208 and ln(48) = 3.8712. Both sit on the floor, identically, to four decimals. A gate asking whether A beats B, where A and B have both learned nothing, cannot pass and cannot fail. Every possible outcome is the same, so running it would produce a number carrying no information.

The premise was falsified by our own later work: G2 assumed family A depth 3 would be learnable, and the depth wall says it is not. **This is a gate whose task stopped existing, not a gate we are avoiding.**

Family B at 1/1/1 is where a loop-count curve exists: depth 6 needs k=3 at tau=0.90 and single-pass accuracy falls to 0.492, so there is something for loops to buy and the comparison can come out either way. Matched compute is well defined: a looped model at k passes does `prelude + k*core + coda` blocks, so the feedforward baseline is given the same block count.

**On the no-tuning rule.** Moving a hard-stop gate to a task where it can pass is exactly the shape of the thing CLAUDE.md forbids, so the distinction has to be explicit. The gate is not being moved because it failed. It is being moved because it returns no information wherever it is pointed at present, in the same way G1.3 demanded an unachievable chance-level score (D-016) and the family B sweep was flat in every cell whatever the truth (D-028). Restoring a measurement's ability to come out either way is not selecting its outcome. **G2 can still fail on family B, and the recent literature says it might:** Gao et al. arXiv 2607.16051 state that parameter scaling usually beats looping at matched compute and needed a specific mixture-of-experts design at 20B to overturn it.

**Consequence:** the looped arm reuses the eight existing `famB_curve111` runs, so only the two baselines are new. The original depth 3 comparison is reported in the paper as an undefined gate with its numbers, not hidden.
**Reversible:** yes, if family A ever gains a learnable depth 3.

### D-031. multirun.pbs asks for 64gb rather than 96gb, because the queue now refuses 96
**Date:** 2026-09-06
**Milestone:** 4
**Type:** deviation
**Decision:** the `#PBS -l select=` directive in `scripts/multirun.pbs` requests `mem=64gb` instead of `mem=96gb`.
**Reason:** the gpu queue carries `resources_max.mem = 64gb` and refuses anything above it at submit time. Probed on 2026-09-06 rather than inferred from the queue configuration:

| request | result |
|---|---|
| `mem=48gb` | accepted |
| `mem=64gb` | accepted |
| `mem=96gb` | rejected, `Job violates queue and/or server resource limits` |

The cap postdates the script. It had never bitten because every real submission passes its own `-l` on the command line, which overrides the directive, so the checked-in default was unreachable and would have failed only for a bare `qsub scripts/multirun.pbs`.
**Consequence:** none for any recorded run. No submitted job ever used the 96gb directive, so no result changes and the paper says nothing about this. The number is advisory in any case: PBS does not enforce memory here, and job 375 on the gpu node is using 1041 GiB against no request at all, so it holds the 32gb default. The binding constraint on that node is cores, measured at 96 of 96 assigned with 41 jobs resident.
**Reversible:** yes, if the cap is lifted. Re-check with `qstat -Qf gpu | grep resources_max`.

### D-032. The feedforward control for the colour deadlock, matched on compute at k=8
**Date:** 2026-09-06
**Milestone:** 4
**Type:** scope
**Decision:** build two feedforward controls, `famA_d2_d4colour_ffwd` and `famA_d2_s3only_ffwd`, each derived from its looped twin and differing from it in exactly three keys: `arch`, `k_schedule` and `k_train`. Both are built at k=8, so `prelude + k*core + coda` is `2 + 16 + 2 = 20` blocks, which is what the looped twin executes at its training operating point.
**Reason:** the deadlock finding says a looped model cannot bootstrap composition on colour from scratch but can from a donor. That is a claim about **looping** only if a non-looped model of the same depth does not deadlock the same way. If it does, the finding is about compositional learning in general, which is a different paper. This has been ranked second in the handoff since 4 September and was never built.

k=8 and not another value because both looped twins train with `k_schedule: sampled` and `k_mean_target: 8`, so 8 is their operating point. Gate G2 matched at k=4 for a different reason (D-030): there k=4 was where the looped model cleared tau at every depth. Here the looped model clears nothing at any k, so no such point exists and the training mean is the only choice that is not arbitrary.
**Consequence:** **matched compute is not matched parameters.** The control holds 35.47M parameters against the looped model's 10.98M, a ratio of 3.23x, because its 20 blocks are untied where the looped model reuses one core. The spec (IMPLEMENTATION.md, baselines) calls for a matched-parameter arm separately and it is not built. So the two outcomes are not symmetric:

- If the control **deadlocks**, the result is clean. A model with three times the parameters and the same depth also fails, and the deadlock is not an artifact of weight tying.
- If the control **solves** the task, parameter count remains an alternative explanation. Run 5190 (wide, deep and big looped models on the same task) addresses that from the other side: a larger looped model that still deadlocks while the feedforward solves it isolates the architecture rather than the size.

Reporting either outcome requires saying which of these two situations we are in.
**Reversible:** yes. Nothing else depends on these configs.

### D-033. TF32 is left alone for the campaign, and the inherited worry was based on an outdated default
**Date:** 2026-09-06
**Milestone:** 4
**Type:** resolution
**Decision:** leave both TF32 settings at their torch 2.4.1 defaults for every run in this campaign. Pin them explicitly near the top of `train/cli.py`, at the values they already hold, once the node is clear of our runs. Not before: `src/` is read live by running jobs, so adding the lines now would change what an in-flight experiment computes at its next chain hop.
**Reason:** the concern arrived from a note carried over from the merging project, which states that on Ampere `torch.backends.cuda.matmul.allow_tf32` defaults to True, so fp32 matmuls run with a 10-bit mantissa. **That was true before PyTorch 1.12 and is false here.** Measured in this project's venv on 6 September:

| setting | value | what it means |
|---|---|---|
| `torch.backends.cuda.matmul.allow_tf32` | **False** | matmuls run at full fp32 |
| `torch.get_float32_matmul_precision()` | `highest` | the same fact, stated the other way |
| `torch.backends.cudnn.allow_tf32` | **True** | convolutions may use TF32 |

Attention and the MLPs, which is nearly all of the arithmetic in this model, are already at full fp32. The entire exposure is cudnn convolutions, and each model holds exactly one convolution: `patchify`, an `nn.Conv2d` applied once per forward pass to embed the image into patch tokens (`model/loopvit.py` lines 88 and 339).

Changing either setting now would make new runs non-comparable with the 44 that have already finished under the current defaults, for a benefit nobody has measured. A certain cost against an unquantified benefit settles it for the campaign. Run count from `analysis/registry.py`, not counted by hand.
**Consequence:** the paper needs no precision caveat for its matmul arithmetic, which is the arithmetic every quoted number depends on. If the patch embedding is ever questioned, the test is cheap and does not need retraining, because the setting affects the forward pass and not the stored weights: set `torch.backends.cudnn.allow_tf32 = False`, re-run one instrument that quotes four decimals against a stored checkpoint, and compare against the recorded value.
**Reversible:** yes. Until the pinning lines exist this is a decision to leave a default alone rather than a change to anything, and the pin itself only fixes the values already in force.
**The timing clause above was overtaken on 2026-09-06. See D-035.**

### D-034. The repository is public, which resolves D-005 four milestones early
**Date:** 2026-09-06
**Milestone:** 4
**Type:** resolution
**Decision:** the full repository, `paper.md`, `findings.md` and `gap-analysis.md` included, was published at https://github.com/K144U/What-Loops-See on 2026-09-06, public, to timestamp priority over the work. D-005 is therefore resolved by action, rather than by the deliberation it scheduled for the end of milestone 8.
**Reason:** priority. The instruction was explicit and repeated. A public repository with a push timestamp is the cheapest defensible claim to having held these results on this date, and three of them are unpublished.
**Consequence:** D-005 asked exactly this question and named the cost, so the cost is recorded here rather than discovered later. It wrote that a preprint "complicates NeurIPS anonymity", and that the earlier submission pair had already flagged cross-paper anonymity risks. That cost is now incurred, and **it cannot be withdrawn by deleting the repository**, because the push is timestamped and public material may be mirrored, cached or indexed within minutes. Any double-blind submission drawn from this work must assume a reviewer can find the repository under the author's own account and read the whole experimental record, including findings that are in no paper yet. Read the venue's policy on preprints and public code before submitting, not after.

Two things were done before publishing, recorded rather than assumed. The history was rewritten once, to remove tooling attribution trailers and to take a private address, a cluster username and two hostnames out of the published record. `docs/sha-map.txt` maps every pre-rewrite commit to its current one, because 49 run directories record the SHA they were produced at and would otherwise point at nothing. A scan for credentials and secrets before publishing found none.
**Reversible:** no. Publication is not reversible, and treating it as reversible is the mistake this entry exists to prevent.

### D-035. The TF32 pin reached the cluster before D-033 said it should
**Date:** 2026-09-06
**Milestone:** 4
**Type:** deviation
**Decision:** the pin from D-033 is live on the cluster now, with four of our jobs running, rather than waiting for the node to be clear of our runs as D-033 required.
**Reason:** publishing required rewriting history, which orphaned the cluster's clone: every commit it held stopped existing. Restoring a coherent clone meant resetting it onto the new history, and the pin is part of that history. The alternative was leaving the cluster on a history unreachable from anywhere, which breaks provenance for every run started from that point onward. That is worse than deploying a change already shown to do nothing.

D-033's caution was written while the pin's effect was unknown. It is now measured. After importing the pinned module the flags read False and True, identical to the unpinned defaults, and the suite passed in a sandbox built against the live tree, 272 passed and 4 skipped at exit 0, then again on the cluster after the reset at 276 passed. A value-preserving change cannot alter what a running job computes.
**Consequence:** any run chaining after 2026-09-06 executes the pinned settings, which are the settings every earlier run already used, so no result is affected and no comparability is broken. The risk actually taken was not a precision change but the ordinary hazard of editing source that running jobs read live: a syntax error would have broken the next chain hop. That is why the sandbox test came first, and it is the argument for testing in a sandbox rather than in place.
**Reversible:** yes, by deleting two lines, though there is no reason to.

### D-036. A second history rewrite, force pushed over an already published repository
**Date:** 2026-09-06
**Milestone:** 4
**Type:** deviation
**Decision:** the history was rewritten a second time, setting the author and committer of all 102 commits to the identity actually linked to the publishing account, and force pushed over the repository published earlier the same day.
**Reason:** the first rewrite set a uniform author, which satisfied the requirement that one name appear, but used an address not registered to the GitHub account. GitHub attributes commits by email, so the contributors list stayed empty and the commits linked to no profile. A priority claim that does not resolve to its author is worth less than one that does.
**Consequence:** every SHA changed again. `docs/sha-map.txt` now maps original SHAs, the ones the 49 run directories actually record, directly to current ones, verified by resolving all 100 targets to real commits. Intermediate SHAs from the first rewrite are deliberately absent, because they never left this machine.

**The force push is the part worth remembering.** The repository was public for roughly twenty minutes carrying the earlier history. Anyone who cloned in that window holds commits that no longer exist upstream and cannot pull. That is acceptable here only because the window was short, the repository was hours old, and its existence had been announced to nobody. **It stops being acceptable the moment anyone else holds a clone**, and no later rewrite should be assumed safe on this precedent.
**Reversible:** no, in the sense that matters. The earlier SHAs were public, and public cannot be undone.

### D-037. The matched-parameter arm is the feedforward baseline at k=1
**Date:** 2026-09-06
**Milestone:** 4
**Type:** scope
**Decision:** the matched-parameter arm of the feedforward baseline is the same control at `k_train: 1`, giving 6 blocks against the looped model's 6 distinct blocks. Configs `famA_d2_d4colour_ffwd_mp` and `famA_d2_s3only_ffwd_mp`, derived from their looped twins, differing from the matched-compute arm in exactly one number.
**Reason:** the matched-compute arm carries 3.23x the looped parameters, because its 20 blocks are untied where the looped model reuses one core. If it solves a task the looped model deadlocks on, capacity is an alternative explanation and the result does not isolate the architecture. The matched-parameter arm removes that explanation.

k=1 is not a tuned choice. A looped model's parameters live in `prelude + core + coda` blocks however many times it iterates, so a feedforward model with that many blocks carries the same parameters, and `prelude + k*core + coda = prelude + core + coda` has the single solution k=1. Measured rather than argued, at d_model 384:

| k | blocks | params | ratio to looped |
|---|---|---|---|
| **1** | **6** | **10.6886M** | **0.973x** |
| 2 | 8 | 14.2291M | 1.295x |
| 8 | 20 | 35.4720M | 3.230x |

**The residual 2.69 percent is the injection adapter** and is not closable. The looped model holds an `adapter` mapping 2d to d, 294912 parameters, which feeds the state back in each iteration. A feedforward model has nothing to inject and no counterpart module. Widening the baseline to close the gap would be fitting the control to a number, which is the thing CLAUDE.md forbids, so the gap is reported instead.

**The direction of the residual is the useful part.** The arm is 2.69 percent *smaller* than the looped model. If a smaller non-looped model solves a task the looped model never leaves chance on, capacity cannot be the explanation and neither can depth, because this arm is also shallower. That is the strongest form the control can take.
**Consequence:** the two arms now bracket the looped model. Matched compute is deeper and larger, matched parameters is shallower and slightly smaller. Read together:

- **both deadlock:** the deadlock is about compositional learning in general, not about looping. Finding 3 becomes a different paper, as D-032 already noted.
- **both solve:** looping itself is the obstacle, and neither depth nor capacity explains it.
- **matched compute solves, matched parameters does not:** the rescue came from capacity or depth rather than from dropping the loop, and the looping claim does not follow.

Reporting any outcome requires saying which of these three it is.
**Reversible:** yes. Nothing depends on these configs.

### D-038. The untied baseline, to decompose the gate G2 failure
**Date:** 2026-09-07
**Milestone:** 4
**Type:** scope
**Decision:** build `famB_g2_untied`, the fourth spec baseline, derived from the failing gate's feedforward arm and differing from it in `arch`, a truncated `eval_k_sweep`, and a pinned `untied_copies: 4`.
**Reason:** G2.1 failed against an arm that changed **two** properties at once. The matched-compute feedforward both untied the weights and removed the loop structure, the recurrent state and the injection. The gate therefore establishes that looping lost without establishing what it lost to. The untied arm restores the loop structure while keeping untied weights, so it sits between the other two and separates them:

| arm | parameters | vs looped | loop structure |
|---|---|---|---|
| looped, tied | 5.6594M | 1.000x | yes |
| **untied** | **10.9701M** | **1.938x** | yes |
| feedforward | 10.6752M | 1.886x | no |

`untied` exceeds `feedforward` by exactly 294912 parameters, the injection adapter mapping 2d to d, which a feedforward model has no counterpart for. Up to that one module the two are parameter matched, so:

- **untied against feedforward** holds parameters and compute fixed and varies only the loop structure.
- **untied against looped** holds compute and depth fixed and varies only weight tying.

**A trap worth recording, because it nearly ate the baseline.** `untied_copies` defaults to `max_k_in_use`, which reads `eval_k_sweep`. The inherited sweep runs to 64, so an unpinned untied model allocates **64 core copies**, roughly twenty times the core parameters, and is a matched comparison to nothing whatever. Pinning the copies is only safe once the sweep is truncated to what the model can serve, or the run raises partway through training. Both halves are asserted, and the mutation check confirms each is detected.
**Consequence:** three outcomes, and each says something different about the failed gate.

- **untied also beats looped:** the win came from untying the weights, not from abandoning the loop. That is a claim about parameter sharing and it leaves the looped architecture's serial story intact, at a cost in parameters.
- **untied does not beat looped, but feedforward still does:** the win came from removing the loop structure itself, which is the harder result to write around.
- **untied matches feedforward:** the loop structure is inert at this operating point, and the whole comparison reduces to a parameter count.

**This does not reopen gate G2.** G2 failed, it is recorded as failed, and no outcome here changes that. This is diagnosis of a failure already reported, not an appeal against it.
**Reversible:** yes. Nothing depends on these configs.

### D-039. The pre-registration is written at milestone 6, and says so
**Date:** 2026-09-07
**Milestone:** 6
**Type:** deviation
**Decision:** `docs/preregistration.md` is written and frozen at tag `prereg-v1` on 2026-09-07, at milestone 6. IMPLEMENTATION.md Section 8 requires it before milestone 5 runs. Milestone 5 has not run, so the ordering is not violated for H1's remaining cells, H2 or H4, but the file arrives after H1's family B result and after three H3 heatmaps were already seen.
**Reason:** H3 has no defined metric anywhere in the repository, and a first read of the M2 grids on 2026-09-07 produced numbers that did not fit the prediction using statistics invented after looking at them. Without a registered metric that reading cannot become a result, and with a metric written afterwards it could become whatever the data suggested. The only honest way out is a document that states what had been seen, at the moment of freezing, per hypothesis.
**Consequence:** Section 0 of the pre-registration is a disclosure table rather than a formality, and it costs the project something real:

- **H2 and H4 are pre-registered in the ordinary sense.** Nothing has been computed for either.
- **H1's family B result is demoted to exploratory.** Eight seeds, already analysed, already written up as supported. The registered form binds only the cells not yet run.
- **H3's three heatmaps are exploratory** and stay that way whatever the confirmation set shows. H3's metrics are now fixed as normalised Shannon entropy over iterations and over positions, chosen to be parameter free so that no later choice of a top-`m` cutoff can move the answer.

The mechanistic alternative that would explain the observed reversal, counting being a global aggregate and composition being strip-local, is registered in Section 4 **before** the confirmation runs, so that adopting it later is a recorded prediction rather than a story fitted to the outcome.
**Reversible:** no. The tag is the point. Deviations from here go in this file.

### D-040. The D4 curriculum needs a third arm, because its control shares the group
**Date:** 2026-09-07
**Milestone:** 4
**Type:** deviation
**Decision:** add `famA_d2_d4colour_curriculum_s3donor`, a third curriculum arm initialised from `famA_d1_s3only_s0`, a depth 1 model that learned the S3 factor and carries no D4 structure at all.
**Reason:** the two D4 arms are both at 1.0000 while still training, the real curriculum at 90 percent of budget and its control at 77 percent. Read carelessly that says the rescue is not donor specific. Read carefully it says something else, because the S3 experiment's control and the D4 experiment's control are not the same kind of control.

| experiment | task | donor in the control arm | shares the task's group | outcome |
|---|---|---|---|---|
| S3 | depth 2 S3 | depth 1 **D4** | no | stayed at 0.1652 for the full budget |
| D4 colour | depth 2 D4 on colour | depth 1 **D4** on position | **yes** | at 1.0000 |

S3's control was a different group. D4's control is the same group on a different substrate. So the existing D4 control tests whether position-trained D4 knowledge transfers to colour, and the answer appears to be yes, which is a real result. It does not test whether the donor's content matters at all, and that is the claim the S3 experiment supports and the D4 experiment is currently assumed to support.
**Consequence:** until this arm finishes, the defensible D4 statement is **"a depth 1 D4 donor rescues depth 2 D4 on colour, whether that donor was trained on position or on colour"**. The stronger statement, that the rescue is specific to the donor rather than to having any pretrained start, is not supported for D4 and must not be written as though the existing control established it.

Three arms now differ from each other in exactly one config line, `init_from`, which is the cleanest form this comparison can take:

- `famA_d1_d4colour_s0`, same group, same substrate
- `famA_d1_d4only_s0`, same group, different substrate
- `famA_d1_s3only_s0`, different group, no D4 content

If the third arm also reaches 1.0000, the rescue is about having any trained depth 1 initialisation and the donor's content is irrelevant, which would weaken the deadlock account considerably and would apply retrospectively to the S3 result as well. That outcome is written here before the run, not after it.
**Reversible:** yes, it is one more run.

### D-041. The queue is consolidated, because the cap counts jobs and not runs
**Date:** 2026-09-07
**Milestone:** 4
**Type:** deviation
**Decision:** the nine queued jobs were deleted and resubmitted as six, ordered by scientific priority rather than by submission history, with runs packed several to a job.
**Reason:** `max_run` is 5 **jobs** per user, and `max_run_res.ncpus` is 16 **cores**. Neither counts runs. We were holding four job slots for five runs, with nine more jobs queued behind carrying nineteen runs between them, several of them single run jobs. `multirun.pbs` exists to put several runs inside one allocation and the queue had drifted away from using it.

Measured at the time: all eight GPUs had more than the 10240 MiB `pick_gpus` requires, we held five claims, and eight cores were about to free. So the ceiling on concurrent runs is roughly eight, set by GPUs, and the job cap was costing us most of the gap between five and eight.

Order now reflects what the results need rather than when things were submitted:

| job | runs | why it is where it is |
|---|---|---|
| 5269 | s3spatial s0, s1, curr_s3donor | closes the 2x2 from 293k checkpoints, about 7k steps each, and starts the D-040 control |
| 5270 | untied s0, s1 | diagnoses the gate G2 failure |
| 5271 | famC s3, s4, famA d4only s2 | H3 confirmation seeds |
| 5272 | famA d4only s3, s4 | H3 confirmation seeds |
| 5274 | wide, deep, big | scale control |
| 5275 | depth ladder d3 to d6 | second family for H1 |

**Two things went wrong and are recorded rather than tidied away.**

First, 5223 had started in the minutes before the `qdel`, so killing it ran its chain block, which correctly resubmitted its unfinished runs as 5276. That is the chain fix working as designed, and it produced a **duplicate**: 5276 and the new 5273 both carried `famA_d2_s3only_ffwd_mp_s{0,1}`, which would have put two processes in one run directory. Caught by comparing run ids across every queued job, and 5273 was deleted because 5276 was already running. **A qdel of a running multirun job resubmits its remainder, so any bulk requeue must check for duplicates afterwards rather than assume the deleted jobs are gone.**

Second, the first verification pass used `tr -d '
	 '`, which deletes the spaces separating run ids, so every job appeared to hold one run and the duplicate was invisible. The check that found it keeps spaces.
**Consequence:** resubmitting costs queue position, which is free here because nothing had started except 5223, and no work is lost because every run resumes from checkpoint under `--resume auto`. The s3spatial pair resumes at 293448 and 292987 steps rather than restarting.
**Reversible:** yes.

### D-003. Compute block, sixteen weeks on the cluster
**Date:** open
**Milestone:** blocks the pre-registration freeze at milestone 5
**Type:** resolution
**Decision:** OPEN. From Section 12 of the spec.
**Question:** is a sixteen-week window on the Jaypee cluster realistic alongside the merging pipeline?
**If access is intermittent:** cut the size ladder from four models to two (d=384 and d=768) and say so in the paper.
**RESOLVED 2026-09-01: access is workable but throughput is not. The size ladder is cut to two models, see D-023.**
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
**Resolved by D-034** on 2026-09-06, four milestones early, by publication rather than by deliberation.

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
