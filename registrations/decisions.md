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

### D-042. The headline moved, so the seed requirement moves with it
**Date:** 2026-09-08
**Milestone:** 6
**Type:** deviation
**Decision:** the contribution list is revised ahead of milestone 9, in `paper.md` Section 3b, and the preregistration's 10 seed rule is applied to the substrate 2x2 rather than to family B.

**Why now rather than at milestone 9.** `paper.md` Section 5 says the objective is revisited only at milestone 9 and any revision logged here. This is that log. The trigger is not preference: Section 3a named one blocking condition for the binding claim, the crossed design, and it landed on 8 September on the registered substrate row in both new cells. Waiting until December to write down that the blocked claim is unblocked would leave the plan describing a project that no longer exists.

**What the seed rule says.** The preregistration, Section 1: minimum 5 per cell, minimum 10 for any cell whose number appears in an abstract, a headline sentence, or a figure caption. Its H1 section then says 10 for families B and C "which carry the headline". They no longer carry it. The rule follows the headline rather than the family, so the 2x2 cells need 10 and family B keeps 5 unless H1 returns to the abstract.

**Where the seeds actually stand**, counting only the four crossed cells:

| cell | outcome | done | in flight | needed for 5 | needed for 10 |
|---|---|---|---|---|---|
| `d4_position` | solves | 4 | 0 | 1 | 6 |
| `s3_position` | solves | 2 | 0 | 3 | 8 |
| `d4_identity` | fails | 2 | 3 (5282) | 0 | 5 |
| `s3_identity` | fails | 2 | 0 | 3 | 8 |

**Tier 1, to the 5 seed minimum: 7 runs.** Three of them solve and cost the full 300000 steps; three fail and are early stoppable. Packed three to a two core job per D-041, that is three job slots and roughly 16 hours of wall clock with all three running.

**Tier 2, to the 10 seed headline requirement: 27 further runs.** Fourteen of them are in solving cells and cost full budget. Thirteen are in failing cells.

**Early stopping does not apply to these cells, which is a correction to how this entry first read.** The existing seeds in every cell ran the full 300000 steps, and the claim they support is that the task never leaves chance across that budget. New seeds that stop at 30000 would leave each cell holding a mix of budgets, and the claim would degrade to something weaker for most of its seeds. **All Tier 1 and Tier 2 seeds in the crossed cells run the full budget.**

Early stopping remains worth having, for exploration and for arms that are not carrying a headline number, where it can be disclosed as such. Commit 22ad755 made every run record `n_distinct` and `modal_share`, which is the measurement; nothing yet acts on it. Measured across the finished campaign, 3880000 of 17500000 steps ran in runs already within 0.01 of their final at step 20000, so the saving is real. It is simply not a saving that may be taken here.

**What is deliberately not being bought.** More architecture variants. The untied arm is a third decimal effect measured near ceiling at 1.938x parameters, and another one would not change what the paper claims. Cores are the binding constraint and seeds on the headline cells are what they should buy.

**Consequence:** the schedule absorbs this. Milestone 6 is running roughly seven weeks ahead of its calendared window, and Tier 1 plus Tier 2 is a few days of wall clock at current throughput, measured at about 28000 steps per hour for a single run in a two core job.
**Reversible:** yes. Section 3 and Section 3a are intact, and reverting means preferring the older list.

### D-043. The scale control had no depth arm, and the fix is a new config rather than an edit
**Date:** 2026-09-08
**Milestone:** 6
**Type:** correction
**Decision:** `configs/m3/famA_d2_d4colour_deep.yaml` is left exactly as it is, and a new arm `famA_d2_d4colour_depth4.yaml` is added at 4/4/4. `famA_d2_d4colour_deep_s0` runs to completion under 5274 and is recorded as a third seed of the base arm.

**What was wrong.** D-032 specified 5190 as "wide, deep and big looped models", three **larger** models, to test whether a larger looped model still deadlocks. Two of the three are larger. `deep` is not:

| arm | d_model | blocks | params | larger than base |
|---|---|---|---|---|
| base | 384 | 2/2/2 | 10983552 | |
| wide | 768 | 1/1/1 | 22552320 | yes |
| big | 768 | 2/2/2 | 43790592 | yes |
| **deep** | **384** | **2/2/2** | **10983552** | **no, it is the base** |

The `deep` config differs from the base config in exactly three keys, `prelude_blocks`, `core_blocks` and `coda_blocks`, all set to 2, which is what the base already uses by default. `famA_d2_d4colour_s0` does not set them and its `summary.json` reads 2/2/2. So the arm restates the defaults and varies nothing.

**Why the file is not being edited.** `cli.train` calls `load_config(args.config, ...)`, which reads the config **from `configs/`** and not from the run directory's own `config.yaml` snapshot, and `multirun.pbs` passes `--config configs/${cfg}` on every launch including a resume. `famA_d2_d4colour_deep_s0` was at step 170000 when this was found. Editing `_deep.yaml` in place would mean that the next time PBS kills that job at the wall, the run resumes by building a 4/4/4 model and loading a 2/2/2 checkpoint. A new file has no such interaction, and it also keeps the run id `..._deep_s0` pointing at the config it actually ran.

**Why 4/4/4.** Measured, not chosen for tidiness. At `d_model` 384 the block counts give 16294272 params at 3/3/3, **21604992 at 4/4/4**, and 26915712 at 5/5/5, against `wide` at 22552320. So 4/4/4 sits at 0.96x of `wide` and is the closest available. That makes the new arm a **matched-parameter width against depth contrast** rather than merely a depth arm, which is the comparison that separates capacity as width from capacity as serial depth. That distinction is the bootstrapping question, so the arm that was missing is the interesting one.

**What the two working arms already say, at one seed each and still running.** `wide` reads 1.0000 at step 250000 and `big` reads 1.0000 at step 100000, while `d_model` 384 fails on three seeds. D-032 set this up as a decision procedure: a larger looped model that still deadlocks isolates the architecture, one that escapes isolates the size. **They escape.** The deadlock is about capacity and not about looping, and the alternative explanation D-032 flagged for the feedforward control solving, that parameter count was doing the work, is the explanation.

**Consequence:** the scale control is two arms until `depth4` runs, and no seeds are queued for `deep`. Four more seeds of a duplicate base arm would cost roughly 80 GPU hours and answer nothing.
**Reversible:** yes. `_deep.yaml` is untouched, so nothing that has run is invalidated.

### D-044. Six configs cannot run, and seven finished runs cannot be re-analysed
**Date:** 2026-09-08
**Milestone:** 6
**Type:** correction
**Decision:** the six configs are left in place and marked here rather than deleted, and the seven affected runs are treated as closed evidence: their stored numbers stand, no instrument will ever be run on them again, and nothing that needs a fresh forward pass may cite them.

**How this surfaced.** Job 5275 spent thirteen hours holding a four core slot with three of its four runs dead within seconds of launch. `famA_d4/d5/d6_d4only` ask to train at depths 4 to 6, and family A's `train` and `iid_val` are defined at depths (1, 2, 3). Depths 4 to 6 exist only in `depth_ood`, which is held out by construction. The guard in `dataset.py` refused correctly and loudly; nothing was watching the per-run logs, so the job looked healthy in `qstat` the whole time.

**The full sweep, because three found by accident is not a survey.** Of 74 configs in the repository, six cannot generate their own training data:

| config | asks for | split allows |
|---|---|---|
| `m3/famA_d4_d4only.yaml` | depth 4 | depths 1 to 3 |
| `m3/famA_d5_d4only.yaml` | depth 5 | depths 1 to 3 |
| `m3/famA_d6_d4only.yaml` | depth 6 | depths 1 to 3 |
| `m3/vark_famB_alldepths.yaml` | breadths 6 to 9 | breadths 12 to 16 |
| `m3/vark_famB_d1.yaml` | breadths 6 to 9 | breadths 12 to 16 |
| `m3/vark_famB_d2.yaml` | breadths 6 to 9 | breadths 12 to 16 |

**The two halves have different causes and only one is a mistake.** The family A depth configs were never runnable: they were written for a depth axis family A does not have. The family B ones were correct when written. D-020 set family B's breadth range at 6 to 9, and D-028 moved it to 12 to 16 so that depth could reach 6, measuring chain-blind accuracy at depth 4 falling from 0.62 to 0.44 and the floor from 0.2897 to 0.1833 as the justification. Those configs are not wrong, they are historical, and they predate D-028.

**The consequence nobody had noticed.** A run's stored `config.yaml` is what every instrument reads to regenerate its evaluation batches. Seven finished runs carry snapshots naming breadths 6 to 9, so with today's code they cannot produce a single batch:

`varkB_fix_s{0,1,2}`, `vark_famB_d1_s{0,1}`, `vark_famB_d2_s{0,1}`

Sixty finished runs are re-analysable and these seven are not. M4 geometry, M5 halting, the collapse check and the input ablations can never be run on them. This is not the same as the runs being invalid: their training was real and their recorded accuracies were produced under a coherent task definition. They are simply closed to any question that needs a new forward pass.

**Why the configs are not deleted.** The three family B ones are the only record of what those seven runs were trained on, and deleting them would leave seven run directories whose configs cannot be reconstructed. The three family A ones stay because deleting a config that was never runnable does not stop the next person writing it again; the entry does.

**What this does not resolve.** Family A has no depth axis beyond 3, so "second family for H1", which is how `STATE.md` described 5275, is not achievable from the current data definition. The choices are to regenerate family A's `train` with deeper programs, which invalidates every family A run in the campaign, or to accept that family B is the only depth family and H1's dissociation rests on it alone. That is a scientific decision and it is not taken here.

**Consequence:** `analysis/check_configs.py` should exist and does not. The sweep above was run by hand, and a config that cannot generate data is exactly the kind of fault that costs a job slot silently. Until it is written this entry is the only record.
**Reversible:** yes. Nothing was deleted.

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

### D-045. Fourteen more runs are closed to re-analysis, and the checker that missed them is fixed

**Date:** 2026-09-10. **Supersedes nothing. Extends D-044.**

**What happened.** D-044 recorded seven finished runs closed to re-analysis, all family B, all because their configs ask for breadths 6 to 9 which D-028 moved out of the split. A geometry pass on 2026-09-10 failed on nine `vark_*` runs with `KeyError: 'prelude_blocks'`, a different cause: those snapshots predate the block count keys entirely, so `build_model` cannot reconstruct the model whatever the split says.

**Why `check_configs` did not see them.** It read run snapshots with `load_config`, which layers `DEFAULTS` over the file, while every instrument reads them with a bare `yaml.safe_load`. The checker was validating a configuration that did not exist on disk and that the run may never have trained with. It also never model checked a snapshot, only its data. Both are fixed: snapshots are read raw, the way instruments read them, and the model is built. See L-027.

**The decision.** Closed to re-analysis now stands at **21 finished runs**, up from 7. They are not deleted and not invalid. Their recorded numbers stand and are quotable; what cannot be done is regenerate their evaluation batches or their states, so no instrument pass can be run over them and no figure may depend on one.

**What this costs.** Nothing currently claimed. The `vark_*` runs are superseded by the family A and family B campaigns and nothing in `paper.md` rests on them. The cost is that the loop-count variance work they represent cannot be re-examined with any instrument built since, and re-running them would mean re-training, not re-analysing.

**What it does not license.** Back-filling the missing keys into those snapshots. A snapshot is a record of what a run used, and writing a plausible `prelude_blocks` into one would make it a record of a guess. If those runs are ever needed again they are re-trained under a current config with a new run id.

**Enforced by:** `analysis/check_configs.py`, which now reports them on every run, and the two tests named in L-027.

### D-046. CLAUDE.md was two revisions behind on cores, in the file every session reads first
**Date:** 2026-09-10
**Milestone:** 6
**Type:** deviation
**Decision:** the `Request ncpus=8` line in CLAUDE.md is replaced by the two core rule, scoped to the gpu queue, with workq named as the exception.

**Reason:** the line had been wrong twice over. D-021 moved the request to 4 on 2026-08-31 and the 2026-09-07 backfill evidence moved it to 2, and neither edit reached CLAUDE.md. That matters more than an ordinary stale line because CLAUDE.md is loaded at the start of every session while STATE.md is read only when a session is told to read it, so the wrong number is the one a cold session sees first. A six core request does not queue slowly here, it waits indefinitely while smaller jobs overtake it, which is how 5269 sat while 5270 and 5272 ran.

**Why it is scoped rather than global.** The first version of this edit said "request ncpus=2" without qualification, which would have been a different error in the same place: `scripts/m5_halting.pbs` and the other workq analysis passes correctly ask for 16, because workq is a separate pool with no `resources_max` and no per-user cap. A rule that shrank those would cost throughput for nothing.

**Consequence:** none to any result. No run changes, and the `.pbs` files already carried the right numbers; only the instruction did not.
**Reversible:** yes, one paragraph.

### D-047. Breadth is recorded per item, and Claim 3 is post-hoc rather than registered
**Date:** 2026-09-10
**Milestone:** 6
**Type:** deviation
**Decision:** `instruments/m5_halting.py` now stores `b_task`, the parallel breadth of each item's generating program, alongside `k*_task`. `analysis/m5_h2.py` regresses the signed error on it and reports the result as **descriptive and post-hoc**, outside the Holm family of fourteen. `scripts/m5_halting.pbs` gains `FORCE=1` so a run can be passed again.

**Reason, and the blocker that was not visible from STATE.md.** STATE.md ranked this second and described it as PENDING for want of code rather than data, with family C already passed. The data was not there either. `_pass` discarded the fifth value `make_batch` returns, which is the per-item breadth, so **no stored `m5_halting.parquet` has ever carried a breadth column**. Breadth is a property of the item and is gone once the pass ends, so this could not be computed from what was on disk however the analysis was written. The module was only half the work.

**Why it is not a registered test, which is the part STATE.md had wrong.** STATE.md item 2 called this "a registered claim". It is not one. The preregistration's Section 3 fixes two tests per criterion and both regress on `k*_task`; no breadth test appears anywhere in the family of fourteen listed in its Section 1. paper.md Section 4 states Claim 3 and says in the same paragraph that it "is unfalsifiable on its own and is reported descriptively". Section 1 of the preregistration then says any statistic not defined in that file is post-hoc, is labelled post-hoc in the paper, and gets a confirmation run at milestone 8. So the breadth slope carries no Holm column, never counts toward supporting or falsifying H2, and prints under a heading that says so. Adding it to the family would also mean editing a frozen document.

**Three states that had to stay apart**, because two of them are not evidence and all three produce no slope:

| state | what it means | what the report says |
|---|---|---|
| breadth varies | a slope exists | the slope, with its levels |
| breadth fixed by design | the axis never moved | "breadth fixed at [6]", no slope |
| breadth never recorded | the pass predates the column | the run is named, with the remedy |

A criterion blind to breadth and a set of runs that never varied breadth both return nothing, and only the first is a result. `_slope` returns NaN rather than 0.0 for an axis with one level for this reason.

**Only runs that actually sweep breadth are offered a re-pass.** The first version recommended re-passing every run lacking the column. **CORRECTED 2026-09-10, and the correction is the interesting part.** This entry originally said family B fixes breadth, so re-passing it would record a constant. That is wrong: `configs/m3/famB_curve_111.yaml` sweeps breadths 12 to 16 as well as depths 1 to 6, so family B is a crossed family and a re-pass would give Claim 3 a second breadth arm at a different breadth range. The `cfg_breadth_levels` check built for this decision is what caught the error, by reading the config instead of taking the claim on trust and then listing all eight famB seeds as worth re-passing. Note the consequence: once family B carries `b_task` it varies both axes, so its slopes become marginal fits that can absorb one another, which is the case `print_blind_axis` warns about. The config knows the sweep and the parquet does not, so `collect` now carries `cfg_breadth_levels` and the recommendation is split. Family C is the family that needs passing again.

**Claim 3 is assembled across families and cannot come from one.** Family B varies depth 1 to 6 and fixes breadth; family C varies breadth 4 to 8 and fixes depth at 1. Each answers one half, so the report names which task each slope came from. The two slopes are per unit of their own axis and are not rescaled, because normalising them onto a common scale would be a second unregistered choice made after seeing the numbers.

**Consequence.** No stored number changes. `b_task` is added data and touches no threshold, no `k*_model`, no `k_hat_c`, so landing it mid-campaign cannot alter what a finished or in-flight run computes. Family C needs a re-pass under `FORCE=1` before Claim 3 can be reported at all; until then the module prints an absent result and says which runs would fix it.

**Verified.** Five deliberate breaks were confirmed to fail the right tests, per the mutation rule: a no-leverage slope returning 0.0 instead of NaN, the breadth slope fitted on depth, a missing column filled with a real level, missing runs left unnamed, and the breadth slope routed through `holm_bound`. Three further breaks cover the re-pass split. The suite is 529 passed, 4 skipped.
**Reversible:** yes. The column is additive and the analysis refuses cleanly without it.

### D-048. What a feedforward trajectory is, and why five geometry jobs died
**Date:** 2026-09-10
**Milestone:** 8
**Type:** deviation
**Decision:** `FeedforwardBaseline.forward` gains `return_states`, sampling the residual stream after the prelude and after each group of `core_blocks`. `m4_geometry` gains `usable_k`, which clamps a requested k to what an architecture can actually run, and records the trajectory it got rather than the one it asked for.

**Two separate faults, one symptom.** Jobs 5458 to 5462 all failed and F5 stayed at 55 files. The logs held two different errors:

    TypeError: FeedforwardBaseline.forward() got an unexpected keyword argument 'return_states'
    ValueError: untied model has 4 cores but was asked for iteration 4

The first is a missing capability. The second is a **working guard doing its job**: the untied baseline refuses to extrapolate past its own cores, because reusing a core is precisely what that baseline exists to avoid. Fixing the second by loosening the guard would have destroyed the control. The actual fault in both cases is the caller: `scripts/m4_geometry.pbs` sends one `K=32` for every run in the batch, and two of the three architectures have a depth fixed when they were built.

**The definition, which is a choice and not a fact.** A looped model yields `s_0 .. s_k`, one state per core application. A feedforward stack has no recurrence, so its trajectory has to be defined. It is the residual stream sampled at the same depths: after the prelude, then after each group of `core_blocks`, with the trailing `coda_blocks` left out because in the looped model the coda runs after the last state. Column i then means the same number of core applications in both architectures. The alternative, sampling every block, would put a 10 block stack and an 8 iteration loop on axes that are not the same length and not the same thing.

**The caveat that cannot be fixed in code, recorded rather than smoothed over.** With `state_init: randn` the looped model's `s_0` is noise drawn independently of the image, so its i=1 column compares noise against the first real state. The feedforward stack's state 0 is the post-prelude activation, a function of the input. **The first column is not the same measurement across architectures.** Every later column is. `m4_geometry` now writes `arch` and `state0` into its result so F5 can say which case a run is in, because F5 draws both on one axis and nothing else would stop a reader assuming they match.

**Clamping is not extrapolation.** `usable_k` returns the built depth for a feedforward stack, `min(requested, untied_copies)` for an untied model, and the request unchanged for a tied looped one. It measures the trajectory a model has instead of demanding one it does not, and the run prints when it clamps. `k` in the result is now the length that came back and `k_requested` is what was asked for, so a clamp is visible in the stored file rather than only in a log.

**Verified.** Nine deliberate breaks, all caught. On the model: boundaries sampled one block early, the coda sampled as a core group, `s0` ignored instead of refused, the requested k stretching a fixed stack, and the recording path perturbing the residual stream. On the instrument: untied not clamped, everything clamped including tied looped, feedforward reporting the request instead of its depth, and the recorded k reverting to the request. The last needed a stub architecture whose trajectory is shorter than its k implies, because `usable_k` and the model agree for every architecture that exists today and no real run can separate the two values.

**The regression that mattered most.** 44 finished runs were trained through `FeedforwardBaseline.forward`. Recording states walks the blocks once, in one code path, whether or not states are kept, and a test asserts the logits are **bit identical** with and without `return_states`. A tolerance there would have hidden exactly the kind of reordering that makes an old checkpoint score differently.

**A defect this found in its own author's work.** The first version reassigned `k` before storing `k_requested`, so both fields recorded the clamped value and the clamp was invisible in the stored file. Caught by the test that asserts the two differ.

**Consequence.** No stored number changes. `return_states` is additive and off by default, `usable_k` only affects instrument calls, and no training path reads either. Geometry can now run on the `ffwd` and `untied` runs it has always skipped. The `vark_*` and `g1_*` failures in 5461 and 5462 are untouched and remain expected, being the runs D-045 closed to re-analysis.
**Reversible:** yes.

### D-049. The contribution order moves again, and this time on a failed prediction
**Date:** 2026-09-10
**Milestone:** 8
**Type:** deviation
**Decision:** `paper.md` Section 3b swaps contributions 4 and 5. The oracle comparison for halting criteria moves above the patching mechanism account. Nothing else in the list moves.

**Why this is logged here.** Section 5 of `paper.md` says the objective is revisited only at milestone 9 and any revision logged in this file. D-042 was that log on 8 September and this is its successor. The trigger is the same as D-042's: waiting until December to write down that the list no longer describes the evidence would leave the plan describing a project that does not exist.

**What changed on 10 September.** Two things, pulling opposite ways on a list explicitly ordered by defensibility.

H3 was evaluated for the first time. Its confirmation set had held one family until jobs 5455 and 5456 landed, and a contrast between a depth family and a breadth family cannot be computed from one family. It came back **split**: the loop contrast indistinguishable from zero, the spatial contrast significant and **opposite to the registered prediction**, neither surviving Holm. Preregistration Section 4 requires that be reported as split with the lead claim weakened rather than asserted on the surviving half.

The halting comparison went from unrun to determinate. It is fully pre-registered in the ordinary sense, which Section 0 of the preregistration makes a short list, and its outcome is neither supported nor falsified in a way that is determinate rather than pending. Claim 3 then gained a second breadth arm at a different breadth range, and the two arms disagree in sign on the criterion that looked breadth sensitive on one of them.

**Why a swap and not a demotion.** The patching instrument is not damaged. Every cell was admitted and every positive control read at the ceiling, and the (loop, spatial position) grid is still untouched by anyone. What failed is the registered directional story, and the paper has to carry that distinction rather than blur it: the map is publishable, the predicted separation is not a finding. The spatial reversal is itself a result, breadth being the spatially uniform case rather than the concentrated one, and it holds in both depth families rather than depending on how the depth group was composed.

**What was deliberately not done.** The loop half's null is a pooling artefact: the two depth families sit far enough apart that the gap dwarfs the contrast, and one lands on the wrong side of the breadth family. That is written into the contribution as the reason the null cannot be read, and **not** promoted into a claim of its own. It is post-hoc, on cells from two families differing in more than one way, and it would need its own registration and its own cells.

**No coefficients entered `paper.md`.** Family C is still gaining seeds toward the preregistration's ten, so every H3 coefficient and halting slope will move. The standing rule is that prose outside the escape-rates markers must not quote a number that moves as seeds land. Directions and signs are stated, magnitudes are not, and the numbers live in `findings.md`.

**Consequence.** No result changes and no run is affected. `escape_rates --check docs/paper.md` passes, 7 counts in the marked region, all matching.
**Reversible:** yes, and the superseded ordering is preserved in Section 3 and in the 8 September text of 3b, per the standing rule that a superseded entry is kept with a pointer rather than deleted.

### D-050. A claim is retired, and the interesting part is that it never had support
**Date:** 2026-09-10
**Milestone:** 8
**Type:** correction
**Decision:** "escape needs capacity, not the removal of the loop" is withdrawn from `STATE.md`. Nothing replaces it. The 2x2 marked region in `paper.md` is regenerated after `famA_d2_s3only_s4` completed it at 21 of 21.

**What happened.** Job 5433 finished. `famA_d2_d4colour_ffwd_mp_s3` wrote DONE at 1.0000 on the full budget, a genuine solve at 8 distinct classes rather than a collapse. The matched-parameter feedforward arm is now 1 of 5 where it had been 0 of 2.

**The control did its job.** D-037 built that arm to remove capacity as an explanation: the matched-compute arm carries 3.23x the looped parameters, so an escape there was confounded. Both feedforward arms now escape at 1 of 5, across a 3.23x range in parameters, Fisher p = 1.000 between them. **Capacity is ruled out.**

**Nothing replaces it, and that is the decision.** The obvious next claim is that removing the loop is what matters, since the looped arm is 0 of 6 and both feedforward arms escape. Fisher exact says otherwise: 0.455 for each feedforward arm against the looped arm, 0.500 pooled. **No arm is distinguishable from any other at these seed counts.** Writing the converse would repeat the original error with the sign flipped.

**The original claim never had support at any point.** It rested on matched compute 1 of 5 against matched parameter 0 of 2. Fisher p on that comparison is **1.000**. Not weak evidence, none. It became a claim because two consecutive failures read as a pattern and because 0 of 2 reads as a zero rather than as two draws. The same `STATE.md` item carried the warning "0 of 2 is consistent with never escaping and with a rate near 1 in 5" and the heading asserted the claim regardless. **The hedge and the headline contradicted each other and the headline won.**

**Why this cost nothing.** The claim reached `STATE.md` and no further. It is absent from `paper.md`, `findings.md` and `claims.md`, so no contribution, no headline number and no figure moves. The documents that gate what the paper says held; the one that summarises for a fresh session did not. That asymmetry is worth keeping: a summary is where an unsupported reading is cheapest to hold and easiest to state.

**The rule this suggests.** A rate quoted from fewer than five draws gets its Fisher p computed against the arm it is being contrasted with, before the contrast is written as a claim. Two of the three contrasts in this campaign that looked decisive at small n have p above 0.45.

**Also regenerated.** `famA_d2_s3only_s4` finished at 0.1681 on the identity floor, taking the 2x2 to 21 of 21 seeds. `escape_rates --check docs/paper.md` failed on the stale counts, which is the third time that check has caught drift before a reader did. The row is unchanged in direction: identity 0 of 11, position 8 of 10.

**Consequence:** no result is withdrawn, one reading is. `escape_rates --check docs/paper.md` passes at 7 counts.
**Reversible:** the claim can return if the seed counts ever separate the arms. At 0 of 6 against 1 of 5 they do not.


### D-051. The queue is reordered so the headline closes before the work that supports it
**Date:** 2026-09-16
**Milestone:** 8
**Type:** operational
**Decision:** the eighteen unfinished runs of the registered substrate 2x2 are moved to the front of the gpu queue with `qorder`, behind the donor stage and ahead of everything else. No job is deleted, no run loses a step, and nothing is resubmitted.

**What happened.** The slate queued on 11 September put the 2x2 first because it is the headline, landing at 1.1 days. On 16 September nine of its seeds were unfinished and all nine sat at the very back of a fifty-three job queue, about 146 runs and ten days out, with 23 GPU hours of work left between them. They were 65 to 93 percent trained. The cause is structural and is written up as L-030: a run that hits its 23 hour wall self-chains, the chained job is a fresh submission with a fresh job number, and priority here is arrival order, so a run falls further back every time it is interrupted. The ordering was correct at submission and decayed on its own.

**`qorder` works here and `qalter` does not.** Both are installed and both return 0. The first swap tried moved no `qtime` at all, which is exactly what the blocked `qalter` looks like, and it was only reading the `qstat` listing back that showed the two jobs had in fact swapped position. Queue rank is what moved; `qtime` is not it. That is the third time L-021 has paid out.

**A tenth cell member was found by the check rather than by hand.** `--order` reported nine `famA_d2_d4colour_deep` seeds as buried at 74 runs. D-043 already recorded that this config duplicates the baseline instead of being a depth arm, and `escape_rates` has counted `deep_s0` as a D4 identity seed ever since, so its nine siblings are cell members too and the cell cannot close without them. They were moved forward as well. **The alternative, declaring the cell closed at 26 finished and treating `deep` as a separate arm, was rejected**: it would shrink a denominator to reach a result sooner, which the first non-negotiable forbids, and D-043 decided the question already on grounds that have not changed.

**What this does not do.** The donor stage keeps the front of the queue. It is 14 remaining runs and it gates the 44 run curriculum block, which is the most valuable thing in the slate and the only thing taking three headline numbers off n=1. Delaying it to reach the 2x2 a few hours sooner would trade the strongest available result for the fastest one. The displaced jobs are tier 2 and tier 3 supporting cells, which go to the back and lose nothing but position.

**Consequence:** the 2x2 closes in roughly two days rather than roughly ten. `check_queue` reports 175 of 175 queued runs still build a model and generate a batch after the reorder, and `check_queue --order` reports a worst headline backlog of 14, within budget.
**Reversible:** entirely. Nine plus three `qorder` calls put it back, and no state other than queue rank was touched.

### D-052. The headline claim becomes a rate, because the cell was never settled
**Date:** 2026-09-21
**Milestone:** 8
**Type:** correction
**Decision:** "a looped model never composes identity permutations" is withdrawn and replaced by a rate: position 15 of 20, identity 15 of 50, Fisher two sided p = 0.0011. `paper.md` section 3b, `findings.md` "SUBSTRATE, NOT GROUP" and Section 1, and `claims.md` rows 2 and 15 are revised. The result is not withdrawn and its direction is unchanged.

**What happened.** The registered substrate 2x2 closed at 70 of 70 seeds, every cell at or above the preregistration's ten. The identity column, which read **0 of 11** when the paper's framing was written and which the word "never" was built on, is **15 of 50**. Roughly three identity seeds in ten solve the task outright, at 1.0000, emitting the full class count at a modal share equal to the label prior. They are real solves and not artifacts; that was checked before anything was rewritten.

**The measurements were never wrong.** `famA_d2_d4colour_s0` and `s1` did finish at 0.1254 on the floor at full budget, and they still do. What was wrong was reading two seeds, and later eleven, as a settled cell. **This is precisely the failure the ten seed rule exists to prevent**, and the sentence it bit was the one the paper leads with. D-042 applied that rule to the 2x2 in September and it took until now to pay out.

**Why 0 of 11 was not evidence of never.** At the observed rate near 0.30, the chance that eleven consecutive identity seeds all fail is about 0.02. Unlikely enough to be worth noticing, and nowhere near unlikely enough to support an absolute. A run of eleven is a small number pretending to be a large one.

**The historical counts stay in the files.** `paper.md` keeps "0 of 11" outside the checked markers with a `count-ok` naming it as the 2026-09-10 reading, and the superseded `findings.md` entry is kept in place with a scope note rather than rewritten. A record that quietly becomes correct cannot be audited, and the distance between the two readings is the argument for the rule.

**Two smaller claims moved with it, both found while checking this one.** Run level escape is nearly rather than strictly all or nothing: of 196 finished family A runs, 104 finish above 0.95, 91 sit on a floor and exactly one lands between, `famA_d2_d4only_s5` at 0.6802 with its budget spent, so it is an intermediate outcome and not a late escape caught early. And the collapse signature is 72 of 78 rather than 13 of 13, the six exceptions emitting two classes at floor accuracy.

**Consequence:** `escape_rates --check docs/paper.md` passes at 7 counts with 0 unannotated. `check_findings` and `--doc docs/claims.md` both pass. `test_claims_sync` passes. 647 tests pass.
**Reversible:** nothing to reverse. The rate is what the runs say at the resolution the preregistration asked for.

### D-053. The registered H4 metric is underspecified in three ways, and each is resolved in the open
**Date:** 2026-09-21
**Milestone:** 8
**Type:** deviation
**Decision:** `instruments/m6_attractor.py` implements the H4 metric of preregistration Section 5. Three gaps in the registered sentence are closed on stated principles rather than by choosing what produces a result. The preregistration is frozen and is not edited; this entry is the deviation record CLAUDE.md requires.

**The registered sentence.** "Cluster final states across seeds. A wrong-attractor landing is an item whose final state joins a cluster whose modal answer is not the correct answer. Rate is the fraction of such items."

**Gap 1, "modal answer" is ambiguous.** It can mean the modal TRUE LABEL of a cluster's members or the modal MODEL PREDICTION for them. These coincide for a model that is right about everything and diverge exactly where H4 is interesting, so the choice is not cosmetic and could move the verdict. **Both are computed, both are reported, and neither is named the answer.** House usage favours the label reading, since "answer" in this codebase means the correct composite rather than the model's output, and that reading also makes "wrong ATTRACTOR" describe a basin rather than a mistake. But a reading selected after seeing the numbers is not a registered metric. The instrument therefore publishes `rate_by_label` and `rate_by_prediction` side by side and flags every run where they disagree, so any disagreement is part of the result rather than a choice buried in code.

**Gap 2, "across seeds" cannot mean what it says.** Two independently trained networks hold their representations in different and arbitrarily rotated bases, so final states from different seeds do not live in a shared space. Clustering them jointly would measure basis alignment and not attractor structure. **Clustering is within a seed and rates are aggregated across seeds**, which is the only reading under which the sentence computes anything at all. This is a correction to the registered text rather than an interpretation of it, and it is recorded as such.

**Gap 3, the cluster count is not given.** It is the number of distinct answers present in the evaluated items, one attractor per answer the task can have. **Fixed by the task and not tuned**, which matters because k is precisely the knob that could be turned until H4 came out. It is reported in the output so a reader can see it was not moved.

**One further choice, made for a reason already in the record.** States are normalised and clustered by direction. `RMSNorm` before the head makes the state norm inert, which STATE.md carries as the standing trap in F5, so euclidean distance on raw states would weight a magnitude the head cannot see. Assignment turns out to be scale invariant either way, because a point's own norm cancels out of the argmin against unit centroids; the normalisation bites only in the centre update, where a single large member would otherwise drag the basin onto itself. That is `spherical_mean`, and it is tested where it bites rather than through the clustering, where the mutation was equivalent and slipped through twice.

**Gap 4, found by a positive control and not by reading: "final state" cannot mean the raw residual stream.** The first version clustered the flattened final state and produced a wrong-attractor rate of 0.83 for `famB_curve111_s0`, a run at 1.0000, against 0.84 for a run at 0.60. Close to what random clustering gives at eleven answers, and unable to tell a perfect model from a failing one, which means it was not measuring attractors at all. The stream is dominated by the image and the answer is a small part of it. Clustering `model.readout(s)`, the state as the head actually sees it after the coda, the output norm and the query-token pool, the same two runs read **0.19 and 0.66**, with mean cluster purities of 0.91 and 0.47. `readout` is now exposed on both architectures and `decode` goes through it, so the instrument cannot drift from the real output path, which is the reason `decode` was exposed for M3 in the first place. **Had the control not been run, H4 would have been resolved against a number that was near chance for every arm.**

**What this does not do.** It computes the metric. It does not resolve H4. The verdict needs the rates aggregated over the four registered widths and the exploratory correction applied separately from the registered family, per Section 5, and H4 stays labelled exploratory in the paper whatever it says.

**Consequence:** H4 has had an instrument since 21 September, having had all its compute spent since milestone 6. Nine mutations, all caught, plus two on the centre after the first attempt at testing it produced an equivalent mutation twice.
**Reversible:** the readings are both reported, so a later decision to privilege one costs a sentence and no recomputation.
### D-054. The self-stop was set to a wall that only one of the two submitters has
**Date:** 2026-09-23
**Milestone:** 8
**Type:** deviation
**Decision:** `MAX_HOURS` in `scripts/multirun.pbs` goes from 23.0 to 90.0, is read from `MAX_HOURS_OVERRIDE` rather than `MAX_HOURS`, and is no longer forwarded by `chain_submit`. `scripts/submit.pbs` is deliberately left at 23.0.

**Why 23.0 was ever right.** `submit.pbs` carries `#PBS -l walltime=24:00:00` and its own comment says the value is "about an hour under the 24 hour walltime, so the process stops itself and checkpoints rather than being killed mid-step". That is correct and stays.

**Why it was wrong in the other file.** `multirun.pbs` carries `#PBS -l walltime=168:00:00`, and its `chain_submit` asks for 168 hours too. The 23.0 was copied across without the wall it was measured against. The `famB_curve111_d768`, `d512` and `d4colour_big` arms each need roughly 30 hours, so every one of them stopped clean at 23, chained, and its remainder **re-entered the queue at the back**, where it waited behind the whole slate to do the rest. This is the L-030 failure reached by a second route: not a self-chain after a genuine wall, but a self-chain after a wall we imposed on ourselves at one seventh of the real one.

**The evidence it was actually costing time.** On 2026-09-23 two jobs closed this way within hours of each other. 8803 stopped at step 166462 after 21.7 hours and its remainder requeued as 9444, last in line; 8805 stopped at step 236359 after 22.8 hours and requeued as 9452, also last. Runs finished per rolling 48 hours went 46, then 42, then 26 across three readings, while nothing failed, nothing errored, and `check_queue` stayed clean at 71 of 71.

**Verified before changing it, because too high is the dangerous direction.** `qmgr -c "print queue gpu"` sets `resources_max.mem` and `resources_max.ncpus` and **no** `resources_max.walltime`. The 24:00:00 on the queue is `resources_default.walltime`, which applies only to a job that does not ask, and every job of ours carries `Resource_List.walltime = 168:0` accepted at submit. So 90 leaves 78 hours of margin and is not near anything.

**Why it reads a different variable name.** `chain_submit` used to forward `MAX_HOURS` into its successor, so a value fixed at submit time was copied into the successor and into the successor's successor without end. Every job queued before today carries `MAX_HOURS=23.0` in its PBS `Variable_List`. PBS snapshots the job script at submit time, so those jobs keep running the old copy whatever this file says; their chains, however, submit the file as it stands when the chain fires. Had the new copy read `MAX_HOURS`, the inherited 23.0 would have won in every one of those successors and this change would never have reached a single run. Reading `MAX_HOURS_OVERRIDE` is what makes the fix land, and `chain_submit` no longer forwards the value at all, because the wall belongs to the script and the header, not to whichever job started the chain.

**Also changed.** `analysis/orchestrator.py` had its own `--max-hours` default of 20.0, a third value for the same quantity, and now defaults to 90.0 and emits `MAX_HOURS_OVERRIDE`.

**Consequence:** the 27 jobs in flight keep 23.0 and will each chop once more, since PBS is running the copy it snapshotted. Every successor after that runs at 90 and the heavy arms stop doing their work twice.
**Reversible:** one number in one file. Nothing in `runs/` depends on it, and a run stopped at either value resumes identically.

### D-055. The paper build could only run on the machine without the results
**Date:** 2026-09-23
**Milestone:** 8
**Type:** dependency
**Decision:** `markdown>=3.10` is added to `dependencies` in `pyproject.toml` and installed into the cluster venv, and `scripts/build_paper.py` now finds its figure directory on either machine.

**How it was found.** The full suite on the cluster at `d6f257e` read **704 passed, 2 failed**, both in `tests/test_build_paper.py`, both `ModuleNotFoundError: No module named 'markdown'`. `scripts/build_paper.py` imports it and it was declared **nowhere**: not in `dependencies`, not in the `dev` extra. It is installed on the Windows machine, which is where the script was written and tested, and that is why nobody noticed.

**Why this was worse than two red tests.** The script's own docstring says it exists because "the point of this one is that it reruns every time a result lands". Results land on the cluster. The cluster could not run it. The only machine that could run the build was the one with none of the runs on it.

**Installed with `--no-deps`, deliberately.** CLAUDE.md pins torch at 2.4.1+cu121 because driver 525 with CUDA 12.0 cannot run a cu124 build, and a resolver given room to move is how that pin gets quietly broken. `markdown` has no runtime dependencies on Python 3.11, so nothing is lost by refusing them. Torch was read back afterwards and is still `2.4.1+cu121`.

**The floor is a floor and not a pin.** Local is 3.10.2 and the cluster took 3.10.3. The skew is acceptable here in a way it would not be for anything that touches a number: this renderer **computes nothing**. Every figure in the output comes from a stored metrics file through the plot scripts, and every number in the prose comes from `draft.md`, which is written against the checked generators. A minor version of a markdown renderer can change whitespace in the HTML and cannot change a result. `>=3.10` records what is actually in use on both machines rather than claiming a feature requirement the code does not have.

**The import was only half of it.** With `markdown` installed the build still failed on the cluster, with "no figure directory at ~/figures". The default was `root.parent / "figures"`, which is the Windows layout, where the repository sits beside its figures in a shared parent. **On the cluster the figures are inside the checkout**, and nine of them were sitting there while the build said there were none. `default_figures` now prefers `root / "figures"` and falls back to `root.parent / "figures"`, so both layouts work. Inside wins when both exist, because that copy sits next to the metrics the figures were drawn from, and the tie is broken on purpose rather than by whichever branch ran first.

**Consequence:** `pytest` on the cluster goes from 704 passed and 2 failed to **709 passed, pytest's own exit code 0**, measured at `b600ade` in 16m33s, and `build_paper.py` runs end to end there for the first time: 1.5 MB with 9 figures embedded. Three mutations on the path resolution, all caught.

**This line first said 707, which was arithmetic and not a measurement.** 706 tests existed, the 2 failures became passes, and 3 were added, which is 709. The habit the rest of this project enforces on result numbers, that they come from a generator rather than from someone's head, applies to its own operational numbers too, and 707 was exactly the kind of number that reads as measured and is not. Corrected once the run reported.
**Reversible:** removing the line from `pyproject.toml` and uninstalling restores the previous state exactly. Nothing in `runs/` or any figure depends on it.

### D-056. The test suite moves off the login node onto workq
**Date:** 2026-09-23
**Milestone:** 8
**Type:** operational
**Decision:** `scripts/pytest.pbs` runs the suite as a `workq` job. The login node is no longer the place to run it.

**Why.** The suite is pure CPU: forward passes over tiny tensors and a few chained smoke runs. It needs cores and no GPU, which is exactly the argument `m4_geometry.pbs` and `factor_lens.pbs` already make for putting analysis on `workq`. On the login node it competes with everyone else's interactive work. On 2026-09-23 the load average there was **62.85 with 43 users signed in** and the suite took **12m15s**. That is slow for us and it makes a shared machine worse for 43 other people, which is the same complaint we were about to take to the admins about somebody else.

**Not the gpu queue, for the reason that queue is capped.** `max_run` is five jobs per user there and all five are training. A test job submitted to `gpu` does not fail, it silently takes a slot from a run we are waiting on. `tests/test_pytest_job.py` asserts the header says `workq`, because this is the kind of thing that gets changed by whoever is in a hurry.

**The exit code is the real content of this script.** On 2026-09-23 a suite run was piped to `tail` and reported **exit 0 while pytest had reported 2 failed**, because a shell pipeline returns the status of its last command and `tail` always succeeds. CLAUDE.md's standing rule is that a milestone is done when its command exits 0, so an exit code that cannot be non-zero turns that rule off without saying anything. The script reads pytest's own status out of `PIPESTATUS` and exits with it, prints it in plain text, and prints the FAILED lines when it is non-zero. Two of the four tests exist only to keep that true: one fails if the captured status never reaches `exit`, and one fails on an unconditional `exit 0` anywhere after the run.

**`OMP_NUM_THREADS` is 4 and not 16.** The suite is hundreds of small tensors where thread setup costs more than the work it spreads. The win here is not competing with 43 interactive users, not a thread count. The number is a guess, it is overridable, and it is labelled as a guess in the script.

**Measured, on the same day and the same commit family.** The login node took **16m33s for 709 tests** at load 62.85. Job 9494 on workq took **6m21s for 713 tests**, pytest's own exit code 0. Roughly **2.6 times faster**, and the counts differ only because the workq run included the four tests that guard this script. The speed was a bonus rather than the argument: the reason to move was not competing with 43 interactive users, and it would have been worth doing even if the two had tied.

**A note on reading `qstat`, since it caught me here.** The short `qstat <id>` form prints **Time Use, which is CPU time**, in the column where the wide `qstat -u` form prints elapsed walltime. Job 9494 showed `00:22:39` while its wall clock was 6m21s, because four threads over six minutes is about twenty-five CPU minutes. Reading that as walltime says the job is slower than the login node when it is two and a half times faster, which is a conclusion exactly backwards from the truth. See L-021: read the state back, and know which state you are reading.

**Consequence:** `qsub scripts/pytest.pbs` is the way to run the suite on the cluster, and it is added to the command list in CLAUDE.md. Four mutations on the new tests, all caught.
**Reversible:** delete one file. Nothing depends on it and `pytest` on the login node still works, it is just rude.

### D-057. Confirmation runs are milestone 8 work, and they are held out of the 2x2
**Date:** 2026-09-23
**Milestone:** 8
**Type:** deviation
**Decision:** Confirmation runs are milestone 8, following the frozen preregistration rather than the tracker. Confirmation seeds carry `confirmation: true` in their own config and `escape_rates.arm_of` holds them out of the registered 2x2.

**The numbering was inconsistent in three places and the frozen file wins.** `preregistration.md` Section 1 says a post-hoc statistic "gets a confirmation run at milestone 8", and its Section 0 says IMPLEMENTATION.md "scheduled the confirmation runs at milestone 8". `IMPLEMENTATION.md` Section 8 agrees. Against that, `progress.md`'s milestone table puts confirmation runs at 9, and `claims.md` says "milestone 9 confirmation" in all seven Table 1 rows **while its own status legend says "owes a confirmation run at milestone 8 under Section 1 of the preregistration"**. One file contradicts itself. The preregistration is frozen at `prereg-v1` and cannot be edited, so it is the one the others are wrong against. This also removes a false blocker: these runs are not milestone N+1 and are not gated behind H4 resolving.

**Why a hold-out was needed before a single run could be queued.** The four rows whose arms are complete are rows 1, 2, 3 and 7, and every one of them is a registered 2x2 cell member. Tested rather than assumed:

    m3/famA_d2_d4only.yaml     -> ('D4', 'position')
    m3/famA_d2_d4colour.yaml   -> ('D4', 'identity')
    m3/famA_d2_s3only.yaml     -> ('S3', 'identity')

Five fresh seeds of each would have taken the grid from **70 seeds to 85**, moved position off 15 of 20 and identity off 15 of 50, and changed the Fisher p of 0.0011 that `paper.md` Section 3b was revised around two days earlier under D-052. The grid closed on 2026-09-21 and nothing announces it reopening.

**Pooling would also have confirmed nothing, which is the deeper fault.** A confirmation run has to be independent of the data the claim was discovered in. A seed that joins the cell it is meant to test stops being a test of that number and becomes part of it. So the naive design fails twice at once: it breaks a sealed result AND produces no confirmation. Held out, the comparison is the real one, a rate measured on data nobody had seen against a rate that was.

**The marker lives in the run's own config**, not in a list in the module, so a run carries its status forever and no later edit can reclassify a run that has already finished. `is_confirmation` treats absent, `false` and `null` as ordinary data and **everything else as held out**, because the two errors are not symmetric: a confirmation wrongly admitted changes a sealed headline silently, while an ordinary seed wrongly held out fails the next `--closed` against its registered minimum. Ambiguity therefore resolves toward holding out.

**`--confirm` reports the register and exits non-zero when a confirmation has failed**, which preregistration Section 1 requires. It also refuses to print the word confirmed for a quiet result: five seeds against a rate of three in ten cannot produce a significant p in the agreeing direction, so a cell that comes back quiet has shown only that it is not loudly inconsistent, and the output says that in those words.

**The regression guard is the whole safety argument.** After the change, `escape_rates --closed` on real cluster data must still report **70 finished and 0 unfinished, cell counts 10 / 40 / 10 / 10**, and the rates must still read 15 of 20 and 15 of 50. Anything else means the hold-out changed the thing it exists to protect.

**Consequence:** `configs/m3/famA_d2_{d4only,d4colour,s3only}_confirm.yaml` exist and differ from their baselines by the marker and nothing else, verified by loading both and diffing the parsed dicts. Eighteen tests in `tests/test_confirmation_holdout.py`, four mutations, all caught, including one that restores the old `arm_of` and one that swaps the strict marker test for plain truthiness. **No confirmation run has been queued.**
**Reversible:** deleting the marker from a config returns that run to the grid, and removing the key from `arm_of` restores the previous behaviour exactly. No finished run is affected either way.

### D-058. Training jobs ask for 16gb of host RAM, not 48gb
**Date:** 2026-09-23
**Milestone:** 8
**Type:** operational
**Decision:** `scripts/multirun.pbs` (header and `chain_submit`), `scripts/submit.pbs` and the orchestrator's `--mem` default go from 48gb to 16gb. The `workq` analysis jobs at 64gb are untouched: nothing here measured them.

**Measured, not guessed.** Five running jobs on 2026-09-23 reported `resources_used.mem` of 2.14 to 3.96 GB against 48gb requested; the 3.96 GB job was running three runs at once. The largest single training process held 1.49 GB of RSS, and all twelve of our processes on the node summed to 12.8 GB. The heavier arms have been measured indirectly: `d4colour_ffwd`, among the larger models in the slate, ran three at once in 3.89 GB. Host memory here is dataloader and interpreter; the model lives on the GPU, so parameter count moves it little.

**Why 16 and not something tighter.** No memory enforcement setting is visible at the server level, and the per-node PBS configuration cannot be read from the login node, so a job that exceeds its request has to be assumed killable. **A killed job does not chain**, which is the L-031 failure that destroyed a whole schedule on 2026-09-17. Over-requesting costs a reservation nobody is competing for; under-requesting costs runs. 16gb is four times the worst job measured, and `tests/test_wall_margin.py` refuses anything under three times that peak, so the margin cannot be trimmed later to look efficient.

**A correction to what this entry's motivation used to say.** The over-request was raised earlier the same day as something that "can block other people's jobs from scheduling". It was not. `pbsnodes <cluster>-gpu01` shows about 2 TB available and 668 GB assigned, so memory was never the scarce resource; cores are, at 73 of 96 assigned. This change is hygiene, and it may help backfill in the way D-021 found smaller core requests do, but it must not be described to the admins as us having been in anyone's way.

**The two numbers in `multirun.pbs` must agree.** `chain_submit` passes its own `-l select` so a successor is sized like its parent rather than falling back to the header. A header and chain that disagree mean every chained job silently runs under a different request from the job that started it. Tested.

**Consequence:** new submissions ask for 16gb. **Jobs already queued keep 48gb for up to two generations**: PBS snapshotted their script at submit time, and the old copy's `chain_submit` hardcodes 48gb into the first successor, which then runs the new script and chains at 16gb. That includes the three confirmation jobs 9497 to 9499. Six tests, four mutations, all caught.
**Correction, 24 September: the safety argument above was overstated.** It
says a job over its request "has to be assumed killable". `STATE.md` Section 6
already recorded, from 6 September, that **neither memory nor walltime is
enforced** here: one neighbouring job held 1041 GiB against the 32gb default,
and several ran past ten times their walltime. So the request is a
reservation, not a limit, and the case for 16gb is not reserving 240 GB we
never touch. The wide margin stays because enforcement could be switched on
without notice, and the three-times floor in `tests/test_wall_margin.py`
stands for that reason, but it should be read as insurance rather than as
protection against an observed risk. That section also already said memory
was "nowhere near binding", which is the correction this entry makes about
blocking other users, found two days late by not reading it first.

**Reversible:** one number in three places, all tested for agreement.

### D-059. Timing predictions on the fifteen milestone 8 confirmation seeds
**Date:** 2026-09-24
**Milestone:** 8
**Type:** registration
**Decision:** Three predictions about training dynamics are registered on the
confirmation seeds D-057 created. They are evaluated only by
`python -m loopvision.analysis.escape_timing --registered` at the commit that adds this entry, which exits 2 while any is pending, 1 if any failed and 0
when all held.

**Why these seeds, and why now.** The staircase and the early window were
found on 24 September by looking at training curves of runs that already
existed (`why-experiments.md` Section 2), so they are post-hoc. The fifteen
confirmation seeds were held out of the grid by D-057 and had not started
when this was written: jobs 9497 to 9499 were queued and no confirmation run
directory existed at 10:40 IST. They are the only held-out data about these
dynamics that exists. The author signed this entry on 24 September.

**Void for any arm whose confirmation run started before this entry was
committed.** The cluster was unreachable when it was signed, so that could
not be checked first, and a leave step can happen within the first hour of
a run. The check, at the first read-back after the commit: the modification
time of each confirmation run's `config.yaml`, which is written when the run
starts, against the commit time, both converted to IST. An arm with a seed
that started earlier is excluded from P1 to P3 and reported as excluded,
never as held.

**Definitions**, all constants in `analysis/escape_timing.py` and pinned by a
test so that any change after registration breaks the suite:

- **leave step**: first logged step at which the trailing 2000-step mean of
  train loss is below ln C minus 0.15 and stays below for 10000 steps, or
  until the log ends;
- **solve step**: first logged step at which the trailing 1000-step mean is
  below 0.1 ln C;
- **plateau**: at least 10000 steps with the trailing 2000-step mean within
  0.15 of ln(C/q), for q a proper divisor of C greater than 1, before the
  solve step. For D4 the levels are ln 4 and ln 2; for S3, ln 3 and ln 2.

**Predictions.**

- **P1.** No identity confirmation seed, `famA_d2_d4colour_confirm` s30 to
  s34 or `famA_d2_s3only_confirm` s10 to s14, has a leave step after 40000.
- **P2.** At least 4 of the 5 `famA_d2_d4only_confirm` seeds, s10 to s14,
  show an ln 4 plateau.
- **P3.** No identity confirmation seed shows a plateau at any quotient
  level.

**How the same rules read on the discovery data**, from
`runs/_why_timing_20260924/timing.txt`, so a reader can judge how much each
prediction risks. Latest leave step: 18k in both D4 identity arms, 12k in
S3 identity, and 22k in any depth 2 cold-start arm without a staircase. The
ln 4 plateau: 9 of 10 D4 position seeds. Quotient plateaus in D4 identity or
S3 identity runs: none. **The limit of 40000 and the four of five were both
chosen with those readings in view.** They are margins, stated as such.

**What failure would mean.**

- **P1 fails**: an identity run left chance late, so the window is a rate
  rather than a wall, and H-L's "lock" is wrong as stated.
- **P2 fails**: the staircase did not replicate on fresh seeds, and H-V loses
  the only evidence it has.
- **P3 fails**: an identity run sat on a quotient level, which contradicts
  H-V's claim that the identity renderings offer no accessible quotient.

**Limits, stated before the data.** Five seeds per arm. P1 can only be broken
by a seed that leaves chance, and at the cells' rates about three in ten are
expected to, so P1 may hold with few or no departures to test it;
`--registered` prints how many there were, and **a P1 that holds with zero
departures confirms nothing and must be reported that way.** A plateau is a
loss level, not a proven quotient; E0.3 and E0.4 test that separately. Identity
runs also park on non-quotient plateaus near 1.0 (`why-experiments.md`
observation 4). Those are not quotient levels, P3 does not count them by
construction, and that is disclosed here rather than discovered later. These
predictions test whether the dynamics replicate. They do not test the causal
claim, which is what E1 and E2 exist for.

**Consequence:** no config, job or run changes. The confirmation seeds run
exactly as D-057 queued them.
**Reversible:** a registration is not reversible once its data exists. Before
then it can be withdrawn by an entry saying so.

### D-060. E0.4: which D4 character a model learns first
**Date:** 2026-09-29
**Milestone:** 8
**Type:** registration
**Decision:** Twelve short runs are registered with two predictions about the
first thing a depth 2 model learns about D4. They are evaluated only by
`python -m loopvision.analysis.character_stage --registered` at the commit that
adds this entry, which exits 1 if a prediction failed, 2 while pending,
3 if undetermined and 0 when all held.

**Why this character.** D4 has three characters onto Z2. On the native
J-tetromino, `rotflip` is the glyph's aspect ratio: it stands tall exactly
when rot + flip is even (`tests/test_d4_characters.py`). That was derived from
the rendering on 24 September and written into `why-experiments.md` before any
character readout existed. D-059's P2 has since held: D4 position climbs
through an ln 4 plateau, one character's worth of loss, in 4 of 5 held-out
seeds. This entry asks which character that plateau is. The author signed this
entry on 29 September; no E0.4 run existed then.

**Runs.** Four D4 position seeds, `why_e04_d4only_s100` to `s103` from
`configs/why/e04_d4only.yaml`, and eight D4 identity seeds,
`why_e04_d4colour_s100` to `s107` from `configs/why/e04_d4colour.yaml`. Each
config is its 2x2 base with five changes and no others, pinned by a test:
30000 steps, the learning rate schedule of the 300000 step arm, an eval every
500 steps, `log_characters: true`, and `experiment: E0.4`. The step budget
keeps them out of the sealed 2x2, also tested.

**Definitions**, constants in `analysis/character_stage.py`, pinned by a test:

- **character accuracy**: the share of eval items whose predicted element
  has the same value of that character as the label, logged by training at
  every eval;
- **single-character stage**: one character at or above 0.9 while both others
  are at or below 0.6, held by the same character for two consecutive evals.
  A solved model and an identity run's location plateau both raise all three
  characters together, so neither can pass for one.

**Predictions.**

- **Q1.** Every D4 position seed that shows a single-character stage shows it
  on `rotflip`. A verdict needs at least three such seeds; with fewer, Q1 is
  undetermined, not held.
- **Q2.** No D4 identity seed shows a single-character stage.

**How much Q1 risks.** If each of the three characters were equally likely
to come first, three informative seeds would all land on `rotflip` by chance
with probability 1 in 27, and four with 1 in 81.

**What failure would mean.**

- **Q1 fails**: the first thing D4 position learns is not the character its
  silhouette shows, and the aspect-ratio account of the staircase is wrong.
- **Q2 fails**: an identity rendering does give a one-character foothold in
  some seeds, against H-V's claim that it offers none.

**Limits, stated before the data.** 30000 steps is enough for the stage:
every D4 position seed that left chance did so by 11k, the ten 2x2 seeds by
7k and the four confirmation seeds that left by 11k
(`runs/_why_timing_20260929/timing.txt`, which also stores D-059's final
verdicts). But a seed that never leaves chance shows nothing, and one of
five D4 position confirmation seeds never did. Q2 can only be broken by an
identity seed that leaves chance; the output of `--registered` counts them,
and **a Q2 that holds with none leaving chance confirms nothing and is
reported that way.** The readout says what a model gets right, not how it
computes it. Evaluating every 500 steps draws more random initial states than
the 300000 step arms did, so these runs match those arms in distribution and
not seed for seed.

**Consequence:** `train/cli.py` gains the character readout. It is off unless
a config sets `log_characters`, the key is absent from `DEFAULTS`, and tests
pin that a run which does not ask stores the same config and writes the same
metrics as before. 16 deliberate breakages of the new code, each caught.
**Reversible:** before its runs exist, by an entry withdrawing it.

### D-061. Milestone 8 confirmation of claims.md rows 4 to 6
**Date:** 2026-09-29
**Milestone:** 8
**Type:** registration
**Decision:** Five held-out seeds for each of the three D4 identity curriculum
arms behind `claims.md` rows 4, 5 and 6, each seed with its own fresh depth 1
donor, and five checks evaluated only by
`python -m loopvision.analysis.curriculum_confirm` at the commit that adds this
entry, which exits 1 if a check failed, 2 while any is pending and 0 otherwise.

**Why now.** D-057 held these rows until their arms stopped gaining seeds.
All three reached ten by 27 September: the same-task donor arm escaped in
10 of 10, the same-group donor on the other substrate in 9 of 10, and the
wrong-group donor in 1 of 10 (`runs/_why_timing_20260929/timing.txt`). The author
signed this entry on 29 September; no donor or recipient run existed then.

**Runs.** Stage one, fifteen donors: `famA_d1_d4colour_s10` to `s14`,
`famA_d1_d4only_s10` to `s14` and `famA_d1_s3only_s10` to `s14`, from the same
depth 1 configs every existing curriculum seed's donor used. They carry no
claim. Stage two, fifteen recipients: `famA_d2_d4colour_curr_confirm_s10` to
`s14`, `famA_d2_d4colour_currctrl_confirm_s10` to `s14` and
`famA_d2_d4colour_curr_s3donor_confirm_s10` to `s14`, each started from the
donor of its own seed. Each recipient config is its arm's template with the
donor line changed and `confirmation: true` added, and nothing else, which
a test pins, and no two share a donor. Recipients are held on PBS `afterany`
of their donors' job, and `cli.py` refuses to start from a donor without
DONE whatever the scheduler does.

**Checks.**

- **R4, R5, R6.** Each held-out arm against its sealed arm fails if the two
  sided Fisher p is below 0.05, the rule `escape_rates --confirm` applies to
  rows 1 to 3 and 7. Not failing is reported as "not contradicted" and never
  as confirmed: five seeds can show a disagreement, not an agreement.
- **C1.** On held-out seeds alone, the same-task donor arm escapes more often
  than the wrong-group donor arm, with two sided Fisher p below 0.05. This is
  row 6's claim itself: the rescue is not generic warm starting.
- **C2.** The same for the same-group donor on the other substrate against
  the wrong-group donor: what transfers is the group, not the substrate
  (`findings.md`, "The D4 curriculum gets its control").

**How much C1 and C2 risk.** At five seeds a side, p below 0.05 needs near
perfect separation: 5 of 5 against at most 1 of 5, or 4 of 5 against 0 of 5.
If the held-out arms behave like the sealed ones, C1 should hold about nine
times in ten and C2 about three times in four, computed by the binomial from
the sealed rates. So a failure of C2 would be unremarkable on its own and a
failure of C1 would not.

**What failure would mean.**

- **R4, R5 or R6 fails**: that row's rate does not replicate on fresh seeds,
  and the row is reported as a failed confirmation.
- **C1 fails**: the difference between the right donor and the wrong-group
  donor is not confirmed on held-out seeds, so row 6's reading, that the
  rescue is specific rather than generic warm starting, stays exploratory.
- **C2 fails**: "the group transfers, not the substrate" stays exploratory.

**Limits, stated before the data.** The S3 side is not tested: its
wrong-group donor rescued in 5 of 10 seeds, every one at about 2k steps, which
bears on how C2 should be read and is reported separately rather than folded
in here. A donor's own accuracy is not checked, only its DONE; every existing
depth 1 donor solved its task.

**Consequence:** 30 runs. Neither depth 1 nor curriculum configs are 2x2
baselines, so none of them can enter the sealed grid; `escape_rates --closed`
is read back after submission all the same.
**Reversible:** before its runs exist, by an entry withdrawing it.

**Correction, 29 September, before any recipient existed: the recipients are
not held on `afterany`.** Found while submitting. `multirun.pbs` chains any run
that ends without DONE straight away (lines 293 to 302), and `afterany` fires
on any exit of the donor job, including a GPU famine requeue. A recipient that
started before its donor finished would be refused by `cli.py`, chain, start
again and be refused again for as long as the donor trained: the L-031 spin.
So the fifteen donors were submitted alone as jobs 9933 to 9935, and the
fifteen recipients are submitted by hand once every donor carries DONE,
checked by listing the fifteen DONE files first. No prediction, threshold,
seed or config changes.

### D-062. E1 and E2: does a visible quotient cause the staircase?
**Date:** 2026-09-29
**Milestone:** 8
**Type:** registration
**Decision:** Eighty runs, twenty seeds (`s200` to `s219`) in each of four new
renderings of D4, and six contrasts, evaluated only by
`python -m loopvision.analysis.why_e1e2` at the commit that adds this entry,
which exits 1 if a contrast failed, 2 while pending, 3 if undetermined and
0 when all held.

**Why.** H-V says D4 position learns reliably because its rendering makes a
quotient of the group visible, and the model climbs through it. It was found
by looking at training curves, so it is post-hoc. D-059 then held on
held-out seeds in the two parts that describe it: D4 position climbs through
ln 4 (P2) and identity runs never sit on a quotient level (P3). That is the
pattern replicating, not its cause. These runs change only the rendering, and
test the cause. The author signed this entry on 29 September; no E1 or E2
run existed then.

**The renderings**, each with its properties pinned in
`tests/test_why_substrates.py`, and each keeping the group, the task, the
labels and the order-blind ceiling:

- **E1a `d4_square`**: D4 moving an F-pentomino. Still a rigid motion, but
  its bounding box is 3 by 3 in every orientation, so aspect ratio, which is
  `rotflip` on the native J, shows nothing.
- **E1b `d4_scrambled`**: exactly the eight native J tiles, each standing for
  a different element than its own. No tile feature equals or opposes any
  character, and the scramble is not an affine map of D4, so composing
  pictures no longer composes elements.
- **E2a `d4_colour_cue`**: the `d4_colour` block, a fixed grey anchor pixel,
  and a grey cue pixel present exactly when `rotflip` is odd. The cue
  composes: the three cue bits of a problem carry one bit about its label.
- **E2b `d4_colour_decoy`**: identical except the pixel marks rotation index
  2 or 3, a balanced partition that agrees with every character on exactly
  four of eight elements and does not compose. Its three cue bits carry under
  a twentieth of a bit about the label.

**Contrasts.** Each is a two sided Fisher test that holds only at p below
0.05 in the predicted direction. "Climbed" is the share of runs that left
chance which showed an ln 4 plateau (`escape_timing`, as in D-059). The
sealed arm is `famA_d2_d4only`, 9 of 10 escaped and 9 of 10 climbed
(`runs/_why_timing_20260929/timing.txt`).

| | measure | predicted lower | predicted higher |
|---|---|---|---|
| V1 | escaped | E1a | sealed D4 position |
| V2 | climbed | E1a | sealed D4 position |
| V3 | escaped | E1b | sealed D4 position |
| V4 | climbed | E1b | sealed D4 position |
| V5 | escaped | E2b | E2a |
| V6 | climbed | E2b | E2a |

**What each account predicts, written before the runs:**

| | V1 | V3 | V5 |
|---|---|---|---|
| **H-V** visible quotient | held | held | held |
| **R1** spatial bias: rigid motion is what the architecture learns | failed | held | failed |
| **R3** the tile set itself | held | failed | failed |
| **R2** salience: distinct silhouettes help, and so does any extra visible bit | failed | failed | failed |

The four rows differ, so the three escape contrasts together name which
account the data fit.

**The verdict on H-V is set now.** Supported only if V1, V3 and V5 all hold.
If V5 fails, H-V is falsified: a visible, composing quotient did not make
D4 identity learnable. If V5 holds but V1 or V3 fails, H-V is not supported
and the rival whose row matches is named. V2, V4 and V6 say whether the
mechanism is the staircase; an undetermined one is reported as such.

**How much each risks.** V1 and V3 hold only if the arm escapes in at most
10 of 20: 98 in 100 if it behaves like an identity cell, at a rate near 0.3,
and 1 in 100 or fewer if it behaves like D4 position. V5 holds 98 in 100 if
the cue makes D4 identity as learnable as D4 position, 73 in 100 if E2a
reaches three in four against E2b's three in ten, and 16 in 100 if E2a only
reaches one in two. **So a failed V5 is strong evidence only against a large
effect**, and is reported with that sentence.

**Limits, stated before the data.** E2 is read against E2b and not against the
sealed `d4colour` arm, because the anchor pixel changes the tiles. On the
F-pentomino some other character could still be visible; every run logs
all three characters at each eval, so a stair in E1a is reported with the
character it is on, and does not rescue H-V. The runs evaluate every 1000
steps where the sealed arm evaluated every 10000, which changes random number
consumption, so the comparison is between distributions, not matched seeds.

**Consequence:** `groups.py` gains four substrates, the existing two
unchanged and tested so; configs `configs/why/e1a_square.yaml`,
`e1b_scrambled.yaml`, `e2a_cue.yaml` and `e2b_decoy.yaml`, each its 2x2 base
with four changes, pinned; none can enter the sealed 2x2. 13 deliberate
breakages each caught, one only after a test was changed to read the scramble
from the pixels. Eighty 300000 step runs, about a week of queue.
**Reversible:** before its runs exist, by an entry withdrawing it.

### D-063. GPU selection counts every process on a card, and a live claim is never cleared for its age
**Date:** 2026-10-06
**Milestone:** 8
**Type:** operational
**Decision:** three changes to `scripts/pick_gpus.sh`, the picker `multirun.pbs` uses.

1. **A claim is cleared only when qstat says its job is gone**, or when qstat fails. Age is a fallback for a host with no qstat at all, and its threshold is 10200 minutes, past the 168 hour wall `multirun.pbs` asks for.
2. **Cards in the same 15 point band of utilisation are ranked by how many processes they carry**, every user's, read from `nvidia-smi --query-compute-apps`, and only then by utilisation and index.
3. **Runs stacked past the first pass go to the card with the fewest processes**, counting every user's and this job's own picks, among cards with free memory for one more. Before, the second pass counted only this job's runs on a card.

**Reason, measured on the node at 04:57 IST on 2026-10-06.** Seven of eight cards read 100 percent utilisation, so the sort on utilisation fell through to index order and every job started on GPU 0. GPU 0 carried 11 processes, six of them ours, and GPU 1 carried 10, while GPU 3 carried 3 at 45 percent. The slowest seeds of D-062, the ones that set its ETA, were on GPUs 0 and 2. Separately, any claim older than 1500 minutes was cleared whatever qstat said, under a comment that the queue wall was 24 hours, which stopped being true when `multirun.pbs` moved to 168. Jobs 9947 and 9948, 42 hours in and training on GPUs 0 to 2, held no claim, and job 9950 claimed GPUs 0 and 1 on top of them when it started.

**What it does not change.** Which runs exist, their configs, seeds or results: a card choice changes only speed. Running jobs keep the cards they launched on. Queued jobs read `scripts/` fresh at launch, so the D-061 recipients, jobs 10341 to 10343, are the first to use it.

**Checked against the real node.** The three `nvidia-smi` readings from the GPU node, taken later the same morning, were replayed through the new script locally, with claims in a temporary directory: bus ids map, process counts of 5 to 8 per card come out, and the three cards under the 10240 MiB floor stay out.

**Consequence:** six tests in `tests/test_pick_gpus.py`, one pinning the age backstop above every walltime in `multirun.pbs` the way `test_wall_margin.py` pins the self-stop. Against the old script four fail and the two controls pass; three single mutations of the new one, dropping the process count from the sort, from the stacking load, and restoring the 1500 minute rule, are each caught by the test written for it. 31 passed in the file.
**Reversible:** yes, one script.

### D-064. A finished run keeps only its latest checkpoint, and a failed prune is loud
**Date:** 2026-10-11
**Milestone:** 8
**Type:** operational
**Decision:** `train/checkpoint.py` gains `prune_to_latest`, which `train/cli.py` and `train/smoke.py` call right after DONE is written, so a finished run keeps exactly the checkpoint `find_latest_checkpoint` returns. `_prune_checkpoints`, the interval prune that keeps two while training, now prints `WARNING checkpoint prune failed` instead of swallowing the error. The watcher reports `runs/` disk use and how many runs hold more than two checkpoints, and headlines either going wrong.

**Reason.** The author reported that the project's disk reached 6 TB once and had to be cleared by hand. Keeping the last two had existed since the first commit, so checkpoints alone could only reach that if pruning failed: a run checkpoints every 30 minutes, about 132 MB a time, so a run whose prunes all fail holds about 21 GB after 80 hours, and about 300 such runs make 6 TB. The old prune swallowed every `OSError`, so that failure would have been invisible. It cannot be confirmed now because everything was cleared.

**Done the same day, by hand, at the author's request.** Every reader of a checkpoint, the instruments, the probes and `init_from`, loads only the latest. So the older of the two checkpoints in each of the 509 finished runs was deleted, but only where the run carried DONE and its `LATEST` pointer named a file that existed: 504 files, 81.2 GB, every path logged in `runs/_cleanup_20261011/deleted_checkpoints.txt`. Read back: every run holds exactly its `LATEST` checkpoint, 509 files, `runs/` at 77 GB.

**Consequence:** seven tests in `tests/test_checkpoint_prune.py`, and the smoke trainer's end-to-end test now expects one checkpoint after DONE. Five deliberate breaks, each caught by the test written for it: the smoke trainer not pruning, the prune keeping the wrong file, the interval prune silent again, `cli.py` not pruning, and `cli.py` pruning before DONE.
**Reversible:** the code, yes; the deleted checkpoints, no, and nothing reads them.

### Shared rules for D-065 to D-070
**Date:** 2026-10-11

**Shared rules, for every entry below.**
- New runs never enter the sealed 2x2. The 2x2's membership test is extended
  to compare learning rate, weight decay, warmup, the early-stop setting and
  the pixel scale too, with a test proving the grid still reads 70 of 70 with
  identical cells. Its `escape_rates --closed` read-back must still pass after
  every submission.
- Seeds start at numbers no existing run uses.
- Where an entry has more than one confirmatory test, Holm is applied across
  them, and the entry says so in advance.
- "Escaped" means final accuracy at or above 0.95, as everywhere else.

### D-065. Early stop for the new runs, family A only
**Date:** 2026-10-11
**Milestone:** 8
**Type:** registration (protocol)
**Signed by the author on 2026-10-11, before any of its runs existed**, from the draft `Registrations-draft-2026-10-11.md`, in answer to the professor's review of 10 October. The shared rules above apply to it.


**Rule.** A run stops when the overall validation accuracy at its evaluation
k has been at or above 0.95 at every evaluation for the last 20000 steps. It
then runs its normal final evaluation, writes `summary.json` and DONE, and
prunes to one checkpoint. Configs turn it on with `early_stop_tau: 0.95` and
`early_stop_window: 20000`; absent means off, so no existing config changes.

**Evidence, a design check and not a result**
(`runs/_earlystop_check_20261011/`, 11 October):
- Of 126 depth 2 family A runs and 47 D-062 runs that ever reached 0.95, none
  finished below it, and this rule would have stopped none of them wrongly.
- It would have stopped them at a median of about 40000 steps instead of
  300000.
- Family B is the exception (6 of 85), so the rule is limited to family A.

**Scope.**
- On for every new recipient and every new cold-start run in D-066 to D-069.
- **Off for donors**, which run their full 150000 steps so they match the
  donors already used.

**What it cannot change.** A run that never reaches 0.95 runs its full 300000
steps, exactly as before, so failures and late escapes are measured as
before. A solved run is classified at its stop. The design check says that
gives the same classification it would have had at 300000.

### D-066. C1: the substrate asymmetry across learning rate and weight decay
**Date:** 2026-10-11
**Milestone:** 8
**Type:** registration
**Signed by the author on 2026-10-11, before any of its runs existed**, from the draft `Registrations-draft-2026-10-11.md`, in answer to the professor's review of 10 October. The shared rules above apply to it.


**Runs.**
- Two substrates: D4 acting on position, from the `famA_d2_d4only` config, and
  D4 acting on appearance, from the `famA_d2_d4colour` config.
- Three learning rates, 1e-4, 3e-4 and 1e-3, crossed with two weight decays,
  0.05 and 0.5. That makes six cells, and 3e-4 with 0.05 is the setting every
  existing run used.
- Five seeds per cell, seeds 300 to 304, for 60 runs. Everything else is
  unchanged from the base configs, with early stop on.

**Predictions, with Holm across G1 and G2.**
- **G1.** Pooled over the six cells, position escapes more often than
  appearance: a Cochran-Mantel-Haenszel test stratified by cell, two sided
  p below 0.05, with the common odds ratio above 1.
- **G2.** In no cell does appearance escape significantly more often than
  position. G2 fails if any cell's Fisher test, two sided and Holm-corrected
  across the six cells, shows that reversal.

**What failure would mean.**
- **G1 fails:** the asymmetry is not shown to survive a change of
  optimisation settings, and the paper must restrict its claim to the one
  setting used.
- **G2 fails:** at some setting the asymmetry reverses. That would be a
  finding in its own right and is reported as one.

**Power.** If the rates resemble the sealed ones, about 0.75 against 0.30,
then G1 holds about 95 times in 100 at 30 seeds a side.

**Module.** A new `analysis/hp_grid.py` evaluates this registration. Its
Mantel-Haenszel test is checked against a published worked example, and its
tests are broken on purpose to confirm they fail.

### D-067. C2: does the asymmetry survive tiles that span several patches?
**Date:** 2026-10-11
**Milestone:** 8
**Type:** registration
**Signed by the author on 2026-10-11, before any of its runs existed**, from the draft `Registrations-draft-2026-10-11.md`, in answer to the professor's review of 10 October. The shared rules above apply to it.


**Runs.**
- Each rendered image is upsampled 2× by nearest neighbour to 64 pixels, with
  a new config key `pixel_scale: 2`. The patch stays 4×4, so every glyph now
  covers 2×2 patches and no patch holds a whole glyph.
- D4 position and D4 appearance, 10 seeds each, seeds 400 to 409, for 20 runs,
  with early stop on. Everything else is unchanged.
- There are 256 image tokens instead of 64, about 4.5 times the compute per
  step.

**Prediction.**
- **T1.** Position escapes more often than appearance: Fisher two sided p
  below 0.05, in that direction.

**What failure would mean.** At the original scale, the asymmetry may be a
property of one glyph per patch, that is, of the tokenizer. The paper then
says so and restricts its claim.

**Reported descriptively, not tested.** Each arm's rate against its
original-scale counterpart. That comparison crosses experiments, so it is
reported as such.

### D-068. C3: more E2a and E2b seeds, and the trap registered for the first time
**Date:** 2026-10-11
**Milestone:** 8
**Type:** registration
**Signed by the author on 2026-10-11, before any of its runs existed**, from the draft `Registrations-draft-2026-10-11.md`, in answer to the professor's review of 10 October. The shared rules above apply to it.


**Runs.**
- The D-062 configs for E2a, the composing cue, and E2b, the non-composing
  pixel, unchanged except for early stop and the seeds.
- Ten seeds each, seeds 220 to 229, for 20 runs.

**Predictions, on the new seeds alone, with Holm across T2 and T3.**
- **T2.** E2b escapes more often than E2a: Fisher two sided p below 0.05, in
  that direction. This replicates V5's reversal.
- **T3, the trap.** Of the new E2a runs that leave chance, at least half
  finish with accuracy between 0.24 and 0.26, the one-bit score. "Leaves
  chance" is defined by `escape_timing` exactly as in D-062. If fewer than four
  runs leave chance, T3 is undetermined.

**What failure would mean.**
- **T2 fails:** the reversal does not replicate on fresh seeds.
- **T3 fails:** the post-hoc trap reading does not hold up. Section 5.6 then
  loses its strongest claim, and says so.

**Reported descriptively, not tested.** The new seeds pooled with D-062's
twenty.

### D-069. C4: more held-out donor seeds, for both specificity checks
**Date:** 2026-10-11
**Milestone:** 8
**Type:** registration
**Signed by the author on 2026-10-11, before any of its runs existed**, from the draft `Registrations-draft-2026-10-11.md`, in answer to the professor's review of 10 October. The shared rules above apply to it.


**Runs.**
- **Donors:** 30 fresh depth 1 donors, seeds 15 to 24, ten each of
  `famA_d1_d4colour`, `famA_d1_d4only` and `famA_d1_s3only`. Each trains its
  full 150000 steps, with early stop off.
- **Recipients:** 30, ten each of the same-task, other-substrate and
  wrong-group arms. Each starts from the donor of its own seed, carries
  `confirmation: true`, and has early stop on.
- **Ordering:** the recipients are submitted by hand once every donor carries
  DONE, for the reason D-061's correction gives.
- **Total:** 60 runs.

**Predictions, on the new seeds alone, with Holm across K1 and K2.**
- **K1.** The same-task donor escapes more often than the wrong-group donor:
  Fisher two sided p below 0.05.
- **K2.** The other-substrate donor escapes more often than the wrong-group
  donor: Fisher two sided p below 0.05.

**What failure would mean.**
- **K1 fails:** the rescue's specificity rests on D-061's narrow margin alone,
  and the paper says so.
- **K2 fails:** "the group transfers, not the substrate" does too.

**Reported descriptively, not tested.** Pooled with D-061's held-out seeds,
which gives 15 seeds per arm.

**Size chosen by the author: the full 60 runs**, so both K1 and K2 get new
data. The cheaper 40 run option without the other-substrate arm was declined.

### D-070. A tile-distinguishability measure, with its prediction, before it is computed
**Date:** 2026-10-11
**Milestone:** 8
**Type:** registration
**Signed by the author on 2026-10-11, before any of its runs existed**, from the draft `Registrations-draft-2026-10-11.md`, in answer to the professor's review of 10 October. The shared rules below apply to it.

**Measure, fixed now.**
- **M_pix:** for a rendering, the mean over all pairs of distinct group
  elements of the Euclidean distance between their rendered tiles, in pixel
  values scaled to 0 to 1, using `render.render_element`.
- **M_emb:** the same distance after the model's patch-embedding convolution at
  initialisation, averaged over 20 seeded initialisations.
- Both are also reported as the minimum over pairs, the hardest pair, as a
  secondary.
- A new tested module, `analysis/tile_distinguishability.py`, computes them. It
  is tested on synthetic tiles only, so no real value exists before this entry
  is signed.

**Renderings.**
- **The seven non-trap renderings:** D4 position, D4 appearance, E1a, E1b, E2b,
  S3 position and S3 appearance.
- **E2a, the composing cue,** is reported but excluded from the test, because
  D-062 already established that salience does not account for it.
- E1b uses the same tiles as D4 position, so the two tie, and ties take their
  average rank.

**Prediction, with M_pix primary and M_emb secondary.**
- **S1.** Across the seven renderings, the Spearman correlation between M_pix
  and the escape rate is at least 0.6.
- **What decides the verdict.** S1 holds if the correlation is at least 0.6,
  fails if it is at or below 0, and is undetermined in between.
- **The escape rates** are the existing sealed and D-062 rates, read from
  their stored files.

**What failure would mean.** Salience, measured this way, does not order
learnability. The paper's salience paragraph then has to come out.

### D-071. One job reads the cards at a time, the scale 2 floor covers what nvidia-smi sees, and every launched run is checked
**Date:** 2026-10-12
**Milestone:** 8
**Type:** operational
**Decision:** three changes, made on the author's go-ahead after nine runs died at launch on 11 October. L-036 is the account of how that went unseen for eleven hours.

1. **`scripts/pick_gpus.sh` takes a lease before it reads `nvidia-smi`.** The lease is a directory, `.pick-lease` beside the claims, made with `mkdir` and holding the second it was taken. It counts for 300 seconds and then stops counting; nothing releases it. A job that finds a live lease waits it out and only then reads the cards. A job that launches nothing gives its lease back at once. A job that has waited five settles goes ahead without one and says so on stderr.
2. **The pixel scale 2 memory floor in `train/mem_floor.py` goes from 30720 to 32768 MiB**, and the fallback in `multirun.pbs` follows it.
3. **`check_queue --launched`** reads, for every run of every running job, the log that job wrote for it, and exits 1 if any run is dead or has written nothing for an hour.

**Reason, from `runs/_launch_oom_20261011/` on the cluster.** Jobs 10952 to 10956 started between 18:28:51 and 18:29:04 on 11 October, 13 seconds apart, carrying 25 runs. Nine wrote a CUDA out of memory traceback between 18:29:58 and 18:30:51 and nothing else: `c2_app_x2` seeds 405 to 408, `famA_d1_d4colour` seeds 15, 18, 22 and 24, and `famA_d1_d4only` seed 19. Each job read free memory before the runs of the others had loaded. The picker's first pass is safe against that, because it places one run per card it has claimed with `mkdir`. The stacking pass is not, and cannot use the claims, since stacking means sharing a card: it divides each card's free memory by the floor, and five jobs divided the same free memory. Ten of the 25 runs were pixel scale 2, which hold about 30 GB each where every earlier run of this project held under 9, so an over-commitment that used to cost a slow card cost the runs.

The settle time is measured from the same event. A scale 2 run of job 10952 held 26.58 GiB 105 seconds after its job started and 29.38 GiB at 120 seconds, with 25 processes importing at once. 300 seconds is two and a half times that. At most five jobs start together, the queue's running cap, so the last of them waits 20 minutes.

The floor was set from the wrong instrument. 30720 came from torch's own peak, 28.3 GB, measured by `scripts/probe_step_cost.py`. `nvidia-smi` on the node at 06:04 on 12 October, eleven hours into the runs, read 30060 to 30164 MiB for each of the four scale 2 runs and 8728 to 8918 MiB for each of the eight scale 1 runs, because the CUDA context and the allocator's reserve sit on top of the tensors. The picker divides what `nvidia-smi` reports, so the old floor cleared the real figure by 556 MiB, under two percent. 32768 clears it by 2604 MiB, 8.6 percent. The scale 1 floor, 10240, clears its figure by 14.8 percent and stays.

**Why all three live in files read at launch.** A queued job runs the `multirun.pbs` it was submitted with and reads `scripts/` and the Python package fresh when it starts. Jobs 10957 to 10977 were queued before this, so a change in `multirun.pbs` would not reach them. The lease is entirely inside `pick_gpus.sh` for that reason, and the floor is read from `mem_floor.py` by the `multirun.pbs` they already carry.

**What it does not change.** Which runs exist, their configs, seeds or results. The nine dead runs wrote no metrics and no checkpoint, so each starts from step 0 with the config and seed it was submitted with when its job chains it. They are the registered runs and not replacements. Running jobs keep the cards they hold.

**What it costs.** A job that starts within five minutes of another waits out the rest of those five minutes. And the higher floor places fewer scale 2 runs on a partly used card: replayed on the 06:04 reading, a card with 61710 MiB free takes one scale 2 run where the old floor gave it two, which would have left 1382 MiB on a card two other users were also on. Five scale 2 runs get four places there instead of five, and the fifth chains.

**What it does not fix, said plainly.**
- Another user's process that grows after we have stacked beside it. Nothing of ours can see that coming.
- `pick_gpu.sh`, the single-run picker that `submit.pbs` uses, takes no lease. Nothing is queued through it.
- A claim stays with a job after its run on that card has finished, so the first pass of a later job skips a card that may by then be the emptiest on the node. The stacking pass still reaches it.
- Taking over a spent lease is not perfectly atomic. Two waiters that find the same spent lease within a few milliseconds of each other could both go on to hold one, and the cost is the old behaviour for that pair.
- The dead runs still re-enter at the back of the queue. They are moved by hand with `qorder`, on the author's go-ahead of 12 October, when their successors appear.

**Checked against the real node.** The stored 06:04 reading, eight cards with their free memory, utilisation and process counts, and the seven claims live at the time, was replayed locally through the new script under `set -uo pipefail`, the mode `multirun.pbs` sources it in. A five-run scale 1 job is given cards 4, 3, 5, 5 and 1. A five-run scale 2 job is given 4, 5, 1 and 2 and told there is room for four of five. A second job started straight after the first prints that the first chose its cards seconds ago, waits, and only then picks.

**Consequence:** nine tests in `tests/test_pick_lease.py`, one of which runs two real pickers at once against a card whose memory shows a second after a run is placed; eight in `tests/test_check_launched.py`; `tests/test_mem_floor.py` now holds the floors to five percent over the stored `nvidia-smi` figures and reads the fallback in `multirun.pbs` against the largest floor instead of a typed number. Deliberate breaks: eight of the lease (no wait, the lease kept in a famine, no cap on the wait, a lease that never ends, the settle at 60, the lease not dot-named, the cards read before the lease, the cap at two settles), three of the floor and its fallbacks, and nine of the launched check. Every one fails a test. One did not at first: reading any job's log for a run, not only the current job's, passed until the test gained the case of a run this job never launched whose earlier job had left a traceback.
**Reversible:** yes. Two scripts and one constant.

**Added the same morning, from the first real reading.** On the cluster at 06:13 `check_queue --launched` named exactly the nine dead runs and exited 1. It also said six running jobs carried the 25 runs when five did: the sixth was the test suite, job 11060, which carries no runs. A running job with no runs is no longer counted, with one more test and one more deliberate break, so that check now has nine tests and ten breaks. The suite read 1217 passed, pytest's own exit code 0, on the cluster at `37e3bfc`. The picker was then run on the GPU node itself, under `set -uo pipefail`, with its claims and lease in a temporary directory so that nothing real was touched: the first job picked five cards at once, and a second started behind it printed that the first had chosen its cards six seconds ago, waited out the lease, and only then read the cards.

### D-072. Memory counts from the moment it is promised: a ledger in the picker, after the lease proved to be a guess about time
**Date:** 2026-10-13
**Milestone:** 8
**Type:** operational
**Decision:** `scripts/pick_gpus.sh` keeps a ledger of promises, `.promised/` beside the claims. A job that places n runs on a card writes one line for that card: the second, the card, what its own arithmetic leaves there (the free memory it counted less n floors) and its job id. Every job reads the ledger after it holds the lease and before it reads `nvidia-smi`, and counts each card at the smaller of what the card shows and the smallest figure a job ahead left on it. A promise counts for 1800 seconds and then stops; nothing has to release it. A job that ends removes its own, in the trap that releases its claims. The lease of D-071 is unchanged. Made on the author's go-ahead of 13 October.

**Reason, read off the cluster on 13 October.** The three donor jobs of the first wave ended together and jobs 10957, 10958 and 10959 started at 22:22:35, 22:22:51 and 22:23:31 on 12 October. This was the first time several jobs started together under D-071.
- The lease did what it was written to do. The process start times of their runs on the node are 22:22:40, 22:27:45 and 22:32:44, five minutes apart.
- It was not enough. `famA_d1_d4only_s20` of job 10957 wrote its `config.yaml` at 22:29:10, 390 seconds after it was launched. `famA_d1_s3only_s19` of job 10958 wrote its at 22:29:25, 100 seconds after it was launched. So the second job read the cards at 22:27:45, before any run of the first had touched one.
- Both runs were on GPU 3, which carried another user's process at 64.31 GiB. Their logs give the card at 14352 MiB free and at 13796 MiB free, room for one run at the 10240 floor. Each job had counted that room for itself. Both died of CUDA out of memory, their logs last written at 22:31:01 and 22:30:57, holding 7.02 and 7.36 GiB with the card full.
- Job 10959 put `famA_d1_s3only_s20` on the same card at 22:32:44, after both had died. It read 14352 MiB free, wrote its `config.yaml` 19 seconds after launch, and finished its 150000 steps.

The readings are on the cluster in `runs/_launch_oom_20261012/`. Why the first job's runs took 390 seconds where the slowest of 11 October took 120 was not measured. My reading is a cold file cache: nothing had started Python on that node since the first wave 28 hours earlier, and the same step took the second job 100 seconds and the third 19. That is a reading and not a finding.

**Why a ledger and not a longer lease.** The settle time was sized at two and a half times the slowest launch seen, and the next day the slowest launch seen was more than three times longer. Any number put there is the same kind of guess. The ledger does not depend on how long a launch takes, only on its lasting less than 1800 seconds, which is three and a half times the 390 seconds to reach the card plus the 120 to fill it.

**Why the smaller of the two figures, and not the reading less the promise.** Subtracting the promise from what the card shows is right while the runs ahead have not loaded and counts them twice once they have. A job starting ten minutes after another would then see too little room, launch fewer runs than fit, and chain the rest to the back of the queue. The figure a job left is right in both states: runs that have loaded take less than a floor each (8728 to 8918 MiB against 10240, 30060 to 30164 against 32768), so a card never has less than the job ahead left on it by its own doing.

**What it costs.** Until a promise lapses the card cannot look fuller than the job ahead left it. Memory another user frees on that card goes unseen for up to half an hour, and so does the room left by a run of ours that dies or finishes inside that half hour. That is some packing lost and never a run.

**What it does not fix, said plainly.**
- Another user's process that grows on a card between our choosing it and our runs loading. The ledger takes the reading at its word for other people's memory.
- A job whose own pick takes longer than the lease. The job behind would then take the lease over and read the ledger before the first had written to it. A pick takes about five seconds on the node against a lease of 300.
- A job that waits past the lease cap and goes ahead without it reads the ledger without holding the lease.
- `pick_gpu.sh`, the single-run picker, keeps no ledger. Nothing is queued through it.
- The two runs that died restart only when jobs 10957 and 10958 end and chain them to the back of the queue.

**Why it lives in `pick_gpus.sh`.** For the reason the lease does: jobs 10961 to 10977 and the three successors were queued before this and run the `multirun.pbs` they were submitted with, which reads this script fresh.

**Consequence:** fourteen tests in `tests/test_pick_promise.py`. They hold the cards still, so that no run ever appears to load, and set the lease to zero, so nothing waits: the arithmetic has to be right with no help from time. The first is the night of 12 October on one card, 14352 MiB read and 4112 counted. Twenty-one deliberate breaks of the ledger each fail a test, two of them only after the test for a damaged ledger file was strengthened: it had written a line wrong in every field, which is thrown out at the first, so a picker that read a missing figure as zero and blacklisted the card passed it. The one test of D-071 that runs two pickers at once had its lease raised from 4 to 12 seconds: it failed three times in four local runs on 13 October before this change, because the first picker was still choosing its cards when a 4 second lease ran out, which is the same fault at the size of a test.
**Reversible:** yes. One script.

**Added the same morning, after shipping.** The cluster suite at `114ec5a` read 1232 passed with pytest's own exit code 0, job 11163. Then a dry run on the GPU node itself, in a temporary claims directory with the lease set to zero and nothing launched, against the real cards as they stood: 37173, 25955, 14765, 11521 and 22557 MiB free on GPUs 0, 2, 3, 5 and 6 and no floor free on the other three, which is room for nine runs at 10240. A first job asking for five took cards 2, 3, 5, 0 and 6 and by its own arithmetic left 4525 MiB on GPU 3. A second job asking for five, reading the same unchanged node at once, counted GPU 3 at 4525 where it read 14765, placed four runs on cards 2, 0, 6 and 0, and said there was room for four of five. A third found no card with a floor free. Five and four are the nine the node holds, and GPU 3 got one run, where on 12 October it got two. In a second pass one job's ledger lines were gone after its release and the other jobs' lines were still there. The real claims, ledger and lease were listed before and after both passes and had not changed.

### D-073. The C4 recipients go in before the last donor finishes, on the author's word
**Date:** 2026-10-14
**Milestone:** 8
**Type:** operational
**Decision:** the 30 recipients of D-069 were submitted at 19:35 on 14 October as jobs 11324 to 11329, five runs each, and moved to the front of the queue with `qorder`, when 29 of the 30 donors carried DONE. D-069 says they go in once every donor does. The donor still training, `famA_d1_d4only_s20`, is the donor of `c4_currctrl_s20`, which was put in the last of the six jobs, 11329. The author chose this on 14 October from four options put to them.

**Reason.** The last donor's job, 11206, was the only running job likely to end within a day: each of the other four held a run still at chance with 200000 or more steps to go. A recipient job already in the queue when 11206 ends takes its slot. One submitted after it ends waits a day or more for the next.

**Why it cannot train a recipient from a partial donor.** `train/cli.py` refuses to initialise from a donor directory without DONE and raises at launch. A recipient that starts before its donor finishes therefore dies at once and restarts when its job ends; it never trains from a half trained donor. D-061's correction was about a PBS dependency that would spin. Nothing here depends on one.

**What it changes.** When the recipients were queued, and nothing else: no run, config or seed. Run ids are the config names and each seed is the donor's, as for D-061.

**Checked.** `check_configs` exit 0. `check_queue` exit 0, 121 of 121 queued runs. `qorder` put 11324 to 11329 in the six places held by 10969 to 10974, read back from `qstat`. `check_queue --order` exit 0. `escape_rates --closed` exit 0, 70 finished and 0 unfinished.
**Reversible:** not applicable once the runs have started.

### D-074. A loop-count curve pools one training setup, and F2 is drawn from the registered arms and stored with them
**Date:** 2026-10-22
**Milestone:** 12
**Type:** correction
**Decision:** `m1_curves` keys every cell by how its run was trained as well as by its task. `setup_key` is the run's own config with the trainer's defaults filled in, less the seed, the data seed, the worker count and the checkpoint interval, and with a recipient's donor reduced to its arm. A summary in which two setups share a task is refused, in `main` and in `curve_summary`, and `--config` keeps only the runs a config file trains. F2's summary is made with `--config configs/m3/famB_curve_111.yaml --config configs/m3/famC_breadth_111.yaml`, the arms H1 has always been reported on (findings.md, H1 SUPPORTED). `scripts/build_figures.sh` now draws every figure the draft cites and stores each figure's input first, in a dated directory under `runs/`, with the commit that made it. F2's censored-cell label names every depth that has censored cells instead of saying "all at" the worst one.

**How it was found.** Preparing the Zenodo release, every cited figure was redrawn on the cluster from stored data into a temporary directory and compared pixel by pixel with its copy in `figures/`. F1, F3, F4, F8 and F9 matched exactly, and F6 matched once its task was named. F2 did not. It was drawn on or about 2026-09-10 from a summary that was never stored, and `m1_curves` grouped by task alone, so family B's curve took in every run that shares the task. On 2026-10-22 that is ten arms and 85 runs: the feedforward, echo and untied baselines of `famB_g2_*`, three model widths and three jitter widths besides the curve arm itself. A feedforward run's k_min is 1 at every depth, an untied run's is 4, and the zero-jitter arm is censored in 122 of its 150 cells. The published figure reads "157 of 1740 cells", 58 runs, the pool as it stood the day it was drawn. The H1 fit's bootstrap drew seed numbers shared across arms as though each were one seed.

**What the registered arms show**, ten seeds each, read from the mirror of 2026-10-22 and again from the stored summary once it exists: median k_min 1, 1, 2, 2, 2 and 3 at depths 1 to 6 in family B, with 8 of its 300 cells censored, 3 at depth 5 and 5 at depth 6; family B flat at 2 across breadths 12 to 16, pooled over its depths; family C flat at 1 across breadths 4 to 8. The draft said 2 at depth 6 and family B's breadth line flat at 1. **H1's direction holds either way. Its numbers were the pool's.**

**Why by config and not by run name.** `task_key` was built from the config so that it could not fall behind a list of names, and the same holds here. Both registered config files resolve through `train/cli.load_config` to exactly the setup of their runs. No arm splits into two setups. Two pairs of arms share one, `famA_d2_d4colour` with `famA_d2_d4colour_deep` (the duplicate D-043 records) and `g1_famC_fixed` with `g1_famC_looped`; neither pair is a curve, and pooling identical setups is correct.

**What it does not touch.** No run, config or seed. Every other figure's input is unchanged and its drawing is identical. The halting analysis, `m5_h2`, has the same task-only key, but only `famB_curve111` and `famC_breadth111` carry its per-run files, so its pool is the registered one today. It is recorded here rather than fixed.

**Checked.** Eight new tests in `test_m1_curves.py` and one in `test_f2_curves.py`; eleven deliberate breaks of the key, the refusals, the selection and the label, each caught. On the mirror, a summary without `--config` is refused and names the arms that share each task; with the two configs it keeps 20 runs.
**Reversible:** yes. The selection is a command-line argument and the old figure is in git history.

### D-075. The release is a scrubbed snapshot of one commit, built by a script that refuses to leave anything private in it
**Date:** 2026-10-22
**Milestone:** 12
**Type:** release
**Decision:** the preprint goes on Zenodo with one archive beside it, built by `scripts/build_release.py` from a single commit. In it: the paper; the preregistration as tagged `prereg-v1`; this log and `experiment-substrate.md`; `ledger.csv`, the commit and time at which each entry of this log was first committed, generated from `git log`; every training run's config, metrics, summary and instrument outputs; the stored directories the draft names and the gate directory Section 4 reads; the seven figures, each copied only if its hash is the one `scripts/build_figures.sh` stored beside its inputs; and the package, its tests and its configs. The author chose a snapshot over public history, and chose to include the code, on 2026-10-22.

**Left out, and why.** Checkpoints, 96 GB, whose instrument outputs are in. The three jitter arms, 15 runs, and the per-run geometry files, because the loop-count jitter result and the geometry figure are in concurrent work under review and the paper points there. The PBS scripts and shell helpers, and the nine test files that exercise them only. The history, because two commits name the cluster. This log is kept whole, D-053 included, on the author's word: it describes the H4 metric, not the result under review.

**Scrubbing.** The identifiers are read from the cluster file outside the repository, never written into it. In text, a home directory becomes `~`, the cluster's name `<cluster>` and a user name `<user>`, as whole tokens in any case. Then every file is searched again and the build fails naming the file. In binary files only the exact case counts, because the cluster's name is four letters and turns up by chance, in some case, inside the base64 images of `paper.html`. A trial build scrubbed 659 files: one line of `check_queue.py`, two lines of this log, and 657 `env.txt` package listings. Nothing was left.

**A run file the builder does not know stops it.** Files are released from a list, so a log nobody read cannot leak; a file neither listed nor excluded is an error rather than a silent omission.

**Licences, the author's choice of 2026-10-22.** The record is Creative Commons Attribution 4.0 and the code MIT. The repository has no licence of its own, so the builder writes `code/LICENSE` naming the holders given by `--licence-holder`, and refuses a release without them; only a trial may leave them out.

**Checked.** 21 tests in `tests/test_build_release.py`, using made-up identifiers only, and 19 deliberate breaks, each caught. In a trial build the package's own suite passes inside the archive, 1138 passed and 6 skipped, and a search for any identifier over every file of the archive finds none.
**Reversible:** until the record is published. A Zenodo record can be versioned and not withdrawn.

### D-076. S1's tie restored as registered, and Appendix B read again from stored files
**Date:** 2026-10-22
**Milestone:** 12
**Type:** correction
**Decision:** five corrections, from a reader's review of the release draft on 2026-10-22 and from checking the appendix it prompted.

1. **S1 is 0.673, not 0.703, and still held.** D-070 registered that "E1b uses the same tiles as D4 position, so the two tie, and ties take their average rank." The two M_pix values were one mean of one set of distances summed in a different order, 2.245522029421255 and 2.2455220294212546, and `average_ranks` compared them exactly, so the tie the registration named was broken by rounding error. Broken that way it gave 0.703; broken the other way it gives 0.631; as registered it gives 0.673, equal to M_emb's, whose two values happened to be bit-identical. `average_ranks` now compares at nine decimals, `TIE_DECIMALS`. The reviewer found it.
2. **Each correlation now carries an exact two-sided permutation p**, over all 5040 orderings of the seven escape rates: 0.107 for both. The registration fixed a threshold and not a test, so the p decides nothing; the paper says the correlation is suggestive and not significant, and "orders" becomes "tracks" wherever the draft said salience orders learnability.
3. **The H3 numbers in Appendix B were from a run nobody kept.** The draft quoted -0.1029 and -0.1462, from 2026-09-10 on 14 cells; findings.md has the 2026-09-11 update on 19 cells; `h3_contrast` pools every `m2_confirm.parquet` present, and on 2026-10-22 there are 30. The verdict has not moved at any of the three: loop contrast null, spatial contrast significant and opposite, neither surviving Holm. `scripts/build_figures.sh` now stores the module's output, and the draft quotes it. The family means it also quoted, 0.1630 and 0.7831, were computed by hand on the 19-cell set and are gone; the stored per-cell table carries the qualitative point that replaces them.
4. **Entropy's slope is +0.24**, as the ten-seed `m5_h2` output says, not the +0.23 carried from an earlier draft.
5. **H2's paragraph explains its verdict again, but not with the sentence the reviewer asked for.** "Every criterion has at least one p above alpha outright" was true at eight seeds, when the second-order rule's slope had p = 0.143, and is false at ten, where it is 0.0412. What holds at ten is findings.md's rule: support needs one criterion with both its error and its slope rejected after Holm, and none has both. It is determinate, though `m5_h2` labels 0.0412 "undetermined until the rest of the family runs": Holm can reject a p of 0.0412 among fourteen tests only if it is the largest, and four tests of the family stored beside it are larger (1.0000, 1.0000, at least 0.2500, 0.2114).

Also from the review: Figure 1 drew a seven-line note that the draft's caption repeated, so the note is gone and its one extra point is in the caption; the E2a and E2b replication line is written in the table's order; and `scripts/build_paper.py` prints on A4 with 20 by 18 mm margins, keeps tables two pixels inside the column, lets code in a table cell wrap inside its cell, and caps figure height at 112 mm so that most figures fit below the text that cites them. On the previous print the right rule of every table sat on the edge of a 0.39 inch margin, where a viewer could clip it.

**Checked.** `tests/test_tile_distinguishability.py`: six new tests, including the stored full-precision values giving 18.5 / 27.5 and a p of 540 / 5040; six deliberate breaks, each caught. `tests/test_f1_tasks.py`: one new test, which fails against the old figure. On the mirror the corrected module prints 0.673 and p = 0.107 for both measures, S1 held. The print layout was measured with PyMuPDF: no rule or text past the column on any page.
**Reversible:** yes; nothing that ran is changed.

### D-077. The release goes on the project's GitHub repository, not on Zenodo
**Date:** 2026-11-12
**Milestone:** 12
**Type:** deviation
**Decision:** The author's professor approved the draft, and the author chose to publish the release on the public GitHub repository instead of a Zenodo record or a preprint server. The snapshot of D-075 is unchanged in what it holds. It is added as one commit on top of the repository's public branch, and the paper's "Code and data" section gives the repository address where it had a placeholder for a record identifier.
**Why one commit and not the history.** The working history carries the cluster's names in two commits and in two tracked files, which is the reason D-075 gave for a snapshot, and that reason does not depend on where the snapshot is published. The public branch stops at the commit of 2026-09-07; everything since is in the snapshot's files, and `registrations/ledger.csv` dates each registration by the commit that first added it.
**What this gives up.** A Zenodo record has a DOI and a deposit date set by a third party. A git commit's date is set by whoever makes it, so the repository alone is weak evidence of when the work existed. An archive of the repository by Software Heritage records the date independently, and is the planned remedy.
**Found on the way.** `SHA256SUMS` was written with `write_text`, which on Windows ends lines with CRLF, so `sha256sum -c` failed on every line of the first snapshot built for GitHub although every hash was right. `write_sums` now writes bytes with LF ends and leaves the checksum file out of its own list.
**Checked.** `tests/test_build_release.py`: one new test, and two deliberate breaks (the old `write_text`, and a checksum file that hashes itself), each caught.
**Reversible:** the draft's line and the builder, yes. A public commit can be removed from the repository but not from any copy made of it.
