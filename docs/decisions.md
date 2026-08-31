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

### D-003. Compute block, sixteen weeks on the cluster
**Date:** open
**Milestone:** blocks the pre-registration freeze at milestone 5
**Type:** resolution
**Decision:** OPEN. From Section 12 of the spec.
**Question:** is a sixteen-week window on the Jaypee cluster realistic alongside the merging pipeline?
**If access is intermittent:** cut the size ladder from four models to two (d=384 and d=768) and say so in the paper.
**Resolve by:** before milestone 5, mid October 2026.

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
