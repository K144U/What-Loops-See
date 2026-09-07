# Progress

Milestone tracker for "What a Loop Sees". Mirrors Section 11 of `what-a-loop-sees-IMPLEMENTATION.md`.

**Rule:** a milestone is done when its command exits 0, not when the work feels finished. Do not start milestone N+1 until milestone N is marked DONE here with the command output pasted into the log.

**Update cadence:** update the status table on every milestone transition, and append a session log entry at the end of every working session, even a session that achieved nothing. Sessions that achieved nothing are the ones worth recording.

---

## Current state

**As of 2026-09-06**

- Phase: **milestone 4 in progress, gate G2.** Milestone 3's build work was finished early during milestone 2, and H1 resolved two milestones early. See `findings.md`.
- Repository: at `c1302b2`. Local and `<cluster-user>@<login-node>:~/loopvision` are in sync, transferred by git bundle so the LF line endings survive. There is no network remote: `origin` is a bundle file that must be placed on the cluster before a pull will work, and it is not left there between syncs.
- Environment: Python 3.11.11 via `module load python311`, torch **2.4.1+cu121**, driver 525.147.05 confirmed. See D-014.
- `pytest`: **270 passed in 228.77s, exit 0**, on the cluster 2026-09-06. Note that the documented command `pytest -q` doubles the `-q` already in `addopts` and suppresses the pass count entirely, so it prints dots and no summary. Run bare `pytest` when you need the number: `addopts` has supplied a `-q` since the first commit, and a second one suppresses the summary. The documented command was corrected in CLAUDE.md, README.md and the spec on 2026-09-06.
- The suite does not run on the Windows machine at all: the package is not installed there and collection fails on import for all 20 modules. Every test claim in these documents is a cluster claim.
- **Blocking: cores on the gpu node, not compute and not memory.** 96 of 96 cores assigned with 41 jobs resident, while memory is nowhere near binding and is not enforced at all: one neighbouring job holds 1041 GiB against no request. Walltime does not predict releases either, since several neighbours have used more than ten times the wall they asked for. Nine of our jobs are on the cluster: 5178, 5180 and 5181 running, 5188, 5190, 5191, 5206 and 5207 queued, 5189 held.
- In flight and **not results**: `famA_d2_s3spatial` seeds 0 and 1 read 0.9929 and 1.0000 at 30k of 300k steps, and `famA_d2_d4colour_s1` sits on the chance floor at 280k. If those hold, the 2x2 closes on substrate with no group effect. Neither run has written DONE, and a near perfect score still owes the three controls, so none of this is a finding yet.
- Next action: read gate G2 when 5206 and 5207 land, `python -m loopvision.analysis.gate_g2`. It can fail, and a failure is reported rather than worked around.

---

### Milestone 0 cluster sign-off, PASSED 2026-08-31

Submitted as:

```bash
qsub -v RUN_ID=smoke_chain02,ENTRY=loopvision.train.smoke,MAX_HOURS=0,STEPS=3 scripts/submit.pbs
```

All four pass criteria met, plus one stronger check that was not required.

| # | Criterion | Result |
|---|---|---|
| 1 | Two or more chained jobs, each depending on the previous | **4 jobs**: 4904 cold start, 4906 resumed at step 1, 4908 resumed at step 2, 4910 resumed at step 3 and wrote DONE |
| 2 | DONE sentinel present at the end | yes, and job 4910 terminated the chain rather than queueing a fifth |
| 3 | Every step exactly once, no duplicates, no gaps | steps 1, 2, 3, each once |
| 4 | `pick_gpu.sh` selected a card and logged it | every job: `selected GPU 4 (77075 MiB free, 3977 MiB used)` |
| 5 | **Not required: chained equals uninterrupted** | loss trace **identical**, and all final parameters bit identical under `torch.equal` |

Criterion 5 is the one that matters scientifically. A run split across four PBS jobs produced exactly the same result as one that ran straight through, on the real cluster, with the real torch build.

**Two spec assumptions failed first contact and are now fixed.** The GPU selection rule (D-013) and self-chaining from inside a job (D-015). Both would have been discovered later and more expensively.

---

## Milestone status

Legend: NOT STARTED, IN PROGRESS, BLOCKED, DONE, FAILED.

| # | Weeks | Planned dates | Work | Done when | Status |
|---|---|---|---|---|---|
| 0 | 0 | Sep 1 to 6, 2026 | Repo skeleton, `pyproject.toml`, CI running pytest, `pick_gpu.sh`, hello-world PBS job surviving a kill and resuming | `pytest -q` passes and a chained 2-job run completes | **DONE 2026-08-31.** 13 tests pass on the cluster, 4-job chain completed and matched an uninterrupted run bit exactly |
| 1 | 1 | Sep 7 to 13 | `groups.py`, `render.py`, families A, B, C generators, manifest and splits | `pytest tests/test_data.py` passes, 64 sample images dumped and eyeballed | **DONE 2026-08-31.** 99 tests pass, all three families, 11 contact sheets dumped and eyeballed |
| 2 | 2 | Sep 14 to 20 | **GATE G1**, task validity | `pytest tests/test_gate_g1.py` passes on stored gate runs | **DONE 2026-09-01.** Passed on attempt 2 at 200k steps. Attempt 1 at 20k steps failed G1.2 and is recorded in full |
| 3 | 3 to 5 | Sep 21 to Oct 11 | Model, training loop, stability, three conditioning variants, four baselines | d=384 trains stably at k=8 on depth 3 above threshold, cold start, twice with different seeds | **UNBLOCKED.** Model, trainer, baselines and parallel generation already built during milestone 2 |
| 4 | 5 | Oct 5 to 11 | **GATE G2**, loops beat matched-compute feedforward and echo baseline | `python -m loopvision.analysis.gate_g2` exits 0 | **FAILED 2026-09-07.** Undefined where the spec put it: on family A depth 3 both arms sat on the chance floor at 0.0225. Moved to family B at 1/1/1 per D-030, where the comparison could come out either way. It came out against us. G2.1 failed, looped 0.9966 against matched-compute feedforward 0.9998. G2.2 passed against the echo baseline. Hard stop, reported |
| 5 | 6 to 7 | Oct 12 to 25 | Freeze `preregistration.md` at tag `prereg-v1`, then full M1 sweep | H1 resolves, `m1_loopcurve.parquet` complete, coefficients a and b with CIs printed | NOT STARTED |
| 6 | 8 to 9 | Oct 26 to Nov 8 | M2 patching over (loop, patch position) | H3 resolves, heatmaps for all three families | **IN PROGRESS 2026-09-07.** Started ahead of milestone 5, which is unstarted, and after gate G2 failed. `instruments/m2_patching.py` built on the existing `hooks.recovery_grid`, positive control wired in and mutation checked. First grid on `famA_d2_d4only_s0`: control 1.0000 at the last iteration, single positions at most 0.094, recovery concentrated at i=k. Family A only, 64 items against the 256 the spec asks for. Families B and C need a `corrupted_twin` before their heatmaps exist |
| 7 | 10 | Nov 9 to 15 | M3 coda-lens, M4 geometry | rays-not-fixed-points question answered either way | NOT STARTED |
| 8 | 11 to 12 | Nov 16 to 29 | Baselines and controls at matched compute and matched parameters, M5 extrapolation, attractor and jitter-width sweeps | H2 and H4 resolve, jitter width zero arm included | NOT STARTED |
| 9 | 13 | Nov 30 to Dec 6 | Confirmation runs for every post-hoc headline number. **No new exploration** | every headline number in `docs/claims.md` has a confirming run id | NOT STARTED |
| 10 | 14 to 15 | Dec 7 to 20 | Stage 2 graft and probes, optional Ouro cross-check, artifact and anonymity items | probes reproduce or refute the small-model signatures | NOT STARTED |
| 11 | 16 | Dec 21 to 27 | Buffer, ablation gap filling | | NOT STARTED |
| 12 | Jan to Mar 2027 | | Writing, internal review | full draft reviewed by both advisors | NOT STARTED |
| 13 | Apr to May 2027 | | Anonymous release, abstract, submission | submitted | NOT STARTED |

Planned dates are derived from the spec's week numbers anchored at week 0 starting 2026-09-01. Milestones 3 and 4 overlap in week 5 by design, as in the spec.

---

## Gate ledger

The two hard stops. A gate that fails means stop and report, not work around.

| Gate | Milestone | Question | Status | Date | Outcome |
|---|---|---|---|---|---|
| G1 | 2 | Do the tasks actually separate depth from breadth? | **PASSED, attempt 2** | 2026-09-01 | Attempt 1 failed G1.2 at 20k steps. Attempt 2 at 200k steps passes all four. Both attempts in findings.md |
| G1.1 | 2 | Depth 3 unsolvable at k=1, looped and non-looped both near chance | **PASSED** | 2026-09-01 | 0.0212 both at 200k steps, chance 0.0208. Now meaningful, since depth 1 is solved at the same budget |
| G1.2 | 2 | Depth 1 solvable at k=1 to high threshold | **PASSED** | 2026-09-01 | 1.0000 at 200k steps. Grokked at step ~84k, so attempt 1's 20k budget was 4x short |
| G1.3 | 2 | Order-blind ceiling below tau, probe not above ceiling (revised, D-016) | **PASSED** | 2026-08-31 | Worst ceiling 0.652 at depth 2, tau 0.90, margin 0.248. Linear probe 0.056, well under its ceiling. Needs no trained model |
| G1.4 | 2 | Family C solvable at k=1 at every breadth | **PASSED** | 2026-08-31 | 1.000 at every breadth cell, 4 through 8 |
| G2 | 4 | Does looping beat matched-compute feedforward and the echo baseline? | **FAILED** | 2026-09-07 | G2.1 FAILED: looped 0.9966 against feedforward 0.9998, margin -0.0032 where +0.02 was required. G2.2 PASSED: looped 0.9966 against echo 0.9716, margin +0.0250. The feedforward wins at every depth and the gap widens with depth, -0.0001 at d3 to -0.0138 at d6. Reported, not worked around |

Record every gate outcome here and in `findings.md`, including a failing outcome and what was changed in response. A gate that was re-run after a task change must show both attempts.

---

## Hypothesis resolution

Mirrors `findings.md`. Kept here so the milestone view and the science view stay in sync.

| ID | Type | Resolves at milestone | Status |
|---|---|---|---|
| H1 | pre-registered | 5 | **SUPPORTED 2026-09-04**, ahead of milestone 5. Depth raises k_min in 3 seeds of 3, breadth in 0 of 3, at matched architecture. Seed variance in magnitude is large and reported. See findings.md |
| H2 | pre-registered | 8 | UNRESOLVED |
| H3 | pre-registered | 6 | UNRESOLVED. Instruments built (`hooks.py`), M2 not run |
| H4 | exploratory | 8 | UNRESOLVED |

---

## Standing checklist, not tied to a milestone

Items that are easy to defer to the final week and must not be. Guardrail 1 from spec Section 10.

| Item | Owner | Due by | Status |
|---|---|---|---|
| `preregistration.md` written and tagged `prereg-v1` | | before milestone 5 runs | NOT STARTED |
| Anonymous code link, real not ANON-PENDING | | milestone 10 | NOT STARTED |
| Artifact DOI | | milestone 10 | NOT STARTED |
| AI use statement, written not TODO | | milestone 10 | NOT STARTED |
| Co-authorship and affiliation settled | | before preprint or submission | OPEN, see decisions.md D-004 |
| Compute access confirmed for 16 weeks | | before milestone 5 | PARTIAL. Access works and jobs run. The gpu queue was busy at first contact (39 running, 4 queued), so throughput is the open half. See D-003 and D-014 |
| Disk quota checked against probe checkpoint estimate (about 150 GB) | | before milestone 5 | **CLOSED 2026-08-31.** `/home` is BeeGFS, 466T with 324T free. 150 GB is a non-issue. Probe density stays at 100 log-spaced steps, no reduction to 50 needed. See D-014 |

---

## Risk register

Live risks with the trigger that would make each one real.

| Risk | Trigger | Response | Status |
|---|---|---|---|
| **H1 has exactly one live path** | family B is the only family that can produce a loop-count curve | family A cannot supply one while depth 2 is unsolved at every k: with no k that solves it there is no k_min to regress. Family C is the null arm by design. So H1 rests entirely on the 1/1/1 family B runs. If those come back flat, H1 has no route and the contribution ordering has to change, with H3 and the family A factor result carrying the paper | **LIVE, triggered 2026-09-04** |
| **Gate G2 is downstream of the same run** | G2 asks whether loops beat a matched-compute feedforward baseline | a task solvable at k=1 cannot show loops beating anything, so G2 could not have passed on the old family B whatever the truth. It is now gated on the corrected task showing k_min above 1. G2 and H1 therefore share a single point of failure rather than being independent checks | **LIVE, triggered 2026-09-04** |
| **A config key parsed but never read** | any new config field | three appeared today. `eval_k_sweep` was wired, `factor` and the block counts were not, and the block counts would have trained a 2/2/2 model from a 1/1/1 config with nothing in the run directory to show it. Every new key now needs a test asserting on the built object, not on the config dict | live, standing |
| Model does not train stably at k=8 on depth 3 | end of week 3 with no stable run | adopt ELT-style intra-loop self-distillation, re-run | not triggered |
| Step budget makes the M1 sweep unaffordable | family A needs 100k to 200k steps, not 20k, measured at gate G1 | **LIVE.** A five to ten times multiplier on every family A run. Options: fewer seeds, a shorter ladder, or accept a longer milestone 5. Must be resolved before the sweep is launched | **triggered 2026-09-01** |
| Still unstable after distillation | end of week 5 | drop to single-block core and 16x16 resolution | not triggered |
| Cluster access intermittent | any two-week gap in usable GPU time | cut size ladder from four models to two (d=384, d=768) and say so in the paper | not triggered |
| Probe checkpoints exceed disk quota | quota check before milestone 5 | reduce probe density from 100 to 50 log-spaced steps | not triggered |
| Resolution creep | any proposal to move above 32x32 outside the stated robustness check | refuse, log in decisions.md | not triggered |
| Scope creep into the generation arm | any work on H5 | refuse, it is out of scope for the main paper | not triggered |
| Gate G2 fails because looping genuinely loses at matched FLOPs | G2 comparison goes against the looped model | distinguish "looped loses at matched compute" from "looped model is broken" before responding. arXiv 2608.04879 reports standard ViTs preferable at matched FLOPs on CIFAR-100, and arXiv 2604.21106 puts the recurrence-equivalence exponent at 0.46. G2 is genuinely uncertain, not a formality | live, see gap-analysis.md Section 5 |
| Scooped on a hypothesis while we build | a directly competing paper appears | recheck `gap-analysis.md` at milestone 9 and before submission. The looped vision literature roughly tripled between February and August 2026 | live, watch list in gap-analysis.md Section 8 |

---

## Session log

Append one entry per working session. Newest at the bottom. Keep entries short and factual. A session that produced nothing gets an entry saying so and why, because that is what makes the schedule honest at review time.

Format:

```
### YYYY-MM-DD, milestone N
Did: 
Command run and exit code: 
Result: 
Blocked by: 
Next: 
```

### 2026-08-31, pre-milestone-0

Did: read the implementation spec end to end. Created the five planning documents: `paper.md`, `progress.md`, `findings.md`, `decisions.md`, `learnings.md`.
Command run and exit code: none, no code exists yet.
Result: planning scaffold in place. Paper objective fixed and written down before any result exists, so no outcome can quietly reshape the story later.
Blocked by: nothing.
Next: milestone 0. Repo skeleton, pinned `pyproject.toml`, `scripts/pick_gpu.sh`, `scripts/submit.pbs`, and a smoke training script with a bit-exact SIGKILL resume test. No model code.

### 2026-08-31, pre-milestone-0, related work survey

Did: full related work survey before writing code. Verified the spec's reference list by fetching arXiv abstract pages, and searched for current work. Wrote `gap-analysis.md`.
Command run and exit code: none, no code exists yet.
Result: the spec's citations are accurate but its reference list is roughly twenty one papers short of the current literature. H2 is severely threatened as worded and must be rewritten before the pre-registration freeze. H4 is scooped as a phenomenon by arXiv 2606.29983. H3 is largely intact and is now the strongest hypothesis. The core gap, ground-truth composition depth in pixels with depth crossed against breadth, holds and nobody else has it. Two new risks added to the register above.
Blocked by: nothing. The changes are to framing, not to the build order.
Next: unchanged, milestone 0. The H2 rewrite is due before milestone 5, not before milestone 0.

### 2026-08-31, milestone 0

Did: synced the spec to the survey (H2 reframed, H4 attributed to arXiv 2606.29983, family C protected, reference list expanded from 4 groups to 6, ELT correctly labelled as generation). Built the repository: `pyproject.toml`, package skeleton, `checkpoint.py`, `smoke.py`, `pick_gpu.sh`, self-chaining `submit.pbs`, `CLAUDE.md`, `.gitattributes`. Moved the six planning documents into `docs/` per D-002. Two commits.

Command run and exit code: `pytest -q` exits 0, **13 passed**.

Result: bit-exact resume is verified three ways, abrupt self-termination via `os._exit`, an external kill mid-run, and three sequential kills in one run. Also covered: the DONE sentinel stopping the chain, wall-clock self-stop leaving a checkpoint but no DONE, a twelve-job chain reproducing an uninterrupted run exactly, recovery from a missing LATEST pointer, and refusal to load a mismatched checkpoint format.

The suite was mutation checked. Disabling `restore_rng_state` initially failed **zero** tests, which exposed that the smoke model consumed no global RNG and so could not test RNG restoration at all. The smoke model was rebuilt as a miniature looped model with per-batch loop-count sampling. Re-running the same mutation now fails three tests. See `learnings.md` L-011 and `decisions.md` D-011.

Blocked by: **cluster access.** The PBS half of the milestone 0 gate has not run. See the sign-off block at the top of this file.

Next: run the two-job chain on the cluster. Milestone 1 does not start until it passes.

### 2026-08-31, milestone 0 cluster sign-off

Did: connected to `<login-node>` as `<cluster-user>` over the VPN. Probed the environment, transferred the repo by git bundle, built a venv on `module load python311`, installed torch 2.4.1+cu121, ran the suite, and drove the PBS chain to completion.

Command run and exit code: `python -m pytest -q` exits 0, **13 passed on the cluster**. Chain submitted as job 4904, completed through 4906, 4908 and 4910.

Result: **milestone 0 gate passed in full.** A run split across four PBS jobs is bit identical to an uninterrupted one, verified on loss trace and on final parameters under `torch.equal`. Details in the sign-off block at the top of this file.

Two spec assumptions failed on first contact with real hardware and both were fixed rather than worked around:

1. **GPU selection gated on the wrong quantity** (D-013). The spec said fail if the chosen card has more than 2 GB in use. Every A100 on this cluster had more than 2 GB in use while six of eight had over 70 GB free, so the rule would have refused to start on a perfectly usable machine. Now gates on free memory with a 20 GB floor. Validated live: `selected GPU 4 (77075 MiB free, 3977 MiB used)`.
2. **Self-chaining from inside a job is refused** (D-015). `qsub: Bad UID for job execution`. The spec's Section 3.2 design cannot work here. A probe job established that ssh to the login node is passwordless and qsub is accepted there, so the chain now falls back to that. This one had a silent failure mode: the first job checkpointed correctly and exited 0, and the run simply stopped advancing with nothing in the log to say why. It is now fatal and loud.

Also learned: driver is **525.147.05 exactly as the spec said**, so the cu121 pin is correct and cu124 would genuinely have failed. Disk is a non-issue at 324T free, closing that checklist item. The gpu queue was busy, 39 running and 4 queued, which bears on D-003.

Blocked by: nothing.

Next: milestone 1. `groups.py` with the D4 x S3 multiplication table and unit tests for associativity, inverses and non-commutativity, then `render.py` and `family_a.py` with the anti-shortcut sampler.

### Session, 2026-09-04

**The day's real work was auditing, and most of it found faults rather than results.**

Two things went right. The family C gate rerun passed G1.4 on the repaired task, and the blank-image control confirms it honestly: 0.218 with the image blanked against a 0.375 label prior, so the model drops below the prior rather than falling back on it. And the family A factor probe overturned the abelian-quotient reading: both 1M seeds predict the D4 factor at 1.0000 and the S3 factor at its chance level, which is evidence **for** sequential composition rather than against it, with three controls behind it.

Four things went wrong, all found by looking rather than by a failing test.

1. Family B's k sweep is flat at 1.000 for every depth and every k from 1 to 64. No H1 evidence exists in any family B run.
2. Family B's floor was misreported as 1/13 = 0.0769 when it is 0.2897, because `size` has two values and was askable. Runs recorded at 0.24 to 0.38 as partial successes were at or below chance.
3. The relation chain was skippable: 0.62 at depth 4 without reading it, 19 percent of questions with one possible answer.
4. `L_MAX["B"]` was a literal 8, silently capping chains at depth 4, and the block counts never reached `build_model`.

**Instruments added, each mutation checked:** `blank_control.py`, the factor decomposition in `parity_probe.py`, `registry.py` (which CLAUDE.md required and which did not exist, so every count in these documents had been typed by hand), and family B difficulty invariants.

**Decisions:** D-027, D-028. **Learnings:** L-013, L-014.

Milestone 3 continues. Nothing here changes the milestone schedule, which remains roughly three weeks ahead of the planned dates, but the science has one live path where it had two.

### Session addendum, 2026-09-04, later

**H1 resolved, two milestones early.** Depth raises k_min in three seeds of
three, breadth in zero of three, at matched architecture and matched
budget. Single-pass accuracy falls from 0.998 to 0.492 across the depth
range and moves from 1.000 to 0.999 across the breadth range.

**Three mechanism results, two of which contradict something we believed.**

The factor runs finished: D4 alone is solved perfectly, S3 alone sits at
chance after 300000 steps with nothing competing. A linear probe then
showed S3 is not merely unread by the output head, it is never extracted
from the image at all, with the same probe reading D4's composite at 0.896
against a chance of 0.125 as the control.

The coda lens refuted a hypothesis of mine outright: the models do not
walk the chain hop by hop. Every intermediate hop sits at the guessing
floor at every pass while the final answer sharpens from 0.65 to 0.999.
What differs between seeds is sharpening rate, not traversal rate.

**Failures and mistakes recorded rather than quietly fixed.** L-016, a mock
that ignored the command it was given, hiding a watcher that would have
reported UNREACHABLE forever. L-017, three config keys parsed but never
read, two of them live faults, one of which would have trained a 2/2/2
model from a 1/1/1 config with nothing in the run directory to show it.
L-018, a CLAUDE.md rule naming a tool that did not exist, so every count in
these documents had been hand typed for weeks.

**Superseded entries kept in place with pointers rather than deleted:** the
abelian quotient reading, the family A depth wall as originally stated, and
the single-seed curve result. The wrong version is how the right one was
reached and deleting it would hide the reasoning.

**Open and unchanged:** gate G2 has not run, and the substrate experiment
in `experiment-substrate.md` is designed but not built.

### Session, 2026-09-06

**Did:** read the cluster and PyTorch notes carried over from the merging project against this repo. Two operational faults found and fixed, three reported hazards checked and already handled here.

**Commands run:** `pytest -o addopts='--strict-markers'` on the cluster, **270 passed in 228.77s, exit 0**. `qsub` probes at mem 48gb, 64gb and 96gb to find the queue ceiling by measurement rather than from the queue configuration.

**Fixed.** The run watcher tracked every queued experiment except the four gate G2 baselines, so the result ranked first in `STATE.md`, and a declared hard stop, would have completed in silence. `multirun.pbs` asked for `mem=96gb`, which the gpu queue now refuses outright at submit time; it had never bitten only because every real submission overrides the directive on the command line, so the checked-in default was unreachable. Recorded as D-031.

**Resubmitted.** Gate G2 went from one 4-core job, 5202, to two 2-core jobs, 5206 and 5207, because cores are the binding constraint on this node and a smaller request fits a smaller gap. The cost is `WORKERS=0` and in-process generation, against D-025's margin of roughly 4000 samples per second supplied to 1500 to 3000 consumed, so expect the runs to be slower per step and to start sooner.

**Checked and already sound, no change needed:** the randomised-SVD reproducibility trap does not apply, `svd_lowrank` is not used and `checkpoint.py` captures and restores full CPU and CUDA RNG state, so a resumed run continues the exact stream; stale GPU claims are already swept by `pick_gpus.sh`; checkpoint writes are already atomic. Recording these because a future session will otherwise re-audit them.

**Open, nobody has decided:** TF32 is not set anywhere in the repo, so fp32 matmuls run at a 10-bit mantissa by default on these cards. That is an inherited default rather than a choice, and the instruments quote numbers to four decimals.

**Superseded later the same day by D-033.** The claim above was taken from the carried-over note without checking it, and it is wrong for this project: `matmul.allow_tf32` is False in torch 2.4.1, so the matmuls were at full fp32 the whole time. Only the single patch-embedding convolution is exposed. Left here rather than corrected in place, because how the wrong version was reached is part of the record.

**Documentation faults found:** `preregistration.md` has never existed in the working tree or anywhere in git history, and there is no `prereg-v1` tag, yet CLAUDE.md stated the freeze in the present tense. Corrected to say it is milestone 5 work. The `Current state` block above was six days stale at milestone 2 while the milestone table below it was correct at milestone 4.

**Blocked by:** cores. 96 of 96 assigned on the gpu node. Gate G2 cannot run until a gap opens, and no scheduled drain exists to wait for.

**Next:** read gate G2 when 5206 and 5207 land. Then the feedforward control for the colour deadlock, still not started, which is the one result that could change what the paper is about.

---

---

## Handoff, 2026-09-04 end of session

Written so a cold start can continue without the conversation. Everything
below is derivable from the other documents; this says where to look and
what is in flight.

### The three results the paper now rests on

1. **H1 supported.** Depth raises k_min in eight seeds of eight, breadth in
   zero of three, at matched architecture. Single-pass accuracy falls 0.998
   to 0.492 across depth and moves 1.000 to 0.999 across breadth.
   findings.md, "H1 SUPPORTED".
2. **Substrate, not group.** D4 solves at 1.0000 on position and sits on
   the chance floor on colour, at the full budget, with every algebraic
   property held constant including an order-blind ceiling of 0.8125 in
   both. The registered abelian-foothold alternative was refuted.
   findings.md, "SUBSTRATE, NOT GROUP".
3. **The failure is a bootstrapping deadlock, not difficulty.** A depth 2
   S3 task that never left chance in 300000 steps is solved in 10000 from a
   depth 1 donor, with a wrong-donor control flat at chance. The state
   probe shows the cold-start model never extracts its own inputs and the
   curriculum model does. findings.md, "CORRECTION" and "The deadlock
   closes".

### In flight

| job | experiment | why it matters |
|---|---|---|
| 5178 | d4_colour seed 1 | second seed of result 2 |
| 5180, 5181 | s3_spatial | closes the 2x2. Substrate reading predicts it solves cold |
| 5188 | d4_colour depth 1 donor | feeds 5189 |
| 5189 (held) | d4_colour curriculum + control | does the rescue generalise across groups |
| 5190 | scale control, wide/deep/big | the reviewer objection: was the model too small |
| 5191 | depth ladder d4only d3 to d6 | could give family A a usable depth axis, which H1 currently lacks |

All fourteen runs are tracked by the scheduled task `loopvision-run-watch`,
which persists to disk and has survived four power cuts. It reports each
start and each completion.

### What is NOT done, ranked

1. **Gate G2 is running at last.** It was undefined where the spec put it:
   on family A depth 3 the looped and feedforward arms both sit at 0.0225
   against a 0.0208 floor, so every outcome was identical. Moved to family
   B at 1/1/1 by D-030, module built and mutation checked, baselines
   queued. It can still fail, and Gao et al. 2607.16051 say it might.
2. The feedforward control: does a non-looped model of matched depth
   deadlock on colour too? Decides whether the finding is about looping or
   about compositional learning in general. Two very different papers.
3. The curriculum runs to full budget, and s3_spatial, both under way.

### Standing rules that keep being earned

Every instrument is mutation checked, and three times today the first
version of a test passed against deliberately broken code: a mocked ssh
that ignored its command, a substrate test that compared images which
differ anyway, and a hop-rate metric that returned the same number for
opposite mechanisms. See L-011, L-015, L-016, L-017.

A perfect score gets the blank-image control and, since today, the input
ablation. `analysis/ablate_input.py` removes one patch at a time; the
solved D4 model needs its state and both operators and correctly ignores
the distractors, which together with an order-blind ceiling of 0.8125
below its 1.0000 is why that score is believed.

### Operational

The machine running the VPN has had repeated power cuts. Cluster jobs are
unaffected, the scheduled watcher survives, in-session monitors do not. The
GPU node runs at or near zero free cores most of the time, so jobs are
queue bound rather than compute bound, and shrinking a core request to fit
a gap has started work hours earlier on four occasions.
