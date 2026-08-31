# Progress

Milestone tracker for "What a Loop Sees". Mirrors Section 11 of `what-a-loop-sees-IMPLEMENTATION.md`.

**Rule:** a milestone is done when its command exits 0, not when the work feels finished. Do not start milestone N+1 until milestone N is marked DONE here with the command output pasted into the log.

**Update cadence:** update the status table on every milestone transition, and append a session log entry at the end of every working session, even a session that achieved nothing. Sessions that achieved nothing are the ones worth recording.

---

## Current state

**As of 2026-08-31**

- Phase: milestone 0, **local half complete, cluster half outstanding**.
- Repository: `loopvision/`, git initialised, two commits.
- `pytest -q`: **13 passed**. Bit-exact resume verified against abrupt death, external kill, and repeated kills.
- Blocking item: **the PBS chain has never run.** Milestone 0's gate is "pytest passes **and** a chained 2-job run completes". The second half needs cluster access and cannot be faked locally.
- Next action: run the two-job chain on the cluster (see the sign-off block below). Do not start milestone 1 until it passes.

### Milestone 0 cluster sign-off, outstanding

The chain *logic* is tested locally in `tests/test_chain.py`, including a twelve-job chain that reproduces an uninterrupted run exactly. What is untested is PBS itself: `qsub`, `-W depend=afterany`, and `pick_gpu.sh` against real cards. Run this on the cluster and paste the result into the session log:

```bash
RUN_ID=smoke_chain01 ENTRY=loopvision.train.smoke qsub scripts/submit.pbs
```

Pass criteria, all four:

1. Two or more jobs appear in `qstat` history for the run id, the second depending on the first.
2. `runs/smoke_chain01/DONE` exists at the end.
3. The metrics file has every step exactly once, no duplicates and no gaps.
4. `pick_gpu.sh` selected a card and the log shows its index and free memory.

If the chain does not terminate, the DONE sentinel logic is wrong and that is a hard stop, not a nuisance: an unterminating chain will silently consume the queue.

---

## Milestone status

Legend: NOT STARTED, IN PROGRESS, BLOCKED, DONE, FAILED.

| # | Weeks | Planned dates | Work | Done when | Status |
|---|---|---|---|---|---|
| 0 | 0 | Sep 1 to 6, 2026 | Repo skeleton, `pyproject.toml`, CI running pytest, `pick_gpu.sh`, hello-world PBS job surviving a kill and resuming | `pytest -q` passes and a chained 2-job run completes | **IN PROGRESS.** pytest half done, 13 passed. PBS chain not yet run, see sign-off block above |
| 1 | 1 | Sep 7 to 13 | `groups.py`, `render.py`, families A, B, C generators, manifest and splits | `pytest tests/test_data.py` passes, 64 sample images dumped and eyeballed | NOT STARTED |
| 2 | 2 | Sep 14 to 20 | **GATE G1**, task validity | `pytest tests/test_gate_g1.py` passes on stored gate runs | NOT STARTED |
| 3 | 3 to 5 | Sep 21 to Oct 11 | Model, training loop, stability, three conditioning variants, four baselines | d=384 trains stably at k=8 on depth 3 above threshold, cold start, twice with different seeds | NOT STARTED |
| 4 | 5 | Oct 5 to 11 | **GATE G2**, loops beat matched-compute feedforward and echo baseline | `python -m loopvision.analysis.gate_g2` exits 0 | NOT STARTED |
| 5 | 6 to 7 | Oct 12 to 25 | Freeze `preregistration.md` at tag `prereg-v1`, then full M1 sweep | H1 resolves, `m1_loopcurve.parquet` complete, coefficients a and b with CIs printed | NOT STARTED |
| 6 | 8 to 9 | Oct 26 to Nov 8 | M2 patching over (loop, patch position) | H3 resolves, heatmaps for all three families | NOT STARTED |
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
| G1 | 2 | Do the tasks actually separate depth from breadth? | PENDING | | |
| G1.1 | 2 | Depth 3 unsolvable at k=1, looped and non-looped both near chance | PENDING | | |
| G1.2 | 2 | Depth 1 solvable at k=1 to high threshold | PENDING | | |
| G1.3 | 2 | Bag-of-operators shortcut probe at chance for depth greater than 1 | PENDING | | |
| G1.4 | 2 | Family C solvable at k=1 at every breadth | PENDING | | |
| G2 | 4 | Does looping beat matched-compute feedforward and the echo baseline? | PENDING | | |

Record every gate outcome here and in `findings.md`, including a failing outcome and what was changed in response. A gate that was re-run after a task change must show both attempts.

---

## Hypothesis resolution

Mirrors `findings.md`. Kept here so the milestone view and the science view stay in sync.

| ID | Type | Resolves at milestone | Status |
|---|---|---|---|
| H1 | pre-registered | 5 | UNRESOLVED |
| H2 | pre-registered | 8 | UNRESOLVED |
| H3 | pre-registered | 6 | UNRESOLVED |
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
| Compute access confirmed for 16 weeks | | before milestone 5 | OPEN, see decisions.md D-003 |
| Disk quota checked against probe checkpoint estimate (about 150 GB) | | before milestone 5 | **CLOSED 2026-08-31.** `/home` is BeeGFS, 466T with 324T free. 150 GB is a non-issue. Probe density stays at 100 log-spaced steps, no reduction to 50 needed. See D-014 |

---

## Risk register

Live risks with the trigger that would make each one real.

| Risk | Trigger | Response | Status |
|---|---|---|---|
| Model does not train stably at k=8 on depth 3 | end of week 3 with no stable run | adopt ELT-style intra-loop self-distillation, re-run | not triggered |
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
