# What a Loop Sees: state as of 7 September 2026

> **GATE G2 FAILED on 7 September**, and three results since have pointed the
> same way. Matched-compute feedforward beats the looped model at every depth,
> the gap widening to -0.0138 at depth 6. The matched-**parameter** feedforward
> arm then sat at chance on both seeds at full budget, which D-037 registered in
> advance as meaning the escape came from capacity or depth and **not** from
> dropping the loop. And a feedforward model that fails the colour task fails it
> the same way a looped one does, reading nothing from the image.
>
> **The evidence has converged on a paper that is not about looping.** The
> mechanism results are strong and stand on their own; the loop framing in
> `paper.md` is the part under strain. That decision is item 1 below and it is
> not mine to make.

**Read this first after clearing context.** Everything here is derivable
from the other documents; this says where to look, what is running, and
what to do next. Written so a session starting cold can continue without
the conversation that produced it.

Local and cluster are at the same commit. `pytest` passes, 270 tests,
exit 0, run on the cluster. **Run it bare, not as `pytest -q`:** `addopts`
has supplied a `-q` since the first commit and a second one suppresses the
count entirely. The suite does not run on the Windows machine at all,
where the package is not installed and collection fails on import, so
every test claim in these documents is a cluster claim.

The scheduled task `loopvision-run-watch` is armed and reports every run
as it starts and finishes, including the four gate G2 baselines, which it
did not track until 6 September.

---

## 1. What the paper rests on

**Read the status label on each, they are not the same kind of evidence.**
The pre-registration was written at milestone 6 and its Section 0 says which
results it can and cannot bind, so some of what follows is exploratory by its
own admission.

**The strongest four, in order of how well controlled they are:**

1. **Substrate, not group.** Two seeds, full budget, three controls, and a
   registered alternative refuted.
2. **The deadlock is bootstrapping, not difficulty**, with a curriculum rescue
   and a genuine wrong-donor control in S3. The D4 replication's control is
   compromised and 5280 fixes it, see D-040.
3. **Escape needs capacity, not the removal of the loop.** The
   matched-parameter arm sat at chance on both seeds at full budget while the
   matched-compute arm solved on one. D-037 registered that meaning first.
4. **Both architectures fail the same way when they fail.** The deadlocked
   feedforward seed scores 0.1250 with the image and 0.1250 blanked, the same
   signature the probe found in cold-start looped models.

**Weaker than it reads elsewhere:**

- **H1 is exploratory now.** Prereg Section 0 demotes it: the family B result
  was analysed before any metric was registered. The registered form binds
  only cells not yet run.
- **H3 is unresolved and its first read was post-hoc.** Metrics are now fixed
  as normalised entropy, the confirmation set is one family of three.
- **The coda lens is a replication** of Lu et al. arXiv 2507.02199, and
  `paper.md` section 3a already says so.

**H1 as originally stated.** Depth raises the loops needed in
eight seeds of eight; breadth raises it in zero of three. Single-pass
accuracy falls 0.998 to 0.492 across depths 1 to 6 and moves 1.000 to 0.999
across breadths 4 to 8. Seed variance in magnitude is large, slopes 0.229 to
0.886, and is reported rather than averaged away: the paper needs a per-seed
random effect. See `findings.md`, "H1 SUPPORTED".

**Substrate, not group.** D4 solves composition at 1.0000 acting on
position and sits on the chance floor acting on colour, at the full 300000
step budget, with every algebraic property held constant including an
order-blind ceiling of 0.8125 in both. The registered abelian-foothold
alternative predicted the opposite and was refuted. See `findings.md`,
"SUBSTRATE, NOT GROUP". **Replicated in a second seed on 6 September:**
`famA_d2_d4colour_s1` wrote DONE at the full 300000 steps and finished at
0.1277, spread 0.1195 to 0.1421, against a 0.125 floor.

**The failure is a bootstrapping deadlock, not difficulty.** A depth 2 S3
task that never left chance in 300000 steps is solved in 10000 from a depth
1 donor of itself, while a wrong-donor control stays at chance for the full
budget. A trained probe shows the cold-start model never extracts its own
inputs and the curriculum model does, with a positive control reading D4 at
0.896 on the same code. See `findings.md`, "CORRECTION" and "The deadlock
closes".

A fourth thing is convergent rather than headline: **three instruments in
two task families all say these models compute answers rather than
trajectories.** The coda lens finds every intermediate hop at the floor
while the answer sharpens, the state probe finds the halfway composite the
least readable thing in a solved model, and both replicate Lu et al.
arXiv 2507.02199 in a different modality.

---

## 2. Where the paper stands

`paper.md` section 3a holds the pivot. The short version: the coda lens
result is a **replication** of Lu et al. and must cite them, and what is
ours is the ground truth they lacked plus a protocol that resolves the
ambiguity they leave open. The binding claim is written in as the intended
headline and explicitly **not** as a result until the crossed design
finishes.

`gap-analysis.md` section 10 has the three papers that moved the novelty,
all verified from their arXiv abstract pages.

---

## 3. Running now

Thirteen jobs, all two to four cores. **Ask for two.** Larger requests are
backfilled past rather than queued behind: 5269 at six cores sat while 5270
and 5272 at four, both submitted later, ran. Four jobs were resized for this
on 7 September. `max_run` counts jobs and not runs, so pack runs into a job
and cores stay small. See D-041 and section 6.

| job | runs | what it decides |
|---|---|---|
| 5270 (R) | untied s0, s1 | **the G2 diagnosis.** At 48 percent it had already passed the looped model's final accuracy at depths 4, 5, 6 |
| 5272 (R) | d4only s3, s4 | H3 confirmation seeds. s3 was at 0.2490 at 20 percent while s4 had solved, worth watching |
| 5276 (R) | s3only ffwd mp s0, s1 | matched parameters in the second family. Both near the 0.1667 floor at 80 percent |
| 5279 | s3spatial s0, s1 | **closes the 2x2.** Resumes at 293400 and 292950, about 7k steps each, both already 1.0000 |
| 5280 | curr_s3donor | **the D-040 control.** Two D4 curriculum arms sit at 1.0000 with nothing to interpret them against |
| 5281 | d4colour ffwd s2, s3, s4 | turns the feedforward seed split into a rate |
| 5282 | d4colour looped s2, s3, s4 | turns "the loop never escapes" into a rate. Currently four runs |
| 5283 | famC s3, s4, famA d4only s2 | the rest of the H3 confirmation seeds |
| 5274, 5275 | scale control, depth ladder | lower priority |

**Finished and analysed:** gate G2 baselines, the matched-compute and
matched-parameter feedforward arms on d4_colour, the D4 curriculum and its
same-group control, five family B M2 confirmation grids, and the three input
controls on the solved feedforward seed.

**Analysis that exists and is waiting for data.** All three were written
before the numbers, on purpose.

- `python -m loopvision.analysis.untied_contrast` refuses until the untied
  runs write DONE, and says why.
- `python -m loopvision.analysis.h3_contrast` refuses with only one axis
  present: an absent result, not a null one.
- `python -m loopvision.instruments.m2_patching --run-id X --items 900
  --min-admissible 256 --out runs/X/m2_confirm.parquet`, or in bulk via
  `scripts/m2_confirm.pbs` on workq.

## 4. What to do next, ranked

1. **Decide what the paper claims.** Three registered or controlled results
   now say the loop is not what the story is about, and the mechanism results
   do not need it to be. `paper.md` section 3a already pivoted once, to
   "looping does not rescue appearance binding", which survives all of this.
   The question is whether the paper is now about **compositional
   bootstrapping** with looping as one architecture among several, and that is
   a judgement call, not a run.
2. **5280, the D-040 control.** Until it lands the defensible D4 claim is only
   that a depth 1 D4 donor rescues depth 2 D4 on colour whether trained on
   position or on colour. Not that the donor's content matters.
3. **5279, closing the 2x2.** Both seeds have read 1.0000 since 50k steps, so
   the science is settled and the bookkeeping is not. About 7k steps each.
4. **The untied contrast when 5270 finishes.** It decides whether gate G2 was
   lost to weight tying or to the loop itself, which is the difference between
   a repairable claim and an abandoned one.
5. **The H3 confirmation set**, blocked on seeds. Family B has 5 accepted
   cells; A has 2 usable seeds and C has 3, against the 5 the prereg requires.
   5283 and 5272 are training the rest.

## 5. Things that will bite a fresh session

**Every instrument is mutation checked, and the first version of a test has
passed against deliberately broken code three times.** A mocked ssh that
ignored the command it was handed, a substrate test comparing images that
differ anyway, and a hop-rate metric returning the same number for opposite
mechanisms. See L-011, L-015, L-016, L-017. Do not trust a new test until
it has failed on purpose.

**A perfect score gets three controls**, not one: the blank image control,
the input ablation in `analysis/ablate_input.py`, and an analytic
order-blind ceiling. The solved D4 model needs its state and both operators
and correctly ignores the distractors, and its 1.0000 sits above a computed
ceiling of 0.8125. That is why the score is believed. **The S3 ceiling is
0.7500**, computed exactly over all 1176 unordered operator pairs and
registered before the substrate runs, so it is not outstanding work.
`findings.md` carries both: 0.8125 for both D4 arms, 0.7500 for both S3
arms, identical within a row because a row is one group.

**Two gates have been moved and both are recorded.** G1.3 demanded an
unachievable chance-level score (D-016) and G2 was undefined where it was
written (D-030). Both were moved because they returned no information
wherever they pointed, not because they failed. That distinction is the one
to preserve.

**Superseded entries are kept in place with pointers**, never deleted: the
abelian quotient reading, the family A depth wall as first stated, the
single-seed curve result, and the "S3 is genuinely hard" conclusion. The
wrong version is how the right one was reached.

**Editing source reaches jobs that are already running.** PBS spools the
`.pbs` script at submit time and nothing else, so a queued or running job
reads `src/` and `configs/` fresh from disk. Editing either while a run is
in flight changes what that experiment computes. Editing a `.pbs` file is
safe by the same rule, because the copy the job runs was taken at `qsub`.

**There is no preregistration and no `prereg-v1` tag.** The file has never
existed in the working tree or anywhere in git history. Writing and
freezing it is milestone 5 work and milestone 5 is NOT STARTED. CLAUDE.md
asserted the freeze in the present tense until 6 September.

**Use `afterany` for dependencies, never `afterok`, and do not trust
`qalter` to change one.** `afterok` never fires if the job it waits on exits
non-zero, and a held job produces no error, no output and no completion, so
it fails by going quiet. `qalter -W depend=...` returns 0 and **appends**
rather than replaces, leaving both conditions in force. The fix is `qdel`
and resubmit, which is free before a run directory exists. See L-021.

**The cluster remote is a file that has to be placed.** `origin` on the
cluster points at `/home/<cluster-user>/loopvision.bundle`, which does not
persist between syncs. To ship a commit: `git bundle create` locally, copy
it to that path, then `git pull origin master` on the cluster. A pull with
no bundle in place fails and says nothing useful about why.

---

## 6. Operational

The machine running the VPN has had repeated power cuts. Cluster jobs are
unaffected, the scheduled watcher survives them, in-session monitors do
not. If a long silence happens, check rather than assume.

**The GPU node is core bound and there is no scheduled drain to wait for.**
Measured 6 September: 96 of 96 cores assigned with 41 jobs resident, while
memory is nowhere near binding. GPUs are claimed atomically so concurrent
jobs stop landing on the same card.

**Ask for two cores. Larger requests are backfilled past, not queued
behind.** On 7 September, 5269 at six cores sat queued while 5270 and 5272
at four cores, both submitted after it, started and ran. The scheduler
fills the gaps that exist, and the gaps here are small, so a larger request
does not wait its turn, it waits indefinitely while smaller jobs overtake
it. Four jobs were resized for this reason in one afternoon: 5269, 5271,
5277 and 5278, all carrying work ranked above what was overtaking them.

The cost of two cores is `PER_RUN=1` and `WORKERS=0`, in-process
generation, which D-025 puts at roughly 30 to 40 percent slower per run.
That is the right trade against not running. **Pack runs into a job rather
than cores into a job**: `max_run` counts jobs and not runs, so three runs
in one two core job costs one slot, and three separate jobs cost three.
See D-041.

**Neither memory nor walltime is enforced.** One neighbouring job holds
1041 GiB against no request at all, so the 32gb default. Several jobs have
used more than ten times the walltime they asked for, one at 286 hours
against a 24 hour request. Two consequences: a memory request protects
nothing, and a neighbour's walltime tells you nothing about when its cores
come back.

**The gpu queue caps a job at `mem=64gb` and `ncpus=16`, and a user at 5
running jobs and 16 running cores.** The memory cap is recent and refuses
anything larger at submit time. `multirun.pbs` asked for 96gb until
6 September, which never bit only because every real submission overrides
it on the command line. See D-031.

**workq is a separate machine pool and is mostly idle.** Six nodes,
`<cpu-node-01>` to `<cpu-node-06>`, 64 cores each, no GPUs, and no `resources_max`
at all. Four of the six were completely idle on 6 September. Nothing that
needs a GPU can go there, but CPU-only analysis never has to queue behind
the gpu node famine.

---

## 7. Open decisions, unchanged

D-003 compute beyond the current block, D-004 co-authorship, D-005 whether
to post a preprint before the deadline, D-006 which family to prioritise.
D-006 now has evidence behind it: family B carries H1, family A carries the
mechanism.

**Resolved on 6 September by D-033, and not the way it was framed.** This
section previously recorded TF32 as an untaken decision degrading fp32
matmuls to a 10-bit mantissa. Measurement says otherwise: in torch 2.4.1
`matmul.allow_tf32` is False and `float32_matmul_precision` is `highest`,
so the arithmetic every quoted number rests on is already full fp32. The
claim came from a note written against PyTorch before 1.12, where the
default was the other way. Only cudnn convolutions have TF32 on, and each
model holds exactly one convolution, the patch embedding. Left at the
defaults for the campaign, because changing them now would break
comparability with 44 finished runs.
