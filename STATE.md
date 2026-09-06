# What a Loop Sees: state as of 6 September 2026

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

## 1. The three results the paper rests on

**H1 is supported, two milestones early.** Depth raises the loops needed in
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

| job | experiment | what it decides |
|---|---|---|
| 5180, 5181 | s3_spatial seeds 0 and 1 | closes the 2x2. Substrate reading predicts it solves cold. At 54600 of 300000 |
| 5188 | d4_colour depth 1 donor | feeds 5189 |
| 5189 (held) | d4_colour curriculum and control | does the rescue generalise across groups |
| 5190 | scale control, wide, deep, big | the reviewer objection: was the model too small |
| 5191 | depth ladder, d4only depths 3 to 6 | could give family A a usable depth axis, which H1 lacks |
| 5206 (running), 5207 | gate G2 baselines, feedforward and echo | the hard-stop gate |
| 5211 | d4_colour feedforward control, two seeds | is the deadlock about looping or about composition |
| 5212 | s3only feedforward control, two seeds | the same question in the group the deadlock was found in |

**5202 no longer exists.** It asked for four cores to run four G2 baselines
and was replaced on 6 September by 5206 and 5207, two cores each, because
cores are the binding constraint and a smaller request fits a smaller gap.
The cost is `WORKERS=0` and in-process generation. D-025 puts one worker at
roughly 4000 samples per second against a GPU consuming 1500 to 3000, so
expect these two to be slower per step than 5180 and 5181 are.

**In flight, and not results.** Both s3_spatial seeds read **1.0000** at
50000 steps, which is the direction the substrate reading predicted in
advance rather than a pattern found afterwards. The colour half of the
2x2 is no longer in flight: d4_colour finished at chance in both seeds.
If the s3_spatial pair holds to its budget the 2x2 closes on substrate
with no group effect, solved on position for both groups and chance on
colour for both. Neither s3_spatial run has written DONE, and a near
perfect score still owes the three controls described in section 5, so
that half is not a finding yet.

**5180 and 5181 will probably not finish inside their wall.** They are at
18.2 percent of the step budget having spent 18.3 percent of `MAX_HOURS=20`,
so a straight extrapolation needs about 20.1 hours against a 20.0 hour
limit. They will most likely stop a few thousand steps short and chain.
The chain itself is sound, but the successor queues into a node with no
free cores, so the 2x2 could stall for days over the last one percent.
Nothing can be done to a running job here: `qalter` is blocked and
`MAX_HOURS` is fixed at submit time.

---

## 4. What to do next, ranked

1. **Read gate G2 when 5206 and 5207 land.**
   `python -m loopvision.analysis.gate_g2`. It can fail: Gao et al.
   arXiv 2607.16051 report parameter scaling usually beating looping at
   matched compute. A failure is reported, not worked around.
2. **The feedforward control is built, verified and not yet queued.** Does
   a non-looped model of matched depth deadlock on colour too? If yes the
   finding is about compositional learning in general rather than about
   looping, which is a different paper. The configs are
   `famA_d2_d4colour_ffwd` and `famA_d2_s3only_ffwd`, two seeds each in
   `configs/sweep/ffwd_control.yaml`, each derived from its looped twin so
   the diff is three lines. Both build 20 blocks against the 20 their twin
   executes at k=8, checked on the cluster rather than assumed. Read D-032
   before reporting either outcome: matched compute is not matched
   parameters here, so a control that deadlocks is clean and a control
   that solves the task leaves parameter count as a rival explanation.
   **Both pairs are queued, d4_colour as 5211 and s3only as 5212**, two
   cores and two seeds each. d4_colour is the cell the substrate result
   rests on; s3only is the group the deadlock and its curriculum rescue
   were originally found in, so the control covers both.
3. **Read the depth ladder.** If d4only solves depths 3 to 6, family A
   gains a depth axis and H1 gets a second independent family.

---

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
ceiling of 0.8125. That is why the score is believed. The ceiling for S3 is
a different number and has not been computed yet.

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
memory is nowhere near binding. Shrinking a core request to fit a gap has
started jobs hours earlier on five occasions. GPUs are claimed atomically
so concurrent jobs stop landing on the same card.

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

An untaken decision, not yet numbered: **TF32 is not set anywhere in the
repository**, so fp32 matmuls run at a 10-bit mantissa by default on these
cards. That is an inherited default rather than a choice, and the
instruments quote numbers to four decimals.
