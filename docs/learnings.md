# Learnings

What we got wrong, what it cost, and the rule that exists now because of it.

This file is not a diary and it is not a list of good intentions. Every entry must name a concrete failure, the cost it imposed, and a rule that is enforced somewhere specific: a test, a gate, a script, a checklist line in another file. A lesson with no enforcement point is a lesson we will learn again.

**Sections 1 and 2 are seeded before any code exists.** The cost of those lessons has already been paid, by the prior submission in this line and by the text-case work. Re-learning them would be the most expensive mistake available to this project.

**Entry format:**

```
### L-NNN. Short title
**Source:** prior work | this project, milestone N
**What happened:** 
**Cost:** 
**Root cause:** 
**Rule now:** 
**Enforced by:** the specific test, script, gate or checklist line.
```

---

## 1. Inherited from the prior submission

Each of these cost real time on the previous paper in this line. Spec Section 10.

### L-001. Artifact and anonymity items left to the final week
**Source:** prior work
**What happened:** the submission went in with an ANON-PENDING code link and a TODO in the AI use statement.
**Cost:** reviewer-visible incompleteness on items that were entirely under our control and cost nothing but calendar time to close.
**Root cause:** the items had no milestone, so they defaulted to the last week, which is the week with the least slack.
**Rule now:** anonymous code link, artifact DOI and AI use statement are all closed at milestone 10, four months before submission, not in the final week.
**Enforced by:** the standing checklist in `progress.md`, which lists all three with milestone 10 due dates and is reviewed at every milestone transition.

### L-002. A promised sweep that had not run
**Source:** prior work
**What happened:** the submission described a confirmation sweep that had not been executed. It was run later and it did not confirm.
**Cost:** a claim in a submitted paper that the data did not support.
**Root cause:** writing ran ahead of computing, and the draft had no mechanism forcing every described run to exist.
**Rule now:** if it is not in `runs/`, it is not in the paper. No exceptions for runs that are queued, or nearly done, or certain to work.
**Enforced by:** milestone 9 requires every headline number in `docs/claims.md` to have a confirming run id, and `findings.md` Section 6 tracks post-hoc claims through confirmation. The jitter-width sweep that was promised last time is scheduled up front at milestone 8 rather than promised at submission.

### L-003. A hand-typed count that was wrong
**Source:** prior work
**What happened:** the draft carried an invented count, 329, where the true number was 344.
**Cost:** an incorrect number reached review.
**Root cause:** a number was typed from memory because regenerating it was mildly inconvenient at the time.
**Rule now:** every count and total in prose comes from `analysis/registry.py`, printed by a script, pasted with the date it was generated.
**Enforced by:** `findings.md` Section 8, which holds every paper count with its generating command and date, and the provenance rule at the top of that file.

### L-004. A reciprocal statistic with no inclusion gate
**Source:** prior work
**What happened:** a speed-like quantity defined as 1/slope blew up near zero slope. One checkpoint at slope 0.036 produced 27.6 and shifted its group mean by 48 percent.
**Cost:** a headline statistic that was driven by a single near-degenerate point.
**Root cause:** the inclusion criterion was chosen after seeing the fits, which is unfalsifiable by construction.
**Rule now:** any statistic that is a reciprocal of a fitted slope ships with an explicit inclusion gate defined before the fits are run, plus a sensitivity script reporting the headline both ways.
**Enforced by:** `gate_sensitivity.py` as a required deliverable of M4, and the statistical hygiene log in `findings.md` Section 7.

### L-005. Convergence diagnostics lie in this model family
**Source:** prior work, text case
**What happened:** recurrent states converged in direction while raw norm kept growing, up to 316x in the small-model regime. Update-norm halting criteria therefore misclassified convergence.
**Cost:** would have produced wrong conclusions about where computation finishes, had the direction-normalised quantity not been logged.
**Root cause:** the natural, cheap diagnostic measures the wrong thing, and it measures it plausibly enough that nobody checks.
**Rule now:** never use an update-norm plateau as evidence of convergence without the direction-normalised counterpart beside it. Log both raw and direction-normalised quantities from day one, during every evaluation pass, so the question is answerable without a re-run.
**Enforced by:** M4 logs all of norm, delta norm, cos(s_i, s_i-1), cos(delta_i, delta_i-1) and the convergence-to-growth ratio on every evaluation pass, not as a separate study. This is also H2 and contribution 4 of the paper.

### L-006. Distribution-free tests on heavy-tailed reparameterisations
**Source:** prior work
**What happened:** Silverman rejected where Hartigan's dip did not, on a heavy-tailed reparameterisation of a bounded quantity.
**Cost:** a bimodality claim that depended on which test was chosen.
**Root cause:** a single test was run, on a transformed quantity whose transformation violated the test's assumptions in a way that was not obvious.
**Rule now:** any bimodality or clustering claim runs at least two tests with different assumptions, and both outcomes are reported including disagreement.
**Enforced by:** `findings.md` Section 7, statistical hygiene log.

### L-007. The echo baseline is not optional
**Source:** prior work
**What happened:** the echo baseline, where the core is replaced by identity plus injection so the state is refreshed but not learned-updated, caught a real problem in the text-case work.
**Cost:** would have been a trivially explainable result presented as a mechanism finding.
**Root cause:** the interesting-looking curve had a boring explanation available, and only a baseline designed to expose boring explanations found it.
**Rule now:** baselines that exist to rule out trivial explanations run before the claim is made, not in response to a reviewer.
**Enforced by:** gate G2 at milestone 4 requires beating both the matched-compute feedforward and the echo baseline before any modelling claim proceeds. It is a hard stop.

### L-008. Public depth-recurrent models hit a composition ceiling
**Source:** prior work
**What happened:** two public depth-recurrent language models tracked at one and two composition steps and collapsed at three, so the intended curve could not be fit at all.
**Cost:** a planned analysis was impossible, discovered late.
**Root cause:** the models' actual capability range was assumed rather than measured first.
**Rule now:** for the Stage 2 graft, measure the composition depth ceiling before planning any analysis that depends on depth greater than 2. If the same ceiling appears, that is the result and it goes in the main text rather than an appendix.
**Enforced by:** `findings.md` Section 4, Stage 2 table, which has a composition depth ceiling row.

### L-009. Deterministic loop counts produce wrong-attractor landings
**Source:** prior work, text case
**What happened:** deterministic loop counts during training produced wrong-attractor landings at 16 of 44, versus 5 of 75 with jitter, p = 7.0e-5.
**Cost:** none yet. This is a lesson we are acting on in advance.
**Root cause:** training at a fixed loop count lets the model settle into solutions that do not survive a change in k.
**Rule now:** jitter is on by default. The jitter-width sweep, including the width zero arm the prior sweep lacked, is scheduled at milestone 8 rather than promised at submission.
**Enforced by:** H4 in `findings.md`, with the width zero row already in the table.

### L-010. Mathematics in the abstract and introduction
**Source:** prior work, Prof. Garg's standing rule
**What happened:** formalism appearing early costs readers who decide in the first page whether the paper is for them.
**Cost:** reviewer attention, which is the scarcest resource in the process.
**Root cause:** writing the paper in the order the work was done rather than the order a reader needs it.
**Rule now:** motivation first, "what could nobody do before" on page one, no formalism before the formal setup section.
**Enforced by:** `paper.md` Sections 2, 5 and 10.

---

## 2. Standing traps specific to this project

Not yet mistakes. Failure modes visible from the start, recorded so that hitting one is recognised immediately rather than after a week.

### T-001. Resolution creep
The compute estimate only works because the resolution is small. Any proposal to move above 32x32, outside the single stated robustness check at 64x64, breaks the schedule rather than merely stretching it. Tracked in the `progress.md` risk register.

### T-002. The generation arm
H5 and the generation stack are out of scope and are the identified creep vector. They are attractive precisely because they would make the paper feel bigger. They would also make it late.

### T-003. Tuning a task until a gate passes
Gate G1 can be made to pass by deepening the task. That is legitimate and the spec allows it. What is not legitimate is failing to report that it happened. Every gate attempt goes in `findings.md` Section 2 with what changed between attempts.

### T-004. The instrument path and the training path disagreeing
If a forward pass at k and the incremental state sequence captured by the M4 hooks can silently differ, every downstream measurement is suspect and nothing will look obviously wrong. Enforced by `tests/test_model_invariance.py` at 1e-5 tolerance, which is a milestone 3 deliverable and not optional.

### T-005. Plot scripts that compute statistics
A plotting script that computes a statistic is a bug, because it creates a number with no stored provenance, which is how L-003 happens. Plot scripts read metrics files and render.

### T-006. Silent double-counting of seeds
A sweep that re-runs a config whose hash already has a DONE sentinel will silently inflate the seed count. The registry refuses to start such a run unless `--force` is passed. If `--force` is ever used, it gets an entry in `decisions.md`.

### T-007. Jobs that cannot survive being killed
`qalter` is blocked on the cluster, so jobs die at the wall with no warning and no extension. Any training script that cannot survive SIGKILL at a random step and resume bit-exact is broken, not merely fragile. This is a milestone 0 deliverable for exactly that reason.

---

## 3. Learned on this project

Empty until we make our own mistakes. Newest at the bottom. When adding an entry, check whether it is a repeat of something in Section 1 or Section 2, and if it is, say so explicitly in the entry, because a repeat means the enforcement point failed and the enforcement point is what needs fixing.

*(no entries yet)*

---

## 4. Near misses

Things caught before they cost anything. Worth recording because the catch mechanism is the reusable part, and because a near miss is evidence a guardrail is working.

### N-001. Nearly cited a paper whose headline results were withdrawn
**Source:** this project, pre-milestone-0
**What happened:** arXiv 2604.09870 entered `gap-analysis.md` from a curated list as a relevant probing study. On verification, v2 carries a prepended erratum: a post-publication audit found the three headline results inflated by two independent evaluation errors, a canonical-ordering artifact and train-test leakage. A 95.2 percent evaluator accuracy became 63.9 percent once antisymmetrised on the full test set, and the central effect fell to 2.3 percentage points.
**Cost:** none. Caught at week zero rather than in review.
**Caught by:** the D-008 rule that no citation enters a draft until its abstract page has been fetched. The curated list showed v1's claims with no indication anything was wrong.
**Rule now:** verify the **version**, not only the identifier. A paper can be clean at v1 and corrected at v2, and citation managers routinely pull v1 metadata.
**Enforced by:** `gap-analysis.md` Section 9, which now requires version confirmation at bibliography freeze.

**The methodological lesson, which is the more valuable half.** The author reports that the two errors were "mutually invisible", each concealing the other, so that finding them required both a split audit and an antisymmetrisation check. Neither alone would have surfaced anything. This is a direct warning about our own instruments: a single validity check that comes back clean is weak evidence when two plausible errors can cancel. Gate G1 already reflects this by running four checks rather than one, and the M2 patching normalisation, where 0 is the corrupted run and 1 is the clean run, should be audited in both directions rather than only checking that clean recovers to 1.

---

## 5. Recurring themes

Filled in during milestone 9 and again before writing. If three entries in Section 3 share a cause, the cause is the real lesson and it belongs here.

*(no entries yet)*
