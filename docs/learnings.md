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

Newest at the bottom. When adding an entry, check whether it is a repeat of something in Section 1 or Section 2, and if it is, say so explicitly in the entry, because a repeat means the enforcement point failed and the enforcement point is what needs fixing.

### L-011. A passing test that could not fail
**Source:** this project, milestone 0
**What happened:** `tests/test_resume.py` passed all six tests. To check the tests had teeth, `restore_rng_state` was mutated into a no-op, so that resumed runs would restore model, optimizer and data cursor but not a single RNG stream. **All six tests still passed.** The smoke trainer at that point was a plain MLP whose forward pass consumed no global RNG after initialisation, so RNG restoration genuinely did not affect the result and the test could not have detected a bug in it.
**Cost:** none, caught at milestone 0. The cost had it survived would have been large and delayed: the real trainer samples a loop count per batch from the global stream (IMPLEMENTATION.md Section 6.1), so RNG restore becomes load bearing at milestone 3. A silently broken restore would have made every resumed run a slightly different experiment, and resumed runs are the normal case on a cluster where jobs die at the wall.
**Root cause:** the test target was built to be simple rather than to be representative. A smoke model that does not exercise the mechanism under test cannot test it, no matter how thorough the assertions look.
**Rule now:** two rules, and the second is the general one.
1. The smoke trainer is a miniature of the real architecture, not an arbitrary stand-in. It is a looped model with per-batch loop-count sampling from the global RNG stream, because that is the structure whose resume we actually need to guarantee.
2. **Every new test gets mutation checked.** Break the thing it claims to test, on purpose, and confirm the test fails. A test that passes against a deliberately broken implementation is not evidence, and the effort to find that out is about two minutes.
**Enforced by:** `CLAUDE.md` workflow section, which now carries the mutation-check rule; the docstring on `SmokeModel`, which records why it is looped so nobody simplifies it back; and the confirmed failure mode, three tests failing with "loss diverged at step 8" once the mutation is reapplied.

**Connection to N-001.** The same week, a paper entered the survey whose author reported two evaluation errors that were "mutually invisible", each concealing the other, requiring two independent checks to surface. This is the same shape: a single check that comes back clean is weak evidence. That is now twice in one week that the lesson has appeared from different directions, which is worth noticing.

### L-012. A perfect score is a reason to look harder, not to celebrate
**Source:** this project, milestone 3
**What happened:** family B returned 1.0000 accuracy on all four runs, both depths, both seeds. The user asked whether 100 percent was even possible or whether something was wrong on our side. It was. At depth 1 the target sprite **is** the anchor, and the query identified the anchor by its colour and shape, so a question asking for colour or shape contained its own answer: 66.8 percent of samples were answerable from the query tokens alone, and a trained model scored **0.833 on a blank image**.
**Cost:** four training runs wasted, and gate G1.4's number invalidated because the same review turned up a second fault in family C. Cheap only because it surfaced in week one.
**Root cause:** the task was validated for *internal consistency*, never for *whether the image was needed*. Every family B test checked that the label followed correctly from the program, and they all passed, because the label did follow correctly. None asked whether the label could be obtained without looking at the picture.
**Rule now:** every task family gets a **blank-image control**. Feed the trained model a blank canvas with the real query and record the score. Whatever it gets is the floor that needs no vision, and any claim about the task has to clear it. This is one forward pass and it should have existed from the first family.
**Enforced by:** `test_anchor_never_names_the_attribute_being_asked_for` in `tests/test_families_bc.py`, and the blank-image measurement recorded in `findings.md` for both families.

**The second fault, found in the same pass.** Family C had no leak but was not a counting task either: 53.5 percent of labels were zero, 93.4 percent were 0 or 1, the count never exceeded 4 of a cap of 10, and a constant "0" scored 0.535. It exists to stress **breadth**, and the answer barely moved when scene size went from 4 sprites to 16. The axis was in the data and not in the task.

**And a test of ours was set at the wrong place.** `test_family_c_labels_are_not_degenerate` asserted fewer than 75 percent zeros and at least three distinct counts. It passed the whole time. It checked that the label varied *at all*, when the property that matters is that it varies *with breadth*. A test can be true, passing, and still not be testing the thing its name claims. Replaced with two that assert the mean count rises with breadth and that no constant answer beats 0.35.

**Connection to the running theme.** This is the fourth time (see [[L-011]], N-001, N-002) that a clean-looking result was wrong and only deliberate adversarial checking found it. The pattern is now unmistakable: for this project, "the number looks good" carries almost no information until someone has tried to break it. The blank-image control is the cheapest instance of that discipline and now applies to every family.

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

### N-002. A shortcut probe that reported no shortcut because of a tie break
**Source:** this project, milestone 1
**What happened:** the first implementation of `bag_of_operators_ceiling` enumerated every ordering of the operator multiset, took the modal composite with `Counter.most_common`, and checked whether it equalled the true target. It reported a ceiling of **1.000** at depth 2. `itertools.permutations` yields the input ordering first, so on a tie `most_common` returns whichever composite was inserted first, which is the true answer by construction. The estimator was reading back its own input.
**Cost:** none. It failed a test written expecting roughly 0.5, which is the only reason it surfaced.
**Root cause:** measuring a quantity by simulating a predictor, where the simulation had access to information the real predictor would not have. The tie break was the leak.
**Rule now:** where a quantity has a closed form, compute the closed form rather than simulating an agent that estimates it. The fix computes `E_M [ max_g P(g | M) ]` directly, which has no tie-breaking step to leak through.
**Enforced by:** a docstring on `bag_of_operators_ceiling` naming this specific trap, because the modal-composite implementation is the obvious one and someone will reach for it again.

**Why this is a near miss rather than a curiosity.** Had the number come out plausible instead of a suspicious 1.000, it would have been recorded as evidence that the anti-shortcut sampler worked perfectly, and gate G1.3 would have passed on a measurement error. The true ceiling is 0.652 at depth 2. Same shape as [[L-011]] and N-001: a check that comes back clean is weak evidence until you have tried to make it come back dirty.

---

## 5. Recurring themes

Filled in during milestone 9 and again before writing. If three entries in Section 3 share a cause, the cause is the real lesson and it belongs here.

**Emerging already, three times in one week.** L-011 (a resume test that passed against a disabled RNG restore), N-001 (a paper whose two errors were each invisible without the other), and N-002 (a shortcut probe that leaked its own answer through a tie break). In every case a check returned a clean result that was wrong, and in every case the only thing that caught it was deliberately trying to make it fail. This is on track to be the methodological theme of the project, and it argues for treating "the validation passed" as the beginning of a check rather than the end of one.

### L-013. A rule that lives in a document gets skipped, one that lives in a module does not

L-012 made "a perfect score is a reason to look harder" a standing rule,
and it was still a habit rather than a check: the blank-image control that
found both faults was run by hand, twice, because someone thought to.

The gap showed immediately. The family C gate re-run came back at 1.000 at
every breadth, the same number the broken task produced, and nothing in the
pipeline would have questioned it.

`analysis/blank_control.py` now reports real, blank-image, and best-constant
accuracy together, and names the gap between the last two as a leak. It
compares against the best constant rather than chance on purpose: a model
riding the label prior is not leaking, and flagging that as a fault would
teach us to ignore the instrument. Mutation checked three ways, including
one that widens the margin until nothing is ever flagged.

**Why:** the faults were not caught by a failing test, they were caught by
a question. Questions do not run in CI.

**How to apply:** when a manual check finds something a test suite missed,
the check becomes a module before the session ends. See [[l-012-a-perfect-score-is-a-reason-to-look-harder]].

### L-014. When two hypotheses predict the same number, the number is not the evidence

The depth 2 plateau was recorded as "the model learns the abelian
quotient", on the strength of accuracy 1/6 and loss ln(6). Both are also
exactly what a model that has learned the whole D4 factor produces, because
the abelianisation and the D4 factor both have order 8. The two readings
are opposites: one says the model is avoiding sequential computation, the
other says it is doing it. The finding asserted the first for three days.

What broke it was not new data. The refuting measurement, the S3 sign at
0.4999, was already in the original table. The abelian reading requires
that number to be 1.0000, and nobody checked, because the headline
accuracy already agreed with the story.

The entry did flag a residual it could not explain, and that residual was
the whole clue: it came from a prediction of 1/12 that assumed two bits
when the hypothesis being defended implied three.

**Why:** a hypothesis that explains the headline number can still be
contradicted by a number already sitting next to it. Agreement with the
statistic you were looking at is the weakest kind of evidence there is.

**How to apply:** before recording a mechanism, write down what a rival
mechanism would predict for every quantity already measured, not just the
one that prompted the claim. If no measured quantity separates them, the
mechanism is not established yet and the entry says so. `parity_probe.py`
now does this in code: `interpret()` returns a refusal when the numbers fit
neither reading cleanly. See [[l-012-a-perfect-score-is-a-reason-to-look-harder]]
and [[l-013-a-rule-that-lives-in-a-document-gets-skipped]].

### L-015. A summary statistic can report the same number for opposite mechanisms

The coda lens was built to measure hops advanced per core pass. Its first
real output was "0.00 hops per pass" for all three seeds, which reads as
these models never advance. They in fact reach the final answer on the
first pass and never need to advance, which is the opposite situation.

Both produce a flat frontier and so both produce a slope of zero. The
statistic was correct and the reading it invited was backwards.

**Why:** a rate summarises a shape, and different shapes can share a rate.
Reporting the rate without the shape hands the reader a number whose
meaning depends on information that was discarded to compute it.

**How to apply:** when an instrument reduces a curve to one number, it
must also report which qualitative case produced it, and refuse to give
the number in the cases where it does not apply. `traversal()` returns
stalled, immediate or walks, and the hop rate is printed only for walks.
See [[l-011-a-passing-test-that-could-not-fail]] and
[[l-014-when-two-hypotheses-predict-the-same-number]].

### L-016. A mock that ignores its input tests only the code after the mock

The cluster run watcher was verified against a fake `ssh` that printed a
canned reply and never looked at the command it was handed. Every test
passed. The command was broken: the run list spans several lines for
readability, and those newlines were interpolated into `for r in $RUNS; do`
on the far side, where a newline ends the list early. The remote shell
answered with a syntax error, stderr was discarded, and an empty reply is
exactly how the script detects an unreachable cluster.

So it would have reported UNREACHABLE on every firing, forever, while ssh
was perfectly healthy, and the tests would have kept passing. It was caught
only because the user asked for a manual run.

The same shape had already been used for the blank-image control's tests,
where it happened not to matter. That is luck, not design.

**Why:** a mock replaces the boundary. Whatever crosses that boundary is
then untested by construction, and the thing crossing it is usually the
thing most likely to be wrong, because it is the part written in another
language for another interpreter.

**How to apply:** a mock must assert on what it received, or the suite must
include one real call. For anything built as a string and executed
elsewhere, prefer the real call: it costs a second and it is the only test
of the string. See [[l-011-a-passing-test-that-could-not-fail]].

### L-017. A config key that is parsed but never read

Three appeared in a single day. `eval_k_sweep` was wired correctly.
`factor`, which selects a subgroup of D4 x S3, was not. `prelude_blocks`,
`core_blocks` and `coda_blocks` were not.

The last of those was the dangerous one. A config asking for a 1/1/1 core
would have trained a 2/2/2 model, produced another flat loop-count sweep,
and left nothing in the run directory to say why: the yaml would have read
1/1/1, the run would have honoured 2/2/2, and the conclusion would have
been that shrinking the core does not help.

**Why:** a key that is parsed has a plausible value everywhere it is
inspected. It is only wrong where it is used, and it is not used anywhere,
so nothing looks wrong.

**How to apply:** a new config key gets a test asserting on the **built
object**, not on the config dict. Copying a field into a dataclass is not
the same as the model being built with it. `test_the_block_counts_reach_the_built_model`
constructs the model and counts its blocks; it fails if the key is dropped
in `build_model`. See [[l-013-a-rule-that-lives-in-a-document-gets-skipped]].

### L-018. A rule that names a tool is violated by the tool not existing

CLAUDE.md lists `python -m loopvision.analysis.registry` as a command and
makes it a non-negotiable that counts and totals in prose come from its
output rather than being typed by hand. The module did not exist.

So the rule had been silently unenforceable since it was written, and
every count in the documents had been typed by hand, which is exactly the
failure L-003 already records happening once.

**Why:** a rule stated as a prohibition reads as satisfied when nobody is
doing the forbidden thing on purpose. Nobody was hand-typing counts *in
defiance* of the rule; there was simply no alternative and the gap never
announced itself.

**How to apply:** when a rule names a command, the command's existence is
part of the rule. Every path and command named in CLAUDE.md should be
checked to resolve. See [[l-003-a-hand-typed-count-that-was-wrong]].

### L-019. Eliminating one of two explanations does not establish the other

The factor experiment asked whether S3 is hard in itself or merely
neglected because D4 is easier and gets learned first. S3 alone, with
nothing competing, stayed at chance. Competition was ruled out, so I
concluded the alternative: genuinely hard.

A curriculum run then solved the same task in 10000 steps by starting from
a depth 1 version of itself, against 300000 steps of chance from a cold
start, with a wrong-donor control staying at chance. S3 was never hard. It
was a deadlock: the composition cannot be learned without the extraction,
the extraction gets no gradient without the composition, and removing a
competitor does nothing about that because the deadlock is internal.

**Why:** the experiment was sound and the inference was not. Two named
hypotheses feel exhaustive when both are stated, and the space of
mechanisms is not covered by the two that occurred to me. Eliminating one
leaves the other **plus everything unlisted**.

**How to apply:** when an experiment eliminates one of two explanations,
write down what would distinguish the survivor from *an unnamed third*
before recording the conclusion. Here that test was cheap and obvious in
hindsight: if the subtask is genuinely hard, handing the model a head start
should not rescue it. That is one run. See
[[l-014-when-two-hypotheses-predict-the-same-number]].

### L-020. A control that supplies its own target verifies nothing
**Source:** this project, milestone 4
**What happened:** the first version of `test_control_is_matched_compute` read the loop count k from the feedforward control's own config and computed the expected block count from it. Changing the control's `k_train` from 8 to 4 moved the expected value and the built value together, so the test passed against a control built at half the depth of the model it exists to match.
**Cost:** none, because the mutation check caught it before the control was queued. Unchecked it would have cost a 300000 step run answering a different question than the one asked, with nothing in the run directory to record the substitution.
**Root cause:** a comparison whose two sides are derived from the same source is not a comparison. The flaw survived writing because the docstring stated the correct rule, "read the expected count from the LOOPED config, not the control's own", and the code obeyed it for the block counts while still taking k from the control. A reader checking code against docstring sees the word looped and moves on.
**Rule now:** in a matched-X test every term of the expected value comes from the reference side, and the test additionally asserts that the control's own declaration equals it. Two assertions, because the second one names the failure rather than merely preventing it.
**Enforced by:** `tests/test_ffwd_control.py::test_control_is_matched_compute`, which reads k from the looped config and asserts `ffwd["k_train"] == k`. Mutation checked: k_train 8 to 4, steps divergence, and arch reverted to looped are all detected.

### L-021. A command that reports success without doing the thing
**Source:** this project, milestone 4
**What happened:** `qalter -W depend=afterany:5188 5189` exited 0. It had not replaced the dependency, it had **appended** to it, leaving `afterok:5188,afterany:5188`. Both conditions then had to be met, so the `afterok` that made the job hangable was still in force and the change had bought exactly nothing. Only reading the attribute back revealed it, and reading it back was almost skipped, because the command had said it worked.
**Cost:** none, this time. Uncaught, job 5189 would have looked repaired and still hung indefinitely the moment its donor exited non-zero, holding the D4 curriculum rescue behind a dependency that could never be satisfied, with nothing anywhere reporting a problem. A held job produces no error, no output and no completion.
**Root cause:** an exit code describes whether a command ran, not whether it accomplished what was intended. PBS received a well-formed request and applied its own semantics for a repeated `-W depend`, which is accumulation rather than replacement. Nothing was broken and nothing was wrong, and the result was still not what was asked for.
**The same shape appeared three times in one day, which is why it gets an entry rather than a shrug.** A mutation harness reported every check as MISSED because its `$`-anchored `sed` never matched CRLF line endings, so no mutation was ever applied and the tests were correctly passing on unmutated code. The documented `pytest -q` prints dots and no count, because `addopts` already supplies one `-q` and a second suppresses the summary, so the command that verifies the suite cannot tell you what it verified. And `qalter` here. In each case the tool succeeded, and in each case the thing the tool existed to do had not happened.
**Rule now:** after any state-changing cluster command, read the state back and assert on what it says, not on the exit code. The read-back is the evidence. The command is only the request.
**Enforced by:** a non-negotiable line in `CLAUDE.md`, and by the submit-then-verify pattern now used for every `qsub` in this project, which reads back `job_state`, the resource request and the full variable list before the job is treated as queued. See [[l-020-a-control-that-supplies-its-own-target]] for the sibling failure, where the check itself was the thing that was wrong.

### L-022. A control that is not wired in produces a perfect looking result
**Source:** this project, milestone 6
**What happened:** the M2 positive control, patching every position at every iteration, read exactly 0.0000 at all nine iterations. That is a clean, orderly, entirely believable negative result, and it was wrong. `patched()` yields a hook to hand to `forward(state_hook=...)` and deliberately does not mutate the model. The call site was `with patched(...):`, which discards the yielded hook, so nothing was ever patched and the model was asked what it thought about an intervention that had not happened.
**Cost:** none, because the control was run against a heatmap that had already been produced and was about to be interpreted. The heatmap read a maximum of 0.094 and a mean near 0.001, which is exactly what "the information is distributed" looks like, and also exactly what "the instrument is disconnected" looks like. Without the control there was no way to tell, and the distributed reading is the one that supports H3, so the pull was toward believing it.
**Root cause:** an API that returns the intervention rather than applying it. That is the right design, because it keeps the scope of a patch visible at the call site, and it means a call that ignores the return value fails silently rather than loudly. The docstring says so in its second sentence.
**Rule now:** every patching result ships with the full-state control computed in the same function, on the same items, and stored in the same file. The control at the last iteration is a mathematical identity: patching every position hands the coda the clean state, so recovery must be exactly 1 whatever the model has learned. Anything else means the intervention did not land, and the heatmap is not evidence of anything.
**Enforced by:** `instruments/m2_patching.py` computes the control inside `patch_grid` and writes it as position -1, and the CLI prints a warning when it falls below 0.9. `tests/test_m2_patching.py::test_control_recovers_exactly_one_at_the_last_iteration` asserts the identity on an untrained model, so it is a unit test rather than an experiment. Mutation checked: reverting to the discarded hook is detected. See [[l-021-a-command-that-reports-success-without-doing-the-thing]], which is the same shape one layer down.

### L-023. An instrument encodes an assumption about where the intervention lives
**Source:** this project, milestone 6
**What happened:** `hooks.recovery_grid` took one `query` and used it for both the clean and the corrupted run. That is correct for family A, whose operators are rendered into the scene, so its counterfactual differs in pixels and the query is genuinely shared. Family B puts the relation chain in the query and its twin is byte identical in the image, so passing one query made the two runs the same run. Every logit difference was zero, every cell NaN, and the driver reported 0 of 120 items admissible.
**Cost:** none. The positive control printed `nan` instead of a number and the admissible count printed zero, so the failure was loud. Without the control the file would still have been written, and an all-NaN heatmap plotted with a colour map that maps NaN to a background colour looks like a clean negative result.
**Root cause:** the instrument was written against the first family that needed it and quietly took on that family's shape. Nothing was wrong for family A and nothing warned when a second family arrived with its counterfactual somewhere else.
**Rule now:** an instrument used across task families states which axis it intervenes on and takes that axis as a parameter rather than assuming it. `recovery_grid` accepts `corrupt_query`, defaulting to the shared query, and the docstring says which family is which and what happens if the wrong one is passed.
**Enforced by:** `instruments/m2_patching.py` passes both queries, and the positive control travels with every grid, so an instrument that intervenes on the wrong axis reads NaN at the control rather than plausible numbers in the cells. See [[l-022-a-control-that-is-not-wired-in-produces-a-perfect-looking-result]].
