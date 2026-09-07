# What a Loop Sees

Mechanistic interpretability of looped (depth-recurrent) vision models.
Full spec: docs/IMPLEMENTATION.md. Read the relevant section before writing code.

## Document map
- STATE.md                 read this first. Where things stand, what is running, what to do next.
- docs/IMPLEMENTATION.md  how to build it. Sections 0 to 15.
- docs/paper.md           what it claims, and what it says under every outcome.
- docs/gap-analysis.md    who else is in this space and where our novelty actually sits.
- docs/progress.md        milestone status, gate ledger, risks, session log. Update every session.
- docs/findings.md        every result, with run_id provenance.
- docs/decisions.md       append-only. Every deviation, dependency, and fallback.
- docs/learnings.md       mistakes, near misses, and the rule each produced.

## Non-negotiables
- Never tune a task, threshold, seed count, or test to improve a result. Failures get reported.
- Every plotted number must be readable from a stored metrics file. Plot scripts do not compute statistics.
- No em dashes or en dashes anywhere, including comments and docstrings.
- docs/preregistration.md is frozen at git tag prereg-v1 as of 2026-09-07. It
  is never edited. Every deviation goes in docs/decisions.md with a date and a
  reason. Read its Section 0 before citing it: H2 and H4 are pre-registered in
  the ordinary sense, H1 and H3 bind only data that did not exist when it was
  frozen, and their current results are exploratory.
- Counts and totals in prose come from analysis/registry.py output, never typed by hand.
- No citation enters a draft until its arXiv abstract page has been fetched and the version confirmed.
- Family C is never cut. It is the breadth arm, and without it H1 is a replication rather than a dissociation.
- After any state-changing cluster command, read the state back and assert on
  what it actually says. An exit code of 0 means the command ran, not that it
  did what was asked. See L-021.

## Environment
- PyTorch cu121 only (driver 525, CUDA 12.0). Never install cu124 or later builds.
- PBS cluster, qalter is blocked, jobs die at the wall. Every run must resume from checkpoint and self-chain.
- Request ncpus=8. Select the GPU with scripts/pick_gpu.sh, never assume CUDA_VISIBLE_DEVICES.

## Workflow
- One milestone at a time (docs/IMPLEMENTATION.md Section 11). Do not start N+1 until N's command exits 0.
- Gates G1 (task validity) and G2 (loops beat baselines) are hard stops. If one fails, stop and report.
- Add a dependency only with a line in docs/decisions.md.
- Instruments use src/loopvision/instruments/hooks.py. Never reimplement the loop inside an instrument.
- New tests get mutation checked: break the thing on purpose and confirm the test fails.
  A test that passes against a deliberately broken implementation is not evidence.

## Commands
    pytest                                           # bare. addopts adds -q, a second -q hides the count
    python -m loopvision.train.smoke --run-id <id> --steps 1000 --resume auto
    python -m loopvision.train.cli --config configs/base.yaml --run-id <id> --resume auto
    python -m loopvision.analysis.registry            # run index and counts
    RUN_ID=<id> CONFIG=<cfg> qsub scripts/submit.pbs
