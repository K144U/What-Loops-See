# Release snapshot

This archive accompanies the preprint *A Rescue That Knows Its Group: Training Deadlocks and Partial-Credit Traps in Learning to Compose from Pixels*, in `paper.pdf`. It
was taken from commit
`359b4147e91996eb78c7fcff369a9ff9122d1a0a` of the project repository on 2026-11-12. Appendix A of
the paper names, for each claim, the decision log entry behind it and the
stored file it is read from; both are here.

## Contents

- `paper.pdf`, the preprint.
- `registrations/preregistration.md`, frozen at tag `prereg-v1`, commit
  `fd8d037df2a9ad48607d2ca88f1ff5888f3e2808`, 2026-09-07T10:35:53+05:30.
- `registrations/decisions.md`, the decision log, and
  `registrations/experiment-substrate.md`, the design of the substrate
  experiment.
- `registrations/ledger.csv`, the commit and time at which each decision log
  entry was first committed, 80 entries.
- `runs/`, 657 training runs in 77 arms: each run's
  config, metrics over training and evaluation, summary, and the output of
  every instrument run on it.
- `runs/_*/`, the stored outputs the verdicts and figures are read from,
  11 directories, each with the commit that wrote it.
- `figures/`, the 13 figures of the paper, each identical to
  the hash stored beside its inputs in `runs/_figures_20261112_d080/`.
- `code/`, the `loopvision` package, its tests and its configs at the same
  commit.
- `SHA256SUMS`, a hash of every other file here.

## Licences

The code under `code/` is under the MIT licence in `code/LICENSE`. Everything
else, the paper, the registrations and the data, is under Creative Commons
Attribution 4.0 International.

## Left out, and why

- Model checkpoints, about 96 GB. Every instrument the paper reports was run
  on them and its output is under `runs/`.
- 15 runs of three arms, and the per-run geometry files,
  which belong to concurrent work in submission:
- `famB_curve111_jit000`: the loop-count jitter sweep, concurrent work in submission
- `famB_curve111_jit025`: the loop-count jitter sweep, concurrent work in submission
- `famB_curve111_jit100`: the loop-count jitter sweep, concurrent work in submission
- The cluster's job scripts and their tests.
- The repository's history, which is not public. `ledger.csv` records when
  each registration was committed.

## Scrubbing

Host and user names of the cluster were replaced in text: home directories
by `~`, the cluster's name by `<cluster>`, user names by `<user>`. That
changed 659 files, 657 of them the per-run
`env.txt` package listings, and nothing else was edited.

## Rerunning

Python 3.10 or later, PyTorch 2 (the runs used the cu121 build).

    pip install -e code
    cd code
    python -m loopvision.analysis.escape_rates --runs ../runs
    python -m loopvision.analysis.m1_curves --runs ../runs --no-fit \
        --summary m1_summary.json \
        --config configs/m3/famB_curve_111.yaml \
        --config configs/m3/famC_breadth_111.yaml

Each figure script under `code/src/loopvision/analysis/plots/` reads one of
the stored inputs in `runs/_figures_20261112_d080/` and computes nothing;
`code/scripts/build_figures.sh` lists the command for each.
