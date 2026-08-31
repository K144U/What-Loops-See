# What a Loop Sees

Mechanistic interpretability of looped (depth-recurrent) vision models.
Targeting NeurIPS 2027.

One question: when the input is pixels, does the number of loop iterations a
task requires track the sequential depth of the computation, or the parallel
breadth of the scene, or neither?

Start with `docs/paper.md` for what this claims, `docs/IMPLEMENTATION.md` for
how it is built, and `docs/progress.md` for where it currently stands.

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -e ".[dev]"
    pytest -q

The cu121 index URL is not optional. The target cluster runs driver 525 with
CUDA 12.0 and cannot load a cu124 or later build.

## Status

Milestone 0. Repository skeleton and the checkpoint-resume guarantee.
No model code yet, by design: gate G1 in milestone 2 decides whether the
tasks are valid, and nothing above it is worth building until it passes.
