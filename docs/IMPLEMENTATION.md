# What a Loop Sees: Implementation Specification

**Project:** mechanistic interpretability of depth-recurrent (looped) vision models
**Target:** NeurIPS 2027 main track (fallbacks: TMLR, then ICLR 2028)
**Status:** implementation spec, August 2026. Supersedes nothing in the research proposal; this is the engineering companion to it.
**Audience:** Claude Code, working in a fresh repository, plus the human reviewing its output.

---

## 0. How to use this file

Put this file at `docs/IMPLEMENTATION.md` in the repo and put the short version (Section 13) at `CLAUDE.md` in the repo root.

Work milestone by milestone (Section 11). Each milestone has a definition of done that is a command that exits 0, not a description. Do not start milestone N+1 until milestone N's gate passes. Two gates are hard stops where the scientific validity of everything downstream depends on the answer (G1 in milestone 2, G2 in milestone 4); if either fails, stop and report rather than working around it.

Three standing rules for the agent:

1. **Never change a task difficulty, threshold, seed count, or statistical test to make a result look better.** If a pre-registered number fails, it fails, and it gets reported as a failure. The prior paper in this line reports fourteen failed registered predictions; that is a feature.
2. **Every number that appears in a plot or a paper draft must be readable back out of a stored metrics file**, not recomputed inline in a notebook. If a plotting script computes a statistic, that is a bug.
3. **No em dashes or en dashes in any generated prose, docstring, comment, or paper text.** Commas, periods, parentheses only.

---

## 1. The scientific contract, compressed

Everything below exists to answer one question: **when the input is pixels, does the number of loop iterations a task requires track the sequential depth of the computation, or the parallel breadth of the scene, or neither?**

The four hypotheses that the code must be able to resolve:

| ID | Status | Claim | Instrument | Falsified if |
|---|---|---|---|---|
| H1 | pre-registered | Minimum loops to threshold grows with composition **depth**, flat in scene **breadth**. The contribution is the dissociation, not the depth trend, which is close to known in text (arXiv 2604.07822, 2607.20594) | M1 | breadth coefficient is significantly nonzero, or depth coefficient is not |
| H2 | pre-registered | Halting criteria are biased against the oracle, and the bias grows with composition depth. Measured as the signed error between where each criterion fires and `k*_model`, regressed on the known `k*_task`. See Section 7 M5 | M5 | the signed error is indistinguishable from zero at the pre-registered tolerance **and** its slope against `k*_task` is indistinguishable from zero. Both must hold |
| H3 | pre-registered | Activation patching over (loop, patch position) is loop-localised and spatially distributed on depth tasks, loop-diffuse and spatially local on breadth tasks. **The lead result**, see `paper.md` | M2 | patching heatmaps do not separate by task family |
| H4 | exploratory | Loop-count jitter during training reduces wrong-attractor landings, replicating the text-case effect. **Cite arXiv 2606.29983, which already reports this in text.** Our version is a vision replication with attractor analysis, not a new idea | M5 | no effect, which is reportable |

**H2 was reframed on 2026-08-31**, see `decisions.md` D-007. The original wording, that convergence diagnostics do not predict extrapolation success, is already published for text (arXiv 2509.23314, 2607.20594, 2607.20519) and as written would have been a replication. The oracle framing is what is ours: prior work can only compare against `k*_model`, an oracle recovered by searching over loop counts, whereas we also hold `k*_task`, the generating program's depth, known by construction. Full statement in `paper.md` Section 4 and `findings.md` H2.

H5 (loop-as-refinement versus loop-as-denoising-timestep in a generation arm) is **out of scope for the main paper**. Do not build the generation stack. If it gets built later it lives behind `--arm generation` and never blocks the main pipeline.

The interesting secondary claim, which comes free from M4: in the text case, recurrent states converge in **direction** while raw norm keeps growing (up to 316x in the small-model regime), which means update-norm halting criteria misclassify convergence. Log both raw and direction-normalised quantities from day one so this is answerable without a re-run.

**Do not claim this geometry as new in vision.** Block-Recurrent Dynamics in Vision Transformers (arXiv 2512.19941) already reports directional convergence into class-dependent angular basins and low-rank collapse in late depth. It studies standard ViTs re-expressed as block-recurrent rather than models trained as looped, and does not isolate raw norm growth from direction-normalised quantities, which is the specific comparison M4 is built around. Position M4 as extending a known geometry to trained looped models and as the **mechanism behind the H2 signed error**, not as an independent verdict on halting.

---

## 2. Repository layout

```
loopvision/
  CLAUDE.md                     # short agent brief, Section 13
  README.md
  pyproject.toml                # single source of dependency truth
  preregistration.md            # frozen before milestone 5, git tag "prereg-v1"
  docs/
    IMPLEMENTATION.md           # this file
    proposal.md                 # the research proposal
    paper.md                    # what the paper claims, and what it says under every outcome
    gap-analysis.md             # related work survey and where the novelty actually sits
    progress.md                 # milestone tracker, gate ledger, risk register, session log
    findings.md                 # every result, with run_id provenance
    decisions.md                # append-only log of every deviation from this spec
    learnings.md                # mistakes, near misses, and the rule each one produced
    claims.md                   # headline numbers, each with a confirming run id (milestone 9)
  configs/
    base.yaml
    model/{d256,d384,d512,d768}.yaml
    task/{famA,famB,famC}.yaml
    sweep/{m1_loopcurve,m2_patch,m5_extrap}.yaml
  src/loopvision/
    __init__.py
    data/
      groups.py                 # dihedral and colour permutation group algebra
      render.py                 # canvas rendering primitives
      family_a.py               # transformational state tracking in pixels
      family_b.py               # relational chain queries
      family_c.py               # counting and aggregation (breadth stressor)
      dataset.py                # torch Dataset, manifest IO, splits
      validate.py               # gate G1 checks
    model/
      blocks.py                 # RMSNorm, attention, SwiGLU, one transformer block
      loopvit.py                # prelude / core / coda assembly
      conditioning.py           # none | embedding | adaLN loop conditioning
      baselines.py              # feedforward, echo, untied-same-depth
    train/
      loop_schedule.py          # loop-count sampling and curriculum
      bptt.py                   # truncated backprop through the loop
      trainer.py
      checkpoint.py             # save, resume, RNG state, probe checkpoints
      cli.py                    # python -m loopvision.train.cli --config ...
    instruments/
      m1_loopcurve.py
      m2_patching.py
      m3_codalens.py
      m4_geometry.py
      m5_extrapolation.py
      hooks.py                  # shared state capture and injection
    analysis/
      stats.py                  # bootstrap CIs, Holm correction, mixture fits
      registry.py               # run index, config hashing
      plots/                    # one script per figure, reads metrics only
    stage2/
      graft.py                  # SigLIP encoder + projector + frozen Huginn-0125
      probes.py                 # M2, M3, M4 rerun on the graft
  scripts/
    pick_gpu.sh                 # nvidia-smi based free-device selection
    submit.pbs                  # single job template
    chain.py                    # submit with -W depend=afterany for wall-clock chaining
    sanity/                     # fast end to end smoke tests
  tests/
  runs/                         # gitignored, one directory per run_id
```

---

## 3. Environment and cluster

Target is the RU cluster at Jaypee Institute of Information Technology, Noida: PBS scheduler, six workq nodes (384 cores) plus one GPU node with 96 cores and 8 GPUs, and a 16 concurrent core per user cap on the gpu queue.

**Hard constraints that shape the code:**

1. **Driver 525.147.05, CUDA 12.0.** Install PyTorch built for cu121 (compatible with driver 525.60.13 and above). Do **not** install a cu124 or later build; it will need a newer driver or forward compatibility packages that are not present. Pin it: `torch==2.4.1+cu121`.
2. **`qalter` is blocked, so a running job cannot have its walltime extended and dies at the wall.** Therefore checkpoint-and-resume is not a nice-to-have, it is load bearing. Every training run must be resumable to the exact optimizer, dataloader, and RNG state, and every job script must self-chain (Section 3.2). Any training script that cannot survive being killed with SIGKILL at a random step is broken.
3. **PBS does not track `ngpus` as a consumable resource.** Never assume `CUDA_VISIBLE_DEVICES` is set for you. Every job sources `scripts/pick_gpu.sh`, which picks the freest card by `nvidia-smi --query-gpu=memory.used,utilization.gpu` and exports `CUDA_VISIBLE_DEVICES`. Re-check inside the Python process and fail loudly if the selected device already has more than 2 GB in use.
4. **16 concurrent cores per user on the gpu queue.** Request `ncpus=8` per job so two jobs can run concurrently, and set `--num-workers 4`. Do not request more.

### 3.1 Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Dependencies to pin in `pyproject.toml`: `torch`, `numpy`, `einops`, `pyyaml`, `pandas`, `pyarrow`, `scipy`, `statsmodels`, `matplotlib`, `tqdm`, `pytest`. Logging goes to a local Parquet file plus optional Weights and Biases in offline mode (`WANDB_MODE=offline`), because compute nodes may not have outbound network. Do not add a dependency without appending a line to `docs/decisions.md`.

### 3.2 Job template

`scripts/submit.pbs`:

```bash
#!/bin/bash
#PBS -N loopvis
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=48gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /home/$USER/loopvision/logs/

set -euo pipefail
cd "$PBS_O_WORKDIR"
source .venv/bin/activate
source scripts/pick_gpu.sh          # exports CUDA_VISIBLE_DEVICES

export RUN_ID="${RUN_ID:?RUN_ID must be set}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# Resume is the default path, not the exception.
python -m loopvision.train.cli \
    --config "configs/${CONFIG}" \
    --run-id "$RUN_ID" \
    --resume auto \
    --max-hours 23.0            # self-stop before the wall, then chain

# If the run has not hit its step budget, queue the successor.
if [ ! -f "runs/$RUN_ID/DONE" ]; then
    qsub -W depend=afterany:"$PBS_JOBID" \
         -v RUN_ID="$RUN_ID",CONFIG="$CONFIG" scripts/submit.pbs
fi
```

`--max-hours` makes the process stop cleanly and write a checkpoint before PBS kills it. Set it about one hour under the walltime. The `DONE` sentinel file is what stops the chain.

---

## 4. Data

All data is procedurally generated so composition depth is **known exactly**, not estimated. This is the property the whole study rests on. Generation is deterministic given `(family, split, index, master_seed)`; do not cache images to disk beyond a small validation dump.

### 4.1 Shared interface

```python
@dataclass(frozen=True)
class Sample:
    image: np.ndarray        # uint8, (C, H, W), C=3, H=W=32 (or 64 for the robustness check)
    query: np.ndarray        # int32, (L,), token ids; L padded to L_max per family
    label: int               # class index
    depth: int               # sequential composition depth, the H1 x-axis
    breadth: int             # parallel scene breadth, the H1 null axis
    program: str             # canonical string form of the ground truth program
    idx: int
```

Every family exposes `generate(idx: int, split: str, cfg: TaskConfig) -> Sample` and a `vocab_size`, `num_classes`, `L_max`. The `program` field is what makes counterfactuals cheap for M2: a corrupted twin is produced by editing exactly one program element and re-rendering with the same layout seed.

### 4.2 Family A: transformational state tracking in pixels (flagship)

The direct pixel analogue of the group word problems from the text-case paper, which is what lets the loop-count curves be plotted against the text results on the same axes.

- **Group.** `D4 x S3`, that is the dihedral group of order 8 (rotations by multiples of 90 degrees and reflections) crossed with the permutation group on 3 colours. Order 48, non-abelian, so operator order genuinely matters. Implement in `groups.py` as an explicit 48 element multiplication table built once and asserted associative in a unit test.
- **Stimulus.** An asymmetric glyph (an L-tromino with a coloured tip works; it must have trivial stabiliser so every group element gives a visually distinct state) rendered on the canvas, plus a strip of `n` operator glyphs rendered along the bottom rows, in order.
- **Task.** Predict the final state after applying the operator sequence, as a 48-way classification. Chance is 1/48 = 0.021.
- **Depth** = `n`, the operator sequence length, from 1 to 6.
- **Breadth** = number of independent objects, each with its own operator strip, from 1 to 6. With breadth b, the label is the state of a queried object indicated by a marker patch.
- **Anti-shortcut sampling.** Sample the operator sequence so that every prefix product is uniform over the group. Concretely: draw the target final element uniformly, then draw a uniformly random factorisation of length `n`. This kills prefix-statistics shortcuts. Assert empirically in `validate.py` that a bag-of-operators logistic regression on the ground truth operator labels gets chance accuracy at every depth greater than 1.

**Presentation variants** (config flag, run the first by default): `strip` (operators as glyphs in the image), `frames` (intermediate states as a filmstrip, which makes the task easier and serves as a control), `tokens` (operators in the query token stream rather than the image, which isolates visual parsing from sequential composition).

### 4.3 Family B: relational chain queries

2D grid scenes of sprites with attributes (colour from 6, shape from 5, size from 2). Queries are generated from functional programs of known chain length, in the CLEVR style but generated locally so the depth and breadth axes cross cleanly.

- Program grammar: `query(attr, chain)` where `chain := object | relate(rel, chain)`, `rel in {left_of, right_of, above, below}`.
- Depth = chain length, 1 to 4. Example depth 3: "is the object above the object left of the red square a circle".
- Breadth = number of distractor objects, 3 to 12.
- **Uniqueness constraint.** Every intermediate referent must resolve to exactly one object, checked at generation time; reject and resample otherwise. Log the rejection rate; if it exceeds 30 percent at any (depth, breadth) cell, the scene sampler is badly parameterised.
- Labels are answers over a small shared vocabulary (yes/no plus attribute values).

### 4.4 Family C: counting and aggregation (breadth stressor)

Deliberately breadth-heavy and depth-shallow. Count objects satisfying a conjunction of two or three attributes; label is the count, capped at 10. Depth is nominally 1. Breadth is 4 to 16 objects. This is the null arm of H1: **if required loops grow here, H1 is falsified**, so it must be built before the loop sweeps, not after.

**Family C is not a candidate for cutting under any circumstance.** The August 2026 survey found the depth half of H1 partly anticipated in text, which makes the breadth axis the genuinely unexplored one and therefore the load-bearing half of the contribution. Without family C there is no breadth arm, and H1 collapses from a dissociation into a replication. If the schedule forces a choice between families A and B (open decision D-006), that choice is between A and B only.

### 4.5 Splits

- `train`: depth 1 to 3, breadth 1 to 4 (family-appropriate ranges).
- `iid_val`: same ranges, disjoint indices.
- `depth_ood`: depth 4 to 6, breadth as in train.
- `breadth_ood`: depth as in train, breadth 5 to 12.
- `combo_ood`: unseen colour and shape pairings.

Splits are index-range based off a single master seed, with a unit test asserting zero overlap of `(program, layout_seed)` pairs between splits.

### 4.6 Gate G1 (hard stop, milestone 2)

Before any architecture work:

1. Depth 3 must be **unsolvable at k=1**. Train a k=1 looped model and a matched non-looped model to convergence on depth 3, family A. Both must be near chance. If either solves it, the task is too easy and there is no loop-count curve to analyse; deepen the task (longer sequences, larger group, smaller canvas) and re-run the gate.
2. Depth 1 must be **solvable at k=1** to a high threshold. If not, the task is broken rather than deep, and the problem is rendering or capacity, not composition.
3. The bag-of-operators shortcut probe must sit at chance for depth greater than 1.
4. Family C must be solvable at k=1 at every breadth. If counting needs loops, the breadth axis is contaminated by sequential structure and the H1 contrast is void.

Write the gate as `pytest tests/test_gate_g1.py` reading from a stored gate run, so it is re-runnable, and record the outcome in `docs/decisions.md`.

---

## 5. Model

Deliberately mirrors the Huginn tripartite layout so Stage 2 comparisons are not confounded by architecture differences.

```
image (3,32,32) --patchify(conv 4x4 stride 4)--> 64 patch tokens
query token ids  --embed-->                      L query tokens
                       concat, + learned positional embeddings
                                  |
                       prelude: 2 untied blocks   -> e   (B, N, d)
                                  |
     s_0 = randn(B,N,d) * d**-0.5 (ablate: zeros)
     repeat k times:  s_{i+1} = Core( adapter([s_i ; e]) , cond=i )
                                  |
                       coda: 2 untied blocks -> head (num_classes)
```

- **Block.** Pre-norm, RMSNorm, multi-head attention with head dim 64, SwiGLU MLP at 8/3 expansion, no dropout, RoPE over a flattened 2D patch index plus a learned segment embedding distinguishing patch tokens from query tokens.
- **Adapter.** `Linear(2d, d)` on the concatenation of the current state and the prelude output, exactly Huginn's recipe, so the core sees the input at every iteration instead of drifting. **Ablation control: no injection** (core sees only `s_i`).
- **Size ladder.** Four models, `d in {256, 384, 512, 768}`, roughly 5M, 11M, 19M and 43M parameters at 6 total blocks. Heads = d/64. Core is 2 blocks tied by default, 1 block as an ablation.
- **Loop conditioning** (`conditioning.py`, three variants, this is where the loop-as-timestep question gets tested cheaply):
  - `none`: nothing added.
  - `embed`: learned embedding of the iteration index added to the state.
  - `adaln`: sinusoidal embedding of `i` (and optionally of `k`) driving scale and shift parameters per block, which is the ELT and time-modulated looped transformer recipe.

**Baselines, all trained at matched compute and separately at matched parameters:**

1. Non-looped feedforward with depth equal to `k * core_depth`. The headline comparison.
2. **Echo baseline**: the core is replaced by the identity plus injection, so the state is refreshed but not learned-updated. Rules out trivial explanations of loop-count curves. This baseline caught a real problem in the text-case work; do not skip it.
3. Untied but same total depth, to separate weight tying from depth.
4. Fixed-k models at each k versus one variable-k model, to check the loop-count curve is not an artifact of training-time loop sampling.

**Required invariance test:** `tests/test_model_invariance.py` asserts that a forward pass at k with `torch.no_grad` equals the incremental state sequence captured by the M4 hooks, to within 1e-5. If the instrument path and the training path can silently disagree, every measurement downstream is suspect.

---

## 6. Training

### 6.1 Loop-count sampling and curriculum

Per batch, not per sample (per sample would require ragged execution):

```python
tau = exp(normal(mu_t, sigma))          # sigma = 0.5 default
k   = clip(1 + poisson(tau), 1, k_max)  # k_max = 32 for training
```

`mu_t` ramps linearly from `log(2)` to `log(k_mean_target)` over the first 40 percent of steps, then holds. `k_mean_target = 8` default. Ablate ramp shape (linear, cosine, step) as in the prior paper.

**Jitter is on by default.** In the text case, deterministic loop counts produced wrong-attractor landings at 16 of 44 versus 5 of 75 with jitter (p = 7.0e-5). Jitter is close to free here. H4 tests replication, and unlike last time the **jitter-width sweep is scheduled up front** (milestone 8) rather than promised at submission. The previous submission carried a confirmation sweep that had not run and it did not confirm; do not repeat that.

**Jitter is not our idea.** Kuo et al. (arXiv 2606.29983, June 2026) report that stochastic training-time loop counts sharply reduce out-of-distribution variance, tracing the instability to a spurious correlation between sequence length and loop count. Cite them as prior work. Our addition is the vision setting and the attractor analysis, which they do not do. Presenting jitter as a contribution would be an overclaim.

### 6.2 Truncated backpropagation through the loop

```python
def looped_forward(core, e, k, bptt_window, cond_fn):
    s = init_state(e)
    n_nograd = max(0, k - bptt_window)
    with torch.no_grad():
        for i in range(n_nograd):
            s = core(adapter(cat(s, e)), cond=cond_fn(i))
    s = s.detach()
    for i in range(n_nograd, k):
        s = core(adapter(cat(s, e)), cond=cond_fn(i))
    return s
```

`bptt_window = 8` default, 4 under memory pressure. This changes gradient semantics relative to full backpropagation, so run **one** ablation at `k_max = 8` with full backpropagation and record the difference in `docs/decisions.md`. Do not run the whole grid both ways.

### 6.3 Optimisation

AdamW, learning rate 3e-4, betas (0.9, 0.95), weight decay 0.05, 2000 step linear warmup then cosine to 10 percent, gradient clip 1.0, batch 256 at 32x32, bf16 autocast with fp32 parameters. Scale the recurrent core's output projection initialisation by `1/sqrt(2 * core_depth * k_mean)` to keep early-training state norms bounded; this and the injection are the two things that most often decide whether a small looped model trains at all.

Do **not** use `torch.compile` initially. Dynamic `k` triggers recompilation per loop count and the graph breaks around the no-grad and grad boundary are subtle. Revisit only if throughput is the binding constraint, and then mark shapes dynamic explicitly.

### 6.4 Checkpointing

Two tiers, because naive dense checkpointing exhausts disk:

- **Resume checkpoints:** full state (model, optimizer, scheduler, dataloader position, all RNG states including CUDA), written every 30 minutes and on SIGTERM, keeping the last 2. Written to node-local scratch first, then `rsync` to home, so a kill during write cannot corrupt the only copy.
- **Probe checkpoints:** weights only, bf16, no optimizer state, at 100 log-spaced steps. These are what the loop-versus-training-time surface is built from. Enable **only** for the configurations that feed M1 and M5 (roughly 40 runs), not all 300. Estimate: a 19M model at bf16 is about 38 MB, so 100 probes is 3.8 GB per run and 40 runs is about 150 GB. Check the quota before milestone 5 and reduce probe density to 50 if it does not fit.

### 6.5 Run identity and the registry

`run_id = f"{family}_{d}_{tag}_{sha1(canonical_yaml)[:8]}"`. The full resolved config, the git commit, and the `pip freeze` output are written to `runs/<run_id>/config.yaml`, `git.txt`, `env.txt` at start. `analysis/registry.py` scans `runs/` into a dataframe. **Refuse to start a run whose config hash already has a DONE sentinel** unless `--force` is passed, otherwise the sweep silently double-counts seeds.

Metrics go to `runs/<run_id>/metrics.parquet` with a long schema: `step, k, split, depth, breadth, metric_name, value, seed`. One schema for everything makes the analysis layer trivial and stops per-instrument ad hoc formats from proliferating.

---

## 7. Instruments

Five instruments, four of them ports of code that already exists for the text case. `hooks.py` provides one shared mechanism: a context manager that captures `s_i` for all `i` and optionally overwrites `s_i[:, positions, :]` from a stored donor run. Every instrument uses it; none of them reimplements the loop.

### M1. Loop-count requirement curves (H1)

For each `(family, depth, breadth, d_model, seed)` cell, evaluate a trained variable-k model at `k = 1..32` and record the minimum `k` at which held-out accuracy reaches threshold `tau = 0.90` (fixed in the pre-registration; also report the full accuracy-versus-k curve so the threshold choice can be shown not to drive the result). Confidence intervals by bootstrap over seeds, minimum 5 seeds per cell and 10 for the headline cells. Control: fixed-k models trained at each k, on a reduced grid.

Analysis: fit `k_min ~ a * depth + b * breadth + c` with bootstrap CIs on `a` and `b`. H1 predicts `a > 0` and `b` indistinguishable from 0. Also fit the saturating alternative `k_min ~ a * log(depth)` and report both, because the text-case work found the linear reading breaks in exactly this kind of place.

### M2. Activation patching over (loop, patch position) (H3)

The direct port of the existing patching code with token position replaced by patch position.

1. Generate a clean sample and a corrupted twin differing in exactly one program element (one operator in family A, one relation hop in family B).
2. Run clean, store all `s_i`.
3. Run corrupted while overwriting `s_i[:, p, :]` with the clean value, for each `(i, p)` in the grid.
4. Record the recovered logit difference, normalised so 0 is the corrupted run and 1 is the clean run.

Output: a `[k, n_positions]` tensor per depth per family, saved as Parquet. This is the visual centrepiece of the paper. Grid size at k=16 and 64 positions is 1024 forward passes per item; batch the positions and use at least 256 items per cell.

### M3. Coda-lens trajectories

Decode `s_i` through the coda and head at every `i`, record accuracy(i) and the Kullback-Leibler divergence from the final distribution. In the text case these trajectories were sharp and discontinuous rather than smoothly improving, which argued against a naive latent chain-of-thought reading. Whether vision is smooth or discontinuous is a genuinely open result either way, so report the shape, do not summarise it to a single scalar.

### M4. Trajectory geometry

Per iteration, log **all** of: `||s_i||`, `||s_i - s_{i-1}||`, `cos(s_i, s_{i-1})`, `cos(Δ_i, Δ_{i-1})`, and the ratio of directional convergence to norm growth. Both raw and direction-normalised. This is the instrument that answers the rays-not-fixed-points question, and it costs nothing to log during every evaluation pass, so log it during every evaluation pass rather than as a separate study.

Known trap from the prior work: a speed-like quantity defined as `1/slope` blows up near zero slope (one checkpoint at slope 0.036 gave 27.6 and shifted its group mean by 48 percent). Any reciprocal-of-a-fitted-slope statistic must ship with an explicit inclusion gate defined **before** the fits are run, plus a `gate_sensitivity.py` that reports the headline both ways.

### M5. Extrapolation and attractors (H2, H4)

- Train at `k_mean = 8`, evaluate at `k in {1, 2, 4, 8, 16, 32, 64}` on iterative (family A, B) and one-shot (family C) tasks. These curves are not the result. They are how `k*_model` is computed.
- **Three quantities per item, all logged during every evaluation pass:**
  - `k*_task`, the composition depth of the generating program. Known exactly by construction, model-independent. This is the quantity no prior work has.
  - `k*_model`, the smallest loop count at which this model answers this item correctly. Oracle-over-iterations. The only oracle available to the text literature.
  - `k_hat_c`, the loop index at which halting criterion c fires.
- **Four criteria, not two.** Update norm below epsilon, Kullback-Leibler below epsilon, predictive entropy below threshold (the LoopViT rule, arXiv 2602.02156), and step-size second difference (Pappone et al., arXiv 2509.23314). Testing only the first two invites the objection that we picked the weakest, and the Pappone rule is the current best published exit criterion, so it belongs in the comparison rather than only in related work.
- **The H2 measurement:** distribution of `k_hat_c - k*_model` per criterion, reporting the sign as well as the magnitude because halting early and late have opposite engineering fixes, then the slope of that signed error against `k*_task`. A constant error is miscalibration and a threshold fixes it. An error that grows with depth means the criterion measures the wrong thing and no threshold rescues it.
- Also report the correlation of `k*_model` with `k*_task`. This separates whether a criterion tracks the model from whether the model tracks the task, and it is available to nobody else because it needs both oracles.
- Attractor analysis: cluster final states across seeds, count wrong-attractor landings, sweep jitter width including width zero (the width-zero arm is what the prior sweep needed and lacked). Cite arXiv 2606.29983 as prior work for the jitter effect itself.
- Overthinking is a real risk at the top of the k range: arXiv 2606.18023 reports performance degrading past two loops from positional mismatch, and arXiv 2604.07822 reports the same effect. Expect the k=64 arm to be worse than k=32 and do not treat that as a bug.

---

## 8. Statistics and pre-registration

`preregistration.md` is written and committed with git tag `prereg-v1` **before** milestone 5 runs, and is never edited afterwards; changes go in `docs/decisions.md` with a date and a reason. It must state, for each hypothesis: the exact metric, the threshold, the seed count, the test, the alpha, and what counts as falsification.

Rules:

- Minimum 5 seeds per cell, 10 for anything that becomes a headline number.
- Holm correction across the pre-registered family of tests. Exploratory tests are corrected separately and **labelled exploratory in the text**, as section 4.5 was in the prior submission.
- Any statistic discovered after looking at the data is post-hoc and is labelled post-hoc in the paper, with a confirmation run scheduled in milestone 8. Confirmation runs that fail get reported as failures.
- Distribution-free tests on heavy-tailed reparameterisations of bounded quantities have already burned this line of work once (Silverman rejecting where Hartigan's dip did not). If a bimodality or clustering claim appears here, run at least two tests with different assumptions and report both outcomes including disagreement.

---

## 9. Stage 2: the Huginn scale check

Purpose is external validity, not benchmark chasing. The small-model mechanism story is worth little if it evaporates at 3.5B.

- Frozen Huginn-0125, frozen SigLIP encoder, trainable multi-layer perceptron projector, optional LoRA on the coda. The expensive parts are frozen so this is a few GPU-days.
- Rerun M2, M3, M4 on the visual tokens. The question is whether the loop-count and geometry signatures found at 5M to 43M appear at 3.5B.
- External benchmarks (CLEVR, then MMVP, VSR, BLINK) as a sanity check that the graft works at all. Do not lead with these numbers; HIVE already owns that framing.
- Known engineering friction: Huginn's custom key-value cache implementation, and the Ouro modelling-code cache fix if a cross-model check is added. Budget two days for cache plumbing alone.
- A cross-model check on Ouro-1.4B is cheap and strengthened the uncertainty work; include it if milestone 9 has slack.

Note the prior negative result to guard against over-claiming here: in the text-case work, two public depth-recurrent language models tracked at one and two composition steps and collapsed at three, so a frontier-speed curve could not be fit at all. If the same ceiling shows up in the visual graft, that is the result, and it belongs in the main text rather than an appendix.

---

## 10. Guardrails learned from the prior submission

These are not hypothetical. Each cost real time on the previous paper in this line.

1. **Artifact blockers close early, not at submission.** Anonymous code link, artifact DOI, and the AI use statement all get written at milestone 10, not in the final week. The last submission went in with an ANON-PENDING link and a TODO statement.
2. **Do not promise a sweep that has not run.** If it is not in `runs/`, it is not in the paper.
3. **Convergence diagnostics lie in this model family.** Never use an update-norm plateau as evidence of convergence without the direction-normalised counterpart beside it.
4. **Count your cones.** The prior draft carried an invented count (329 versus the true 344) into review. Every count in the paper must come from `analysis/registry.py`, printed by a script, not typed by hand.
5. **Reciprocal statistics need a gate defined before the fits.**
6. **Abstract and introduction stay free of mathematics** (Prof. Garg's standing rule): motivation first, "what could nobody do before" on page one, formalism no earlier than the formal setup section.

---

## 11. Milestones and gates

Each milestone's definition of done is a command. Sixteen weeks of core work from September 2026, which lands results well before the NeurIPS 2027 abstract deadline (historically early to mid May).

| # | Weeks | Work | Done when |
|---|---|---|---|
| 0 | 0 | Repo skeleton, `pyproject.toml`, CI running pytest, `pick_gpu.sh` and a hello-world PBS job that survives a kill and resumes | `pytest -q` passes and a chained 2-job run completes |
| 1 | 1 | `groups.py` plus `render.py`, families A, B, C generators, manifest and splits | `pytest tests/test_data.py` passes, 64 sample images dumped to `runs/_gate/samples/` and eyeballed |
| 2 | 2 | **Gate G1** (Section 4.6) | `pytest tests/test_gate_g1.py` passes on stored gate runs |
| 3 | 3 to 5 | Model, training loop, stability work, all three conditioning variants, all four baselines | a d=384 model trains stably at k=8 on depth 3 above threshold, from a cold start, twice with different seeds |
| 4 | 5 | **Gate G2**: looped model beats matched-compute feedforward and the echo baseline on depth 3 | `python -m loopvision.analysis.gate_g2` exits 0 |
| 5 | 6 to 7 | Freeze `preregistration.md`, then the full M1 sweep across the depth-by-breadth grid and the size ladder | H1 resolves, `m1_loopcurve.parquet` complete, `a` and `b` coefficients with CIs printed |
| 6 | 8 to 9 | M2 patching ported to (loop, patch) | H3 resolves, heatmaps for all three families |
| 7 | 10 | M3 coda-lens and M4 geometry | rays-not-fixed-points question answered either way |
| 8 | 11 to 12 | Baselines and controls at matched compute and matched parameters, M5 extrapolation and attractor and jitter-width sweeps | H2 and H4 resolve, jitter width zero arm included |
| 9 | 13 | Confirmation runs for every post-hoc headline number. **No new exploration** | every headline number in `docs/claims.md` has a confirming run id |
| 10 | 14 to 15 | Stage 2 graft and probes, optional Ouro cross-check, artifact and anonymity items | probes reproduce or refute the small-model signatures |
| 11 | 16 | Buffer, ablation gap filling | |
| 12 | Jan to Mar 2027 | Writing, internal review | |
| 13 | Apr to May 2027 | Anonymous release, abstract, submission | |

**Compute estimate.** A 19M model at 32x32 with k up to 16 trains in a few hours on one A100 class card. The full grid is roughly 200 to 400 runs including seeds and ablations, on the order of two to four GPU-weeks of wall clock if runs are batched well and two jobs run concurrently under the 16 core cap. Stage 2 adds a few GPU-days since the backbone is frozen. **This fits because the resolution is small. Resist resolution creep.** The generation arm is the main creep vector and it is out of scope.

**Stability time-box.** If a stable model at k=8 on depth 3 is not training by the end of week 3, adopt ELT-style intra-loop self-distillation and re-run. If that also fails by week 5, drop to a single-block core and 16x16 resolution. Record whichever fallback is taken in `docs/decisions.md`.

---

## 12. Open decisions to resolve before milestone 5

These are from Section 13 of the proposal and are still open. They block the pre-registration freeze, not the code.

1. **Compute block.** Is a sixteen-week window on the cluster realistic alongside the merging pipeline? If access is intermittent, cut the size ladder from four models to two (d=384 and d=768) and say so in the paper.
2. **Co-authorship and affiliation.** Prof. Garg and Dr. Saini both in play; the Indian Institute of Information Technology Allahabad affiliation affects anonymity handling if a preprint goes up first.
3. **Preprint timing.** A preprint after milestone 8 protects priority but complicates NeurIPS anonymity, and the earlier submission pair already flagged cross-paper anonymity risks.
4. **Task family priority.** Family A is the strongest scientific transfer from the state-tracking work. Family B is the most legible to vision reviewers. If only one can be done well, which audience is being optimised for?

---

## 13. `CLAUDE.md` (copy this to the repo root)

```markdown
# What a Loop Sees

Mechanistic interpretability of looped (depth-recurrent) vision models.
Full spec: docs/IMPLEMENTATION.md. Read the relevant section before writing code.

## Document map
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
- preregistration.md is frozen at git tag prereg-v1. Deviations go in docs/decisions.md, never in the prereg.
- Counts and totals in prose come from analysis/registry.py output, never typed by hand.
- No citation enters a draft until its arXiv abstract page has been fetched and the version confirmed.
- Family C is never cut. It is the breadth arm, and without it H1 is a replication rather than a dissociation.

## Environment
- PyTorch cu121 only (driver 525, CUDA 12.0). Never install cu124 or later builds.
- PBS cluster, qalter is blocked, jobs die at the wall. Every run must resume from checkpoint and self-chain.
- Request ncpus=8. Select the GPU with scripts/pick_gpu.sh, never assume CUDA_VISIBLE_DEVICES.

## Workflow
- One milestone at a time (docs/IMPLEMENTATION.md Section 11). Do not start N+1 until N's command exits 0.
- Gates G1 (task validity) and G2 (loops beat baselines) are hard stops. If one fails, stop and report.
- Add a dependency only with a line in docs/decisions.md.
- Instruments use src/loopvision/instruments/hooks.py. Never reimplement the loop inside an instrument.

## Commands
    pytest -q
    python -m loopvision.train.cli --config configs/base.yaml --run-id <id> --resume auto
    python -m loopvision.analysis.registry            # run index and counts
    RUN_ID=<id> CONFIG=<cfg> qsub scripts/submit.pbs
```

---

## 14. First Claude Code session

Ordered prompts for a cold start. Keep each session to one milestone.

1. "Read docs/IMPLEMENTATION.md sections 0 to 3. Create the repo skeleton from section 2, the pyproject with the pinned dependencies, scripts/pick_gpu.sh, and scripts/submit.pbs. Add a smoke training script that counts to 1000 steps, checkpoints, and resumes correctly after SIGKILL, and a pytest that verifies the resume is bit-exact. Do not write any model code yet."
2. "Read section 4. Implement groups.py with the D4 x S3 multiplication table and unit tests for associativity, inverses, and non-commutativity. Then render.py and family_a.py with the anti-shortcut sampler. Dump 64 samples per depth to runs/_gate/samples/."
3. "Implement family_b.py and family_c.py with the uniqueness constraint and rejection-rate logging, then dataset.py with the splits from 4.5 and the no-overlap test."
4. "Read section 4.6. Implement tests/test_gate_g1.py and the gate training configs. Run the gate. Report the four gate outcomes as a table and stop; do not proceed to the model."

Only after gate G1 passes: "Read section 5 and implement blocks.py, loopvit.py, conditioning.py, and the invariance test in tests/test_model_invariance.py."

---

## 15. Key references to keep on hand

Looped architectures: Geiping et al. (arXiv 2502.05171, Huginn); Zhu et al. (arXiv 2510.25741, Ouro); Bae et al. (arXiv 2507.10524, Mixture-of-Recursions); Dehghani et al. (Universal Transformers, ICLR 2019).

Looped vision: Shu et al. (arXiv 2602.02156, LoopViT, our nearest vision neighbour: 18M parameters, ARC-AGI, dynamic exit on predictive entropy); Byra et al. (arXiv 2605.10661, bViT, single-block recurrence on ImageNet-1K); Gruszczynski et al. (arXiv 2608.04879, recurrent ViTs at matched FLOPs, relevant to gate G2); Zhang et al. (arXiv 2602.05359, HIVE); RecursiveVLM (arXiv 2602.09080); Lee and Park (arXiv 2607.00774, Soft-MoR and SR-ViT); Madan et al. (arXiv 2607.09061, locality and length generalisation, closest published work to our shortcut concern); Deja View (arXiv 2605.30215, multi-view 3D); Recurrent-Depth VLA (arXiv 2602.07845); LoopVLA (arXiv 2605.09948). **Goyal et al. (arXiv 2604.09168, ELT) is a visual generation paper**, not recognition; cite it for Intra-Loop Self Distillation, which is the week-3 stability fallback, and do not imply otherwise.

Mechanism and geometry: Pappone, Crisostomi, Rodola (arXiv 2509.23314, two-scale latent dynamics, and the step-size second-difference exit rule that beats Kullback-Leibler); Lu et al. (arXiv 2507.02199, coda-lens and latent chain of thought); Zhang et al. (arXiv 2607.20594, when does recurrence become an algorithm, **our nearest neighbour scientifically, group word problems in text, position against it in the introduction**); Block-Recurrent Dynamics in Vision Transformers (arXiv 2512.19941, directional convergence in vision, see Section 1); Blayney et al. (arXiv 2604.11791, mechanistic analysis of looped language models, per-layer fixed points and input injection); Shen, Su, Kyrillidis (arXiv 2605.17811, asymmetric input injection induces role specialisation).

Halting, early exit and adaptive depth, the H2 cluster: Popescu, Saez de Ocariz Borde, Lio (arXiv 2607.20519, learned halting gates versus post-hoc confidence readouts); Kuo et al. (arXiv 2606.29983, **stochastic training-time loop counts, which is H4 already published in text**); ANIRA (arXiv 2602.08864, dynamic compute allocation and extrapolation failure); Movahedi et al. (arXiv 2606.18206, fixed-point convergence as halting signal); Viakhirev et al. (arXiv 2608.18222, settling versus marginal versus drifting regimes predict whether extra iterations help); Gunn Kim (arXiv 2608.26556, initialisation selects the dynamical phase that sets compute scaling, bears on our init choices); Sapunov (arXiv 2604.21999, Universal Transformers need memory); Yu et al. (arXiv 2607.10110, looped state-space models); Kamiya et al. (arXiv 2608.24136, readout feedback steering).

Scaling, efficiency and overthinking: Kohli et al. (arXiv 2604.07822, inference-time recurrence unlocks depth generalisation, and overthinking degrades it, **closest published work to H1**); Schwethelm, Rueckert, Kaissis (arXiv 2604.21106, recurrence-equivalence exponent 0.46, one loop is worth well under one layer); Yang et al. (arXiv 2606.18023, LoopCoder-v2, performance degrades past two loops from positional mismatch); Deng et al. (arXiv 2605.20670, LT2); Chen et al. (arXiv 2605.23872, training-free looped transformers); Lu et al. (arXiv 2606.18208, looped world models).

**Do not cite arXiv 2604.09870** (Kirin, relational preference encoding) without the v2 erratum, which reports the three headline results inflated by two independent evaluation errors. See `learnings.md` N-001.

Equivalences and theory: Svete and Sabharwal, "On the Reasoning Abilities of Masked Diffusion Language Models" (arXiv 2510.13117, ICLR 2026 oral). Cite it for one specific result, the equivalence between masked diffusion models and polynomially-padded looped transformers, not as looped-transformer theory generally; it is a masked diffusion paper and describing it otherwise will be noticed. See also the same group's follow-up on which architectural choices preserve those equivalences, Svete, Merrill, Cotterell and Sabharwal (arXiv 2605.30523). Saunshi et al. (ICLR 2025, latent thoughts); Giannou et al. (ICML 2023); Xu and Sato, "On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding" (arXiv 2410.01405).

Weight-tied vision precedent: Bai, Kolter, Koltun (deep equilibrium models, NeurIPS 2019); Bai et al. (multiscale deep equilibrium models, NeurIPS 2020); Teed and Deng (RAFT, ECCV 2020); Bai and Melas-Kyriazi (arXiv 2401.08741, fixed point diffusion).

Own prior work to cite and to reuse code from: the looped-transformers state-tracking submission (ICLR 2027 submission 6190), the consistency-based uncertainty study on Huginn-0125, and the test-time routing pair.
