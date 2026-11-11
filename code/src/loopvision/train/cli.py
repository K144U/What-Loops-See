"""Training entry point.

    python -m loopvision.train.cli --config configs/gate/g1_famA_d3_looped.yaml \\
        --run-id g1_famA_d3_looped_s0 --resume auto

Reuses train/checkpoint.py rather than reimplementing resume, so a run
killed at the PBS wall continues bit exactly. Metrics land in
runs/<run_id>/metrics.parquet under the one long schema every instrument
reads: step, k, split, depth, breadth, metric_name, value, seed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from loopvision.data import dataset as D
from loopvision.data import groups as G
from loopvision.data import render
from loopvision.model.loopvit import FeedforwardBaseline, LoopViT, LoopViTConfig
from loopvision.train import loop_schedule
from loopvision.train.checkpoint import (
    GracefulKiller,
    find_latest_checkpoint,
    load_checkpoint,
    prune_to_latest,
    save_checkpoint,
)

# TF32 is pinned at the values this campaign actually ran under, so that a
# torch upgrade cannot silently change what the recorded numbers mean.
# These ARE the torch 2.4.1 defaults, measured rather than assumed, so
# setting them changes nothing today: matmuls stay at full fp32 and the
# single patch-embedding convolution keeps TF32. That is the point. The
# five instruments that build models import this module, so pinning here
# covers training and analysis together. Changing either value would break
# comparability with every finished run. See docs/decisions.md D-033.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True

DEFAULTS = {
    "family": "A",
    "d_model": 384,
    "arch": "looped",        # looped | feedforward | echo | untied
    "untied_copies": None,   # untied only. Defaults to the max k in use
    "k_schedule": "fixed",   # fixed | sampled, see train/loop_schedule.py
    "k_train": 1,            # used when k_schedule is "fixed"
    "k_mean_target": 8,      # used when k_schedule is "sampled"
    "jitter_sigma": 0.5,     # the H4 sweep axis. 0.0 is the deterministic arm
    "k_max_train": 32,
    "k_eval": None,          # defaults to k_train
    "eval_k_sweep": None,    # list of k to sweep at the end, for the M1 curve
    "factor": "full",        # family A only: full | d4 | s3, see groups.SUBGROUPS
    # Curriculum: start from another run's weights instead of from scratch.
    # Model weights only. The optimizer, the step counter and the data
    # cursor all start fresh, so this is a different run and not a
    # continuation of the donor.
    "init_from": None,
    # native | s3_spatial | d4_colour. Crosses group against substrate,
    # see docs/experiment-substrate.md.
    "substrate": "native",
    # Sequential depth in one forward pass is prelude + k*core + coda. If
    # that already covers the task's composition depth at k=1 then k binds
    # on nothing and the loop-count sweep is flat whatever the truth. See
    # docs/decisions.md D-028.
    "prelude_blocks": 2,
    "core_blocks": 2,
    "coda_blocks": 2,
    "depths": None,
    "breadths": None,
    "steps": 20000,
    "batch_size": 256,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "warmup": 2000,
    # Decoupled from "steps" on purpose. lr_at previously derived the cosine
    # length from the step budget, so a 20k run and a 200k run differed in
    # learning rate schedule as well as in length, which confounded the two
    # gate G1 attempts. None means "follow steps", as before.
    "lr_schedule_steps": None,
    "grad_clip": 1.0,
    "bptt_window": 8,
    "eval_every": 500,
    "eval_batches": 8,
    "checkpoint_every_minutes": 30,
    "seed": 0,
    "master_seed": 20260901,
    # Training is generation bound. One less than ncpus, leaving a core
    # for the main process and the GPU feed.
    "num_workers": 3,
    "state_init": "randn",
    "conditioning": "none",
    "inject": True,
    # Early stop, D-065. Off unless both are set. A run stops once the
    # overall validation accuracy has been at or above tau at every
    # evaluation for the last `window` steps. Family A only: in family B six
    # runs that reached 0.95 later finished below it, and in family A none
    # did (runs/_earlystop_check_20261011/).
    "early_stop_tau": None,
    "early_stop_window": None,
    # D-067. Each rendered image is upsampled this many times by nearest
    # neighbour before the model sees it, and the model's canvas grows to
    # match while the patch stays 4. At 2 every glyph covers 2x2 patches
    # instead of exactly one.
    "pixel_scale": 1,
}


def load_config(path: str | None, overrides: dict) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            cfg.update(yaml.safe_load(fh) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if cfg["k_eval"] is None:
        cfg["k_eval"] = (
            cfg["k_mean_target"] if cfg["k_schedule"] == "sampled" else cfg["k_train"]
        )
    validate_early_stop(cfg)
    if int(cfg["pixel_scale"]) != cfg["pixel_scale"] or cfg["pixel_scale"] < 1:
        raise ValueError(f"pixel_scale must be a positive integer, got {cfg['pixel_scale']}")
    return cfg


def validate_early_stop(cfg: dict) -> None:
    """Refuse an early-stop setting D-065 did not register.

    Both keys or neither; family A only; and a window that is a whole number
    of evaluation intervals, because the rule needs an evaluation exactly at
    the start of the window. A window that never lines up with an
    evaluation would never fire, and the run would quietly train to its full
    budget while its config claimed otherwise.
    """
    tau, window = cfg.get("early_stop_tau"), cfg.get("early_stop_window")
    if tau is None and window is None:
        return
    if tau is None or window is None:
        raise ValueError("early_stop_tau and early_stop_window are set together or not at all")
    if cfg.get("family") != "A":
        raise ValueError("early stop is registered for family A only (D-065)")
    if window <= 0 or window % cfg["eval_every"] != 0:
        raise ValueError(
            f"early_stop_window {window} must be a positive multiple of "
            f"eval_every {cfg['eval_every']}"
        )


def early_stop_due(history: list[tuple[int, float]], step: int,
                   tau: float, window: int) -> bool:
    """D-065: every evaluation in [step - window, step] is at or above tau.

    Evaluations must exist at both ends of the window, so the whole window
    was observed rather than assumed: a run resumed partway through it, or
    one whose first evaluation is younger than the window, does not stop.
    """
    inside = [(s, a) for s, a in history if step - window <= s <= step]
    steps = {s for s, _ in inside}
    if step not in steps or (step - window) not in steps:
        return False
    return all(a >= tau for _, a in inside)


def eval_history(run_dir: Path, through_step: int) -> list[tuple[int, float]]:
    """Overall validation accuracy by step, as stored, up to a step.

    How a resumed run gets back the evaluations its predecessor made, so an
    early-stop window can span a job boundary.
    """
    import pandas as pd

    path = Path(run_dir) / "metrics.parquet"
    if not path.is_file():
        return []
    df = pd.read_parquet(path)
    a = df[(df["split"] == "iid_val") & (df["metric_name"] == "accuracy")
           & (df["depth"] == -1) & (df["breadth"] == -1) & (df["step"] <= through_step)]
    a = a.sort_values("step").drop_duplicates("step", keep="last")
    return [(int(s), float(v)) for s, v in zip(a["step"], a["value"])]


#: Keys a relaunch is allowed to change, because the allocation decides
#: them rather than the experiment. `multirun.pbs` computes `num_workers`
#: from the cores the job actually got, so it differs between launches of
#: the same run and is not drift. Everything else that differs between the
#: source config and the run's own snapshot is drift and gets reported.
RELAUNCH_KEYS = frozenset({"num_workers"})


def config_drift(source: dict, snapshot: dict) -> dict:
    """Keys where the run's own config disagrees with the source config.

    Returned as key -> (what the run has, what the source says now), in
    that order, because the run's value is the one that will be used.
    """
    return {
        k: (snapshot.get(k), source.get(k))
        for k in sorted(set(source) | set(snapshot))
        if k not in RELAUNCH_KEYS and source.get(k) != snapshot.get(k)
    }


def config_hash(cfg: dict) -> str:
    canonical = yaml.safe_dump(cfg, sort_keys=True)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def task_config(cfg: dict) -> D.TaskConfig:
    return D.TaskConfig(
        master_seed=cfg["master_seed"],
        depths=tuple(cfg["depths"]) if cfg["depths"] else None,
        breadths=tuple(cfg["breadths"]) if cfg["breadths"] else None,
        factor=cfg.get("factor", "full"),
        substrate=cfg.get("substrate", "native"),
    )


_POOL = None


def _gen_chunk(args):
    """Worker entry point. Must be module level to be picklable."""
    family, split, indices, tcfg = args
    out = []
    for idx in indices:
        s = D.generate(family, idx, split, tcfg)
        out.append((s.image, s.query, s.label, s.depth, s.breadth))
    return out


def get_pool(num_workers: int):
    """One pool per process, created lazily and reused across steps.

    Prefers fork, which is what the Linux cluster uses and which avoids
    re-importing torch in every worker. Falls back to spawn on platforms
    without fork, so the same code runs on the Windows dev machine.
    """
    global _POOL
    if num_workers > 0 and _POOL is None:
        import multiprocessing as mp

        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")
        _POOL = ctx.Pool(num_workers)
    return _POOL


def close_pool() -> None:
    """Shut the worker pool down explicitly.

    Leaving it to interpreter teardown raises an AttributeError from
    multiprocessing during shutdown, which is harmless but looks like a
    crash in a PBS log and would waste someone's time.
    """
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
        _POOL = None


def make_batch(cfg: dict, split: str, cursor: int, device):
    """Generate one batch from the procedural stream at a given cursor.

    The cursor is the entire dataloader state, which is why resume can be
    exact. Parallel generation preserves that exactly: every sample is a
    pure function of its global index, so splitting the batch across
    processes and reassembling in index order gives byte-identical results
    to generating it serially. Asserted in tests/test_parallel_gen.py.

    Measured on the gate runs, training was entirely generation bound at
    around 4200 samples per second while the A100 idled, so this is where
    the wall clock actually goes.
    """
    tcfg = task_config(cfg)
    n = cfg["batch_size"]
    indices = [D.global_index(split, cursor + i) for i in range(n)]
    workers = cfg.get("num_workers", 0)

    if workers > 0:
        pool = get_pool(workers)
        chunk = (n + workers - 1) // workers
        chunks = [
            (cfg["family"], split, indices[i : i + chunk], tcfg)
            for i in range(0, n, chunk)
        ]
        rows = [r for part in pool.map(_gen_chunk, chunks) for r in part]
    else:
        rows = _gen_chunk((cfg["family"], split, indices, tcfg))

    images, queries, labels, depths, breadths = zip(*rows)
    image = torch.from_numpy(np.stack(images)).to(device).float().div_(255.0)
    scale = int(cfg.get("pixel_scale", 1))
    if scale > 1:
        # Nearest neighbour, so every pixel becomes a scale x scale block of
        # the same value and no colour is blended across a glyph edge (D-067).
        image = image.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)
    query = torch.from_numpy(np.stack(queries)).to(device).long()
    label = torch.tensor(labels, device=device, dtype=torch.long)
    return image, query, label, np.array(depths), np.array(breadths)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def max_k_in_use(cfg: dict) -> int:
    """Deepest iteration this run can ask for.

    An untied model must be built with at least this many cores or it will
    raise mid-run, so the number is derived rather than guessed.
    """
    candidates = [cfg["k_train"], cfg["k_eval"]]
    if cfg["k_schedule"] == "sampled":
        candidates.append(cfg["k_max_train"])
    if cfg.get("eval_k_sweep"):
        candidates.extend(cfg["eval_k_sweep"])
    return max(int(k) for k in candidates if k)


def build_model(cfg: dict, device):
    family = D.get_family(cfg["family"])
    arch = cfg["arch"]
    model_cfg = LoopViTConfig(
        d_model=cfg["d_model"],
        prelude_blocks=cfg["prelude_blocks"],
        core_blocks=cfg["core_blocks"],
        coda_blocks=cfg["coda_blocks"],
        num_classes=family.NUM_CLASSES,
        vocab_size=D.VOCAB_SIZE,
        query_len=D.L_MAX[cfg["family"]],
        state_init=cfg["state_init"],
        conditioning=cfg["conditioning"],
        inject=cfg["inject"],
        core_mode="echo" if arch == "echo" else "blocks",
        tied=arch != "untied",
        untied_copies=cfg["untied_copies"] or max_k_in_use(cfg),
        canvas=render.CANVAS * int(cfg.get("pixel_scale", 1)),
    )
    if arch == "feedforward":
        model = FeedforwardBaseline(model_cfg, k=cfg["k_train"])
    elif arch in ("looped", "echo", "untied"):
        if arch == "untied" and model_cfg.untied_copies > 16:
            print(
                f"warning: untied baseline with {model_cfg.untied_copies} cores is "
                f"{model_cfg.untied_copies}x the core parameters of the tied model. "
                f"Matched-parameter comparisons need a separate config."
            )
        model = LoopViT(model_cfg)
    else:
        raise ValueError(
            f"unknown arch {arch!r}, expected looped, feedforward, echo or untied"
        )
    return model.to(device)


def forward(model, image, query, k, bptt_window=None):
    if isinstance(model, FeedforwardBaseline):
        return model(image, query)
    return model(image, query, k=k, bptt_window=bptt_window)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, cfg: dict, split: str, device, batches: int) -> dict:
    """Held-out accuracy, broken out by depth and breadth.

    The per-cell breakdown is what gate G1.4 reads ("family C solvable at
    k=1 at every breadth"), so it is produced here rather than recomputed
    later from raw predictions.

    It also counts WHAT was predicted, not only whether it was right. A
    model that has collapsed onto one output class scores the frequency of
    that class in the eval set, which lands exactly where a failure is
    expected to land and is therefore believed. Ten runs failed that way
    before anyone looked, and three of them, across two parameter counts,
    reported an identical accuracy to sixteen digits because they had
    collapsed onto the same class. `n_distinct` and `modal_share` put that
    in every metrics file and every summary, so the next one is visible
    without somebody deciding to check. See L-024.
    """
    model.eval()
    total, correct = 0, 0
    by_cell: dict[tuple[int, int], list[int]] = {}
    pred_counts: collections.Counter = collections.Counter()
    # E0.4 of docs/why-experiments.md: which D4 character a model gets
    # right, read at every eval. Off unless a config asks, and the key is
    # deliberately absent from DEFAULTS, so no other run's stored config or
    # metrics change when this code reaches the cluster.
    char_hits = (
        {name: 0 for name in G.D4_CHARACTERS} if cfg.get("log_characters") else None
    )
    for b in range(batches):
        cursor = b * cfg["batch_size"]
        image, query, label, depths, breadths = make_batch(cfg, split, cursor, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = forward(model, image, query, cfg["k_eval"])
        pred = logits.float().argmax(-1)
        hit = (pred == label).cpu().numpy()
        pred_counts.update(int(c) for c in pred.cpu().numpy())
        if char_hits is not None:
            pairs = list(zip(pred.cpu().numpy().tolist(), label.cpu().numpy().tolist()))
            for name, f in G.D4_CHARACTERS.items():
                char_hits[name] += sum(f(p) == f(t) for p, t in pairs)
        correct += int(hit.sum())
        total += len(hit)
        for h, d, br in zip(hit, depths, breadths):
            by_cell.setdefault((int(d), int(br)), []).append(int(h))
    model.train()
    modal_pred, modal_n = pred_counts.most_common(1)[0]
    out = {
        "accuracy": correct / total,
        "n": total,
        "by_cell": {f"{d}x{b}": float(np.mean(v)) for (d, b), v in sorted(by_cell.items())},
        "n_distinct": len(pred_counts),
        "modal_pred": int(modal_pred),
        "modal_share": modal_n / total,
    }
    if char_hits is not None:
        out["characters"] = {name: n / total for name, n in char_hits.items()}
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class MetricLog:
    """Append-only long-schema metrics, flushed to parquet.

    One schema for everything, so the analysis layer never has to know
    which instrument wrote a row (IMPLEMENTATION.md Section 6.5).
    """

    def __init__(self, run_dir: Path, seed: int) -> None:
        self.path = run_dir / "metrics.parquet"
        self.seed = seed
        self.rows: list[dict] = []

    def add(self, step, k, split, metric_name, value, depth=-1, breadth=-1) -> None:
        self.rows.append(
            {
                "step": int(step),
                "k": int(k),
                "split": split,
                "depth": int(depth),
                "breadth": int(breadth),
                "metric_name": metric_name,
                "value": float(value),
                "seed": int(self.seed),
            }
        )

    def flush(self, keep_through_step: int | None = None) -> None:
        import pandas as pd

        frame = pd.DataFrame(self.rows)
        if self.path.exists():
            existing = pd.read_parquet(self.path)
            if keep_through_step is not None:
                existing = existing[existing["step"] <= keep_through_step]
            frame = pd.concat([existing, frame], ignore_index=True) if len(frame) else existing
        if len(frame):
            frame.to_parquet(self.path, index=False)
        self.rows = []


def truncate_metrics(run_dir: Path, keep_through_step: int) -> None:
    """Drop rows after the resumed step, so a restart cannot duplicate them."""
    import pandas as pd

    path = run_dir / "metrics.parquet"
    if not path.exists():
        return
    frame = pd.read_parquet(path)
    frame[frame["step"] <= keep_through_step].to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def write_provenance(run_dir: Path, cfg: dict) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    (run_dir / "git.txt").write_text(commit + "\n", encoding="utf-8")
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
    except Exception:
        freeze = ""
    (run_dir / "env.txt").write_text(freeze, encoding="utf-8")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def lr_at(step: int, cfg: dict) -> float:
    if step < cfg["warmup"]:
        return cfg["lr"] * step / max(1, cfg["warmup"])
    horizon = cfg.get("lr_schedule_steps") or cfg["steps"]
    progress = (step - cfg["warmup"]) / max(1, horizon - cfg["warmup"])
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def train(args) -> int:
    overrides = {
        "steps": args.steps,
        "seed": args.seed,
        "k_train": args.k,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    cfg = load_config(args.config, overrides)
    run_dir = Path(args.runs_root) / args.run_id

    # A resumed run continues under the config it STARTED with, not under
    # whatever configs/ says today. multirun.pbs passes --config on every
    # relaunch, so without this a config edited while a run is in flight
    # silently changes what that run computes at its next wall kill, and
    # the snapshot written by write_provenance stops describing the run it
    # belongs to. See L-025.
    snapshot = run_dir / "config.yaml"
    if snapshot.is_file():
        resumed = load_config(str(snapshot), overrides)
        drift = config_drift(cfg, resumed)
        if drift:
            print(
                f"WARNING config drift on {args.run_id}. {args.config} has "
                f"changed since this run started. Continuing under the run's "
                f"own config.yaml, which is what its checkpoints match. To "
                f"run the new config, use a new run id.",
                flush=True,
            )
            for key, (was, now) in drift.items():
                print(f"    {key}: run has {was!r}, {args.config} now says {now!r}")
        cfg = resumed

    run_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "DONE"
    if done_path.is_file() and not args.force:
        print(f"run {args.run_id} already has a DONE sentinel, nothing to do")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        free_mib = free / 1024**2
        if free_mib < 4096:
            raise RuntimeError(
                f"selected GPU has only {free_mib:.0f} MiB free. Refusing to "
                f"start rather than dying on OOM later. See decisions D-013."
            )
        print(f"device: {torch.cuda.get_device_name(0)}, {free_mib:.0f} MiB free")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=cfg["weight_decay"],
    )

    start_step, cursor = 0, 0
    if args.resume == "auto":
        ckpt = find_latest_checkpoint(run_dir)
        if ckpt is not None:
            state = load_checkpoint(ckpt, model, optimizer, None, map_location=device.type)
            start_step, cursor = state.step, state.data_cursor
            truncate_metrics(run_dir, start_step)
            print(f"resumed from {ckpt.name} at step {start_step}")
        else:
            print("no checkpoint found, cold start")

    if start_step == 0 and cfg.get("init_from"):
        # Deliberately after the resume block. A self-chained successor
        # resumes at a nonzero step and must NOT be re-seeded from the
        # donor: its own checkpoint already carries everything the donor
        # gave it plus whatever it has learned since. Re-initialising there
        # would silently throw away the run's progress at every wall clock
        # boundary, and the metrics would look like a run that kept
        # restarting for no reason.
        donor_dir = Path(args.runs_root) / cfg["init_from"]
        donor_ckpt = find_latest_checkpoint(donor_dir)
        if donor_ckpt is None:
            raise FileNotFoundError(
                f"init_from names {cfg['init_from']} but no checkpoint exists "
                f"under {donor_dir}. Refusing to cold start silently: a "
                f"curriculum run that quietly started from scratch would be "
                f"indistinguishable from its own control."
            )
        if not (donor_dir / "DONE").is_file():
            raise RuntimeError(
                f"init_from names {cfg['init_from']}, which has no DONE "
                f"sentinel and is therefore still training. Initialising from "
                f"a partial donor would hand this run a half learned "
                f"extraction and the curriculum would measure the donor's "
                f"training progress rather than the effect being tested. "
                f"Correctness cannot depend on the queue happening to order "
                f"these two jobs correctly, so this is checked here."
            )
        payload = torch.load(donor_ckpt, map_location=device.type, weights_only=False)
        model.load_state_dict(payload["model"])
        print(
            f"initialised weights from {cfg['init_from']} at step "
            f"{payload.get('step')} ({donor_ckpt.name}). Optimizer, step and "
            f"data cursor start fresh."
        )

    if start_step == 0:
        write_provenance(run_dir, cfg)
        print(
            f"{cfg['arch']} model, {model.num_parameters() / 1e6:.1f}M params, "
            f"family {cfg['family']}, k={cfg['k_train']}, "
            f"depths={cfg['depths']}, breadths={cfg['breadths']}"
        )

    log = MetricLog(run_dir, cfg["seed"])
    killer = GracefulKiller()
    started = time.monotonic()
    last_ckpt = time.monotonic()
    # D-065. Seeded from the stored metrics so a window spanning a job
    # boundary is judged on every evaluation in it, not only this job's.
    early_stop = cfg["early_stop_tau"] is not None
    history = eval_history(run_dir, start_step) if early_stop else []
    stopped_at = None

    for step in range(start_step + 1, cfg["steps"] + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)

        image, query, label, _, _ = make_batch(cfg, "train", cursor, device)
        cursor += cfg["batch_size"]

        if cfg["k_schedule"] == "sampled":
            k_step = loop_schedule.sample_loop_count(
                step, cfg["steps"], cfg["k_mean_target"],
                cfg["jitter_sigma"], cfg["k_max_train"],
            )
        else:
            k_step = cfg["k_train"]

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = forward(model, image, query, k_step, cfg["bptt_window"])
            loss = F.cross_entropy(logits.float(), label)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        if step % 50 == 0:
            log.add(step, k_step, "train", "loss", loss.item())
            log.add(step, k_step, "train", "grad_norm", float(grad_norm))

        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            result = evaluate(model, cfg, "iid_val", device, cfg["eval_batches"])
            log.add(step, cfg["k_eval"], "iid_val", "accuracy", result["accuracy"])
            for cell, value in result["by_cell"].items():
                d, b = cell.split("x")
                log.add(
                    step, cfg["k_eval"], "iid_val", "accuracy", value, depth=d, breadth=b
                )
            # What was predicted, not only whether it was right. See L-024.
            log.add(step, cfg["k_eval"], "iid_val",
                    "n_distinct_predictions", result["n_distinct"])
            log.add(step, cfg["k_eval"], "iid_val",
                    "modal_prediction_share", result["modal_share"])
            for name, value in result.get("characters", {}).items():
                log.add(step, cfg["k_eval"], "iid_val", f"character_{name}", value)
            log.flush()
            elapsed = (time.monotonic() - started) / 60
            print(
                f"step {step:6d}  loss {loss.item():.4f}  "
                f"val_acc {result['accuracy']:.4f}  lr {lr_at(step, cfg):.2e}  "
                f"{elapsed:.1f}min",
                flush=True,
            )
            if early_stop:
                history.append((step, float(result["accuracy"])))
                if early_stop_due(history, step, cfg["early_stop_tau"],
                                  cfg["early_stop_window"]):
                    stopped_at = step

        stop_for_wall = (
            args.max_hours is not None
            and (time.monotonic() - started) / 3600.0 >= args.max_hours
        )
        due = (time.monotonic() - last_ckpt) / 60 >= cfg["checkpoint_every_minutes"]
        if (killer.should_stop or stop_for_wall or due or step == cfg["steps"]
                or stopped_at == step):
            log.flush()
            save_checkpoint(run_dir, step, model, optimizer, None, cursor)
            last_ckpt = time.monotonic()
            if killer.should_stop or stop_for_wall:
                print(f"stopping cleanly at step {step}, checkpoint written")
                return 0
        if stopped_at == step:
            print(
                f"early stop at step {step}: validation accuracy at or above "
                f"{cfg['early_stop_tau']} for {cfg['early_stop_window']} steps (D-065)",
                flush=True,
            )
            break

    # The step the run's final numbers belong to: its budget, or where the
    # early-stop rule ended it.
    final_step = stopped_at if stopped_at is not None else cfg["steps"]

    sweep = cfg.get("eval_k_sweep")
    if sweep:
        # The M1 loop-count curve: one trained model, evaluated across k.
        # This is why a variable-k model is worth more than one model per k.
        print("loop-count curve (accuracy against k):", flush=True)
        for k in sweep:
            sub = dict(cfg)
            sub["k_eval"] = k
            r = evaluate(model, sub, "iid_val", device, cfg["eval_batches"] * 2)
            log.add(final_step, k, "iid_val", "sweep_accuracy", r["accuracy"])
            for cell, value in r["by_cell"].items():
                d, b = cell.split("x")
                log.add(final_step, k, "iid_val", "sweep_accuracy", value,
                        depth=d, breadth=b)
            print(f"    k={k:3d}  acc={r['accuracy']:.4f}", flush=True)

    result = evaluate(model, cfg, "iid_val", device, cfg["eval_batches"] * 4)
    log.add(final_step, cfg["k_eval"], "iid_val", "final_accuracy", result["accuracy"])
    for cell, value in result["by_cell"].items():
        d, b = cell.split("x")
        log.add(
            final_step, cfg["k_eval"], "iid_val", "final_accuracy", value,
            depth=d, breadth=b,
        )
    log.add(final_step, cfg["k_eval"], "iid_val",
            "final_n_distinct_predictions", result["n_distinct"])
    log.add(final_step, cfg["k_eval"], "iid_val",
            "final_modal_prediction_share", result["modal_share"])
    log.flush()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "config_hash": config_hash(cfg),
                "final_accuracy": result["accuracy"],
                "by_cell": result["by_cell"],
                "n_eval": result["n"],
                "n_distinct_predictions": result["n_distinct"],
                "modal_prediction": result["modal_pred"],
                "modal_prediction_share": result["modal_share"],
                "params": model.num_parameters(),
                "stopped_early_at": stopped_at,
                "config": cfg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    done_path.write_text("done\n", encoding="utf-8")
    # Nothing resumes a finished run, so its resume spare is dead weight.
    # See prune_to_latest; the disk reached 6 TB once without this.
    prune_to_latest(run_dir)
    print(f"completed. final val accuracy {result['accuracy']:.4f}")
    if result["n_distinct"] == 1:
        # Loud, because this is the number that reaches summary.json and
        # from there a table. A collapsed model reports the frequency of
        # its one class, which sits exactly where a failure is expected to.
        print(
            f"WARNING COLLAPSED. all {result['n']} items predicted class "
            f"{result['modal_pred']}. The accuracy above is that class's "
            f"frequency in the eval set, not a measurement of the model. "
            f"Do not quote it as a score or compare it across seeds. "
            f"See docs/learnings.md L-024.",
            flush=True,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a loopvision model")
    p.add_argument("--config")
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--resume", choices=["auto", "never"], default="never")
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None,
                   help="generation worker processes. multirun.pbs sets this "
                        "from the cores each run is given")
    p.add_argument("--force", action="store_true", help="overwrite a DONE run")
    return p


def main() -> int:
    return train(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
