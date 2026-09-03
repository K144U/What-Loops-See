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
from loopvision.model.loopvit import FeedforwardBaseline, LoopViT, LoopViTConfig
from loopvision.train import loop_schedule
from loopvision.train.checkpoint import (
    GracefulKiller,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)

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
    return cfg


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
        num_classes=family.NUM_CLASSES,
        vocab_size=D.VOCAB_SIZE,
        query_len=D.L_MAX[cfg["family"]],
        state_init=cfg["state_init"],
        conditioning=cfg["conditioning"],
        inject=cfg["inject"],
        core_mode="echo" if arch == "echo" else "blocks",
        tied=arch != "untied",
        untied_copies=cfg["untied_copies"] or max_k_in_use(cfg),
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
    """
    model.eval()
    total, correct = 0, 0
    by_cell: dict[tuple[int, int], list[int]] = {}
    for b in range(batches):
        cursor = b * cfg["batch_size"]
        image, query, label, depths, breadths = make_batch(cfg, split, cursor, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = forward(model, image, query, cfg["k_eval"])
        hit = (logits.float().argmax(-1) == label).cpu().numpy()
        correct += int(hit.sum())
        total += len(hit)
        for h, d, br in zip(hit, depths, breadths):
            by_cell.setdefault((int(d), int(br)), []).append(int(h))
    model.train()
    return {
        "accuracy": correct / total,
        "n": total,
        "by_cell": {f"{d}x{b}": float(np.mean(v)) for (d, b), v in sorted(by_cell.items())},
    }


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
    cfg = load_config(
        args.config,
        {
            "steps": args.steps,
            "seed": args.seed,
            "k_train": args.k,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
    )

    run_dir = Path(args.runs_root) / args.run_id
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
            log.flush()
            elapsed = (time.monotonic() - started) / 60
            print(
                f"step {step:6d}  loss {loss.item():.4f}  "
                f"val_acc {result['accuracy']:.4f}  lr {lr_at(step, cfg):.2e}  "
                f"{elapsed:.1f}min",
                flush=True,
            )

        stop_for_wall = (
            args.max_hours is not None
            and (time.monotonic() - started) / 3600.0 >= args.max_hours
        )
        due = (time.monotonic() - last_ckpt) / 60 >= cfg["checkpoint_every_minutes"]
        if killer.should_stop or stop_for_wall or due or step == cfg["steps"]:
            log.flush()
            save_checkpoint(run_dir, step, model, optimizer, None, cursor)
            last_ckpt = time.monotonic()
            if killer.should_stop or stop_for_wall:
                print(f"stopping cleanly at step {step}, checkpoint written")
                return 0

    sweep = cfg.get("eval_k_sweep")
    if sweep:
        # The M1 loop-count curve: one trained model, evaluated across k.
        # This is why a variable-k model is worth more than one model per k.
        print("loop-count curve (accuracy against k):", flush=True)
        for k in sweep:
            sub = dict(cfg)
            sub["k_eval"] = k
            r = evaluate(model, sub, "iid_val", device, cfg["eval_batches"] * 2)
            log.add(cfg["steps"], k, "iid_val", "sweep_accuracy", r["accuracy"])
            for cell, value in r["by_cell"].items():
                d, b = cell.split("x")
                log.add(cfg["steps"], k, "iid_val", "sweep_accuracy", value,
                        depth=d, breadth=b)
            print(f"    k={k:3d}  acc={r['accuracy']:.4f}", flush=True)

    result = evaluate(model, cfg, "iid_val", device, cfg["eval_batches"] * 4)
    log.add(cfg["steps"], cfg["k_eval"], "iid_val", "final_accuracy", result["accuracy"])
    for cell, value in result["by_cell"].items():
        d, b = cell.split("x")
        log.add(
            cfg["steps"], cfg["k_eval"], "iid_val", "final_accuracy", value,
            depth=d, breadth=b,
        )
    log.flush()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "config_hash": config_hash(cfg),
                "final_accuracy": result["accuracy"],
                "by_cell": result["by_cell"],
                "n_eval": result["n"],
                "params": model.num_parameters(),
                "config": cfg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    done_path.write_text("done\n", encoding="utf-8")
    print(f"completed. final val accuracy {result['accuracy']:.4f}")
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
