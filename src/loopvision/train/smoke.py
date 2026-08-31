"""Milestone 0 smoke trainer.

Deliberately not a real model. This exists to prove one property before any
science depends on it: that a run killed at an arbitrary step and resumed
produces bit identical results to a run that was never interrupted.

The cluster makes this load bearing. `qalter` is blocked, jobs die at the
wall with no warning, and every training run therefore has to survive being
killed and continue exactly where it stopped. Building that after the model
exists means debugging it against a moving target, so it is built first
against a target that cannot move.

The checkpoint machinery here is the real machinery. `train/trainer.py` at
milestone 3 imports it rather than reimplementing it.

    python -m loopvision.train.smoke --run-id smoke01 --steps 200
    python -m loopvision.train.smoke --run-id smoke01 --steps 200 --resume auto
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from loopvision.train.checkpoint import (
    GracefulKiller,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def set_determinism(seed: int) -> None:
    """Make a CPU run reproducible bit for bit.

    GPU determinism is a separate and harder problem, deferred to milestone
    3 where it can be tested against the real model. The resume guarantee
    proved here is a CPU guarantee, and the test asserts it as such rather
    than implying more.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


class SmokeModel(nn.Module):
    """A miniature looped model: prelude, tied core applied k times, coda.

    Deliberately mirrors the real architecture's shape rather than being an
    arbitrary MLP. Two properties matter for milestone 0 and neither is
    cosmetic.

    First, the forward pass depends on k, so the loop-count sampling below
    genuinely affects the result. That is what makes RNG restoration
    load bearing and therefore testable. An earlier version of this file
    used a plain MLP with no stochastic element, and disabling RNG restore
    entirely did not fail a single test. See docs/learnings.md L-011.

    Second, weight tying across iterations means optimizer state
    accumulates through repeated application of the same parameters, which
    is the regime the real trainer runs in.
    """

    def __init__(self, d_in: int = 16, d_hidden: int = 32, d_out: int = 4) -> None:
        super().__init__()
        self.prelude = nn.Linear(d_in, d_hidden)
        self.core = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.coda = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor, k: int) -> torch.Tensor:
        s = self.prelude(x)
        for _ in range(k):
            # Residual application, so the state does not blow up as k grows.
            s = s + self.core(s)
        return self.coda(s)


def sample_loop_count(k_max: int = 4) -> int:
    """Draw a loop count from the global torch RNG stream.

    Mirrors the real schedule in IMPLEMENTATION.md Section 6.1, which draws
    tau from a lognormal and k from a Poisson, per batch rather than per
    sample. The distribution here is cruder; what matters for milestone 0
    is that the draw comes from the *global* stream, so that restoring RNG
    state is required for a resumed run to follow the same loop schedule.
    """
    tau = torch.exp(torch.randn(())) * 1.5
    k = int(torch.poisson(tau.clamp(min=1e-6)).item()) + 1
    return max(1, min(k, k_max))


def make_batch(
    cursor: int, master_seed: int, batch_size: int = 8, d_in: int = 16, d_out: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch as a pure function of (cursor, master_seed).

    This mirrors how the real data works: every sample is deterministic
    given (family, split, index, master_seed), so the dataloader position
    collapses to a single integer and a resume can restore it exactly.
    Nothing here reads global RNG, so batch content is independent of how
    many times the process has been restarted.
    """
    gen = torch.Generator().manual_seed((master_seed * 1_000_003 + cursor) % (2**31 - 1))
    x = torch.randn(batch_size, d_in, generator=gen)
    # Fixed teacher, so the target is a stable function of the input.
    tgen = torch.Generator().manual_seed(master_seed)
    w = torch.randn(d_in, d_out, generator=tgen)
    y = x @ w
    return x, y


def _truncate_metrics(path: Path, keep_through_step: int) -> None:
    """Drop metric rows after the resumed step.

    Without this a resumed run leaves duplicate rows for the steps between
    the last checkpoint and the kill, which would quietly corrupt any
    downstream analysis that counts rows.
    """
    if not path.is_file():
        return
    kept = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A partially written final line is exactly what a kill
                # during a write leaves behind. Dropping it is correct.
                continue
            if row["step"] <= keep_through_step:
                kept.append(line)
    with open(path, "w", encoding="utf-8") as fh:
        for line in kept:
            fh.write(line + "\n")


def train(args: argparse.Namespace) -> int:
    run_dir = Path(args.runs_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    done_path = run_dir / "DONE"

    if done_path.is_file():
        print(f"run {args.run_id} already has a DONE sentinel, nothing to do")
        return 0

    set_determinism(args.seed)

    model = SmokeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    start_step = 0
    data_cursor = 0

    if args.resume == "auto":
        ckpt = find_latest_checkpoint(run_dir)
        if ckpt is not None:
            state = load_checkpoint(ckpt, model, optimizer, scheduler)
            start_step = state.step
            data_cursor = state.data_cursor
            _truncate_metrics(metrics_path, keep_through_step=start_step)
            print(f"resumed from {ckpt.name} at step {start_step}")
        else:
            print("no checkpoint found, cold start")

    killer = GracefulKiller()
    started_at = time.monotonic()
    loss_fn = nn.MSELoss()

    for step in range(start_step + 1, args.steps + 1):
        x, y = make_batch(data_cursor, args.seed, batch_size=args.batch_size)
        data_cursor += 1

        # Drawn from the global RNG stream, so a resume that failed to
        # restore RNG state would follow a different loop schedule from
        # here on and the loss trace would diverge.
        k = sample_loop_count(args.k_max)

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x, k), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": f"{loss.item():.17g}",
                        "k": k,
                        "lr": f"{scheduler.get_last_lr()[0]:.17g}",
                        "data_cursor": data_cursor,
                    }
                )
                + "\n"
            )

        if step % args.checkpoint_every == 0:
            save_checkpoint(run_dir, step, model, optimizer, scheduler, data_cursor)

        # Abrupt self termination for the resume test. os._exit skips
        # atexit handlers, finally blocks and buffer flushes, which is the
        # closest a process can come to killing itself the way SIGKILL
        # would. Using a normal exit here would test nothing.
        if args.die_at is not None and step == args.die_at:
            print(f"dying abruptly at step {step} as instructed")
            os._exit(137)

        if killer.should_stop:
            save_checkpoint(run_dir, step, model, optimizer, scheduler, data_cursor)
            print(f"caught a termination signal, checkpointed at step {step}")
            return 0

        if args.max_hours is not None:
            if (time.monotonic() - started_at) / 3600.0 >= args.max_hours:
                save_checkpoint(run_dir, step, model, optimizer, scheduler, data_cursor)
                print(f"hit max-hours at step {step}, checkpointed for the successor")
                return 0

    save_checkpoint(run_dir, args.steps, model, optimizer, scheduler, data_cursor)
    done_path.write_text("done\n", encoding="utf-8")
    print(f"completed {args.steps} steps")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Milestone 0 smoke trainer")
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--k-max", type=int, default=4)
    p.add_argument("--resume", choices=["auto", "never"], default="never")
    p.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="stop cleanly and checkpoint before the PBS wall, then let the chain resubmit",
    )
    p.add_argument(
        "--die-at",
        type=int,
        default=None,
        help="testing only: terminate abruptly at this step via os._exit",
    )
    return p


def main() -> int:
    return train(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
