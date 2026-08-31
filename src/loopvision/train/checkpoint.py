"""Checkpoint save and resume.

This module is load bearing, not a convenience. `qalter` is blocked on the
target cluster, so a running job cannot have its walltime extended and dies
at the wall with no warning. Every training run must be resumable to the
exact optimizer, data cursor and RNG state. A training script that cannot
survive SIGKILL at a random step is broken.

Two tiers, per docs/IMPLEMENTATION.md Section 6.4:

  resume checkpoints  full state, written on an interval and on SIGTERM,
                      last 2 kept. These are what a job restart reads.
  probe checkpoints   weights only, bf16, no optimizer state, at log-spaced
                      steps. Only for configurations feeding M1 and M5.

Writes are atomic. We write to a temporary file in the destination
directory and then os.replace onto the final name, which is atomic on POSIX
and on Windows for same-volume renames. A kill during a write therefore
leaves the previous checkpoint intact rather than truncating the only copy.
"""

from __future__ import annotations

import os
import random
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Bumped whenever the on-disk layout changes in a way that makes old
# checkpoints unreadable. Resume refuses to load a mismatched version
# rather than silently misinterpreting fields.
CHECKPOINT_FORMAT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG that can influence training.

    Missing any one of these makes a resume merely close rather than exact,
    and "merely close" is indistinguishable from a real effect at the sample
    sizes we run.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.set_rng_state wants a ByteTensor on CPU. Checkpoints loaded with
    # map_location can arrive as something else, so normalise.
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_states = [
            torch.as_tensor(s, dtype=torch.uint8).cpu() for s in state["torch_cuda"]
        ]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        # A different device count than the one that wrote the checkpoint is
        # not fatal for CPU-deterministic resume, so we do not raise. It is
        # recorded by the caller in docs/decisions.md if it ever happens.


@dataclass
class TrainState:
    """Everything needed to continue a run as if it had never stopped.

    `data_cursor` is the index of the next sample to be generated. All data
    in this project is procedural and deterministic given
    (family, split, index, master_seed), so a single integer cursor is a
    complete description of dataloader position. That is a property worth
    protecting: it is why resume can be bit exact here when it usually
    cannot be with a shuffled DataLoader and worker processes.
    """

    step: int
    data_cursor: int
    rng: dict[str, Any]


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def save_checkpoint(
    run_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    data_cursor: int,
    keep_last: int = 2,
) -> Path:
    """Write a full resume checkpoint atomically and prune old ones."""
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "data_cursor": data_cursor,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": capture_rng_state(),
    }

    path = ckpt_dir / f"step_{step:09d}.pt"
    _atomic_torch_save(payload, path)

    # Pointer file so resume does not have to guess. Written after the
    # checkpoint itself, so a kill between the two leaves the pointer
    # stale-but-valid rather than pointing at a file that does not exist.
    _atomic_write_text(ckpt_dir / "LATEST", path.name)

    _prune_checkpoints(ckpt_dir, keep_last=keep_last)
    return path


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _prune_checkpoints(ckpt_dir: Path, keep_last: int) -> None:
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    for stale in ckpts[:-keep_last] if keep_last > 0 else []:
        try:
            stale.unlink()
        except OSError:
            # A pruning failure must never take down a training run.
            pass


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the newest usable checkpoint, or None for a cold start.

    Falls back to a directory scan if the LATEST pointer is missing or
    points at a file that is not there, which is the state a kill between
    the checkpoint write and the pointer write would leave behind.
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.is_dir():
        return None

    pointer = ckpt_dir / "LATEST"
    if pointer.is_file():
        candidate = ckpt_dir / pointer.read_text(encoding="utf-8").strip()
        if candidate.is_file():
            return candidate

    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    return ckpts[-1] if ckpts else None


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    map_location: str = "cpu",
) -> TrainState:
    """Restore model, optimizer, scheduler and RNG from a checkpoint."""
    payload = torch.load(path, map_location=map_location, weights_only=False)

    found = payload.get("format_version")
    if found != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            f"checkpoint format version {found} does not match expected "
            f"{CHECKPOINT_FORMAT_VERSION}. Refusing to load rather than "
            f"silently misreading fields: {path}"
        )

    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"])

    return TrainState(
        step=payload["step"],
        data_cursor=payload["data_cursor"],
        rng=payload["rng"],
    )


class GracefulKiller:
    """Set a flag on SIGTERM so the trainer can checkpoint before dying.

    PBS sends SIGTERM before it sends SIGKILL. That window is how a run
    saves its work when it hits the wall. SIGKILL itself is uncatchable by
    design, which is exactly why the interval checkpoint has to exist too:
    the SIGTERM path is the polite case, not the guarantee.

    Windows does not deliver SIGTERM the way POSIX does, so registration is
    guarded. Local development is not the environment this protects against.
    """

    def __init__(self) -> None:
        self.should_stop = False
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError, AttributeError):
                # Not in the main thread, or the platform does not support
                # this signal. Interval checkpointing still covers us.
                pass

    def _handle(self, signum, frame) -> None:  # noqa: ANN001, ARG002
        self.should_stop = True
