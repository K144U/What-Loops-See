"""Shared state capture and injection. Every instrument uses this.

IMPLEMENTATION.md Section 7: "`hooks.py` provides one shared mechanism: a
context manager that captures `s_i` for all `i` and optionally overwrites
`s_i[:, positions, :]` from a stored donor run. Every instrument uses it;
none of them reimplements the loop."

That last clause is the whole point. The intervention happens through
`LoopViT.forward(state_hook=...)`, so a patched run goes down exactly the
same code path as a training step. An instrument with its own copy of the
loop can drift from the real one, and the failure is invisible: training
proceeds, losses fall, and the heatmaps describe a model that never ran.
`tests/test_model_invariance.py` guards the same boundary from the other
side.

Index convention, fixed here and relied on everywhere: state index `i`
means the state after `i` core applications, so `states[0]` is the initial
state and `states[k]` is what the coda decodes. A hook receives `(i, s)`
with the same indexing.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

StateHook = Callable[[int, torch.Tensor], "torch.Tensor | None"]


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


@torch.no_grad()
def capture(
    model,
    image: torch.Tensor,
    query: torch.Tensor,
    k: int,
    s0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the model and return (logits, states) with states[0..k].

    Pass `s0` whenever two runs must be comparable. With the default randn
    initialisation each call starts somewhere new, so a clean run and its
    corrupted twin would differ in their starting point as well as in the
    thing under study.
    """
    logits, states = model(image, query, k=k, return_states=True, s0=s0)
    return logits, states


def shared_init_state(model, image: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """An s_0 to hand to both runs of a counterfactual pair."""
    x = model.embed_input(image, query)
    return model.init_state(x)


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


def patch_hook(
    donor_states: Sequence[torch.Tensor],
    iteration: int,
    positions: Sequence[int],
) -> StateHook:
    """Overwrite `s_iteration[:, positions, :]` from a donor run.

    Every batch element gets the same positions patched. For the M2 grid,
    prefer `position_scan_hook`, which patches a different position per
    batch element and turns 64 forward passes into one.
    """
    idx = torch.as_tensor(list(positions), dtype=torch.long)

    def hook(i: int, s: torch.Tensor):
        if i != iteration:
            return None
        donor = donor_states[i]
        if donor.shape != s.shape:
            raise ValueError(
                f"donor state at iteration {i} has shape {tuple(donor.shape)} "
                f"but the running state is {tuple(s.shape)}. The two runs must "
                f"share batch size, sequence length and width."
            )
        out = s.clone()
        out[:, idx.to(s.device), :] = donor[:, idx.to(donor.device), :].to(s.dtype)
        return out

    return hook


def position_scan_hook(
    donor_states: Sequence[torch.Tensor],
    iteration: int,
    positions: Sequence[int],
) -> StateHook:
    """Patch one position per batch element, batch element j gets positions[j].

    The M2 grid is k iterations by 64 patch positions, which is 1024 forward
    passes per item done naively. Replicating the item across the batch and
    patching a different position in each replica collapses the position
    axis into the batch, so the grid costs k passes rather than k times 64.

    The caller is responsible for replicating the input so that batch size
    equals len(positions); a mismatch raises rather than broadcasting,
    because a silent broadcast here would patch the wrong positions and the
    heatmap would still look plausible.
    """
    idx = torch.as_tensor(list(positions), dtype=torch.long)

    def hook(i: int, s: torch.Tensor):
        if i != iteration:
            return None
        if s.shape[0] != len(idx):
            raise ValueError(
                f"position scan needs batch size {len(idx)}, one replica per "
                f"position, but the running state has batch {s.shape[0]}. "
                f"Replicate the input before calling."
            )
        donor = donor_states[i]
        out = s.clone()
        rows = torch.arange(s.shape[0], device=s.device)
        cols = idx.to(s.device)
        out[rows, cols, :] = donor[rows.to(donor.device), cols.to(donor.device), :].to(s.dtype)
        return out

    return hook


def compose_hooks(*hooks: StateHook) -> StateHook:
    """Apply several hooks in order at the same forward pass."""

    def hook(i: int, s: torch.Tensor):
        changed = None
        for h in hooks:
            out = h(i, s if changed is None else changed)
            if out is not None:
                changed = out
        return changed

    return hook


@contextmanager
def patched(model, donor_states, iteration: int, positions: Sequence[int], scan: bool = False):
    """Context manager form, for readability at a call site.

    Yields a hook to pass to `model.forward(state_hook=...)`. The model is
    not mutated, so nothing needs undoing on exit; the context manager
    exists to make the scope of an intervention obvious in the code.
    """
    builder = position_scan_hook if scan else patch_hook
    yield builder(donor_states, iteration, positions)


# ---------------------------------------------------------------------------
# The M2 metric
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogitDiff:
    """Normalised recovery, 0 at the corrupted run and 1 at the clean run.

    Defined on the difference between the clean answer's logit and the
    corrupted answer's logit, which is the standard patching readout and is
    less noisy than accuracy at these sample sizes.
    """

    clean: float
    corrupted: float

    def recovered(self, patched_value: float) -> float:
        denom = self.clean - self.corrupted
        if abs(denom) < 1e-8:
            # The twin did not move the readout, so recovery is undefined
            # rather than zero. Returning 0.0 here would quietly fill a
            # heatmap cell with a number that means "no effect measured".
            return float("nan")
        return (patched_value - self.corrupted) / denom


def logit_difference(
    logits: torch.Tensor, clean_label: int, corrupt_label: int
) -> torch.Tensor:
    """Per-example readout: logit(clean answer) minus logit(corrupt answer)."""
    return logits[:, clean_label] - logits[:, corrupt_label]


def recovery_grid(
    model,
    clean_image: torch.Tensor,
    corrupt_image: torch.Tensor,
    query: torch.Tensor,
    clean_label: int,
    corrupt_label: int,
    k: int,
    positions: Sequence[int] | None = None,
    corrupt_query: torch.Tensor | None = None,
) -> torch.Tensor:
    """The M2 heatmap for one item: recovery at every (iteration, position).

    Returns a (k+1, n_positions) tensor. Row i is the effect of patching
    the state after i core applications, so row 0 is the initial state and
    row k is what the coda sees.

    Both runs share one s_0, so the only difference between them is the
    corrupted input and whatever the patch restores.

    ``corrupt_query`` exists because the counterfactual does not always live
    in the image. Family A renders its operators into the scene, so the twin
    differs in pixels and the query is shared. Family B puts the relation
    chain in the query, so its twin differs in one query token and the image
    is byte identical. Passing one query for both runs there makes the two
    runs the same run, every logit difference zero, and every cell NaN.
    """
    device = clean_image.device
    n_positions = model.cfg.seq_len if positions is None else len(positions)
    positions = list(range(model.cfg.seq_len)) if positions is None else list(positions)

    corrupt_query = query if corrupt_query is None else corrupt_query
    s0 = shared_init_state(model, clean_image, query)
    clean_logits, clean_states = capture(model, clean_image, query, k, s0=s0)
    corrupt_logits, _ = capture(model, corrupt_image, corrupt_query, k, s0=s0)

    scale = LogitDiff(
        clean=float(logit_difference(clean_logits, clean_label, corrupt_label)[0]),
        corrupted=float(logit_difference(corrupt_logits, clean_label, corrupt_label)[0]),
    )

    # Replicate the corrupted item once per position, and replicate the
    # donor states to match, so one pass covers the whole position axis.
    rep_image = corrupt_image.expand(n_positions, -1, -1, -1).contiguous()
    rep_query = corrupt_query.expand(n_positions, -1).contiguous()
    rep_s0 = s0.expand(n_positions, -1, -1).contiguous()
    rep_donor = [st.expand(n_positions, -1, -1).contiguous() for st in clean_states]

    grid = torch.full((k + 1, n_positions), float("nan"), device=device)
    for i in range(k + 1):
        hook = position_scan_hook(rep_donor, i, positions)
        with torch.no_grad():
            logits = model(rep_image, rep_query, k=k, s0=rep_s0, state_hook=hook)
        diffs = logit_difference(logits, clean_label, corrupt_label)
        grid[i] = torch.tensor(
            [scale.recovered(float(d)) for d in diffs], device=device
        )
    return grid
