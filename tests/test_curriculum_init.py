"""Starting from another run's weights, without becoming that run.

The curriculum arm hands a depth 2 model the extraction a depth 1 model
already learned, to ask whether S3 composition is unlearnable in itself or
only unreachable because nothing upstream ever supplies it.

That only means anything if three things hold, and each has a way of
failing silently:

  the weights actually arrive         or it is just a cold start
  the step counter does not arrive    or it is a continuation, not a curriculum
  a chained restart does not re-seed  or the run resets at every wall clock
                                         boundary and looks like noise
"""

from __future__ import annotations

import pathlib

import pytest
import torch
import yaml

from loopvision.train import checkpoint as C
from loopvision.train.cli import DEFAULTS, build_model


def _cfg(**over):
    cfg = dict(DEFAULTS)
    cfg.update({
        "family": "A", "arch": "looped", "d_model": 64,
        "prelude_blocks": 1, "core_blocks": 1, "coda_blocks": 1,
        "k_train": 2, "k_eval": 2, "untied_copies": None,
        "state_init": "randn", "conditioning": "none", "inject": True,
        "factor": "s3", "depths": [1], "breadths": [4],
        "master_seed": 20260901, "seed": 0,
    })
    cfg.update(over)
    return cfg


def test_init_from_is_a_known_config_key() -> None:
    """A key that is parsed but never read is the L-017 failure, and here
    it would produce a curriculum run that silently cold started."""
    assert "init_from" in DEFAULTS
    assert DEFAULTS["init_from"] is None, "the default must be a cold start"


def test_the_donor_weights_actually_change_the_model(tmp_path) -> None:
    """Loading must move the parameters. If it did not, the curriculum arm
    would be its own control and the comparison would be empty."""
    cfg = _cfg()
    device = torch.device("cpu")

    donor = build_model(cfg, device)
    with torch.no_grad():
        for prm in donor.parameters():
            prm.add_(torch.randn_like(prm) * 0.5)

    fresh = build_model(cfg, device)
    before = [p.detach().clone() for p in fresh.parameters()]
    fresh.load_state_dict(donor.state_dict())
    after = list(fresh.parameters())

    moved = sum(
        1 for b, a in zip(before, after) if not torch.allclose(b, a)
    )
    assert moved > 0, "loading the donor state changed nothing"
    for d, a in zip(donor.parameters(), after):
        assert torch.allclose(d, a), "the loaded weights are not the donor's"


def test_a_shape_mismatch_is_loud(tmp_path) -> None:
    """A curriculum across different architectures is a silent disaster if
    it half loads. torch raises, and this pins that it is not suppressed."""
    small = build_model(_cfg(d_model=64), torch.device("cpu"))
    large = build_model(_cfg(d_model=128), torch.device("cpu"))
    with pytest.raises(RuntimeError):
        large.load_state_dict(small.state_dict())


def test_the_init_only_fires_at_step_zero() -> None:
    """The guard that keeps a self-chained successor from being re-seeded.

    Read from the source rather than executed, because the alternative is
    running a full training step inside a unit test. The condition is one
    line and its correctness is entirely in the `start_step == 0` clause:
    without it, every wall clock restart would throw away the run's
    progress and the loss curve would sawtooth back to the donor forever.
    """
    src = pathlib.Path("src/loopvision/train/cli.py").read_text(encoding="utf-8")
    assert "if start_step == 0 and cfg.get(\"init_from\"):" in src, (
        "the init_from block must be guarded on start_step == 0, or a "
        "chained restart re-seeds from the donor and discards its progress"
    )
    init_at = src.index('if start_step == 0 and cfg.get("init_from")')
    resume_at = src.index('if args.resume == "auto"')
    assert resume_at < init_at, (
        "the init must come after the resume block, or a resumed run is "
        "overwritten by the donor before its own checkpoint is applied"
    )


def test_a_missing_donor_refuses_rather_than_cold_starts() -> None:
    """A curriculum run that quietly started from scratch would be
    indistinguishable from its own control, and would be reported as
    evidence that the curriculum did not help."""
    src = pathlib.Path("src/loopvision/train/cli.py").read_text(encoding="utf-8")
    block = src[src.index('if start_step == 0 and cfg.get("init_from")'):]
    block = block[: block.index("if start_step == 0:")]
    assert "FileNotFoundError" in block
    assert "Refusing to cold start silently" in block


def test_checkpoints_carry_what_the_curriculum_needs(tmp_path) -> None:
    """init_from reads `model` out of a checkpoint payload. If the format
    changed, the curriculum would break at submission time rather than
    here."""
    cfg = _cfg()
    model = build_model(cfg, torch.device("cpu"))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run = tmp_path / "donor"
    run.mkdir()
    C.save_checkpoint(run, 1234, model, opt, None, 5678)

    ckpt = C.find_latest_checkpoint(run)
    assert ckpt is not None
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model" in payload, "init_from loads payload['model']"
    assert payload.get("step") == 1234
