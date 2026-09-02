"""The echo and untied baselines, IMPLEMENTATION.md Section 5.

Both exist to rule out explanations of a loop-count curve that have nothing
to do with computation:

  echo    the core is the identity and only injection remains, so the state
          is refreshed from the input every iteration but never
          learned-updated. If a loop-count curve looks the same here, the
          curve is about re-reading the input, not about computing with it.
          This baseline caught a real problem in the text-case work.

  untied  one distinct core per iteration, same compute per iteration and
          many times the parameters. Separates weight tying from depth.

Both are modes of LoopViT rather than separate classes, so each differs
from the real model in exactly one respect. These tests exist mostly to
hold that "exactly one respect" claim to account.
"""

from __future__ import annotations

import pytest
import torch

from loopvision.data import dataset as D
from loopvision.model.loopvit import LoopViT, LoopViTConfig


def build(**kw) -> LoopViT:
    torch.manual_seed(0)
    cfg = LoopViTConfig(
        d_model=128, vocab_size=D.VOCAB_SIZE, state_init="zeros", **kw
    )
    return LoopViT(cfg).eval()


def batch(n: int = 2):
    torch.manual_seed(1)
    return torch.randn(n, 3, 32, 32), torch.zeros(n, 8, dtype=torch.long)


def core_param_count(model: LoopViT) -> int:
    total = sum(p.numel() for p in model.core.parameters())
    if model.untied_cores is not None:
        total += sum(p.numel() for p in model.untied_cores.parameters())
    return total


# ---------------------------------------------------------------------------
# Echo
# ---------------------------------------------------------------------------


def test_echo_has_no_core_parameters() -> None:
    """The whole point: nothing in the loop body is learned per iteration."""
    assert core_param_count(build(core_mode="echo")) == 0
    assert core_param_count(build()) > 0


def test_echo_still_depends_on_k() -> None:
    """Injection alone must still change the state as k grows.

    If echo were insensitive to k it would be a trivial baseline that
    proves nothing. It has to be a plausible rival explanation.
    """
    model = build(core_mode="echo")
    image, query = batch()
    with torch.no_grad():
        outs = [model(image, query, k=k) for k in (1, 2, 4, 8)]
    for a, b in zip(outs, outs[1:]):
        assert not torch.allclose(a, b, atol=1e-6), "echo output is insensitive to k"


def test_echo_state_is_the_adapter_output() -> None:
    """s_{i+1} = adapter([s_i ; e]) exactly, with no core applied."""
    model = build(core_mode="echo")
    image, query = batch()
    with torch.no_grad():
        _, states = model(image, query, k=3, return_states=True)
        x = model.embed_input(image, query)
        from loopvision.model.blocks import HEAD_DIM, build_rope_cache

        cos, sin = build_rope_cache(x.shape[1], HEAD_DIM, x.device, x.dtype)
        e = x
        for block in model.prelude:
            e = block(e, cos, sin)
        s = states[0]
        for i in range(3):
            s = model.adapter(torch.cat([s, e], dim=-1))
            assert torch.allclose(s, states[i + 1], atol=1e-5)


def test_echo_without_injection_is_refused() -> None:
    """No core and no injection is an empty loop body, not a baseline."""
    model = build(core_mode="echo", inject=False)
    image, query = batch()
    with pytest.raises(ValueError, match="empty loop body"):
        model(image, query, k=2)


# ---------------------------------------------------------------------------
# Untied
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("copies", [2, 4, 8])
def test_untied_scales_core_parameters_exactly(copies: int) -> None:
    tied = core_param_count(build())
    untied = core_param_count(build(tied=False, untied_copies=copies))
    assert untied == copies * tied


def test_untied_uses_a_different_core_each_iteration() -> None:
    """Perturbing one core must change only the iterations that use it.

    This is the test that would catch an indexing bug quietly reusing core
    zero every time, which would make the untied baseline a tied model with
    wasted parameters and no one would notice from the loss.
    """
    model = build(tied=False, untied_copies=4)
    image, query = batch()
    with torch.no_grad():
        before = [model(image, query, k=k) for k in (1, 2, 3, 4)]
        # Perturb the core used at iteration 2 only.
        for p in model.untied_cores[2].parameters():
            p.add_(0.05)
        after = [model(image, query, k=k) for k in (1, 2, 3, 4)]

    assert torch.allclose(before[0], after[0], atol=1e-6), "k=1 must not use core 2"
    assert torch.allclose(before[1], after[1], atol=1e-6), "k=2 must not use core 2"
    assert not torch.allclose(before[2], after[2], atol=1e-6), "k=3 must use core 2"
    assert not torch.allclose(before[3], after[3], atol=1e-6), "k=4 must use core 2"


def test_untied_refuses_to_extrapolate_past_its_cores() -> None:
    """An untied model cannot run deeper than it was built for.

    Silently reusing the last core would turn it into a partly tied model
    at large k and make its extrapolation curve meaningless, which is
    exactly the comparison it exists to provide.
    """
    model = build(tied=False, untied_copies=4)
    image, query = batch()
    with torch.no_grad():
        model(image, query, k=4)  # fine
        with pytest.raises(ValueError, match="untied model has 4 cores"):
            model(image, query, k=5)


# ---------------------------------------------------------------------------
# All modes share the real model's guarantees
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [{}, {"core_mode": "echo"}, {"tied": False, "untied_copies": 8}],
)
def test_state_capture_matches_the_forward_pass(kw: dict) -> None:
    """The instrument path must agree with the training path in every mode.

    A baseline whose captured states disagree with its own forward pass
    would produce M2 and M4 measurements of something that never ran.
    """
    model = build(**kw)
    image, query = batch()
    with torch.no_grad():
        logits = model(image, query, k=4)
        logits2, states = model(image, query, k=4, return_states=True)
        assert torch.allclose(logits, logits2, atol=1e-5)
        assert torch.allclose(logits, model.decode(states[-1]), atol=1e-5)
        assert len(states) == 5


@pytest.mark.parametrize(
    "kw",
    [{}, {"core_mode": "echo"}, {"tied": False, "untied_copies": 8}],
)
def test_every_mode_shares_prelude_coda_and_head(kw: dict) -> None:
    """Baselines must differ in the core only.

    If a baseline also differed in, say, the coda, a difference in results
    could not be attributed to the loop body, and the comparison would be
    confounded in a way no downstream test would reveal.
    """
    reference, model = build(), build(**kw)
    for part in ("prelude", "coda"):
        a = sum(p.numel() for p in getattr(reference, part).parameters())
        b = sum(p.numel() for p in getattr(model, part).parameters())
        assert a == b, f"{part} differs between the real model and {kw}"
    assert reference.head.weight.shape == model.head.weight.shape
    assert reference.adapter.weight.shape == model.adapter.weight.shape
