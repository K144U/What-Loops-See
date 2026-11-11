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
from loopvision.model.loopvit import (
    HEAD_DIM,
    FeedforwardBaseline,
    LoopViT,
    LoopViTConfig,
    build_rope_cache,
)


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


# ---------------------------------------------------------------------------
# The feedforward baseline's trajectory, added 2026-09-10.
#
# M4 geometry failed on every ffwd and untied run for want of a return_states
# path here, so F5 has had no feedforward arm at all. The risk in adding one
# is not that the trajectory is wrong, it is that recording it perturbs the
# logits: 44 finished runs were trained through this forward pass and a
# second code path that merely looks equivalent is how they stop matching.
# ---------------------------------------------------------------------------


def build_ffwd(k: int = 4, **kw) -> FeedforwardBaseline:
    torch.manual_seed(0)
    cfg = LoopViTConfig(d_model=128, vocab_size=D.VOCAB_SIZE, **kw)
    return FeedforwardBaseline(cfg, k=k).eval()


def test_recording_states_does_not_move_the_logits() -> None:
    """The regression guard the rest of this section depends on. Bit
    identical, not close: a tolerance here would hide exactly the kind of
    reordering that makes an old checkpoint score differently."""
    model = build_ffwd()
    image, query = batch()
    with torch.no_grad():
        plain = model(image, query)
        logits, _ = model(image, query, return_states=True)
    assert torch.equal(plain, logits)


def test_the_trajectory_is_one_state_per_core_group() -> None:
    """k + 1 states, matching the looped model's s_0 to s_k, so column i of
    M4's geometry means the same number of core applications in both."""
    for k in (1, 3, 4):
        model = build_ffwd(k=k)
        image, query = batch()
        with torch.no_grad():
            _, states = model(image, query, return_states=True)
        assert len(states) == k + 1


@pytest.mark.parametrize("blocks", [
    {"prelude_blocks": 2, "core_blocks": 2, "coda_blocks": 2},
    {"prelude_blocks": 1, "core_blocks": 1, "coda_blocks": 1},
    {"prelude_blocks": 0, "core_blocks": 2, "coda_blocks": 1},
    {"prelude_blocks": 3, "core_blocks": 1, "coda_blocks": 0},
])
def test_states_are_sampled_at_the_group_boundaries(blocks: dict) -> None:
    """Recomputed by walking the blocks by hand. Without this the sampling
    could be off by one group and every quantity would still look plausible,
    because a residual stream one block earlier is not visibly different."""
    k = 3
    model = build_ffwd(k=k, **blocks)
    image, query = batch()
    p, c = blocks["prelude_blocks"], blocks["core_blocks"]

    with torch.no_grad():
        _, states = model(image, query, return_states=True)

        patches = model.patchify(image).flatten(2).transpose(1, 2)
        tokens = model.query_embed(query)
        x = torch.cat([patches, tokens], dim=1)
        seg = torch.zeros(x.shape[1], dtype=torch.long)
        seg[patches.shape[1]:] = 1
        x = x + model.segment(seg)[None, :, :]
        cos, sin = build_rope_cache(x.shape[1], HEAD_DIM, x.device, x.dtype)

        expected = []
        for idx, block in enumerate(model.blocks):
            if idx == p:
                expected.append(x)
            x = block(x, cos, sin)
        if p == len(model.blocks):
            expected.append(x)

    # expected[0] is the activation entering the first core group, which is
    # the state after the prelude. Rebuild the rest by stepping groups.
    with torch.no_grad():
        y = expected[0]
        got = [y]
        idx = p
        for _ in range(k):
            for _ in range(c):
                y = model.blocks[idx](y, cos, sin)
                idx += 1
            got.append(y)

    assert len(states) == len(got)
    for i, (a, b) in enumerate(zip(states, got)):
        assert torch.equal(a, b), f"state {i} is not the group boundary"


def test_an_initial_state_is_refused_rather_than_ignored() -> None:
    """M2 passes s0 so a clean run and its twin start together. A stack has
    no state to set, and accepting the argument silently would report a
    counterfactual that never happened."""
    model = build_ffwd()
    image, query = batch()
    with pytest.raises(ValueError, match="no initial state"):
        model(image, query, return_states=True, s0=torch.zeros(2, 2, 128))


def test_the_requested_k_cannot_stretch_a_fixed_stack() -> None:
    """The pbs asks for k=32 and these models are built at k_train=8. A
    looped model can honour that and this one cannot, so the trajectory must
    follow the architecture rather than the request."""
    model = build_ffwd(k=4)
    image, query = batch()
    with torch.no_grad():
        _, states = model(image, query, k=32, return_states=True)
    assert len(states) == 5


def test_geometry_reads_a_feedforward_trajectory() -> None:
    """The end the whole change exists for: M4's own function, on states
    this class produced, without special casing."""
    from loopvision.instruments.m4_geometry import METRICS, geometry

    model = build_ffwd(k=4)
    image, query = batch(n=3)
    with torch.no_grad():
        _, states = model(image, query, return_states=True)
    out = geometry([s.cpu() for s in states])
    for m in METRICS:
        assert out[m].shape == (3, 4)


# ---------------------------------------------------------------------------
# usable_k, added 2026-09-10. Five geometry jobs died because the pbs sends
# one K for every run and two of the three architectures have a depth fixed
# when they were built.
# ---------------------------------------------------------------------------


def test_a_tied_looped_model_keeps_the_requested_k() -> None:
    """The recurrent case. Nothing should be clamped here, or the looped
    arm of F5 would silently lose depth it can actually run."""
    from loopvision.instruments.m4_geometry import usable_k
    from loopvision.model.loopvit import LoopViT, LoopViTConfig

    m = LoopViT(LoopViTConfig(d_model=64, vocab_size=D.VOCAB_SIZE)).eval()
    assert usable_k(m, 32) == 32


def test_an_untied_model_is_clamped_to_its_own_cores() -> None:
    """The real failure: the untied guard raises past untied_copies, by
    design, so the request has to come down to what it has rather than the
    model being loosened to accept it."""
    from loopvision.instruments.m4_geometry import usable_k
    from loopvision.model.loopvit import LoopViT, LoopViTConfig

    m = LoopViT(LoopViTConfig(d_model=64, vocab_size=D.VOCAB_SIZE,
                              tied=False, untied_copies=4)).eval()
    assert usable_k(m, 32) == 4
    assert usable_k(m, 2) == 2


def test_a_feedforward_stack_reports_its_built_depth() -> None:
    """Its depth is structural, so the request never applies either way."""
    from loopvision.instruments.m4_geometry import usable_k

    m = build_ffwd(k=8)
    assert usable_k(m, 32) == 8
    assert usable_k(m, 2) == 8


def test_an_untied_model_at_its_clamped_k_actually_runs() -> None:
    """The end to end point. Before this the same call raised ValueError and
    took the whole geometry job with it."""
    from loopvision.instruments.m4_geometry import geometry, usable_k
    from loopvision.model.loopvit import LoopViT, LoopViTConfig

    m = LoopViT(LoopViTConfig(d_model=64, vocab_size=D.VOCAB_SIZE,
                              tied=False, untied_copies=4)).eval()
    image, query = batch(n=2)
    k = usable_k(m, 32)
    with torch.no_grad():
        _, states = m(image, query, k=k, return_states=True)
    assert len(states) == 5
    assert geometry([s.cpu() for s in states])["norm"].shape == (2, 4)
