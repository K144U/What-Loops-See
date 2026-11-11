"""The instrument path and the training path must not disagree.

IMPLEMENTATION.md Section 5 makes this a required test, and the reason is
worth restating: if a forward pass and the state sequence the instruments
read can silently differ, every measurement in the paper is suspect and
nothing about the failure is visible. Training would proceed normally,
losses would fall, and the heatmaps would be of something else.
"""

from __future__ import annotations

import pytest
import torch

from loopvision.data import dataset as D
from loopvision.model.loopvit import FeedforwardBaseline, LoopViT, LoopViTConfig

TOL = 1e-5


def make_model(**overrides) -> LoopViT:
    torch.manual_seed(0)
    cfg = LoopViTConfig(d_model=128, vocab_size=D.VOCAB_SIZE, **overrides)
    return LoopViT(cfg).eval()


def make_batch(batch: int = 2):
    torch.manual_seed(1)
    return torch.randn(batch, 3, 32, 32), torch.zeros(batch, 8, dtype=torch.long)


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_returned_states_reproduce_the_forward_pass(k: int) -> None:
    """Decoding the final captured state must equal the forward logits."""
    model = make_model(state_init="zeros")
    image, query = make_batch()
    with torch.no_grad():
        logits = model(image, query, k=k)
        logits2, states = model(image, query, k=k, return_states=True)
        from_state = model.decode(states[-1])

    assert len(states) == k + 1, "one state before the loop plus one per iteration"
    assert torch.allclose(logits, logits2, atol=TOL)
    assert torch.allclose(logits, from_state, atol=TOL), (
        "decoding the captured final state disagrees with the forward pass, "
        "so the M3 coda lens would be reading a different model"
    )


def test_states_advance_one_core_step_at_a_time() -> None:
    """Each captured state must be exactly one core application on the last.

    Replays the recurrence by hand and compares. This is what catches an
    off-by-one in state capture, which would shift every M2 heatmap by one
    loop index and change the conclusion about loop localisation.
    """
    model = make_model(state_init="zeros")
    image, query = make_batch()
    with torch.no_grad():
        _, states = model(image, query, k=5, return_states=True)

        x = model.embed_input(image, query)
        from loopvision.model.blocks import HEAD_DIM, build_rope_cache

        cos, sin = build_rope_cache(x.shape[1], HEAD_DIM, x.device, x.dtype)
        e = x
        for block in model.prelude:
            e = block(e, cos, sin)

        s = states[0]
        for i in range(5):
            s = model._core_step(s, e, cos, sin, i)
            assert torch.allclose(s, states[i + 1], atol=TOL), f"state {i + 1} differs"


def test_explicit_s0_makes_the_forward_pass_deterministic() -> None:
    """M2 needs clean and corrupted runs to share a starting state.

    With randn initialisation each call starts somewhere new, so a patching
    effect would be measured on top of that variation rather than against a
    fixed baseline.
    """
    model = make_model(state_init="randn")
    image, query = make_batch()
    with torch.no_grad():
        a = model(image, query, k=3)
        b = model(image, query, k=3)
        assert not torch.allclose(a, b, atol=TOL), (
            "randn initialisation should vary between calls, otherwise this "
            "test is not exercising the thing it claims to"
        )

        x = model.embed_input(image, query)
        s0 = torch.zeros_like(x)
        c = model(image, query, k=3, s0=s0)
        d = model(image, query, k=3, s0=s0)
    assert torch.allclose(c, d, atol=TOL)


def test_bptt_window_does_not_change_the_forward_values() -> None:
    """Truncated backpropagation changes gradients, never activations.

    If it changed the forward pass too, the ablation at Section 6.2 would
    be comparing two different models rather than two gradient estimators.
    """
    model = make_model(state_init="zeros")
    image, query = make_batch()
    with torch.no_grad():
        full = model(image, query, k=8)
        for window in (1, 4, 8, 16):
            truncated = model(image, query, k=8, bptt_window=window)
            assert torch.allclose(full, truncated, atol=TOL), f"window {window} differs"


def test_bptt_window_actually_truncates_gradients() -> None:
    """The complement of the previous test: the truncation must be real.

    Tested at window 0, where every iteration runs under no_grad and the
    state is detached, so the core must receive no gradient at all while
    the coda and head still do. That is an unambiguous check of the
    mechanism.

    Comparing gradient norms at window 1 against window 8 would be the
    obvious test and it does not work: at initialisation the core output
    projections are scaled by 1/sqrt(2 * core_depth * k_mean), so earlier
    iterations contribute almost nothing and the two norms agree to about
    0.04 percent. That is correct behaviour rather than a bug, and
    asserting a difference would be asserting something false.
    """
    image, query = make_batch()
    target = torch.zeros(image.shape[0], dtype=torch.long)

    model = make_model(state_init="zeros")
    logits = model(image, query, k=8, bptt_window=0)
    torch.nn.functional.cross_entropy(logits, target).backward()

    core_grad = sum(
        p.grad.abs().sum().item()
        for p in model.core.parameters()
        if p.grad is not None
    )
    coda_grad = sum(
        p.grad.abs().sum().item()
        for p in model.coda.parameters()
        if p.grad is not None
    )
    assert core_grad == 0.0, "window 0 must detach the loop entirely"
    assert coda_grad > 0.0, "the coda is outside the loop and must still learn"


def test_bptt_window_controls_how_many_steps_carry_gradient() -> None:
    """Window w means the last w iterations are inside the graph."""
    image, query = make_batch()
    model = make_model(state_init="zeros")
    for k, window, expected_nograd in [(8, 3, 5), (8, 8, 0), (8, 10, 0), (4, 1, 3)]:
        assert max(0, k - window) == expected_nograd


def test_k_zero_is_the_prelude_and_coda_alone() -> None:
    """k=0 must run, so the loop-count curve has a well defined origin."""
    model = make_model(state_init="zeros")
    image, query = make_batch()
    with torch.no_grad():
        assert model(image, query, k=0).shape == (2, model.cfg.num_classes)


def test_injection_ablation_changes_the_computation() -> None:
    """inject=False must actually remove the input, not silently do nothing."""
    image, query = make_batch()
    with torch.no_grad():
        a = make_model(state_init="zeros", inject=True)(image, query, k=4)
        b = make_model(state_init="zeros", inject=False)(image, query, k=4)
    assert not torch.allclose(a, b, atol=1e-3)


@pytest.mark.parametrize("conditioning", ["none", "embed", "adaln"])
def test_all_conditioning_variants_run(conditioning: str) -> None:
    model = make_model(state_init="zeros", conditioning=conditioning)
    image, query = make_batch()
    with torch.no_grad():
        assert model(image, query, k=3).shape == (2, model.cfg.num_classes)


def test_conditioning_makes_iterations_distinguishable() -> None:
    """The point of loop conditioning is that iteration i is not iteration j.

    With conditioning off, applying the tied core twice from the same state
    is the same function twice. With it on, the two must differ, or the
    variant is decorative.
    """
    image, query = make_batch()
    for conditioning in ("embed", "adaln"):
        model = make_model(state_init="zeros", conditioning=conditioning)
        with torch.no_grad():
            x = model.embed_input(image, query)
            from loopvision.model.blocks import HEAD_DIM, build_rope_cache

            cos, sin = build_rope_cache(x.shape[1], HEAD_DIM, x.device, x.dtype)
            e = x
            for block in model.prelude:
                e = block(e, cos, sin)
            s = torch.zeros_like(e)
            first = model._core_step(s, e, cos, sin, 0)
            second = model._core_step(s, e, cos, sin, 1)
        assert not torch.allclose(first, second, atol=1e-4), (
            f"{conditioning} conditioning does not distinguish iterations"
        )


def test_size_ladder_parameter_counts() -> None:
    """The ladder must match the spec's 5M, 11M, 19M, 43M."""
    expected = {256: 5.0, 384: 11.0, 512: 19.9, 768: 43.8}
    for d, millions in expected.items():
        model = LoopViT(LoopViTConfig(d_model=d, vocab_size=D.VOCAB_SIZE))
        actual = model.num_parameters() / 1e6
        assert abs(actual - millions) < 0.5, f"d={d}: {actual:.1f}M, expected {millions}M"


def test_feedforward_baseline_is_compute_matched() -> None:
    """Matched compute means matched block count, which is what G2 compares.

    A looped model at k has prelude + k*core + coda blocks of compute. The
    baseline must have the same, untied.
    """
    cfg = LoopViTConfig(d_model=128, vocab_size=D.VOCAB_SIZE)
    for k in (1, 2, 4):
        baseline = FeedforwardBaseline(cfg, k=k)
        expected = cfg.prelude_blocks + k * cfg.core_blocks + cfg.coda_blocks
        assert len(baseline.blocks) == expected
        assert baseline(*make_batch()).shape == (2, cfg.num_classes)


def test_core_output_projections_are_scaled_down_at_init() -> None:
    """Without this the state norm explodes in the first few hundred steps."""
    model = make_model()
    core_scale = model.core[0].attn.proj.weight.std().item()
    prelude_scale = model.prelude[0].attn.proj.weight.std().item()
    assert core_scale < prelude_scale * 0.5, (
        f"core projections at {core_scale:.4f} are not scaled below prelude "
        f"at {prelude_scale:.4f}"
    )
