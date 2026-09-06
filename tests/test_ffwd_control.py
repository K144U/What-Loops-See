"""The feedforward control must stay matched to the task it controls for.

The control answers one question: does a non-looped model of matched depth
deadlock on colour too? That question is only answered if the control
differs from its looped twin in the architecture and nothing else. A single
edit to one config and not the other turns the control into a comparison of
two different experiments, and nothing in the training run would complain.

These tests are cheap on purpose so they run in the normal suite rather
than behind the slow marker: block counts do not depend on d_model, so the
models are built narrow.
"""

from __future__ import annotations

import pytest

from loopvision.train.cli import build_model, load_config

# (label, looped config, feedforward control config)
PAIRS = [
    ("d4_colour", "configs/m3/famA_d2_d4colour.yaml", "configs/m3/famA_d2_d4colour_ffwd.yaml"),
    ("s3_native", "configs/m3/famA_d2_s3only.yaml", "configs/m3/famA_d2_s3only_ffwd.yaml"),
]

# The control is allowed to differ from its looped twin in exactly these
# keys. arch selects the baseline, and k_schedule/k_train pin the depth it
# is built at. Anything else means the two runs are different experiments.
ALLOWED_DIFF = {"arch", "k_schedule", "k_train"}

NARROW = {"d_model": 64}


@pytest.mark.parametrize("label,looped_path,ffwd_path", PAIRS, ids=[p[0] for p in PAIRS])
def test_control_differs_only_in_architecture(label, looped_path, ffwd_path):
    looped = load_config(looped_path, {})
    ffwd = load_config(ffwd_path, {})
    differing = {k for k in set(looped) | set(ffwd) if looped.get(k) != ffwd.get(k)}
    assert differing <= ALLOWED_DIFF, (
        f"{label}: control diverges from its looped twin in "
        f"{sorted(differing - ALLOWED_DIFF)}, so it is not a control"
    )
    assert looped["arch"] == "looped"
    assert ffwd["arch"] == "feedforward"


@pytest.mark.parametrize("label,looped_path,ffwd_path", PAIRS, ids=[p[0] for p in PAIRS])
def test_control_is_matched_compute(label, looped_path, ffwd_path):
    """Built depth equals the depth the looped twin executes at k_train.

    Read the expected count from the LOOPED config, not the control's own,
    so that changing the control's block counts cannot make this pass.
    """
    looped = load_config(looped_path, NARROW)
    ffwd = load_config(ffwd_path, NARROW)

    # k comes from the LOOPED twin's training operating point, never from
    # the control. Reading it from the control lets the control redefine
    # its own target: k_train drops, the expected count drops with it, and
    # the assertion still passes while the compute is no longer matched.
    # The first version of this test did exactly that and the mutation
    # check caught it. See L-020.
    k = looped["k_mean_target"] if looped["k_schedule"] == "sampled" else looped["k_train"]
    assert ffwd["k_train"] == k, (
        f"{label}: control is built at k={ffwd['k_train']} but its looped twin "
        f"trains at k={k}, so the compute is not matched"
    )
    expected = looped["prelude_blocks"] + k * looped["core_blocks"] + looped["coda_blocks"]

    model = build_model(ffwd, "cpu")
    assert len(model.blocks) == expected, (
        f"{label}: control builds {len(model.blocks)} blocks but the looped twin "
        f"executes {expected} at k={k}, so the compute is not matched"
    )


@pytest.mark.parametrize("label,looped_path,ffwd_path", PAIRS, ids=[p[0] for p in PAIRS])
def test_control_ignores_k_at_forward(label, looped_path, ffwd_path):
    """Depth is fixed at build time, so k must not change the output.

    If the control ever responded to k it would be a looped model wearing a
    different name, and the comparison would be meaningless.
    """
    import torch

    ffwd = load_config(ffwd_path, NARROW)
    model = build_model(ffwd, "cpu")
    model.eval()

    from loopvision.train.cli import forward

    torch.manual_seed(0)
    image = torch.randn(2, 3, model.cfg.canvas, model.cfg.canvas)
    query = torch.zeros(2, model.cfg.query_len, dtype=torch.long)

    with torch.no_grad():
        a = forward(model, image, query, k=1)
        b = forward(model, image, query, k=32)
    assert torch.equal(a, b), f"{label}: control output changed with k"

# ---------------------------------------------------------------------------
# The matched-PARAMETER arm. A different baseline answering a different
# question, and the two are one number apart in the configs, so the tests
# have to hold that number down from the looped side.
# ---------------------------------------------------------------------------

MP_PAIRS = [
    ("d4_colour", "configs/m3/famA_d2_d4colour.yaml", "configs/m3/famA_d2_d4colour_ffwd_mp.yaml"),
    ("s3_native", "configs/m3/famA_d2_s3only.yaml", "configs/m3/famA_d2_s3only_ffwd_mp.yaml"),
]

# The looped model carries an injection adapter that the feedforward baseline
# has no counterpart for, so exact parity is not reachable. Measured at 2.69
# percent on 2026-09-06. The bound is loose enough to survive that and tight
# enough to fail if the depth is wrong: k=2 is already 1.295x.
PARAM_TOLERANCE = 0.05

# The matched-parameter arm sits at a different k from its looped twin, and
# k_eval is derived from k, so it moves with it. Listed explicitly rather
# than folded into ALLOWED_DIFF, which stays strict for the matched-compute
# control where the two happen to coincide at k=8.
MP_ALLOWED_DIFF = ALLOWED_DIFF | {"k_eval"}


@pytest.mark.parametrize("label,looped_path,mp_path", MP_PAIRS, ids=[p[0] for p in MP_PAIRS])
def test_mp_control_differs_only_in_architecture(label, looped_path, mp_path):
    looped = load_config(looped_path, {})
    mp = load_config(mp_path, {})
    differing = {k for k in set(looped) | set(mp) if looped.get(k) != mp.get(k)}
    assert differing <= MP_ALLOWED_DIFF, (
        f"{label}: matched-parameter arm diverges from its looped twin in "
        f"{sorted(differing - MP_ALLOWED_DIFF)}, so it is not a control"
    )
    assert mp["arch"] == "feedforward"

    # k_eval is allowed to differ ONLY because it is derived, never set by
    # hand: load_config sets it to k_mean_target under a sampled schedule and
    # to k_train under a fixed one. Assert the derivation rather than merely
    # tolerating the difference, so a hand-set k_eval still fails here. It is
    # inert either way, because the architecture ignores k at forward, which
    # test_control_ignores_k_at_forward pins down separately.
    assert mp["k_schedule"] == "fixed"
    assert mp["k_eval"] == mp["k_train"], (
        f"{label}: k_eval is {mp['k_eval']} where the derivation from a fixed "
        f"schedule gives {mp['k_train']}, so it was set by hand"
    )


@pytest.mark.parametrize("label,looped_path,mp_path", MP_PAIRS, ids=[p[0] for p in MP_PAIRS])
def test_mp_control_matches_the_looped_unique_block_count(label, looped_path, mp_path):
    """Built depth equals the number of DISTINCT blocks the looped model holds.

    The looped model reuses one core, so its parameters live in
    prelude + core + coda blocks however many times it iterates. A
    feedforward model with that many blocks therefore carries the same
    parameters. Every term is read from the looped config, never from the
    control, because a control that supplies its own target verifies
    nothing. See L-020.
    """
    looped = load_config(looped_path, NARROW)
    mp = load_config(mp_path, NARROW)

    unique_blocks = looped["prelude_blocks"] + looped["core_blocks"] + looped["coda_blocks"]
    # k solving prelude + k*core + coda == prelude + core + coda
    implied_k = 1
    assert mp["k_train"] == implied_k, (
        f"{label}: matched-parameter arm is built at k={mp['k_train']}, but parity "
        f"with the looped block count needs k={implied_k}"
    )

    model = build_model(mp, "cpu")
    assert len(model.blocks) == unique_blocks, (
        f"{label}: arm builds {len(model.blocks)} blocks against the looped "
        f"model's {unique_blocks} distinct blocks, so parameters are not matched"
    )


@pytest.mark.parametrize("label,looped_path,mp_path", MP_PAIRS, ids=[p[0] for p in MP_PAIRS])
def test_mp_control_parameter_count_is_within_tolerance(label, looped_path, mp_path):
    """And it must be no LARGER than the looped model.

    The direction matters more than the magnitude. If the baseline solves a
    task the looped model deadlocks on while carrying fewer parameters, then
    capacity cannot be the explanation. A baseline that crept above the
    looped model would reopen exactly the question it exists to close.

    Built at the real d_model, because the adapter's share of the total
    depends on width and a narrow model would not measure the real ratio.
    """
    looped = build_model(load_config(looped_path, {}), "cpu")
    mp = build_model(load_config(mp_path, {}), "cpu")
    ratio = mp.num_parameters() / looped.num_parameters()
    assert ratio <= 1.0, (
        f"{label}: matched-parameter arm has {ratio:.3f}x the looped parameters, "
        f"which is the wrong side of parity for this control"
    )
    assert abs(ratio - 1.0) <= PARAM_TOLERANCE, (
        f"{label}: parameter ratio {ratio:.3f} is outside {PARAM_TOLERANCE:.0%} of parity"
    )
