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
