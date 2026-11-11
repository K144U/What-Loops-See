"""The untied baseline decomposes the gate G2 failure.

G2.1 failed against a matched-compute feedforward arm, but that arm changed
two things at once: it untied the weights AND removed the loop structure,
the state and the injection. The gate therefore says looping lost without
saying which property lost it.

The untied arm restores the loop structure while keeping the untied
weights, so it sits between the other two:

    untied vs feedforward   same parameters, same compute  -> the loop
    untied vs looped        same compute, same depth       -> weight tying

That only works if the arm stays pinned to its reference. These tests hold
it there, and every expected value is read from the reference side rather
than from the arm itself. See L-020.
"""

from __future__ import annotations

import pytest

from loopvision.train.cli import build_model, load_config, max_k_in_use

LOOPED = "configs/m3/famB_curve_111.yaml"
FFWD = "configs/m3/famB_g2_ffwd.yaml"
UNTIED = "configs/m3/famB_g2_untied.yaml"

# The untied arm is derived from the feedforward arm, so only these may
# differ. eval_k_sweep must shrink because an untied model cannot serve a k
# above its copy count, and untied_copies does not exist on the other arms.
ALLOWED_DIFF = {"arch", "eval_k_sweep", "untied_copies"}

# untied carries the injection adapter that the feedforward baseline has no
# counterpart for. Measured at 294912 parameters on 2026-09-07.
ADAPTER_PARAMS = 294912


def test_untied_differs_from_the_feedforward_arm_only_where_it_must():
    ffwd = load_config(FFWD, {})
    untied = load_config(UNTIED, {})
    differing = {k for k in set(ffwd) | set(untied) if ffwd.get(k) != untied.get(k)}
    assert differing <= ALLOWED_DIFF, (
        f"untied arm diverges from the feedforward arm in "
        f"{sorted(differing - ALLOWED_DIFF)}, so the comparison is confounded"
    )
    assert untied["arch"] == "untied"


def test_untied_copies_are_pinned_and_sufficient():
    """The trap this baseline nearly fell into.

    untied_copies defaults to max_k_in_use, which reads eval_k_sweep. The
    inherited sweep runs to 64, so an unpinned untied model allocates 64
    core copies, roughly 20x the core parameters, and is a matched
    comparison to nothing at all. Pinning it is only safe if the sweep was
    also truncated, or the run raises partway through.
    """
    untied = load_config(UNTIED, {})
    copies = untied["untied_copies"]
    assert copies is not None, "untied_copies must be pinned, not derived from the sweep"
    assert copies == untied["k_train"], (
        f"untied arm has {copies} core copies but trains at k={untied['k_train']}; "
        f"spare copies are parameters the comparison did not ask for"
    )
    needed = max_k_in_use(untied)
    assert needed <= copies, (
        f"the run can ask for k={needed} but only {copies} cores exist, "
        f"so it would raise partway through training"
    )


def test_untied_builds_exactly_the_pinned_number_of_cores():
    untied = load_config(UNTIED, {"d_model": 64})
    model = build_model(untied, "cpu")
    assert hasattr(model, "untied_cores")
    assert len(model.untied_cores) == untied["untied_copies"]


def test_untied_matches_the_feedforward_arm_on_parameters():
    """Same parameters as the feedforward arm, up to the adapter.

    This is what makes untied vs feedforward a clean read on the loop
    structure. The expected value comes from the feedforward arm, never
    from the untied one.
    """
    ffwd = build_model(load_config(FFWD, {}), "cpu")
    untied = build_model(load_config(UNTIED, {}), "cpu")
    delta = untied.num_parameters() - ffwd.num_parameters()
    assert delta == ADAPTER_PARAMS, (
        f"untied exceeds feedforward by {delta} parameters, expected exactly "
        f"{ADAPTER_PARAMS}, the injection adapter. Any other difference means "
        f"the two arms are not parameter matched"
    )


def test_untied_costs_more_parameters_than_tying():
    """And it must be larger than the looped arm, or it is not untied.

    Untying is precisely the decision to spend parameters instead of
    reusing them. An untied arm at or below the looped arm's size would
    mean the copies are not real.
    """
    looped = build_model(load_config(LOOPED, {}), "cpu")
    untied = build_model(load_config(UNTIED, {}), "cpu")
    assert untied.num_parameters() > looped.num_parameters()
