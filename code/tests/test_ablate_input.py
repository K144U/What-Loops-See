"""Blanking one input at a time, to catch a shortcut a blank image cannot.

A blank-image control passes as long as the model reads something. It
cannot tell a model that composes its operators from one that found a way
to answer using only part of the scene. This instrument removes exactly one
thing at a time, so the pattern of what hurts says which shortcut exists.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from loopvision.analysis import ablate_input as AB
from loopvision.data import render


def test_blanking_actually_clears_the_patch() -> None:
    """The whole instrument rests on this. If the patch were left intact
    every ablation would read as harmless and the report would say the
    model is clean no matter what it does."""
    img = torch.ones(3, render.CANVAS, render.CANVAS) * 200.0
    AB.blank_patch(img, 2, 3)
    p = render.PATCH
    region = img[:, 2 * p : 3 * p, 3 * p : 4 * p]
    for c in range(3):
        assert torch.allclose(
            region[c], torch.full_like(region[c], float(render.BACKGROUND[c]))
        ), "the patch was not set to the background colour"


def test_blanking_touches_only_the_named_patch() -> None:
    """An ablation that spilled into its neighbours would blame the wrong
    input: op1 would look required because blanking it also removed op2."""
    img = torch.ones(3, render.CANVAS, render.CANVAS) * 200.0
    before = img.clone()
    AB.blank_patch(img, 1, 1)
    p = render.PATCH
    changed = (img != before).any(dim=0)
    ys, xs = torch.where(changed)
    assert ys.min() >= p and ys.max() < 2 * p
    assert xs.min() >= p and xs.max() < 2 * p


def test_a_model_that_ignores_an_operator_is_flagged() -> None:
    """The failure this exists to catch, stated as a table."""
    acc = {"none": 1.0, "state": 0.13, "op1": 0.13, "op2": 0.98,
           "distractors": 1.0, "marker": 0.13, "all": 0.12}
    lines = AB.verdict(acc, chance=0.125)
    joined = " ".join(lines)
    assert "WARNING" in joined
    assert "op2" in joined, "the ignored operator must be named"


def test_a_model_that_uses_the_distractors_is_flagged() -> None:
    """Distractor rows carry no information about the answer by
    construction, so using them means the sampler is leaking."""
    acc = {"none": 1.0, "state": 0.13, "op1": 0.13, "op2": 0.13,
           "distractors": 0.60, "marker": 0.13, "all": 0.12}
    joined = " ".join(AB.verdict(acc, chance=0.125))
    assert "WARNING" in joined and "distractor" in joined


def test_an_honest_model_produces_no_warnings() -> None:
    acc = {"none": 1.0, "state": 0.12, "op1": 0.13, "op2": 0.12,
           "distractors": 0.99, "marker": 0.13, "all": 0.12}
    lines = AB.verdict(acc, chance=0.125)
    assert not any("WARNING" in x for x in lines), lines
    assert any("required" in x for x in lines)
    assert any("correctly ignored" in x for x in lines)
