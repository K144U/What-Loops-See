"""Restricting family A to one factor of D4 x S3.

The depth 2 runs solve D4 exactly and leave S3 at chance. Training on each
factor alone separates a hard subtask from gradient competition, but only
if the restricted task is really a task: closed under composition, uniform
in its labels, and leaving the full task untouched.

That last point is the one with teeth. Two family A runs are in flight
against this code, and a change to the "full" sampling path would corrupt
them silently at the next chained restart.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from loopvision.data import dataset as D
from loopvision.data import family_a as FA
from loopvision.data import groups as G


@pytest.mark.parametrize("name", ["full", "d4", "s3"])
def test_subgroups_are_closed_under_the_group_operation(name: str) -> None:
    m = set(G.subgroup_members(name))
    assert G.IDENTITY in m, "a subgroup must contain the identity"
    for a in m:
        assert G.inverse(a) in m, f"{name} is not closed under inverse"
        for b in m:
            assert G.multiply(a, b) in m, f"{name} is not closed under multiply"


def test_subgroup_sizes_match_the_factors() -> None:
    assert len(G.subgroup_members("d4")) == G.D4_ORDER
    assert len(G.subgroup_members("s3")) == G.S3_ORDER
    assert len(G.subgroup_members("full")) == G.GROUP_ORDER


def test_the_d4_subgroup_is_the_d4_factor_and_likewise_s3() -> None:
    """Named for the factor they isolate, so the name must be true."""
    for g in G.subgroup_members("d4"):
        assert G.s3_of(g) == G.S3_ELEMENTS[0], "d4 subgroup must fix the S3 part"
    assert {G.d4_of(g) for g in G.subgroup_members("d4")} == set(G.D4_ELEMENTS)

    for g in G.subgroup_members("s3"):
        assert G.d4_of(g) == G.D4_ELEMENTS[0], "s3 subgroup must fix the D4 part"
    assert {G.s3_of(g) for g in G.subgroup_members("s3")} == set(G.S3_ELEMENTS)


def test_the_full_path_is_byte_identical_to_before_the_option_existed() -> None:
    """Two 1M step runs are training against this path right now.

    `random_factorisation_in` over the full group must make the same draws
    in the same order as `random_factorisation`, or a chained restart would
    resume on a different data stream than it was checkpointed on.
    """
    full = G.subgroup_members("full")
    for target in (0, 5, 23, 47):
        for n in (1, 2, 3, 5):
            a = G.random_factorisation(np.random.default_rng(11), target, n)
            b = G.random_factorisation_in(np.random.default_rng(11), target, n, full)
            assert a == b, f"full path diverged at target {target}, n {n}"


def test_full_samples_are_unchanged_end_to_end() -> None:
    for i in range(40):
        s = FA.generate(D.global_index("train", i), "train")
        t = FA.generate(D.global_index("train", i), "train", D.TaskConfig(factor="full"))
        assert s.program == t.program
        assert s.label == t.label
        assert np.array_equal(s.image, t.image)


@pytest.mark.parametrize("name,order", [("d4", 8), ("s3", 6)])
def test_restricted_samples_stay_inside_the_subgroup(name: str, order: int) -> None:
    members = set(G.subgroup_members(name))
    cfg = D.TaskConfig(factor=name)
    labels = []
    for i in range(300):
        s = FA.generate(D.global_index("train", i), "train", cfg)
        assert s.label in members, f"label {s.label} escaped the {name} subgroup"
        labels.append(s.label)
        initial, strips, queried, _, _ = FA.parse_program(s.program)
        for g in initial:
            assert g in members, "an initial state escaped the subgroup"
        for strip in strips:
            for op in strip:
                assert op in members, "an operator escaped the subgroup"
        assert G.multiply(G.compose_sequence(strips[queried]), initial[queried]) == s.label
    assert len(set(labels)) == order, (
        f"expected all {order} labels of {name}, saw {len(set(labels))}"
    )


@pytest.mark.parametrize("name,order", [("d4", 8), ("s3", 6)])
def test_restricted_labels_are_uniform(name: str, order: int) -> None:
    """Chance must be 1/|subgroup| or the accuracy numbers mean nothing."""
    cfg = D.TaskConfig(factor=name)
    c = collections.Counter(
        FA.generate(D.global_index("train", i), "train", cfg).label
        for i in range(4000)
    )
    p = np.array([c[g] for g in G.subgroup_members(name)]) / 4000
    assert abs(p.max() - 1 / order) < 0.03, f"labels not uniform: {p}"
    assert abs(p.min() - 1 / order) < 0.03, f"labels not uniform: {p}"


@pytest.mark.parametrize("name", ["d4", "s3"])
def test_the_corrupted_twin_stays_inside_the_subgroup(name: str) -> None:
    """A twin containing an operator the task never shows would make a
    patching result measure novelty rather than the counterfactual."""
    members = set(G.subgroup_members(name))
    cfg = D.TaskConfig(factor=name)
    for i in range(60):
        s = FA.generate(D.global_index("train", i), "train", cfg)
        t = FA.corrupted_twin(s, cfg)
        _, strips, queried, _, _ = FA.parse_program(t.program)
        for op in strips[queried]:
            assert op in members, f"twin operator {op} escaped the {name} subgroup"
        assert t.label in members


def test_a_restricted_task_is_still_non_abelian() -> None:
    """If a factor were abelian, order would not matter and the experiment
    would answer a different question than the one being asked."""
    for name in ("d4", "s3"):
        m = G.subgroup_members(name)
        assert any(
            G.multiply(a, b) != G.multiply(b, a) for a in m for b in m
        ), f"{name} is abelian, so it cannot test sequential composition"


def test_the_factor_key_actually_reaches_the_data() -> None:
    """A config key that is parsed but never read is the worst kind of bug:
    the run looks right, the yaml looks right, and the data is the old task.

    `eval_k_sweep` sat one line away from exactly this. So the check is not
    that `task_config` copies the field, it is that a config asking for the
    S3 factor produces samples that are actually inside S3.
    """
    from loopvision.train.cli import DEFAULTS, task_config

    assert "factor" in DEFAULTS, "factor must have a default or configs cannot set it"
    assert DEFAULTS["factor"] == "full", "the default must be the real task"

    cfg = dict(DEFAULTS)
    cfg.update({"master_seed": 20260901, "depths": [2], "breadths": [4], "factor": "s3"})
    tcfg = task_config(cfg)
    assert tcfg.factor == "s3"

    members = set(G.subgroup_members("s3"))
    for i in range(50):
        s = FA.generate(D.global_index("train", i), "train", tcfg)
        assert s.label in members, (
            "the config asked for the S3 factor and got a full-group sample"
        )


def test_an_unknown_factor_is_rejected_rather_than_ignored() -> None:
    with pytest.raises(ValueError, match="unknown subgroup"):
        G.subgroup_members("D4")  # right factor, wrong case
    with pytest.raises(ValueError, match="unknown subgroup"):
        FA.generate(D.global_index("train", 0), "train", D.TaskConfig(factor="s4"))
