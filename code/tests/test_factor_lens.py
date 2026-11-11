"""A lens that reads chance because it asked the wrong question looks
exactly like a model that computed nothing.

That is the whole hazard here. The lens exists to make a null trustworthy,
so every way it could produce a null for the wrong reason has to be closed:
scoring against a regenerated program that is not the one the batch asked,
projecting the head's prediction onto the wrong factor, or averaging per
batch so a short batch quietly reweights the answer.

The pair with `state_probe` also has to stay a pair. If the two modules
drift apart on which five intermediates they mean, the panel compares two
different decompositions while claiming to compare one.
"""

from __future__ import annotations

import numpy as np
import pytest

from loopvision.analysis import factor_lens as FL
from loopvision.analysis import state_probe as SP


def test_the_targets_are_shared_with_the_probe_not_copied() -> None:
    """F8 puts the lens row and the probe row on the same states. If the
    two modules kept separate lists of what the five intermediates are,
    one could be reordered and the panel would compare `op1` against
    `partial` while both rows looked correct."""
    assert FL.TARGETS is SP.TARGETS
    assert FL.FACTORS is SP.FACTORS


def test_a_perfect_reading_of_the_answer_scores_one_there_and_chance_elsewhere() -> None:
    """The positive control's shape. A model whose head reads the composite
    exactly should show 1.0 in that column and nothing anywhere else."""
    rng = np.random.default_rng(0)
    n = 600
    targets = {t: rng.integers(0, 6, n) for t in FL.TARGETS}
    grid = FL.score([targets["composite"]], targets)
    row = dict(zip(FL.TARGETS, grid[0]))
    assert row["composite"] == pytest.approx(1.0)
    for t in FL.TARGETS:
        if t != "composite":
            assert row[t] < 0.35, f"{t} read {row[t]} by coincidence"


def test_a_constant_prediction_scores_each_target_at_its_own_frequency() -> None:
    """What a collapsed model produces. The row is not a measurement of
    the model, it is the frequency of one class in each target, and the
    arithmetic has to give exactly that so the number can be recognised."""
    targets = {t: np.array([0, 0, 1, 2, 3, 4]) for t in FL.TARGETS}
    targets["op1"] = np.array([0, 1, 1, 1, 1, 1])
    grid = FL.score([np.zeros(6, dtype=int)], targets)
    row = dict(zip(FL.TARGETS, grid[0]))
    assert row["initial"] == pytest.approx(2 / 6)
    assert row["op1"] == pytest.approx(1 / 6)


def test_the_grid_has_one_row_per_pass_and_one_column_per_target() -> None:
    targets = {t: np.arange(10) % 6 for t in FL.TARGETS}
    grid = FL.score([np.zeros(10, dtype=int)] * 4, targets)
    assert len(grid) == 4
    assert all(len(r) == len(FL.TARGETS) for r in grid)


def test_the_columns_are_in_the_declared_target_order() -> None:
    """The printed panel and the stored grid are read positionally, so a
    row whose columns are permuted is wrong in a way nothing else catches.

    Mutation checked: iterating a set instead of TARGETS reorders these.
    """
    targets = {t: np.zeros(4, dtype=int) for t in FL.TARGETS}
    targets["partial"] = np.ones(4, dtype=int)
    grid = FL.score([np.zeros(4, dtype=int)], targets)
    assert grid[0][FL.TARGETS.index("partial")] == pytest.approx(0.0)
    assert grid[0][FL.TARGETS.index("initial")] == pytest.approx(1.0)


def test_the_wrong_question_raises_rather_than_returning_a_null() -> None:
    """The guard that matters most.

    A lens scored against a program the batch did not ask reads chance,
    and chance is exactly what the interesting result looks like. So the
    disagreement has to stop the run rather than enter the panel.

    Mutation checked: deleting the guard makes this return a grid.
    """
    with pytest.raises(RuntimeError, match="wrong question"):
        FL.assert_same_question(np.array([1, 2, 3]), np.array([1, 2, 4]))


def test_the_guard_says_how_many_items_disagreed() -> None:
    """One item off is a different problem from every item off: the first
    is an indexing slip at a boundary, the second is the wrong split."""
    with pytest.raises(RuntimeError, match="1 of 3"):
        FL.assert_same_question(np.array([1, 2, 3]), np.array([1, 2, 4]))


def test_the_guard_passes_when_the_question_matches() -> None:
    FL.assert_same_question(np.array([1, 2, 3]), np.array([1, 2, 3]))


def test_scoring_is_over_the_whole_set_not_a_mean_of_batch_means() -> None:
    """Batches are concatenated before scoring. Averaging per batch would
    weight a short final batch as heavily as a full one, which moves the
    number without changing the data."""
    targets = {t: np.array([0] * 100 + [1] * 4) for t in FL.TARGETS}
    grid = FL.score([np.zeros(104, dtype=int)], targets)
    assert grid[0][0] == pytest.approx(100 / 104)
    assert grid[0][0] != pytest.approx(0.5)


def test_the_factor_projection_matches_the_probe_on_group_elements() -> None:
    """The lens projects the head's predicted element onto the factor with
    the same functions the probe projects its targets with. If the two used
    different conventions the panel would compare S3 against D4."""
    s3, s3n = FL.FACTORS["s3"]
    d4, d4n = FL.FACTORS["d4"]
    elements = np.arange(48)
    assert s3n == 6 and d4n == 8
    assert np.array_equal(s3(elements), elements % 6)
    assert np.array_equal(d4(elements), elements // 6)


def test_both_factors_partition_the_group() -> None:
    """48 elements, 6 times 8. If the projections did not tile the group,
    some element would have no factor and the lens would silently score it
    against the wrong index."""
    s3, _ = FL.FACTORS["s3"]
    d4, _ = FL.FACTORS["d4"]
    elements = np.arange(48)
    pairs = set(zip(d4(elements).tolist(), s3(elements).tolist()))
    assert len(pairs) == 48


def test_there_is_no_site_option() -> None:
    """A lens can only see what the head sees, and decode pools over the
    query tokens. A patch-pooled lens would be reaching past the head and
    calling the result what the head reads."""
    import inspect

    params = inspect.signature(FL.lens).parameters
    assert "site" not in params
    assert "site" in SP.run.__code__.co_varnames or hasattr(SP, "SITES")


def test_a_missing_run_is_refused(tmp_path) -> None:
    assert FL.main(["--run", str(tmp_path / "nope")]) == 2
