"""The coda lens must measure the hop rate, not manufacture one.

The seeds disagree about how many core passes a depth 6 chain needs, and
the explanation on offer is that they advance the chain at different rates.
That explanation is only worth anything if the measurement could have come
out flat, so these tests check the arithmetic that turns a frontier into a
rate, including the cases where it must refuse to answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from loopvision.analysis import coda_lens as CL
from loopvision.data import dataset as D
from loopvision.data import family_b as FB


def test_hop_rate_reads_a_clean_one_hop_per_pass() -> None:
    assert CL.hop_rate([0, 1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_hop_rate_reads_two_hops_per_pass() -> None:
    assert CL.hop_rate([0, 2, 4, 6]) == pytest.approx(2.0)


def test_a_flat_frontier_is_a_rate_of_zero_not_a_refusal() -> None:
    """A model that parses the question and never advances is a real
    result, and it must be reported as zero rather than as missing."""
    assert CL.hop_rate([0, 0, 0, 0]) == pytest.approx(0.0)


def test_the_finished_tail_does_not_drag_the_rate_down() -> None:
    """Once the chain is walked the frontier flattens.

    Including that tail would penalise exactly the models that finish
    fastest, which would invert the quantity being measured.
    """
    fast = CL.hop_rate([0, 2, 4, 5, 5, 5, 5, 5, 5])
    slow = CL.hop_rate([0, 1, 2, 3, 4, 5])
    assert fast > slow, f"fast model measured {fast}, slow model {slow}"
    assert fast == pytest.approx(2.0, abs=0.35)


def test_a_frontier_that_never_starts_returns_none() -> None:
    """-1 means no hop was reached at all. That is not a rate of zero, it
    is an absence of evidence, and the two must not be conflated."""
    assert CL.hop_rate([-1, -1, -1]) is None
    assert CL.hop_rate([-1, -1, 0]) is None


def test_hop_labels_end_where_the_sample_says_it_ends() -> None:
    """The lens scores against regenerated labels. If regeneration drifted
    from the real sample the whole grid would be measured against the
    wrong question, silently."""
    for i in range(60):
        idx = D.global_index("iid_val", i)
        s = FB.generate(idx, "iid_val")
        hl = FB.hop_labels(idx, "iid_val")
        assert len(hl) == s.depth, f"depth {s.depth} gave {len(hl)} hop labels"
        assert hl[-1] == s.label


def test_hop_labels_respect_a_restricted_depth() -> None:
    """The lens fixes one depth per call, so the labels must follow."""
    cfg = D.TaskConfig(depths=(4,), breadths=tuple(FB.SPLIT_RANGES["iid_val"][1]))
    for i in range(30):
        idx = D.global_index("iid_val", i)
        assert len(FB.hop_labels(idx, "iid_val", cfg)) == 4
        assert FB.generate(idx, "iid_val", cfg).label == FB.hop_labels(idx, "iid_val", cfg)[-1]


def test_the_reached_threshold_sits_well_above_the_task_floor() -> None:
    """A hop counted as reached at the guessing floor would make every
    model look like it had walked the whole chain immediately."""
    assert CL.REACHED > FB.effective_chance() * 2
