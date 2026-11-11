"""The registered H4 metric, and the three gaps in the sentence that fixes it.

Preregistration Section 5 says to cluster final states and count items whose
cluster's modal answer is not the correct answer. It does not say whether
"modal answer" means the modal true label or the modal prediction, it says
"across seeds" where seeds do not share a basis, and it does not give a
cluster count. The instrument resolves each on a stated principle and these
tests pin the resolutions, because a metric that quietly changes shape is
worse than one that was never written.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from loopvision.instruments import m6_attractor as M


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _blobs() -> np.ndarray:
    """Three directions, far apart on the unit sphere."""
    rng = np.random.default_rng(0)
    a = np.array([1.0, 0.0, 0.0]) + rng.normal(0, 0.01, (20, 3))
    b = np.array([0.0, 1.0, 0.0]) + rng.normal(0, 0.01, (20, 3))
    c = np.array([0.0, 0.0, 1.0]) + rng.normal(0, 0.01, (20, 3))
    return np.concatenate([a, b, c], axis=0)


def test_separable_directions_come_back_as_separate_clusters() -> None:
    assign = M.kmeans(_blobs(), k=3)
    groups = [set(assign[0:20]), set(assign[20:40]), set(assign[40:60])]
    assert all(len(g) == 1 for g in groups), "each blob must be one cluster"
    assert len({g.pop() for g in groups}) == 3, "and the three must differ"


def test_clustering_is_by_direction_and_not_by_magnitude() -> None:
    """RMSNorm before the head makes the state norm inert, so a cluster that
    splits on magnitude is splitting on something the model cannot see. This
    is the F5 trap from STATE.md, arriving in a second instrument.

    Stated as the defining invariant rather than as one example: rescaling
    every row by an arbitrary positive factor changes no assignment. A weaker
    version of this test passed against an unnormalised implementation,
    because with unit centroids a large row's distances are all dominated by
    its own magnitude and direction wins anyway. The invariant does not care
    how the property is achieved, only that it holds."""
    x = _blobs()
    rng = np.random.default_rng(7)
    scale = rng.uniform(0.02, 50.0, (x.shape[0], 1))
    assert np.array_equal(M.kmeans(x, k=3), M.kmeans(x * scale, k=3))


def test_the_same_states_cluster_the_same_way_twice() -> None:
    """L-028: an instrument nobody ran twice had its variance found by
    accident months later. k-means init is seeded so this cannot recur."""
    x = _blobs()
    assert np.array_equal(M.kmeans(x, k=3), M.kmeans(x, k=3))


def test_fewer_items_than_clusters_refuses_rather_than_returning_empty_basins() -> None:
    with pytest.raises(ValueError, match="cannot form"):
        M.kmeans(np.eye(3), k=8)


# ---------------------------------------------------------------------------
# The modal answer
# ---------------------------------------------------------------------------

def test_modal_takes_the_most_common() -> None:
    assert M.modal(np.array([4, 4, 7])) == 4


def test_modal_breaks_ties_deterministically() -> None:
    """An evenly split cluster must not make the rate depend on dict order."""
    assert M.modal(np.array([9, 9, 2, 2])) == 2
    assert M.modal(np.array([2, 2, 9, 9])) == 2


# ---------------------------------------------------------------------------
# The rate, and the ambiguity in the registered sentence
# ---------------------------------------------------------------------------

def test_a_pure_cluster_has_no_wrong_landings() -> None:
    assign = np.array([0, 0, 0, 1, 1, 1])
    labels = np.array([3, 3, 3, 5, 5, 5])
    got = M.wrong_attractor(assign, labels, labels)
    assert got["rate_by_label"] == 0.0
    assert got["rate_by_prediction"] == 0.0


def test_an_item_in_a_basin_of_another_answer_is_a_wrong_landing() -> None:
    """One item of answer 5 sitting among four of answer 3. Its cluster's
    modal answer is 3, so it landed in the wrong attractor."""
    assign = np.array([0, 0, 0, 0, 0])
    labels = np.array([3, 3, 3, 3, 5])
    got = M.wrong_attractor(assign, labels, labels)
    assert got["rate_by_label"] == pytest.approx(1 / 5)


def test_the_two_readings_can_disagree_and_both_are_reported() -> None:
    """The whole reason both are computed. Here the cluster is mostly answer
    3 by truth, while the model predicts 5 for every member. By label the odd
    item is the wrong landing; by prediction all four of the 3s are."""
    assign = np.array([0, 0, 0, 0, 0])
    labels = np.array([3, 3, 3, 3, 5])
    preds = np.array([5, 5, 5, 5, 5])
    got = M.wrong_attractor(assign, labels, preds)
    assert got["rate_by_label"] == pytest.approx(1 / 5)
    assert got["rate_by_prediction"] == pytest.approx(4 / 5)
    assert got["readings_agree"] is False


def test_readings_agree_is_true_when_they_do() -> None:
    """The control. Without it the flag could be hardwired false and the
    disagreement test above would still pass."""
    assign = np.array([0, 0, 0, 1, 1, 1])
    labels = np.array([3, 3, 3, 5, 5, 5])
    got = M.wrong_attractor(assign, labels, labels)
    assert got["readings_agree"] is True


def test_every_cluster_is_described_so_a_rate_can_be_audited() -> None:
    """A single rate with no per-cluster detail cannot be checked by anyone,
    and this metric has too many free choices to be taken on trust."""
    assign = np.array([0, 0, 1, 1])
    labels = np.array([3, 3, 5, 5])
    got = M.wrong_attractor(assign, labels, labels)
    assert [c["cluster"] for c in got["clusters"]] == [0, 1]
    assert all(c["size"] == 2 for c in got["clusters"])
    assert all(c["purity"] == 1.0 for c in got["clusters"])


def test_purity_falls_when_a_basin_is_mixed() -> None:
    assign = np.array([0, 0, 0, 0])
    labels = np.array([3, 3, 3, 5])
    got = M.wrong_attractor(assign, labels, labels)
    assert got["clusters"][0]["purity"] == pytest.approx(0.75)


def test_a_clusters_centre_is_not_dragged_by_one_large_member() -> None:
    """Assignment is scale invariant however this is implemented, because a
    point's own norm cancels out of the argmin against unit centroids. The
    centre is the one place normalising the input changes an answer, so it is
    tested where it bites rather than through kmeans, where the mutation was
    equivalent and slipped through twice."""
    members = np.array([[1.0, 0.0], [0.0, 100.0]])
    got = M.spherical_mean(members)
    # Both directions weight equally, so the centre bisects them.
    assert got[0] == pytest.approx(got[1], abs=1e-9)
    # A raw mean would land almost exactly on the large member instead.
    raw = members.mean(axis=0)
    raw = raw / np.linalg.norm(raw)
    assert raw[1] > 0.99


def test_the_centre_is_a_direction_of_unit_length() -> None:
    got = M.spherical_mean(np.array([[3.0, 4.0], [0.0, 7.0]]))
    assert np.linalg.norm(got) == pytest.approx(1.0)


def test_kmeans_updates_centres_with_the_spherical_mean() -> None:
    """An equivalent mutation on separable data, and a live bug on borderline
    data. Swapping the spherical mean for a raw mean inside kmeans changes no
    assignment on well parted blobs, because the centre has to move a long way
    before a point switches basin, so no behavioural test on such data can
    fail. It is pinned in the source instead, the same way the GPU slot index
    is, rather than left to a comment."""
    src = (Path(M.__file__)).read_text(encoding="utf-8")
    body = src.split("def kmeans(")[1].split("def modal(")[0]
    assert "spherical_mean(members)" in body
    assert "members.mean(" not in body, "a raw mean here is magnitude deciding the basin"


def test_the_instrument_clusters_the_readout_and_not_the_raw_stream() -> None:
    """Settled by a positive control, so it is pinned rather than trusted.
    On the raw residual stream a run at 1.0000 and a run at 0.60 both score
    about 0.83, because the stream is dominated by the image and the answer
    is a small part of it. Through model.readout they read 0.19 and 0.66. A
    metric that cannot tell those two apart is not measuring attractors, and
    reverting this line would restore that silently."""
    src = Path(M.__file__).read_text(encoding="utf-8")
    body = src.split("def run(")[1]
    assert "model.readout(states[-1])" in body
    assert ".reshape(states[-1].shape[0], -1)" not in body, "raw flatten is back"


def test_both_architectures_expose_the_same_readout() -> None:
    """M6 treats looped and feedforward runs alike, so the name has to mean
    the same thing on both. It also has to be the path the head really uses:
    a copy of the coda and pool inside the instrument would drift from the
    model, which is the reason decode was exposed for M3 in the first place."""
    from loopvision.model import loopvit

    for cls in (loopvit.LoopViT, loopvit.FeedforwardBaseline):
        assert hasattr(cls, "readout"), f"{cls.__name__} has no readout"
    head_path = Path(loopvit.__file__).read_text(encoding="utf-8")
    assert "self.head(self.readout(" in head_path, (
        "decode must go through readout, or the two can disagree")


def test_the_sweep_selector_takes_the_jitter_axis_and_not_the_width_axis() -> None:
    """The first run of --all swept in nine famB_curve111_d256, d512 and d768
    runs, which vary WIDTH. That is a different axis and no part of Section 5,
    and they sat in the output looking like sweep members."""
    for name in ("famB_curve111_s0", "famB_curve111_jit000_s4",
                 "famB_curve111_jit025_s0", "famB_curve111_jit100_s3"):
        assert M.H4_ARM.fullmatch(name), f"{name} is a registered arm"
    for name in ("famB_curve111_d256_s1", "famB_curve111_d512_s0",
                 "famB_curve111_d768_s2"):
        assert not M.H4_ARM.fullmatch(name), f"{name} varies width, not jitter"


def test_the_untagged_runs_are_the_half_sigma_arm() -> None:
    """0.5 is the default jitter in cli.py, so the base famB_curve111 runs
    ARE that arm and carry no jit tag. A pattern matching only jit... would
    drop a quarter of the registered sweep while looking complete."""
    assert M.H4_ARM.fullmatch("famB_curve111_s7")
