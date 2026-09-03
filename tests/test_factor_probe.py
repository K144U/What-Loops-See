"""The factor probe has to separate two readings the parities cannot.

Accuracy 1/6 and loss ln(6) are produced equally by a model that knows the
abelianisation of D4 x S3 and by one that knows the whole D4 factor. The
first needs no operator order, the second cannot work without it, so
mislabelling one as the other would invert what the result says about
sequential computation. These tests build both models exactly and check
the probe tells them apart.
"""

from __future__ import annotations

import numpy as np

from loopvision.analysis import parity_probe as P
from loopvision.data import groups as G

RNG = np.random.default_rng(7)


def _knows_d4_only(labels):
    """Right D4 part, S3 part drawn uniformly at random."""
    out = []
    for l in labels:
        d4 = G.d4_of(int(l))
        s3 = G.s3_of(int(RNG.integers(0, G.GROUP_ORDER)))
        out.append(G.element_index(d4, s3))
    return np.array(out)


def _knows_abelianisation_only(labels):
    """Right on all three parities, otherwise uniform inside the fibre."""
    fibre = {}
    for g in range(G.GROUP_ORDER):
        key = tuple(f(g) for f in P.PARITIES.values())
        fibre.setdefault(key, []).append(g)
    out = []
    for l in labels:
        key = tuple(f(int(l)) for f in P.PARITIES.values())
        cands = fibre[key]
        out.append(cands[int(RNG.integers(0, len(cands)))])
    return np.array(out)


def _labels(n=24000):
    return RNG.integers(0, G.GROUP_ORDER, size=n)


def test_both_models_score_identically_on_accuracy_and_loss() -> None:
    """The premise: accuracy alone cannot separate them, so it must not be
    what the interpretation rests on."""
    labels = _labels()
    a = P.analyse(_knows_d4_only(labels), labels)["full_accuracy"]
    b = P.analyse(_knows_abelianisation_only(labels), labels)["full_accuracy"]
    assert abs(a - 1 / 6) < 0.02, a
    assert abs(b - 1 / 6) < 0.02, b
    assert abs(a - b) < 0.02, "if these differed the probe would be unnecessary"


def test_a_d4_only_model_is_read_as_sequential() -> None:
    labels = _labels()
    res = P.analyse(_knows_d4_only(labels), labels)
    f = res["factors"]
    assert f["d4_accuracy"] > 0.95, f["d4_accuracy"]
    assert abs(f["s3_accuracy"] - 1 / 6) < 0.03, f["s3_accuracy"]
    assert res["parities"]["sign (S3 permutation)"]["accuracy"] < 0.60
    assert "REQUIRES sequential composition" in res["reading"]


def test_an_abelian_model_is_not_read_as_sequential() -> None:
    labels = _labels()
    res = P.analyse(_knows_abelianisation_only(labels), labels)
    f = res["factors"]
    assert res["parities"]["sign (S3 permutation)"]["accuracy"] > 0.95, (
        "a model that knows the abelianisation knows the S3 sign by definition"
    )
    assert f["d4_accuracy"] < 0.95, f["d4_accuracy"]
    assert "abelianisation" in res["reading"]
    assert "REQUIRES sequential" not in res["reading"]


def test_the_two_readings_are_never_returned_together() -> None:
    labels = _labels()
    for maker in (_knows_d4_only, _knows_abelianisation_only):
        r = P.analyse(maker(labels), labels)["reading"]
        assert ("abelianisation" in r) != ("REQUIRES sequential" in r)


def test_a_solved_model_is_called_solved_not_sequential_on_one_factor() -> None:
    labels = _labels()
    res = P.analyse(labels.copy(), labels)
    assert res["full_accuracy"] == 1.0
    assert "task is solved" in res["reading"]


def _knows_d4_and_the_s3_sign_but_never_the_element(labels):
    """D4 exact, and an S3 element of the right sign that is never the right one.

    This is the case the `sign` guard in `interpret` exists for. The factor
    accuracies alone look like "D4 solved, S3 unknown", because the S3 part
    is wrong every single time. It is not: the sign is known perfectly, so
    S3 is partly known, and calling it unknown would overstate how cleanly
    the two factors separate.
    """
    all_s3 = sorted({G.s3_of(g) for g in range(G.GROUP_ORDER)})
    out = []
    for l in labels:
        d4, s3 = G.d4_of(int(l)), G.s3_of(int(l))
        sign = P.s3_sign(s3)
        same_sign_wrong = [t for t in all_s3 if P.s3_sign(t) == sign and t != s3]
        pick = same_sign_wrong[int(RNG.integers(0, len(same_sign_wrong)))]
        out.append(G.element_index(d4, pick))
    return np.array(out)


def test_a_known_sign_is_not_reported_as_an_unknown_s3_factor() -> None:
    """Mutation check target: deleting `sign < 0.60` from interpret()
    makes this model read as cleanly sequential, which it is not."""
    labels = _labels()
    res = P.analyse(_knows_d4_and_the_s3_sign_but_never_the_element(labels), labels)
    f = res["factors"]

    assert f["d4_accuracy"] > 0.95, f["d4_accuracy"]
    assert f["s3_accuracy"] == 0.0, "by construction the S3 element is never right"
    assert res["parities"]["sign (S3 permutation)"]["accuracy"] > 0.95, (
        "by construction the sign is always right"
    )
    assert "REQUIRES sequential composition" not in res["reading"], (
        "S3 is partly known, so this must not be read as S3 being unknown"
    )
    assert "do not label the mechanism" in res["reading"]
