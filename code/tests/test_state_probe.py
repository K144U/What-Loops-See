"""The probe must find information that is there and miss what is not.

Its whole job is to distinguish "the model never computed S3" from "the
model computed S3 and the output head cannot read it". A probe that
reports high accuracy on noise would manufacture the second answer, and
one that reports chance on a clean signal would manufacture the first.
Both are checked here against synthetic features where the truth is known.
"""

from __future__ import annotations

import numpy as np
import pytest

from loopvision.analysis import state_probe as SP
from loopvision.data import dataset as D
from loopvision.data import family_a as FA
from loopvision.data import groups as G

CHANCE = 1.0 / G.S3_ORDER


def _split(X, y, frac=0.8):
    cut = int(len(y) * frac)
    return X[:cut], y[:cut], X[cut:], y[cut:]


def test_the_probe_reads_a_signal_that_is_linearly_present() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, G.S3_ORDER, size=3000)
    X = np.eye(G.S3_ORDER, dtype=np.float32)[y] * 4.0
    X = X + rng.normal(0, 0.3, size=X.shape).astype(np.float32)
    acc = SP.fit_probe(*_split(X, y))
    assert acc > 0.95, f"a clean linear signal read at only {acc:.3f}"


def test_the_probe_reports_chance_on_noise() -> None:
    """If this could be beaten, every 'the information is there' claim the
    probe makes would be an artefact."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, G.S3_ORDER, size=3000)
    X = rng.normal(0, 1, size=(3000, 64)).astype(np.float32)
    acc = SP.fit_probe(*_split(X, y))
    assert acc < CHANCE + 0.06, f"probe found {acc:.3f} of structure in noise"


def test_the_probe_is_not_fooled_by_scale_alone() -> None:
    """Residual-stream norms grow across core passes. If the probe tracked
    scale it would report rising information where none was added."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, G.S3_ORDER, size=3000)
    X = rng.normal(0, 1, size=(3000, 64)).astype(np.float32) * 500.0
    acc = SP.fit_probe(*_split(X, y))
    assert acc < CHANCE + 0.06, f"scale alone gave {acc:.3f}"


def test_targets_decompose_the_actual_computation() -> None:
    """The five targets must be the real quantities, not plausible ones.

    composite has to equal the sample's own label in the S3 factor, or the
    probe would be scoring against a question the model was never asked.
    """
    cfg = D.TaskConfig(factor="s3", depths=(2,))
    for i in range(80):
        idx = D.global_index("iid_val", i)
        s = FA.generate(idx, "iid_val", cfg)
        t = SP.targets_for(s.program)
        assert t["composite"] == SP.s3_index(s.label), (
            "the probe's composite target disagrees with the sample's label"
        )
        for name in SP.TARGETS:
            assert 0 <= t[name] < G.S3_ORDER


def test_partial_is_between_initial_and_composite() -> None:
    """`partial` is the halfway state, so applying op2 to it must give the
    answer. If that failed the middle column would be meaningless."""
    cfg = D.TaskConfig(factor="s3", depths=(2,))
    for i in range(60):
        idx = D.global_index("iid_val", i)
        s = FA.generate(idx, "iid_val", cfg)
        initial, strips, queried, _, _ = FA.parse_program(s.program)
        ops = strips[queried]
        partial = G.multiply(ops[0], initial[queried])
        assert SP.s3_index(partial) == SP.targets_for(s.program)["partial"]
        assert G.multiply(ops[1], partial) == s.label


def test_a_target_that_is_constant_would_be_caught() -> None:
    """A degenerate target reads as perfect probe accuracy and would be
    misreported as the model knowing everything."""
    cfg = D.TaskConfig(factor="s3", depths=(2,))
    seen = {t: set() for t in SP.TARGETS}
    for i in range(400):
        idx = D.global_index("iid_val", i)
        t = SP.targets_for(FA.generate(idx, "iid_val", cfg).program)
        for name, v in t.items():
            seen[name].add(v)
    for name, vals in seen.items():
        assert len(vals) == G.S3_ORDER, (
            f"target {name} takes only {len(vals)} values, so probe accuracy "
            f"on it would not mean what the table says it means"
        )


def test_a_faint_signal_beside_a_loud_dimension_is_still_found() -> None:
    """Residual-stream dimensions differ in scale by orders of magnitude,
    and the S3 signal has no reason to live in the loudest one.

    Without standardisation a fixed learning rate is dominated by whichever
    dimension happens to be large, and a faint but perfectly clean signal
    reads as chance. That would be reported as "the model never computed
    S3", which is exactly the wrong conclusion.
    """
    rng = np.random.default_rng(3)
    n = 4000
    y = rng.integers(0, G.S3_ORDER, size=n)
    faint = np.eye(G.S3_ORDER, dtype=np.float32)[y] * 0.01
    loud = rng.normal(0, 1, size=(n, 8)).astype(np.float32) * 800.0
    X = np.concatenate([faint, loud], axis=1)
    acc = SP.fit_probe(*_split(X, y))
    assert acc > 0.90, (
        f"a clean signal at 1/80000 the scale of its neighbours read at "
        f"{acc:.3f}, so the probe is measuring loudness rather than content"
    )
