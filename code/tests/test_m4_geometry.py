"""The geometry has to be right on trajectories whose answer is known.

Every quantity M4 reports is a closed-form function of two consecutive
states, so it can be checked against synthetic trajectories rather than
against a model. That is the point of these tests: a geometry instrument
validated only on real states can be wrong in a way that looks plausible,
because nobody knows what the real answer was.

The trajectories below are chosen so each isolates one failure. A state
that does not move, a state that only grows, and a state that only turns.
The third is the one the paper's claim rests on: if a pure rotation at
constant norm reports norm growth, the rays-not-fixed-points reading is an
artifact of the instrument.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from loopvision.instruments import m4_geometry as M4

#: Half the registered gate, so the gated tests move with the constant
#: rather than hardcoding a value that would silently stop testing the
#: boundary if the gate were ever changed.
RATIO_HALF_GATE = M4.RATIO_MIN_GROWTH / 2


def _states(vectors: list[list[float]]) -> list[torch.Tensor]:
    """One item, one position, so [1, 1, d] per state."""
    return [torch.tensor([[v]], dtype=torch.float32) for v in vectors]


def test_a_state_that_does_not_move() -> None:
    """The degenerate case every other number is measured against."""
    g = M4.geometry(_states([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]))
    assert np.allclose(g["norm"][0], [5.0, 5.0])
    assert np.allclose(g["step_norm"][0], [0.0, 0.0], atol=1e-6)
    assert np.allclose(g["cos_state"][0], [1.0, 1.0], atol=1e-6)
    assert np.allclose(g["norm_growth"][0], [0.0, 0.0], atol=1e-6)
    assert np.allclose(g["unit_step_norm"][0], [0.0, 0.0], atol=1e-6)


def test_pure_growth_along_a_fixed_direction() -> None:
    """Norm doubles each step, direction never changes.

    Mutation checked: normalising before taking the norm, or computing
    unit_step_norm on the raw vectors, both break this.
    """
    g = M4.geometry(_states([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]]))
    assert np.allclose(g["norm"][0], [2.0, 4.0])
    assert np.allclose(g["norm_growth"][0], [1.0, 1.0])
    assert np.allclose(g["cos_state"][0], [1.0, 1.0], atol=1e-6)
    # The direction never moves, so the direction-normalised step is zero
    # even though the raw step is large. This is the separation the whole
    # instrument exists to make.
    assert np.allclose(g["step_norm"][0], [1.0, 2.0])
    assert np.allclose(g["unit_step_norm"][0], [0.0, 0.0], atol=1e-6)


def test_pure_rotation_at_constant_norm() -> None:
    """Turns by 60 degrees each step, norm fixed.

    If this reports any norm growth the rays-not-fixed-points reading is an
    artifact of the instrument rather than a property of the model.
    """
    a = math.radians(60)
    v = [[1.0, 0.0], [math.cos(a), math.sin(a)], [math.cos(2 * a), math.sin(2 * a)]]
    g = M4.geometry(_states(v))
    assert np.allclose(g["norm"][0], [1.0, 1.0], atol=1e-6)
    assert np.allclose(g["norm_growth"][0], [0.0, 0.0], atol=1e-6)
    assert np.allclose(g["cos_state"][0], [math.cos(a), math.cos(a)], atol=1e-6)
    # Chord length of a 60 degree turn on the unit circle is exactly 1.
    assert np.allclose(g["unit_step_norm"][0], [1.0, 1.0], atol=1e-6)


def test_cos_step_compares_consecutive_deltas() -> None:
    """Undefined at i=1 because there is no previous delta, then exact."""
    # Deltas are (1,0) then (0,1): perpendicular, so cos is 0.
    g = M4.geometry(_states([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))
    assert np.isnan(g["cos_step"][0, 0])
    assert np.allclose(g["cos_step"][0, 1], 0.0, atol=1e-6)


def test_the_ratio_is_gated_where_growth_is_tiny() -> None:
    """The registered trap, in the shape that caused it.

    A near-zero denominator must not produce a number, because that number
    is enormous and moves a group mean. Growth here is half the gate.

    Mutation checked: removing the RATIO_MIN_GROWTH condition makes this
    fail with a finite ratio roughly two hundred times the gated case.
    """
    a = math.radians(10)
    grow = 1.0 + RATIO_HALF_GATE
    v = [[1.0, 0.0], [grow * math.cos(a), grow * math.sin(a)]]
    g = M4.geometry(_states(v))
    assert np.isclose(g["norm_growth"][0, 0], RATIO_HALF_GATE, atol=1e-5)
    assert np.isnan(g["ratio"][0, 0]), "a denominator below the gate produced a number"


def test_the_ratio_is_computed_where_growth_is_real() -> None:
    """The other half of the gate. Gating everything would be just as wrong
    as gating nothing, and would be much harder to notice."""
    a = math.radians(10)
    grow = 1.5
    v = [[1.0, 0.0], [grow * math.cos(a), grow * math.sin(a)]]
    g = M4.geometry(_states(v))
    expected = (1.0 - math.cos(a)) / (grow - 1.0)
    assert np.isclose(g["ratio"][0, 0], expected, atol=1e-5)


def test_the_gate_is_a_constant_not_a_magic_number() -> None:
    """The gate has to be readable and changeable without editing logic,
    because gate_sensitivity has to report the headline both ways."""
    assert M4.RATIO_MIN_GROWTH > 0
    assert "ratio" in M4.METRICS
    assert "norm_growth" in M4.METRICS, "the ungated component must be stored too"


def test_geometry_refuses_a_trajectory_with_no_geometry() -> None:
    with pytest.raises(ValueError):
        M4.geometry(_states([[1.0, 0.0]]))


def test_shape_is_items_by_k() -> None:
    """Column j is iteration j+1. Nothing is defined at i=0, and an
    off-by-one here would misalign every quantity against the loop index."""
    states = [torch.randn(4, 3, 8) for _ in range(6)]
    g = M4.geometry(states)
    for m in M4.METRICS:
        assert g[m].shape == (4, 5), f"{m} has shape {g[m].shape}"


def _fake_run(tmp_path, arch: str, k_train: int = 4, **extra):
    """A run directory complete enough for m4_geometry.run to open.

    Built rather than mocked. The recorded k is the x axis F5 draws on, and
    a mocked capture would let the recording agree with itself while the
    real forward pass returned a trajectory of a different length.
    """
    import yaml

    from loopvision.train.cli import build_model

    d = tmp_path / f"{arch}_run"
    (d / "checkpoints").mkdir(parents=True)
    cfg = {
        "arch": arch, "family": "A", "depths": [2], "breadths": [2],
        "d_model": 64, "batch_size": 4, "k_train": k_train, "k_eval": k_train,
        "prelude_blocks": 1, "core_blocks": 1, "coda_blocks": 1,
        "seed": 0, "master_seed": 0, "untied_copies": k_train, "inject": True,
        "num_workers": 0,
        "state_init": "zeros", "conditioning": "none",
    }
    cfg.update(extra)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    model = build_model(cfg, torch.device("cpu"))
    torch.save({"model": model.state_dict()}, d / "checkpoints" / "step_1.pt")
    return d, cfg


@pytest.mark.parametrize("arch,requested,expected", [
    ("feedforward", 32, 4),
    ("untied", 32, 4),
    ("looped", 6, 6),
])
def test_run_records_the_trajectory_it_got_not_the_one_it_asked_for(
        tmp_path, arch, requested, expected) -> None:
    """F5 draws k on one axis across architectures. Recording the request
    would put a feedforward stack measured at depth 4 and a looped model
    measured at depth 32 under the same column name."""
    from loopvision.instruments.m4_geometry import run

    d, _ = _fake_run(tmp_path, arch)
    result = run(d, items=4, k=requested, device=torch.device("cpu"))
    assert result["k"] == expected
    assert result["k_requested"] == requested
    assert result["metrics"]["norm"].shape[1] == expected


def test_run_records_what_state_zero_actually_is(tmp_path) -> None:
    """The one thing about this comparison that cannot be fixed in code: a
    looped model initialised with randn starts from noise and a feedforward
    stack starts from a real activation, so their first columns are not the
    same measurement."""
    from loopvision.instruments.m4_geometry import run

    d, _ = _fake_run(tmp_path, "feedforward")
    assert run(d, 4, 8, torch.device("cpu"))["state0"] == "post_prelude_activation"

    d, _ = _fake_run(tmp_path, "looped", state_init="randn")
    assert run(d, 4, 4, torch.device("cpu"))["state0"] == "randn"


def test_the_recorded_k_follows_the_states_not_the_arithmetic(
        tmp_path, monkeypatch) -> None:
    """`usable_k` and the model agree today, so the recorded k and the
    requested k coincide for every architecture that exists and no real run
    can tell the two apart. The invariant is still "record what came back",
    and this is the only way to hold it: a stub whose trajectory is shorter
    than its k implies, standing in for a future architecture rather than
    for any code under test. Without it that line is unfalsifiable.
    """
    import loopvision.instruments.m4_geometry as M4

    d, _ = _fake_run(tmp_path, "looped")
    real = M4.build_model

    class ShortTrajectory(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.cfg = inner.cfg

        def forward(self, image, query, k, return_states=False, **kw):
            logits, states = self.inner(image, query, k=k, return_states=True)
            return (logits, states[:3]) if return_states else logits

        def load_state_dict(self, *a, **kw):
            return self.inner.load_state_dict(*a, **kw)

        def eval(self):
            self.inner.eval()
            return self

    monkeypatch.setattr(M4, "build_model",
                        lambda cfg, dev: ShortTrajectory(real(cfg, dev)))
    result = M4.run(d, items=4, k=4, device=torch.device("cpu"))
    assert result["k"] == 2, "recorded k did not follow the states it got"
    assert result["k_requested"] == 4
    assert result["metrics"]["norm"].shape[1] == 2
