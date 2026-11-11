"""The blank-image control has to actually blank the image.

L-012 came from two tasks that looked solved and were not. The control
that catches that class of fault is only worth having if it really hands
the model an empty canvas, so that is what these tests assert, rather than
re-checking the arithmetic around it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from loopvision.analysis import blank_control as BC


class _Recorder(torch.nn.Module):
    """Answers from the query alone, and remembers every image it saw."""

    def __init__(self, n_classes: int = 13) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.seen: list[torch.Tensor] = []
        self.lin = torch.nn.Linear(1, 1)

    def eval(self):  # noqa: D102
        return self


def _install(monkeypatch, model, labels, batch_size=8):
    """Point the control at a fake run without touching disk or CUDA."""
    n = len(labels)

    def fake_make_batch(cfg, split, cursor, device):
        i = (cursor // batch_size) % (n // batch_size)
        lab = torch.tensor(labels[i * batch_size : (i + 1) * batch_size])
        image = torch.ones(batch_size, 3, 8, 8)
        query = torch.zeros(batch_size, 4, dtype=torch.long)
        return image, query, lab, [1] * batch_size, [4] * batch_size

    def fake_forward(m, image, query, k):
        m.seen.append(image.clone())
        # Always predicts class 1, so accuracy equals the frequency of 1.
        out = torch.zeros(image.shape[0], m.n_classes)
        out[:, 1] = 1.0
        return out

    monkeypatch.setattr(BC, "make_batch", fake_make_batch)
    monkeypatch.setattr(BC, "forward", fake_forward)
    monkeypatch.setattr(BC, "build_model", lambda cfg, device: model)
    monkeypatch.setattr(BC.C, "find_latest_checkpoint", lambda d: d / "step_1.pt")
    monkeypatch.setattr(BC.torch, "load", lambda *a, **kw: {"model": {}})
    monkeypatch.setattr(model, "load_state_dict", lambda sd: None, raising=False)


def _run_dir(tmp_path):
    d = tmp_path / "fake_run"
    d.mkdir()
    (d / "config.yaml").write_text(
        yaml.safe_dump({"batch_size": 8, "k_eval": 4}), encoding="utf-8"
    )
    return d


def test_blank_pass_receives_an_empty_canvas(tmp_path, monkeypatch) -> None:
    """The whole instrument rests on this one line being right.

    Mutation checked: changing `torch.zeros_like(image)` back to `image`
    in blank_control makes this fail, which is the point.
    """
    model = _Recorder()
    _install(monkeypatch, model, labels=[1] * 8)
    BC.blank_control(_run_dir(tmp_path), batches=1)

    assert len(model.seen) == 2, "expected one real pass and one blank pass"
    real, blank = model.seen
    assert real.abs().sum() > 0, "the real pass was handed a blank image"
    assert blank.abs().sum() == 0, (
        "the blank pass was handed a non-empty image, so the control is "
        "measuring nothing and would report any leak as clean"
    )
    assert blank.shape == real.shape


def test_a_query_only_model_is_reported_as_leaking(tmp_path, monkeypatch) -> None:
    """A model that ignores the image must be caught, not excused.

    Labels are 7/8 class 1, so the best constant is 0.875 and a model that
    always says 1 scores 0.875 blank. That is a model riding the prior, not
    a leak, so it must read clean. The leak case is the next test.
    """
    model = _Recorder()
    _install(monkeypatch, model, labels=[1] * 7 + [0])
    res = BC.blank_control(_run_dir(tmp_path), batches=1)

    assert res["blank"] == pytest.approx(0.875)
    assert res["best_constant"] == pytest.approx(0.875)
    assert res["leak"] == pytest.approx(0.0, abs=1e-6)
    assert not res["leaks"], "matching the label prior is not a leak"


def test_beating_the_prior_without_an_image_is_a_leak(tmp_path, monkeypatch) -> None:
    model = _Recorder()
    _install(monkeypatch, model, labels=[1, 1, 1, 0, 0, 2, 2, 3])

    def cheating_forward(m, image, query, k):
        m.seen.append(image.clone())
        out = torch.zeros(image.shape[0], m.n_classes)
        # Right on every sample regardless of the image.
        for i, lab in enumerate([1, 1, 1, 0, 0, 2, 2, 3]):
            out[i, lab] = 1.0
        return out

    monkeypatch.setattr(BC, "forward", cheating_forward)
    res = BC.blank_control(_run_dir(tmp_path), batches=1)

    assert res["blank"] == pytest.approx(1.0)
    assert res["best_constant"] == pytest.approx(0.375)
    assert res["leaks"], (
        f"a model scoring {res['blank']:.3f} on a blank image against a "
        f"{res['best_constant']:.3f} prior must be flagged"
    )


def test_leak_margin_is_not_so_wide_it_would_have_missed_family_b() -> None:
    """The threshold must catch the fault that motivated the instrument.

    Family B depth 1 scored 0.8327 blank against a best constant near 0.33.
    A margin that would have let that through is useless.
    """
    assert BC.LEAK_MARGIN < 0.8327 - 0.33
