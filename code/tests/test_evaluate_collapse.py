"""Accuracy alone cannot tell a guessing model from a collapsed one.

L-024. Ten runs failed by emitting a single output class for every item,
and three of them, across two different parameter counts, reported an
identical accuracy to sixteen digits because they had collapsed onto the
same class. The reported figure was the frequency of that class in the
eval set. It looked exactly like a failure is supposed to look, so nobody
looked further.

The test that matters is the first one below: two models with the SAME
accuracy, one collapsed and one not. If `evaluate` only returned the hit
rate there would be nothing to assert, which is the whole point.
"""

from __future__ import annotations

import torch

from loopvision.train import cli

BATCH = 8
N_CLASSES = 4

#: Four classes, four items each, in a fixed order.
LABELS = [0, 1, 2, 3] * 4

#: Right on exactly the four items whose label is 0, and wrong everywhere
#: else, but spread across every class. Same hit count as CONSTANT below.
SCATTERED = [0, 2, 3, 1] * 4

#: Right on exactly the same four items, by never saying anything else.
CONSTANT = [0] * 16


class _Fixed(torch.nn.Module):
    """Emits a predetermined prediction per item, as one-hot logits."""

    def __init__(self, preds: list[int]) -> None:
        super().__init__()
        self.preds = preds


def _install(monkeypatch, preds: list[int]) -> None:
    def fake_make_batch(cfg, split, cursor, device):
        i = cursor // BATCH
        lo, hi = i * BATCH, (i + 1) * BATCH
        label = torch.tensor(LABELS[lo:hi])
        image = torch.zeros(BATCH, 3, 8, 8)
        query = torch.zeros(BATCH, 4, dtype=torch.long)
        return image, query, label, [2] * BATCH, [1] * BATCH

    def fake_forward(model, image, query, k, *args, **kwargs):
        # The cursor is not passed to forward, so the batch index is
        # recovered from how many batches have already been asked for.
        i = fake_forward.calls
        fake_forward.calls += 1
        out = torch.zeros(BATCH, N_CLASSES)
        for j, p in enumerate(preds[i * BATCH : (i + 1) * BATCH]):
            out[j, p] = 1.0
        return out

    fake_forward.calls = 0
    monkeypatch.setattr(cli, "make_batch", fake_make_batch)
    monkeypatch.setattr(cli, "forward", fake_forward)


def _evaluate(monkeypatch, preds: list[int]) -> dict:
    _install(monkeypatch, preds)
    return cli.evaluate(
        _Fixed(preds),
        {"batch_size": BATCH, "k_eval": 1},
        "iid_val",
        torch.device("cpu"),
        batches=2,
    )


def test_accuracy_cannot_separate_them_and_n_distinct_can(monkeypatch) -> None:
    """The reason this module exists, stated as one assertion pair.

    Mutation checked: making `pred_counts` count labels rather than
    predictions makes both models report 4 distinct and this fails.
    """
    collapsed = _evaluate(monkeypatch, CONSTANT)
    scattered = _evaluate(monkeypatch, SCATTERED)

    assert collapsed["accuracy"] == scattered["accuracy"] == 0.25
    assert collapsed["n_distinct"] == 1
    assert scattered["n_distinct"] == N_CLASSES


def test_a_collapsed_model_reports_its_one_class(monkeypatch) -> None:
    """The class id is recorded, because comparing it across runs is what
    showed the constant tracks architecture rather than seed."""
    r = _evaluate(monkeypatch, CONSTANT)
    assert r["modal_pred"] == 0
    assert r["modal_share"] == 1.0


def test_modal_share_catches_near_collapse(monkeypatch) -> None:
    """A model one item short of collapse is still degenerate.

    n_distinct is 2 here and would not trip an equality check, which is why
    the share is recorded alongside it.
    """
    preds = [0] * 15 + [1]
    r = _evaluate(monkeypatch, preds)
    assert r["n_distinct"] == 2
    assert r["modal_share"] == 15 / 16


def test_a_model_that_answers_uses_every_class(monkeypatch) -> None:
    """The positive control. A solved model must not look collapsed."""
    r = _evaluate(monkeypatch, LABELS)
    assert r["accuracy"] == 1.0
    assert r["n_distinct"] == N_CLASSES
    assert r["modal_share"] == 0.25


def test_the_hit_rate_is_unchanged_by_any_of_this(monkeypatch) -> None:
    """Guards the 44 finished runs: accuracy and by_cell keep their meaning."""
    r = _evaluate(monkeypatch, SCATTERED)
    assert r["accuracy"] == 0.25
    assert r["n"] == 16
    assert r["by_cell"] == {"2x1": 0.25}
