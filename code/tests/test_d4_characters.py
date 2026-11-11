"""E0.4 reads which D4 character a model gets right during training.

Three ways this could lie, and one way it could do harm. The characters
could fail to be homomorphisms, and then a model could "know" one without
composing anything. `rotflip` could fail to be the glyph's aspect ratio, and
then E0.4's prediction would be about the wrong character. The readout could
count the wrong thing. And the harm: the code reaches the cluster while the
D-059 confirmation seeds are queued, so a run that does not ask for it must
store exactly the config and write exactly the metrics it did before.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from loopvision.data import groups as G
from loopvision.train import cli

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- the characters -------------------------------------------------------

@pytest.mark.parametrize("name", sorted(G.D4_CHARACTERS))
def test_each_character_is_a_homomorphism_onto_z2(name) -> None:
    f = G.D4_CHARACTERS[name]
    for a in range(G.GROUP_ORDER):
        for b in range(G.GROUP_ORDER):
            assert f(G.multiply(a, b)) == (f(a) + f(b)) % 2


@pytest.mark.parametrize("name", sorted(G.D4_CHARACTERS))
def test_each_character_splits_the_group_in_half(name) -> None:
    """A constant map is a homomorphism too, and a useless one."""
    values = [G.D4_CHARACTERS[name](g) for g in range(G.GROUP_ORDER)]
    assert values.count(0) == values.count(1) == G.GROUP_ORDER // 2


def test_the_three_characters_are_distinct() -> None:
    table = {
        name: tuple(f(g) for g in range(G.GROUP_ORDER))
        for name, f in G.D4_CHARACTERS.items()
    }
    assert len(set(table.values())) == 3


def test_rotflip_is_the_aspect_ratio_of_the_native_glyph() -> None:
    """The structural fact E0.4's prediction rests on."""
    for g in range(G.GROUP_ORDER):
        cells = G.act_on_glyph(g)
        w = max(x for x, _, _ in cells) + 1
        h = max(y for _, y, _ in cells) + 1
        assert (h > w) == (G.D4_CHARACTERS["rotflip"](g) == 0)


def test_no_character_is_readable_from_the_d4_colour_silhouette() -> None:
    """The other half of the fact: one silhouette for all eight elements."""
    shapes = {
        frozenset((x, y) for x, y, _ in G.act_d4_colour(g))
        for g in G.subgroup_members("d4")
    }
    assert len(shapes) == 1


# --- the readout ----------------------------------------------------------

BATCH = 8


def _install(monkeypatch, labels, preds) -> None:
    def fake_make_batch(cfg, split, cursor, device):
        i = cursor // BATCH
        lab = torch.tensor(labels[i * BATCH:(i + 1) * BATCH])
        return (torch.zeros(BATCH, 3, 8, 8), torch.zeros(BATCH, 4, dtype=torch.long),
                lab, [2] * BATCH, [1] * BATCH)

    def fake_forward(model, image, query, k, *args, **kwargs):
        i = fake_forward.calls
        fake_forward.calls += 1
        out = torch.zeros(BATCH, G.GROUP_ORDER)
        for j, p in enumerate(preds[i * BATCH:(i + 1) * BATCH]):
            out[j, p] = 1.0
        return out

    fake_forward.calls = 0
    monkeypatch.setattr(cli, "make_batch", fake_make_batch)
    monkeypatch.setattr(cli, "forward", fake_forward)


def _evaluate(monkeypatch, labels, preds, **cfg_over):
    _install(monkeypatch, labels, preds)
    cfg = {"batch_size": BATCH, "k_eval": 1, **cfg_over}
    return cli.evaluate(torch.nn.Module(), cfg, "iid_val", torch.device("cpu"),
                        batches=len(labels) // BATCH)


D4 = list(G.subgroup_members("d4"))


def _same_rotflip_other(g: int) -> int:
    """An element with the same rotflip as g and a different flip."""
    f = G.D4_CHARACTERS
    for h in D4:
        if f["rotflip"](h) == f["rotflip"](g) and f["flip"](h) != f["flip"](g):
            return h
    raise AssertionError


def test_a_model_knowing_only_rotflip_reads_as_exactly_that(monkeypatch) -> None:
    labels = D4 * 2
    preds = [_same_rotflip_other(g) for g in labels]
    r = _evaluate(monkeypatch, labels, preds, log_characters=True)
    assert r["characters"]["rotflip"] == 1.0
    assert r["characters"]["flip"] == 0.0
    assert r["characters"]["rot2"] == 0.0
    assert r["accuracy"] == 0.0


def test_a_solved_model_reads_one_on_every_character(monkeypatch) -> None:
    labels = D4 * 2
    r = _evaluate(monkeypatch, labels, list(labels), log_characters=True)
    assert r["characters"] == {"flip": 1.0, "rot2": 1.0, "rotflip": 1.0}


def test_without_the_key_evaluate_returns_what_it_always_did(monkeypatch) -> None:
    labels = D4 * 2
    r = _evaluate(monkeypatch, labels, list(labels))
    assert set(r) == {"accuracy", "n", "by_cell", "n_distinct", "modal_pred",
                      "modal_share"}


def test_the_key_is_not_a_default_so_no_stored_config_gains_it() -> None:
    """If it were in DEFAULTS, every run started after this code reached the
    cluster would store it, including the D-059 confirmation seeds, whose
    configs D-057 requires to differ from their baselines by the marker
    alone."""
    assert "log_characters" not in cli.DEFAULTS
    cfg = cli.load_config(str(REPO_ROOT / "configs/m3/famA_d2_d4colour_confirm.yaml"), {})
    assert "log_characters" not in cfg


# --- end to end, through the real training loop ----------------------------

def _train(tmp_path: Path, run_id: str, **extra) -> pd.DataFrame:
    cfg = {
        "family": "A", "factor": "d4", "depths": [1], "breadths": [1],
        "d_model": 64, "prelude_blocks": 1, "core_blocks": 1, "coda_blocks": 1,
        "k_schedule": "fixed", "k_train": 1, "batch_size": 8, "steps": 4,
        "eval_every": 2, "eval_batches": 1, "num_workers": 0,
        "eval_k_sweep": None, **extra,
    }
    path = tmp_path / f"{run_id}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "loopvision.train.cli", "--config", str(path),
         "--run-id", run_id, "--runs-root", str(tmp_path / "runs")],
        capture_output=True, text=True, timeout=600, cwd=REPO_ROOT,
        # "-1", not "": on Windows an empty value still reports CUDA available
        # with no valid device, and the run dies naming a GPU it cannot see.
        env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return pd.read_parquet(tmp_path / "runs" / run_id / "metrics.parquet")


def test_training_logs_all_three_characters_when_asked(tmp_path) -> None:
    m = _train(tmp_path, "chars_on", log_characters=True)
    names = set(m.metric_name)
    assert {"character_flip", "character_rot2", "character_rotflip"} <= names
    rows = m[m.metric_name == "character_rotflip"]
    assert len(rows) >= 2 and rows.value.between(0.0, 1.0).all()


def test_training_logs_no_character_when_not_asked(tmp_path) -> None:
    m = _train(tmp_path, "chars_off")
    assert not any(n.startswith("character_") for n in set(m.metric_name))
    stored = yaml.safe_load((tmp_path / "runs/chars_off/config.yaml").read_text())
    assert "log_characters" not in stored
