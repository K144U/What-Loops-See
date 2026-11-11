"""D-065 early stop and D-067 pixel scale, both registered on 2026-10-11.

Early stop changes how long a run trains, so it is pinned at the rule, at
the config guard, at the resume path and end to end through the real
trainer. The pixel scale changes what the model sees, so it is pinned at
the pixels: an upscaled batch must be the original batch with every pixel
turned into a block, and nothing else.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from loopvision.train import cli

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- the rule ---------------------------------------------------------------

def test_stops_when_every_evaluation_in_the_window_clears_tau() -> None:
    h = [(10000, 0.5), (20000, 0.96), (30000, 0.97), (40000, 0.99)]
    assert cli.early_stop_due(h, 40000, 0.95, 20000)


def test_one_evaluation_below_tau_in_the_window_blocks_the_stop() -> None:
    h = [(20000, 0.96), (30000, 0.94), (40000, 0.99)]
    assert not cli.early_stop_due(h, 40000, 0.95, 20000)


def test_an_evaluation_below_tau_before_the_window_does_not_block_it() -> None:
    h = [(10000, 0.10), (20000, 0.96), (30000, 0.97), (40000, 0.99)]
    assert cli.early_stop_due(h, 40000, 0.95, 20000)


def test_a_window_not_fully_observed_does_not_stop() -> None:
    """No evaluation at the start of the window means the run has not been
    watched for the whole window, for instance its first evaluation is too
    recent."""
    assert not cli.early_stop_due([(30000, 0.99), (40000, 0.99)], 40000, 0.95, 20000)


# --- the guard --------------------------------------------------------------

def _cfg(**over) -> dict:
    return cli.load_config(None, {"family": "A", "eval_every": 10000, **over})


def test_early_stop_needs_both_keys() -> None:
    with pytest.raises(ValueError, match="together"):
        _cfg(early_stop_tau=0.95)


def test_early_stop_is_refused_outside_family_a() -> None:
    with pytest.raises(ValueError, match="family A only"):
        cli.load_config(None, {"family": "B", "eval_every": 10000,
                               "early_stop_tau": 0.95, "early_stop_window": 20000})


def test_a_window_that_never_meets_an_evaluation_is_refused() -> None:
    with pytest.raises(ValueError, match="multiple of"):
        _cfg(early_stop_tau=0.95, early_stop_window=25000)


def test_absent_early_stop_is_off_and_unchanged() -> None:
    cfg = _cfg()
    assert cfg["early_stop_tau"] is None and cfg["early_stop_window"] is None


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_pixel_scale_must_be_a_positive_integer(bad) -> None:
    with pytest.raises(ValueError, match="pixel_scale"):
        _cfg(pixel_scale=bad)


# --- the resume path --------------------------------------------------------

def test_history_is_read_back_from_stored_metrics(tmp_path) -> None:
    """A window that spans a job boundary needs the evaluations the previous
    job made; only overall rows count, and only up to the resume step."""
    rows = [
        {"step": 10000, "k": 8, "split": "iid_val", "depth": -1, "breadth": -1,
         "metric_name": "accuracy", "value": 0.96, "seed": 0},
        {"step": 10000, "k": 8, "split": "iid_val", "depth": 2, "breadth": 1,
         "metric_name": "accuracy", "value": 0.10, "seed": 0},
        {"step": 20000, "k": 8, "split": "iid_val", "depth": -1, "breadth": -1,
         "metric_name": "accuracy", "value": 0.97, "seed": 0},
        {"step": 30000, "k": 8, "split": "iid_val", "depth": -1, "breadth": -1,
         "metric_name": "accuracy", "value": 0.98, "seed": 0},
        {"step": 20000, "k": 8, "split": "train", "depth": -1, "breadth": -1,
         "metric_name": "loss", "value": 0.01, "seed": 0},
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "metrics.parquet", index=False)
    assert cli.eval_history(tmp_path, 20000) == [(10000, 0.96), (20000, 0.97)]
    h = cli.eval_history(tmp_path, 20000) + [(30000, 0.99)]
    assert cli.early_stop_due(h, 30000, 0.95, 20000)


# --- pixel scale, at the pixels ---------------------------------------------

def _batch_cfg(**over) -> dict:
    return cli.load_config(None, {
        "family": "A", "factor": "d4", "depths": [2], "breadths": [1, 2],
        "batch_size": 4, "num_workers": 0, **over})


def test_an_upscaled_batch_is_the_original_with_every_pixel_a_block() -> None:
    dev = torch.device("cpu")
    small = cli.make_batch(_batch_cfg(), "train", 0, dev)[0]
    big = cli.make_batch(_batch_cfg(pixel_scale=2), "train", 0, dev)[0]
    assert tuple(small.shape[-2:]) == (32, 32) and tuple(big.shape[-2:]) == (64, 64)
    for dy in (0, 1):
        for dx in (0, 1):
            assert torch.equal(big[..., dy::2, dx::2], small)


def test_the_model_sees_four_times_the_patches_at_scale_two() -> None:
    cfg = _batch_cfg(pixel_scale=2, d_model=64, prelude_blocks=1, core_blocks=1,
                     coda_blocks=1, k_schedule="fixed", k_train=1)
    model = cli.build_model(cfg, torch.device("cpu"))
    assert model.cfg.num_patches == 256
    image, query, _, _, _ = cli.make_batch(cfg, "train", 0, torch.device("cpu"))
    assert model(image, query, 1).shape[0] == cfg["batch_size"]


# --- end to end, through the real trainer -----------------------------------

def _train(tmp_path: Path, run_id: str, **extra) -> Path:
    cfg = {
        "family": "A", "factor": "d4", "depths": [1], "breadths": [1],
        "d_model": 64, "prelude_blocks": 1, "core_blocks": 1, "coda_blocks": 1,
        "k_schedule": "fixed", "k_train": 1, "batch_size": 8, "steps": 20,
        "eval_every": 2, "eval_batches": 1, "num_workers": 0,
        "eval_k_sweep": None, **extra,
    }
    path = tmp_path / f"{run_id}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "loopvision.train.cli", "--config", str(path),
         "--run-id", run_id, "--runs-root", str(tmp_path / "runs")],
        capture_output=True, text=True, timeout=600, cwd=REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return tmp_path / "runs" / run_id


def _eval_steps(run: Path) -> list[int]:
    m = pd.read_parquet(run / "metrics.parquet")
    a = m[(m.metric_name == "accuracy") & (m.depth == -1)]
    return sorted(a.step.unique().tolist())


def test_a_run_meeting_the_rule_stops_and_still_finishes_properly(tmp_path) -> None:
    """tau 0 is met at every evaluation, so with evaluations at 2 and 4 and a
    window of 2 the rule fires at step 4 of 20."""
    run = _train(tmp_path, "es_on", early_stop_tau=0.0, early_stop_window=2)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["stopped_early_at"] == 4
    assert "final_accuracy" in summary
    assert (run / "DONE").is_file()
    assert _eval_steps(run) == [2, 4]
    assert len(list((run / "checkpoints").glob("step_*.pt"))) == 1


def test_a_run_never_meeting_the_rule_trains_its_full_budget(tmp_path) -> None:
    run = _train(tmp_path, "es_never", early_stop_tau=1.01, early_stop_window=2)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["stopped_early_at"] is None
    assert _eval_steps(run)[-1] == 20


def test_without_early_stop_nothing_changes(tmp_path) -> None:
    run = _train(tmp_path, "es_off")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["stopped_early_at"] is None
    assert _eval_steps(run)[-1] == 20


def test_the_trainer_seeds_the_window_from_the_stored_evaluations() -> None:
    """Forcing a job boundary inside a window is not reproducible in a test,
    so pin the line: without it a resumed run judges the window on its own
    job's evaluations only, and stops later than the rule says."""
    src = (REPO_ROOT / "src" / "loopvision" / "train" / "cli.py").read_text(encoding="utf-8")
    assert "eval_history(run_dir, start_step) if early_stop else []" in src
