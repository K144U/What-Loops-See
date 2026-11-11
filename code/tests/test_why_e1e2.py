"""D-062 decides H-V. The ways the evaluator could lie: accept a contrast in
the wrong direction, accept a quiet one, count climbing over runs that never
left chance, read before every seed has finished, or run configs that differ
from their base in more than the rendering.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from loopvision.analysis import escape_rates as ER
from loopvision.analysis import why_e1e2 as W
from loopvision.train import cli

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rows(arm, n, escaped, left, climbed, unfinished=0):
    """n finished runs: the first `escaped` escaped, the first `left` left
    chance, the first `climbed` of those climbed."""
    stem = W.ARMS[arm]
    rows = []
    for i in range(n):
        rows.append({"run_id": f"{stem}_s{200 + i}", "stem": stem, "done": True,
                     "outcome": "escaped" if i < escaped else "collapsed",
                     "left": 3000 if i < left else None, "climbed": i < climbed})
    for i in range(unfinished):
        rows.append({"run_id": f"{stem}_s{300 + i}", "stem": stem, "done": False,
                     "outcome": "unfinished", "left": None, "climbed": False})
    return rows


def _hv_holds():
    """What H-V predicts: E1 arms fall to identity-like rates without a stair,
    E2a climbs and escapes, E2b does neither."""
    return (_rows("sealed", 10, 9, 10, 9)
            + _rows("E1a", 20, 5, 6, 0) + _rows("E1b", 20, 5, 6, 0)
            + _rows("E2a", 20, 16, 17, 14) + _rows("E2b", 20, 5, 6, 0))


def test_h_v_holding_reads_held_everywhere() -> None:
    assert {v for v, _ in W.registered(_hv_holds()).values()} == {"held"}


def test_spatial_bias_rival_fails_e1a_and_holds_e1b() -> None:
    """R1: the square glyph is still a rigid motion, so it stays learnable."""
    rows = [r for r in _hv_holds() if r["stem"] != W.ARMS["E1a"]] + _rows("E1a", 20, 18, 20, 3)
    v = W.registered(rows)
    assert v["V1"][0] == "failed" and v["V3"][0] == "held"


def test_a_contrast_in_the_wrong_direction_fails_however_small_p() -> None:
    rows = [r for r in _hv_holds() if r["stem"] not in (W.ARMS["E2a"], W.ARMS["E2b"])]
    rows += _rows("E2a", 20, 2, 3, 0) + _rows("E2b", 20, 18, 19, 15)
    v = W.registered(rows)
    assert v["V5"][0] == "failed" and v["V6"][0] == "failed"


def test_a_quiet_contrast_fails() -> None:
    rows = [r for r in _hv_holds() if r["stem"] != W.ARMS["E2a"]] + _rows("E2a", 20, 7, 8, 1)
    assert W.registered(rows)["V5"][0] == "failed"


def test_climbing_is_counted_over_runs_that_left_chance_only() -> None:
    """12 of 20 E2a runs never left; 8 left and 8 climbed is a climb rate of 1."""
    t = W.tally(_rows("E2a", 20, 8, 8, 8))
    assert W._counts(t["E2a"], "climbed") == (8, 8)
    assert W._counts(t["E2a"], "escaped") == (8, 20)


def test_no_departures_on_one_side_is_undetermined_not_held() -> None:
    rows = [r for r in _hv_holds() if r["stem"] != W.ARMS["E1a"]] + _rows("E1a", 20, 0, 0, 0)
    v = W.registered(rows)
    assert v["V2"][0] == "undetermined"
    assert v["V1"][0] == "held"


def test_a_running_seed_keeps_its_contrasts_pending() -> None:
    rows = _hv_holds() + _rows("E2b", 0, 0, 0, 0, unfinished=1)
    v = W.registered(rows)
    assert v["V5"][0] == "pending" and v["V6"][0] == "pending"
    assert v["V1"][0] == "held"


def test_a_missing_seed_keeps_its_contrasts_pending() -> None:
    rows = [r for r in _hv_holds() if r["run_id"] != "why_e1b_scrambled_s219"]
    assert W.registered(rows)["V3"][0] == "pending"


def test_the_constants_are_the_ones_d062_registers() -> None:
    assert W.ALPHA == 0.05 and W.LEVEL == "ln4"
    assert W.SEEDS == {"E1a": 20, "E1b": 20, "E2a": 20, "E2b": 20, "sealed": 10}
    assert W.CONTRASTS["V5"] == ("escaped", "E2b", "E2a")
    assert W.ARMS["sealed"] == "famA_d2_d4only"


@pytest.mark.parametrize("run_id", ["why_e1a_square_s205", "why_e1b_scrambled_s219",
                                    "why_e2a_cue_s200", "why_e2b_decoy_s211", "famA_d2_d4only_s3"])
def test_run_ids_resolve_to_their_arm(run_id) -> None:
    assert W.stem(run_id) in W.ARMS.values()
    assert set(ER.stem_rates([{"run_id": run_id, "outcome": "escaped"}])) == {W.stem(run_id)}


# --- scanning real run directories -----------------------------------------------

def _write(root, run_id, cfg, levels, done=True, acc=1.0):
    d = root / run_id
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    if done:
        (d / "DONE").write_text("done\n", encoding="utf-8")
        (d / "summary.json").write_text(
            '{"final_accuracy": %s, "n_distinct_predictions": %d}' % (acc, 8 if acc > 0.5 else 1),
            encoding="utf-8")
    steps = np.arange(50, 300001, 50)
    values = np.full(len(steps), math.log(8))
    for start, end, level in levels:
        values[(steps > start) & (steps <= end)] = level
    pd.DataFrame({"step": steps, "k": 4, "split": "train", "depth": -1, "breadth": -1,
                  "metric_name": "loss", "value": values, "seed": 0}).to_parquet(d / "metrics.parquet")


def test_scan_reads_departure_and_climbing_from_the_losses(tmp_path) -> None:
    root = tmp_path / "runs"
    cfg = {"family": "A", "factor": "d4", "substrate": "d4_square", "experiment": "E1a"}
    _write(root, "why_e1a_square_s200", cfg, [(8000, 60000, math.log(4)), (60000, 300000, 0.0)])
    _write(root, "why_e1a_square_s201", cfg, [(15000, 300000, 0.0)])
    _write(root, "why_e1a_square_s202", cfg, [], acc=0.125)
    _write(root, "famA_d2_d4colour_s0", {"family": "A", "substrate": "d4_colour"}, [(9000, 300000, 0.0)])
    rows = {r["run_id"]: r for r in W.scan(root)}
    assert set(rows) == {"why_e1a_square_s200", "why_e1a_square_s201", "why_e1a_square_s202"}
    assert rows["why_e1a_square_s200"]["climbed"] and rows["why_e1a_square_s200"]["left"] is not None
    assert not rows["why_e1a_square_s201"]["climbed"] and rows["why_e1a_square_s201"]["left"] is not None
    assert rows["why_e1a_square_s202"]["left"] is None


# --- the configs -----------------------------------------------------------------

CONFIGS = [("e1a_square", "famA_d2_d4only", "d4_square", "E1a"),
           ("e1b_scrambled", "famA_d2_d4only", "d4_scrambled", "E1b"),
           ("e2a_cue", "famA_d2_d4colour", "d4_colour_cue", "E2a"),
           ("e2b_decoy", "famA_d2_d4colour", "d4_colour_decoy", "E2b")]


@pytest.mark.parametrize("name,base,sub,exp", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_each_config_differs_from_its_base_only_where_intended(name, base, sub, exp) -> None:
    b = cli.load_config(str(REPO_ROOT / f"configs/m3/{base}.yaml"), {})
    n = cli.load_config(str(REPO_ROOT / f"configs/why/{name}.yaml"), {})
    changed = {k for k in set(b) | set(n) if b.get(k) != n.get(k)}
    assert changed == {"substrate", "eval_every", "log_characters", "experiment"}
    assert (n["substrate"], n["experiment"], n["steps"]) == (sub, exp, 300000)
    assert ER.arm_of(n) is None, "a new rendering must never enter the sealed 2x2"


def test_e2a_and_e2b_differ_only_in_substrate() -> None:
    a = cli.load_config(str(REPO_ROOT / "configs/why/e2a_cue.yaml"), {})
    b = cli.load_config(str(REPO_ROOT / "configs/why/e2b_decoy.yaml"), {})
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"substrate", "experiment"}


def test_main_is_pending_on_an_empty_runs_directory(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    assert W.main(["--runs", str(tmp_path / "runs")]) == 2
