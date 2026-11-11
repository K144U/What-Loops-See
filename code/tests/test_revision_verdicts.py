"""D-066 to D-069 are decided by revision_verdicts. The ways it could lie:
accept a pooled effect in the wrong direction, miss a reversed cell, skip
Holm, test T2 at 0.05 when its registered reading is 0.025, count trapped
runs that never left chance, or read before every seed has finished.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from loopvision.analysis import revision_verdicts as V

ROOT = Path(__file__).resolve().parents[1]


def _rows(stem, n, escaped, left=None, finals=None, unfinished=0, first_seed=300):
    """n finished runs of a stem: the first `escaped` escaped. `left` runs
    left chance, and `finals` gives their final accuracies if supplied."""
    left = n if left is None else left
    rows = []
    for i in range(n):
        final = finals[i] if finals is not None else (1.0 if i < escaped else 0.125)
        rows.append({"run_id": f"{stem}_s{first_seed + i}", "stem": stem, "done": True,
                     "outcome": "escaped" if i < escaped else "collapsed",
                     "final": final, "left": 3000 if i < left else None})
    for i in range(unfinished):
        rows.append({"run_id": f"{stem}_s{first_seed + 100 + i}", "stem": stem, "done": False,
                     "outcome": "unfinished", "final": None, "left": None})
    return rows


def _grid(pos, app, n=5):
    rows = []
    for c in V.C1_CELLS:
        rows += _rows(f"c1_pos_{c}", n, pos) + _rows(f"c1_app_{c}", n, app)
    return rows


# --- Holm itself ----------------------------------------------------------------

def test_holm_matches_a_hand_worked_example() -> None:
    assert V.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_holm_never_adjusts_above_one() -> None:
    assert max(V.holm([0.6, 0.7])) == 1.0


# --- D-066 ----------------------------------------------------------------------

def test_a_consistent_asymmetry_holds_both() -> None:
    v = V.d066(V.tally(_grid(5, 1)))
    assert v["G1"][0] == "held" and v["G2"][0] == "held"


def test_no_asymmetry_fails_g1() -> None:
    assert V.d066(V.tally(_grid(2, 2)))["G1"][0] == "failed"


def test_a_reversed_grid_fails_both_however_small_the_pooled_p() -> None:
    v = V.d066(V.tally(_grid(0, 5)))
    assert v["G1"][0] == "failed", "a pooled effect in the wrong direction must not hold"
    assert v["G2"][0] == "failed", "six fully reversed cells must count as a reversal"


def test_a_missing_seed_keeps_the_grid_pending() -> None:
    rows = [r for r in _grid(5, 1) if r["run_id"] != "c1_app_lr1e3_wd05_s304"]
    assert V.d066(V.tally(rows))["G1"][0] == "pending"


# --- D-067 ----------------------------------------------------------------------

def test_t1_holds_for_a_clear_asymmetry_and_fails_reversed() -> None:
    held = V.d067(V.tally(_rows("c2_pos_x2", 10, 10) + _rows("c2_app_x2", 10, 3)))
    rev = V.d067(V.tally(_rows("c2_pos_x2", 10, 3) + _rows("c2_app_x2", 10, 10)))
    assert held["T1"][0] == "held" and rev["T1"][0] == "failed"


# --- D-068 ----------------------------------------------------------------------

def _trap(e2b_escaped, e2a_escaped, e2a_left, e2a_finals):
    return (_rows("c3_e2b_decoy", 10, e2b_escaped, first_seed=220)
            + _rows("c3_e2a_cue", 10, e2a_escaped, left=e2a_left, finals=e2a_finals,
                    first_seed=220))


def test_t2_is_held_to_0_025_as_fixed_before_any_result() -> None:
    """10 of 10 against 5 of 10 is p = 0.0325: it would pass at 0.05 and must
    not pass at the conservative 0.025 this module registered."""
    finals = [0.25] * 6 + [1.0] * 4
    assert V.d068(V.tally(_trap(10, 5, 10, finals)))["T2"][0] == "failed"
    assert V.d068(V.tally(_trap(9, 1, 10, [0.25] * 9 + [1.0])))["T2"][0] == "held"


def test_t3_counts_only_runs_that_left_chance_and_includes_the_band_edges() -> None:
    finals = [0.24, 0.26, 0.2499, 0.30, 0.70, 0.125, 0.125, 0.125, 0.125, 0.125]
    v = V.d068(V.tally(_trap(9, 0, 6, finals)))
    assert v["T3"][0] == "held", v["T3"]
    assert "3 of 6" in v["T3"][1]


def test_t3_fails_when_most_departures_escape_the_band() -> None:
    finals = [0.25, 0.70, 0.80, 0.90, 1.0, 1.0, 0.125, 0.125, 0.125, 0.125]
    assert V.d068(V.tally(_trap(9, 2, 6, finals)))["T3"][0] == "failed"


def test_t3_is_undetermined_with_too_few_departures() -> None:
    finals = [0.25] * 3 + [0.125] * 7
    assert V.d068(V.tally(_trap(9, 0, 3, finals)))["T3"][0] == "undetermined"


# --- D-069 ----------------------------------------------------------------------

def test_holm_can_fail_a_pair_that_passes_uncorrected() -> None:
    """10 of 10 and 9 of 10 against 5 of 10 give p = 0.0325 and 0.141; Holm
    doubles the smaller to 0.065, so neither holds."""
    rows = (_rows("c4_curr", 10, 10, first_seed=15) + _rows("c4_currctrl", 10, 9, first_seed=15)
            + _rows("c4_s3donor", 10, 5, first_seed=15))
    v = V.d069(V.tally(rows))
    assert v["K1"][0] == "failed" and v["K2"][0] == "failed"


def test_clear_specificity_holds_both_k_checks() -> None:
    rows = (_rows("c4_curr", 10, 10, first_seed=15) + _rows("c4_currctrl", 10, 10, first_seed=15)
            + _rows("c4_s3donor", 10, 1, first_seed=15))
    v = V.d069(V.tally(rows))
    assert v["K1"][0] == "held" and v["K2"][0] == "held"


# --- consistency with the configs, and reading real directories -------------------

def test_the_stems_read_are_exactly_the_configs_written() -> None:
    spec = importlib.util.spec_from_file_location("mrc", ROOT / "scripts" / "make_revision_configs.py")
    mrc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mrc)
    from_configs = {n.rsplit("_s", 1)[0] if n.startswith("c4_") else n for n in mrc.changes()}
    assert V.wanted_stems() == from_configs


def _write(root, run_id, levels, acc):
    d = root / run_id
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump({"family": "A", "substrate": "d4_colour_cue"}),
                                   encoding="utf-8")
    (d / "DONE").write_text("done\n", encoding="utf-8")
    (d / "summary.json").write_text(
        '{"final_accuracy": %s, "n_distinct_predictions": 2}' % acc, encoding="utf-8")
    steps = np.arange(50, 300001, 50)
    values = np.full(len(steps), math.log(8))
    for start, end, level in levels:
        values[(steps > start) & (steps <= end)] = level
    pd.DataFrame({"step": steps, "k": 4, "split": "train", "depth": -1, "breadth": -1,
                  "metric_name": "loss", "value": values, "seed": 0}).to_parquet(d / "metrics.parquet")


def test_scan_reads_final_accuracy_and_departure_and_ignores_other_runs(tmp_path) -> None:
    root = tmp_path / "runs"
    _write(root, "c3_e2a_cue_s220", [(5000, 300000, math.log(4))], 0.2494)
    _write(root, "c3_e2a_cue_s221", [], 0.125)
    _write(root, "why_e2a_cue_s200", [(5000, 300000, math.log(4))], 0.2494)
    rows = {r["run_id"]: r for r in V.scan(root)}
    assert set(rows) == {"c3_e2a_cue_s220", "c3_e2a_cue_s221"}
    assert rows["c3_e2a_cue_s220"]["left"] is not None
    assert abs(rows["c3_e2a_cue_s220"]["final"] - 0.2494) < 1e-9
    assert rows["c3_e2a_cue_s221"]["left"] is None


def test_main_is_pending_on_an_empty_runs_directory(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    assert V.main(["--runs", str(tmp_path / "runs")]) == 2


def test_the_pooled_test_reproduces_r_mantelhaen_example() -> None:
    """R's documented mantelhaen.test example, the penicillin rabbits data, one
    2x2 table per penicillin level (rows delay none or 1.5h, columns cured or
    died). R reports X-squared 3.9286 and p = 0.04747 with its default
    continuity correction. The pooled Mantel-Haenszel odds ratio, worked by
    hand from sum(a d / n) over sum(b c / n), is 3.5 / 0.5 = 7.0."""
    rabbits = [[[0, 6], [0, 5]], [[3, 3], [0, 6]], [[6, 0], [2, 4]],
               [[5, 1], [6, 0]], [[2, 0], [5, 0]]]
    p, odds = V.cmh(rabbits)
    assert abs(p - 0.04747) < 5e-5
    assert abs(odds - 7.0) < 1e-9


def test_a_strong_forward_effect_in_every_cell_is_not_read_as_a_reversal() -> None:
    """5 of 5 against 0 of 5 is p = 0.0079 per cell and still under 0.05
    after Holm across six, so a G2 that ignored direction would call this
    grid reversed. It must only count cells leaning toward appearance."""
    v = V.d066(V.tally(_grid(5, 0)))
    assert v["G1"][0] == "held" and v["G2"][0] == "held"
