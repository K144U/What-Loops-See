"""The timing module feeds a registration, so each test is a way it could lie.

Four matter most. A centred window would date a solve before it happened. A
plateau detector without a duration floor would call any passing value a
level. A level list that included non-quotient values would find staircases
everywhere. And pooling a confirmation seed with the data it tests would
make the registered check read its own discovery data.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from loopvision.analysis import escape_rates as ER
from loopvision.analysis import escape_timing as T

LN8, LN4, LN2 = math.log(8), math.log(4), math.log(2)


def _trace(segments, *, end=300000, every=50, noise=0.03, seed=0):
    """Steps and a loss that holds each (until_step, level) segment in turn."""
    steps = np.arange(every, end + 1, every)
    values = np.empty(len(steps))
    lo = 0
    for until, level in segments:
        values[(steps > lo) & (steps <= until)] = level
        lo = until
    values[steps > lo] = segments[-1][1]
    rng = np.random.default_rng(seed)
    return steps, values + rng.normal(0.0, noise, len(steps))


def _metrics(steps, values):
    n = len(steps)
    return pd.DataFrame({
        "step": steps, "k": np.full(n, 4), "split": ["train"] * n,
        "depth": np.full(n, -1), "breadth": np.full(n, -1),
        "metric_name": ["loss"] * n, "value": values, "seed": np.zeros(n, int),
    })


def _run(tmp_path, name, steps, values, *, factor="full", substrate=None,
         done=True, acc=1.0, confirmation=None, init_from=None, seed=0):
    d = tmp_path / "runs" / name
    d.mkdir(parents=True)
    cfg = dict(ER.BASELINE)
    cfg.update({"family": "A", "breadths": [1, 2, 3, 4], "factor": factor,
                "seed": seed})
    if substrate is not None:
        cfg["substrate"] = substrate
    if confirmation is not None:
        cfg["confirmation"] = confirmation
    if init_from is not None:
        cfg["init_from"] = init_from
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    if done:
        (d / "DONE").write_text("done\n", encoding="utf-8")
    summary = {"final_accuracy": acc}
    if acc < 0.2:
        summary["n_distinct_predictions"] = 1
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _metrics(steps, values).to_parquet(d / "metrics.parquet")
    return d


# --- the registered constants -----------------------------------------------

def test_the_constants_are_the_ones_d059_registers() -> None:
    """D-059 registers these values. Changing one after the confirmation
    seeds run is exactly the tuning CLAUDE.md forbids, so it has to break a
    test and appear in a diff rather than slip through a refactor."""
    assert (T.SOLVE_WINDOW, T.SOLVE_FRACTION) == (1000, 0.1)
    assert (T.PLATEAU_WINDOW, T.PLATEAU_BAND, T.PLATEAU_MIN_STEPS) == (2000, 0.15, 10000)
    assert (T.WINDOW_LIMIT, T.STAIRCASE_MIN, T.SEEDS_PER_ARM) == (40000, 4, 5)
    assert T.WINDOW_ARMS == (("D4", "identity"), ("S3", "identity"))
    assert T.STAIRCASE_ARM == ("D4", "position")


# --- the levels -------------------------------------------------------------

def test_levels_are_the_proper_quotients_and_nothing_else() -> None:
    """ln 3 is not a D4 level: 8 has no divisor 3. A detector that searched
    for it would find staircases in runs that have none."""
    assert set(T.quotient_levels(8)) == {"ln4", "ln2"}
    assert set(T.quotient_levels(6)) == {"ln3", "ln2"}
    assert T.quotient_levels(8)["ln4"] == pytest.approx(LN4)


def test_the_order_comes_from_the_config_not_the_name() -> None:
    assert T.group_order({"factor": "d4"}) == 8
    assert T.group_order({"factor": "s3"}) == 6
    assert T.group_order({"factor": "full", "substrate": "d4_colour"}) == 8
    assert T.group_order({"factor": "full", "substrate": "s3_spatial"}) == 6


def test_the_bands_of_neighbouring_levels_do_not_overlap() -> None:
    """If they did, one stretch could be reported at two levels at once."""
    for order in T.SUPPORTED_ORDERS:
        lv = sorted(T.quotient_levels(order).values())
        for a, b in zip(lv, lv[1:]):
            assert b - a > 2 * T.PLATEAU_BAND


# --- the smoothing ----------------------------------------------------------

def test_the_mean_is_trailing_so_a_drop_is_never_dated_early() -> None:
    steps = np.arange(50, 20001, 50)
    values = np.where(steps <= 10000, LN8, 0.0)
    sm = T.trailing_mean(steps, values, 1000)
    before = steps < 10050
    assert np.all(sm[before & ~np.isnan(sm)] == pytest.approx(LN8))


def test_the_mean_is_undefined_until_its_window_is_full() -> None:
    steps = np.arange(50, 5001, 50)
    sm = T.trailing_mean(steps, np.ones(len(steps)), 1000)
    assert np.isnan(sm[steps < 1000]).all()
    assert not np.isnan(sm[steps >= 1000]).any()


def test_a_repeated_step_keeps_the_later_write() -> None:
    df = _metrics(np.array([50, 100, 100, 150]), np.array([2.0, 9.0, 1.0, 1.0]))
    steps, values = T.train_loss(df)
    assert list(steps) == [50, 100, 150]
    assert values[1] == 1.0


# --- solve step -------------------------------------------------------------

def test_a_direct_drop_solves_when_it_drops() -> None:
    steps, values = _trace([(15000, LN8), (300000, 0.01)])
    s = T.solve_step(steps, values, 8)
    assert 15000 < s <= 15000 + T.SOLVE_WINDOW


def test_one_low_batch_is_not_a_solve() -> None:
    steps, values = _trace([(300000, LN8)])
    values[1000] = 0.0
    assert T.solve_step(steps, values, 8) is None


def test_a_quotient_plateau_is_not_a_solve() -> None:
    """The fault that split the scratch pass's one number into two."""
    steps, values = _trace([(10000, LN8), (300000, LN2)])
    assert T.solve_step(steps, values, 8) is None


# --- leave step ------------------------------------------------------------

def test_a_direct_drop_leaves_when_it_drops() -> None:
    steps, values = _trace([(15000, LN8), (300000, 0.0)])
    s = T.leave_step(steps, values, 8)
    assert 15000 < s <= 15000 + T.PLATEAU_WINDOW


def test_a_collapsed_run_never_leaves() -> None:
    steps, values = _trace([(300000, LN8)], noise=0.05)
    assert T.leave_step(steps, values, 8) is None


def test_a_brief_dip_is_not_a_departure() -> None:
    brief = T.PLATEAU_MIN_STEPS // 2
    steps, values = _trace([(50000, LN8), (50000 + brief, 1.0), (300000, LN8)])
    assert T.leave_step(steps, values, 8) is None


def test_a_departure_that_lasts_to_the_end_counts_however_short() -> None:
    """Otherwise a run escaping in its last few thousand steps would read as
    never having left, which is the direction that would hide a late escape."""
    steps, values = _trace([(295000, LN8), (300000, 0.0)])
    s = T.leave_step(steps, values, 8)
    assert s is not None and s > 295000


def test_a_location_plateau_separates_leave_from_solve() -> None:
    """The identity pattern: off chance early, parked near 1.0, solved late."""
    steps, values = _trace([(15000, LN8), (120000, 1.0), (300000, 0.0)])
    assert T.leave_step(steps, values, 8) < 20000
    assert T.solve_step(steps, values, 8) > 120000
    assert T.plateaus(steps, values, 8, before=T.solve_step(steps, values, 8)) == []


# --- plateaus ---------------------------------------------------------------

def test_a_staircase_is_found_in_order_and_ends_at_the_solve() -> None:
    steps, values = _trace([(8000, LN8), (60000, LN4), (90000, LN2), (300000, 0.0)])
    s = T.solve_step(steps, values, 8)
    found = T.plateaus(steps, values, 8, before=s)
    assert [p["level"] for p in found] == ["ln4", "ln2"]
    assert found[0]["start"] < 12000 and found[0]["end"] > 55000
    assert all(p["end"] < s for p in found)


def test_a_direct_drop_has_no_plateau() -> None:
    steps, values = _trace([(15000, LN8), (300000, 0.0)])
    assert T.plateaus(steps, values, 8, before=T.solve_step(steps, values, 8)) == []


def test_a_short_visit_is_not_a_plateau() -> None:
    short = T.PLATEAU_MIN_STEPS // 2
    steps, values = _trace([(20000, LN8), (20000 + short, LN4), (300000, LN8)])
    assert T.plateaus(steps, values, 8) == []


def test_a_level_between_quotients_is_not_reported() -> None:
    """famA_d2_d4only_s4 sat near 1.1 in the scratch pass. ln 3 is not a D4
    quotient and must not be reported as one."""
    steps, values = _trace([(8000, LN8), (80000, math.log(3)), (300000, 0.0)])
    assert T.plateaus(steps, values, 8) == []


def test_a_gap_in_logging_breaks_a_stretch() -> None:
    steps, values = _trace([(8000, LN8), (300000, LN4)])
    keep = (steps < 14000) | (steps > 30000)
    found = T.plateaus(steps[keep], values[keep], 8)
    assert all(not (p["start"] < 14000 < 30000 < p["end"]) for p in found)


# --- scanning, and the hold-out ---------------------------------------------

def test_unfinished_runs_are_skipped_and_counted(tmp_path) -> None:
    steps, values = _trace([(300000, LN8)])
    _run(tmp_path, "famA_d2_d4colour_s0", steps, values, substrate="d4_colour",
         done=False)
    rows, unfinished = T.scan(tmp_path / "runs")
    assert rows == [] and unfinished == 1


def test_the_full_group_is_out_of_scope(tmp_path) -> None:
    steps, values = _trace([(300000, math.log(48))])
    _run(tmp_path, "vark_famA_d2_s0", steps, values, factor="full", acc=0.02)
    rows, unfinished = T.scan(tmp_path / "runs")
    assert rows == [] and unfinished == 0


def test_confirmation_seeds_never_share_a_group_with_the_arm(tmp_path) -> None:
    steps, values = _trace([(300000, LN8)])
    _run(tmp_path, "famA_d2_d4colour_s0", steps, values, substrate="d4_colour",
         acc=0.1254)
    _run(tmp_path, "famA_d2_d4colour_confirm_s30", steps, values,
         substrate="d4_colour", acc=0.1254, confirmation=True, seed=30)
    rows, _ = T.scan(tmp_path / "runs")
    groups = {r["run_id"]: r["group"] for r in rows}
    assert groups["famA_d2_d4colour_s0"] != groups["famA_d2_d4colour_confirm_s30"]
    assert "[confirmation]" in groups["famA_d2_d4colour_confirm_s30"]


def test_a_donor_run_is_marked(tmp_path) -> None:
    steps, values = _trace([(1000, LN8), (300000, 0.0)])
    _run(tmp_path, "famA_d2_s3only_curr_s0", steps, values, factor="s3",
         init_from="famA_d1_s3only_s0")
    rows, _ = T.scan(tmp_path / "runs")
    assert rows[0]["group"].endswith("[init_from]")


# --- the registered predictions ---------------------------------------------

def _confirm(arm, run_id, solve=None, plateaus=(), leave=None):
    return {"run_id": run_id, "confirm_arm": list(arm), "solve_step": solve,
            "leave_step": leave if leave is not None else solve,
            "plateaus": [{"level": p} for p in plateaus]}


def _all_held():
    rows = []
    for i in range(5):
        rows.append(_confirm(("D4", "identity"), f"c{i}", solve=None))
        rows.append(_confirm(("S3", "identity"), f"s{i}", solve=12000 if i == 0 else None))
        rows.append(_confirm(("D4", "position"), f"p{i}", solve=90000,
                             plateaus=("ln4",)))
    return rows


def test_the_registered_check_holds_when_every_prediction_holds() -> None:
    assert {v for v, _ in T.registered(_all_held()).values()} == {"held"}


def test_one_late_identity_departure_fails_the_window() -> None:
    rows = _all_held()
    rows[0]["leave_step"] = T.WINDOW_LIMIT + 50
    assert T.registered(rows)["P1"][0] == "failed"


def test_a_departure_exactly_at_the_limit_is_inside_the_window() -> None:
    rows = _all_held()
    rows[0]["leave_step"] = T.WINDOW_LIMIT
    assert T.registered(rows)["P1"][0] == "held"


def test_the_window_reads_departure_not_solve() -> None:
    """An identity run that leaves chance early and solves late did not
    escape late. The scratch pass could not tell these apart."""
    rows = _all_held()
    rows[0]["leave_step"], rows[0]["solve_step"] = 15000, 127000
    assert T.registered(rows)["P1"][0] == "held"


def test_two_staircase_seeds_without_ln4_fail_and_one_does_not() -> None:
    rows = _all_held()
    stair = [r for r in rows if r["confirm_arm"] == ["D4", "position"]]
    stair[0]["plateaus"] = []
    assert T.registered(rows)["P2"][0] == "held"
    stair[1]["plateaus"] = [{"level": "ln2"}]
    assert T.registered(rows)["P2"][0] == "failed"


def test_any_plateau_in_a_colour_seed_fails_p3() -> None:
    rows = _all_held()
    rows[0]["plateaus"] = [{"level": "ln2"}]
    assert T.registered(rows)["P3"][0] == "failed"


def test_a_plateau_in_an_s3_identity_seed_fails_p3_too() -> None:
    rows = _all_held()
    s3 = [r for r in rows if r["confirm_arm"] == ["S3", "identity"]]
    s3[1]["plateaus"] = [{"level": "ln3"}]
    assert T.registered(rows)["P3"][0] == "failed"


def test_a_held_window_says_how_many_seeds_left_chance() -> None:
    """Held on zero departures is vacuous, so the count has to be visible."""
    rows = _all_held()
    for r in rows:
        if r["confirm_arm"] in (["D4", "identity"], ["S3", "identity"]):
            r["leave_step"] = None
    verdict, detail = T.registered(rows)["P1"]
    assert verdict == "held" and "0 of 10" in detail


def test_the_departure_count_counts_real_departures() -> None:
    """The zero case alone passes against a counter that never counts."""
    verdict, detail = T.registered(_all_held())["P1"]
    assert verdict == "held" and "1 of 10 seeds left chance" in detail


def test_missing_seeds_are_pending_not_held() -> None:
    """Quiet is not confirmed. With seeds still running, nothing may say held."""
    rows = [r for r in _all_held() if r["run_id"] not in ("c4", "p4")]
    v = T.registered(rows)
    assert v["P1"][0] == "pending" and v["P2"][0] == "pending" and v["P3"][0] == "pending"


def test_ordinary_seeds_are_never_read_by_the_registered_check() -> None:
    """The discovery data must not be able to confirm itself."""
    rows = [dict(r, confirm_arm=None) for r in _all_held()]
    v = T.registered(rows)
    assert {s for s, _ in v.values()} == {"pending"}


# --- the command line -------------------------------------------------------

def test_main_writes_json_and_exits_zero(tmp_path) -> None:
    steps, values = _trace([(8000, LN8), (60000, LN4), (300000, 0.0)])
    _run(tmp_path, "famA_d2_d4only_s0", steps, values, factor="d4")
    out = tmp_path / "t.json"
    assert T.main(["--runs", str(tmp_path / "runs"), "--json", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["rows"][0]["plateaus"][0]["level"] == "ln4"


def test_main_registered_is_pending_with_no_confirmation_seeds(tmp_path) -> None:
    steps, values = _trace([(300000, LN8)])
    _run(tmp_path, "famA_d2_d4colour_s0", steps, values, substrate="d4_colour",
         acc=0.1254)
    assert T.main(["--runs", str(tmp_path / "runs"), "--registered"]) == 2


def test_main_refuses_a_missing_runs_directory(tmp_path) -> None:
    assert T.main(["--runs", str(tmp_path / "nowhere")]) == 2
