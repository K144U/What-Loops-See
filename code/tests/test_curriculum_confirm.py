"""D-061 decides whether claims.md rows 4 to 6 survive fresh seeds.

The ways it could lie: count a held-out seed into the arm it tests, call a
wrong-direction contrast held, call a quiet contrast confirmed, say anything
before every seed has finished, or run configs that differ from their arm in
more than the donor.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from loopvision.analysis import curriculum_confirm as C
from loopvision.analysis import escape_rates as ER
from loopvision.train import cli

REPO_ROOT = Path(__file__).resolve().parents[1]


def _runs(stem, escaped, finished, unfinished=0, seed0=0, neither=0):
    rows = []
    for i in range(finished + unfinished):
        if i < escaped:
            out = "escaped"
        elif i < escaped + neither:
            out = "neither"
        else:
            out = "collapsed"
        rows.append({"run_id": f"{stem}_s{seed0 + i}", "done": i < finished,
                     "outcome": out if i < finished else "unfinished"})
    return rows


def _sealed():
    return (_runs("famA_d2_d4colour_curr", 10, 10)
            + _runs("famA_d2_d4colour_currctrl", 9, 10)
            + _runs("famA_d2_d4colour_curr_s3donor", 1, 10))


def _held(curr=5, ctrl=5, s3=0):
    return (_runs("famA_d2_d4colour_curr_confirm", curr, 5, seed0=10)
            + _runs("famA_d2_d4colour_currctrl_confirm", ctrl, 5, seed0=10)
            + _runs("famA_d2_d4colour_curr_s3donor_confirm", s3, 5, seed0=10))


# --- stems -------------------------------------------------------------------

@pytest.mark.parametrize("run_id,want", [
    ("famA_d2_d4colour_curr_s3donor_confirm_s12", "famA_d2_d4colour_curr_s3donor_confirm"),
    ("famA_d2_d4colour_curr_s3donor_s7", "famA_d2_d4colour_curr_s3donor"),
    ("famA_d2_d4colour_currctrl_confirm_s10", "famA_d2_d4colour_currctrl_confirm"),
    ("famA_d2_d4colour_curr_s0", "famA_d2_d4colour_curr"),
])
def test_run_ids_resolve_to_the_stem_escape_rates_uses(run_id, want) -> None:
    assert C.stem(run_id) == want
    assert set(ER.stem_rates([{"run_id": run_id, "outcome": "escaped"}])) == {want}


def test_a_held_out_seed_is_never_counted_in_the_arm_it_tests() -> None:
    t = C.tally(_sealed() + _held())
    assert t["famA_d2_d4colour_curr"]["finished"] == 10
    assert t["famA_d2_d4colour_curr_confirm"]["finished"] == 5


def test_neither_is_finished_but_not_escaped() -> None:
    t = C.tally(_runs("famA_d2_d4colour_curr_confirm", 3, 5, neither=1))
    assert t["famA_d2_d4colour_curr_confirm"] == {"escaped": 3, "finished": 5, "unfinished": 0}


# --- verdicts ------------------------------------------------------------------

def test_the_expected_outcome_reads_not_contradicted_and_held() -> None:
    v = C.registered(_sealed() + _held())
    assert [v[k][0] for k in ("R4", "R5", "R6")] == ["not contradicted"] * 3
    assert v["C1"][0] == "held" and v["C2"][0] == "held"


def test_a_held_out_wrong_donor_that_rescues_fails_its_row() -> None:
    v = C.registered(_sealed() + _held(s3=4))
    assert v["R6"][0] == "failed"


def test_a_quiet_contrast_is_not_the_claim_held() -> None:
    """5 of 5 against 2 of 5 is p = 0.17: in the right direction, not enough."""
    v = C.registered(_sealed() + _held(s3=2))
    assert v["C1"][0] == "failed"


def test_a_significant_contrast_in_the_wrong_direction_fails() -> None:
    v = C.registered(_sealed() + _held(curr=0, s3=5))
    assert v["C1"][0] == "failed"


def test_the_row_rule_is_the_confirm_register_rule_exactly() -> None:
    """Fisher two sided below 0.05, as escape_rates --confirm applies."""
    assert C.ALPHA == 0.05
    h = {"escaped": 3, "finished": 5, "unfinished": 0}
    s = {"escaped": 10, "finished": 10, "unfinished": 0}
    assert C._p(h, s) == ER.fisher_two_sided(3, 2, 10, 0)


def test_nothing_is_said_while_a_seed_is_running() -> None:
    rows = _sealed() + _held()[:-1] + _runs(
        "famA_d2_d4colour_curr_s3donor_confirm", 0, 0, unfinished=1, seed0=14)
    v = C.registered(rows)
    assert v["R6"][0] == "pending" and v["C1"][0] == "pending" and v["C2"][0] == "pending"


def test_a_sixth_seed_still_running_keeps_the_arm_pending() -> None:
    """Five finished is not enough while another is running: separates the
    running check from the count check, which the test above tests together."""
    rows = _sealed() + _held() + _runs(
        "famA_d2_d4colour_curr_s3donor_confirm", 0, 0, unfinished=1, seed0=15)
    assert C.registered(rows)["R6"][0] == "pending"


def test_a_seed_that_never_started_keeps_the_arm_pending() -> None:
    """Four finished and nothing running: a missing seed is not a verdict."""
    rows = _sealed() + _held()[:-1]
    assert C.registered(rows)["R6"][0] == "pending"


def test_missing_arms_are_pending() -> None:
    v = C.registered(_sealed())
    assert {s for s, _ in v.values()} == {"pending"}


def test_the_constants_are_the_ones_d061_registers() -> None:
    assert (C.SEEDS, C.ALPHA) == (5, 0.05)
    assert set(C.ROWS) == {"R4", "R5", "R6"} and set(C.CLAIMS) == {"C1", "C2"}


# --- the configs -----------------------------------------------------------------

TEMPLATES = {
    "famA_d2_d4colour_curriculum_confirm": ("famA_d2_d4colour_curriculum", "famA_d1_d4colour"),
    "famA_d2_d4colour_curriculum_ctrl_confirm": ("famA_d2_d4colour_curriculum_ctrl", "famA_d1_d4only"),
    "famA_d2_d4colour_curriculum_s3donor_confirm": ("famA_d2_d4colour_curriculum_s3donor", "famA_d1_s3only"),
}
CONFIGS = [(stem, s) for stem in TEMPLATES for s in range(10, 15)]


@pytest.mark.parametrize("stem,seed", CONFIGS, ids=lambda x: str(x))
def test_each_confirmation_config_differs_from_its_arm_only_in_donor_and_marker(stem, seed) -> None:
    template, donor = TEMPLATES[stem]
    base = cli.load_config(str(REPO_ROOT / f"configs/m3/{template}.yaml"), {})
    new = cli.load_config(str(REPO_ROOT / f"configs/m3/{stem}_s{seed}.yaml"), {})
    changed = {k for k in set(base) | set(new) if base.get(k) != new.get(k)}
    assert changed == {"init_from", "confirmation"}
    assert new["init_from"] == f"{donor}_s{seed}", "each seed has its own fresh donor"
    assert new["confirmation"] is True
    assert ER.is_confirmation(new)


def test_there_are_exactly_fifteen_and_no_donor_is_shared() -> None:
    found = sorted((REPO_ROOT / "configs/m3").glob("famA_d2_d4colour_curriculum*_confirm_s*.yaml"))
    assert len(found) == 15
    donors = [cli.load_config(str(p), {})["init_from"] for p in found]
    assert len(set(donors)) == 15
    assert all(re.search(r"_s1[0-4]$", d) for d in donors)


def test_main_is_pending_on_an_empty_runs_directory(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    assert C.main(["--runs", str(tmp_path / "runs")]) == 2
