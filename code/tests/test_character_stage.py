"""The stage detector feeds a registration, so each test is a way it could lie.

It must not call a solved model a stage, nor an identity run's location
plateau, nor one noisy eval. It must name the character that is actually
alone. The registered check must read only E0.4 runs, must not say held on
too few informative seeds, and must fail on the first wrong character. And
the E0.4 configs must stay out of the sealed 2x2.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from loopvision.analysis import character_stage as S
from loopvision.analysis import escape_rates as ER
from loopvision.train import cli

REPO_ROOT = Path(__file__).resolve().parents[1]
NAMES = ("flip", "rot2", "rotflip")


def _series(rows):
    """rows: list of (flip, rot2, rotflip) per eval, 500 steps apart."""
    steps = np.arange(500, 500 * (len(rows) + 1), 500)
    arr = np.array(rows, dtype=float)
    return steps, {n: arr[:, i] for i, n in enumerate(NAMES)}


CHANCE = (0.5, 0.5, 0.5)
ROTFLIP = (0.5, 0.5, 1.0)
FLIP = (1.0, 0.5, 0.5)
SOLVED = (1.0, 1.0, 1.0)
LOCATION = (0.76, 0.76, 0.76)


# --- the stage ----------------------------------------------------------------

def test_a_held_single_character_is_a_stage_named_correctly() -> None:
    steps, s = _series([CHANCE] * 4 + [ROTFLIP] * 6 + [SOLVED] * 4)
    assert S.first_stage(steps, s) == {"character": "rotflip", "step": 2500}


def test_the_other_character_is_named_when_it_is_the_one_alone() -> None:
    steps, s = _series([CHANCE] * 3 + [FLIP] * 3)
    assert S.first_stage(steps, s)["character"] == "flip"


def test_a_single_eval_is_not_a_stage() -> None:
    steps, s = _series([CHANCE] * 3 + [ROTFLIP] + [CHANCE] * 3)
    assert S.first_stage(steps, s) is None


def test_a_direct_jump_to_solved_has_no_stage() -> None:
    steps, s = _series([CHANCE] * 5 + [SOLVED] * 5)
    assert S.first_stage(steps, s) is None


def test_the_location_plateau_is_not_a_stage() -> None:
    """Observation 4's plateau raises all three characters together."""
    steps, s = _series([CHANCE] * 3 + [LOCATION] * 20 + [SOLVED] * 3)
    assert S.first_stage(steps, s) is None


def test_two_characters_high_is_not_a_single_stage() -> None:
    steps, s = _series([CHANCE] * 3 + [(1.0, 0.5, 1.0)] * 5)
    assert S.first_stage(steps, s) is None


def test_a_high_character_with_a_middling_other_is_not_alone() -> None:
    steps, s = _series([CHANCE] * 3 + [(0.5, 0.7, 1.0)] * 5)
    assert S.first_stage(steps, s) is None


def test_two_different_characters_in_a_row_do_not_join_into_one_stage() -> None:
    steps, s = _series([CHANCE, FLIP, ROTFLIP, CHANCE])
    assert S.first_stage(steps, s) is None


def test_series_are_read_from_the_logged_metric_names() -> None:
    rows = []
    for step, vals in zip((500, 1000), (CHANCE, ROTFLIP)):
        for n, v in zip(NAMES, vals):
            rows.append({"step": step, "metric_name": f"character_{n}", "value": v})
    rows.append({"step": 500, "metric_name": "accuracy", "value": 0.1})
    steps, s = S.character_series(pd.DataFrame(rows))
    assert list(steps) == [500, 1000]
    assert s["rotflip"][1] == 1.0 and s["flip"][1] == 0.5


# --- the registered check -------------------------------------------------------

def _row(arm, i, stage=None, leave=None):
    return {"run_id": f"{arm[1]}_{i}", "arm": list(arm), "leave_step": leave,
            "stage": {"character": stage, "step": 3000} if stage else None}


POS, IDN = ("D4", "position"), ("D4", "identity")


def _all_held():
    return ([_row(POS, i, "rotflip", 2000) for i in range(S.POSITION_SEEDS)]
            + [_row(IDN, i, None, 10000 if i < 2 else None)
               for i in range(S.IDENTITY_SEEDS)])


def test_held_when_every_prediction_holds() -> None:
    assert {v for v, _ in S.registered(_all_held()).values()} == {"held"}


def test_one_position_seed_learning_another_character_first_fails_q1() -> None:
    rows = _all_held()
    rows[0]["stage"]["character"] = "flip"
    assert S.registered(rows)["Q1"][0] == "failed"


def test_too_few_informative_seeds_is_undetermined_not_held() -> None:
    rows = _all_held()
    for r in rows[:2]:
        r["stage"] = None
    assert S.registered(rows)["Q1"][0] == "undetermined"


def test_exactly_the_minimum_informative_seeds_gives_a_verdict() -> None:
    rows = _all_held()
    rows[0]["stage"] = None
    assert S.registered(rows)["Q1"][0] == "held"


def test_any_identity_stage_fails_q2() -> None:
    rows = _all_held()
    rows[-1]["stage"] = {"character": "rotflip", "step": 4000}
    assert S.registered(rows)["Q2"][0] == "failed"


def test_q2_says_how_many_identity_seeds_left_chance() -> None:
    verdict, detail = S.registered(_all_held())["Q2"]
    assert verdict == "held" and f"2 of {S.IDENTITY_SEEDS} left chance" in detail


def test_missing_seeds_are_pending() -> None:
    rows = _all_held()[1:-1]
    v = S.registered(rows)
    assert v["Q1"][0] == "pending" and v["Q2"][0] == "pending"


def test_the_constants_are_the_ones_d060_registers() -> None:
    assert (S.STAGE_HIGH, S.STAGE_LOW, S.STAGE_MIN_EVALS) == (0.9, 0.6, 2)
    assert (S.POSITION_SEEDS, S.IDENTITY_SEEDS, S.MIN_INFORMATIVE) == (4, 8, 3)
    assert S.PREDICTED_FIRST == "rotflip" and S.EXPERIMENT == "E0.4"


# --- scanning -----------------------------------------------------------------

def _write_run(root, name, cfg, char_rows, done=True):
    d = root / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    if done:
        (d / "DONE").write_text("done\n", encoding="utf-8")
    rows = []
    for i, vals in enumerate(char_rows):
        for n, v in zip(NAMES, vals):
            rows.append({"step": 500 * (i + 1), "k": 8, "split": "iid_val", "depth": -1,
                         "breadth": -1, "metric_name": f"character_{n}", "value": v, "seed": 0})
    for s in range(50, 500 * len(char_rows) + 1, 50):
        rows.append({"step": s, "k": 4, "split": "train", "depth": -1, "breadth": -1,
                     "metric_name": "loss", "value": math.log(8), "seed": 0})
    pd.DataFrame(rows).to_parquet(d / "metrics.parquet")


def test_only_finished_e04_runs_are_read(tmp_path) -> None:
    root = tmp_path / "runs"
    base = {"family": "A", "factor": "d4", "substrate": "native"}
    _write_run(root, "why_e04_d4only_s100", {**base, "experiment": "E0.4"}, [ROTFLIP] * 3)
    _write_run(root, "famA_d2_d4only_s0", base, [ROTFLIP] * 3)
    _write_run(root, "why_e04_d4only_s101", {**base, "experiment": "E0.4"},
               [ROTFLIP] * 3, done=False)
    rows = S.scan(root)
    assert [r["run_id"] for r in rows] == ["why_e04_d4only_s100"]
    assert rows[0]["arm"] == ["D4", "position"]
    assert rows[0]["stage"]["character"] == "rotflip"


def test_main_registered_is_pending_with_no_runs(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    assert S.main(["--runs", str(tmp_path / "runs"), "--registered"]) == 2


# --- the configs --------------------------------------------------------------

E04 = sorted((REPO_ROOT / "configs/why").glob("e04_*.yaml"))


def test_both_e04_configs_exist() -> None:
    assert {p.name for p in E04} == {"e04_d4only.yaml", "e04_d4colour.yaml"}


@pytest.mark.parametrize("path", E04, ids=lambda p: p.name)
def test_an_e04_config_never_enters_the_sealed_2x2(path) -> None:
    """Fresh seeds of two registered arms. Held out by their step budget,
    which arm_of reads, and checked on the config as a run would store it."""
    cfg = cli.load_config(str(path), {})
    assert ER.arm_of(cfg) is None
    assert cfg["steps"] != ER.BASELINE["steps"]


@pytest.mark.parametrize("stem", ["famA_d2_d4only", "famA_d2_d4colour"])
def test_the_base_configs_are_2x2_arms_so_the_check_above_means_something(stem) -> None:
    cfg = cli.load_config(str(REPO_ROOT / f"configs/m3/{stem}.yaml"), {})
    assert ER.arm_of(cfg) is not None


@pytest.mark.parametrize("stem", ["d4only", "d4colour"])
def test_an_e04_config_differs_from_its_base_only_where_intended(stem) -> None:
    """Anything else changed would be a second variable in the experiment."""
    base = cli.load_config(str(REPO_ROOT / f"configs/m3/famA_d2_{stem}.yaml"), {})
    new = cli.load_config(str(REPO_ROOT / f"configs/why/e04_{stem}.yaml"), {})
    changed = {k for k in set(base) | set(new) if base.get(k) != new.get(k)}
    assert changed == {"steps", "lr_schedule_steps", "eval_every",
                       "log_characters", "experiment"}
    assert new["lr_schedule_steps"] == base["steps"], (
        "the learning rate must follow the 300000 step schedule the arms ran")
