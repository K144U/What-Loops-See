"""Confirmation seeds are held out of the registered 2x2.

The four confirmation configs differ from their baselines in nothing but the
seed. That is what makes them confirmations, and it is also what makes
`arm_of` put them straight into the sealed grid unless something stops it.
Five fresh seeds of `famA_d2_d4colour` would take D4/identity from 40 to 45,
move identity off 15 of 50, and change the Fisher p of 0.0011 that
`paper.md` Section 3b was revised around on 2026-09-21 under D-052.

A confirmation that joins the cell it is meant to test also stops being a
test of that number and becomes part of it, so pooling fails twice over: it
breaks a sealed result AND produces no confirmation.
"""

from __future__ import annotations

import pytest

from loopvision.analysis import escape_rates as E


def _cfg(**over) -> dict:
    """A config that IS a registered 2x2 baseline, so only the hold-out can
    keep it out of the grid. Anything less and these tests would pass for
    the wrong reason."""
    cfg = dict(E.BASELINE)
    cfg.update({"factor": "full", "substrate": "d4_colour", "seed": 30})
    cfg.update(over)
    return cfg


def test_the_fixture_really_is_a_2x2_member_without_the_marker() -> None:
    """The control for every test below. If the fixture were not a grid
    member to begin with, a hold-out that did nothing would still pass."""
    assert E.is_baseline(_cfg())
    assert E.arm_of(_cfg()) == ("D4", "identity")


def test_a_confirmation_seed_is_not_in_the_grid() -> None:
    assert E.arm_of(_cfg(confirmation=True)) is None


def test_a_confirmation_seed_still_knows_which_cell_it_tests() -> None:
    """Held out is not the same as unattributable. A confirmation seed that
    nothing can tie back to a cell cannot confirm anything."""
    assert E.confirmation_arm_of(_cfg(confirmation=True)) == ("D4", "identity")


def test_an_ordinary_run_is_not_a_confirmation() -> None:
    assert E.confirmation_arm_of(_cfg()) is None


@pytest.mark.parametrize("val", [False, None])
def test_false_and_null_are_ordinary_data(val) -> None:
    """These say "not a confirmation" and must not be held out, or a real
    seed silently vanishes from a registered cell."""
    assert not E.is_confirmation(_cfg(confirmation=val))
    assert E.arm_of(_cfg(confirmation=val)) == ("D4", "identity")


@pytest.mark.parametrize("val", [True, "yes", 1, "true", 0, ""])
def test_anything_other_than_false_is_held_out(val) -> None:
    """The two mistakes are not symmetric. A seed wrongly admitted changes a
    sealed number silently; a seed wrongly held out fails the next --closed
    against its registered minimum. So the ambiguous cases hold out."""
    assert E.is_confirmation(_cfg(confirmation=val))
    assert E.arm_of(_cfg(confirmation=val)) is None


def test_a_run_is_never_in_both_populations() -> None:
    """confirmation_tally adds `arm` to one bucket and `confirm_arm` to the
    other, so a run reaching both would be counted twice and the hold-out
    would not be a hold-out."""
    for cfg in (_cfg(), _cfg(confirmation=True)):
        assert (E.arm_of(cfg) is None) != (E.confirmation_arm_of(cfg) is None)


def test_a_confirmation_seed_cannot_change_cell_closure() -> None:
    """The regression guard, in miniature. Closure counts only `arm`, so a
    held-out seed must leave all four cell counts exactly where they were."""
    sealed = [{"arm": ("D4", "identity"), "confirm_arm": None, "done": True,
               "outcome": "escaped"}] * 10
    before = E.cell_closure(sealed)
    held = sealed + [{"arm": None, "confirm_arm": ("D4", "identity"),
                      "done": True, "outcome": "escaped"}] * 5
    after = E.cell_closure(held)
    assert after["cells"] == before["cells"]
    assert after["finished"] == before["finished"] == 10


def test_the_two_populations_are_tallied_apart() -> None:
    rows = (
        [{"arm": ("D4", "identity"), "confirm_arm": None, "done": True,
          "outcome": "escaped"}] * 3
        + [{"arm": ("D4", "identity"), "confirm_arm": None, "done": True,
            "outcome": "collapsed"}] * 7
        + [{"arm": None, "confirm_arm": ("D4", "identity"), "done": True,
            "outcome": "escaped"}] * 2
        + [{"arm": None, "confirm_arm": ("D4", "identity"), "done": False,
            "outcome": "unfinished"}]
    )
    t = E.confirmation_tally(rows)
    assert t["sealed"][("D4", "identity")] == {"finished": 10, "escaped": 3}
    assert t["held"][("D4", "identity")] == {
        "finished": 2, "escaped": 2, "unfinished": 1}


def test_an_empty_register_claims_nothing(capsys) -> None:
    rc = E.report_confirmation([])
    assert rc == 2
    assert "nothing is claimed" in capsys.readouterr().out.lower()


def test_a_disagreeing_confirmation_is_reported_as_a_failure(capsys) -> None:
    """The preregistration requires a failed confirmation to be reported as
    a failure, so the exit code has to carry it. A sealed cell at 3 of 40
    against a held-out 10 of 10 is about as loud as this can get."""
    rows = (
        [{"arm": ("D4", "identity"), "confirm_arm": None, "done": True,
          "outcome": "escaped"}] * 3
        + [{"arm": ("D4", "identity"), "confirm_arm": None, "done": True,
            "outcome": "collapsed"}] * 37
        + [{"arm": None, "confirm_arm": ("D4", "identity"), "done": True,
            "outcome": "escaped"}] * 10
    )
    rc = E.report_confirmation(rows)
    out = capsys.readouterr().out
    assert rc == 1, "a contradicted confirmation must not exit 0"
    assert "CONFIRMATION HAS FAILED" in out


def test_quiet_is_not_reported_as_confirmed(capsys) -> None:
    """Five seeds against a rate of three in ten cannot produce a
    significant p in the agreeing direction, so the absence of a complaint
    must not be printed as though the claim had been confirmed."""
    rows = (
        [{"arm": ("D4", "identity"), "confirm_arm": None, "done": True,
          "outcome": "escaped"}] * 12
        + [{"arm": ("D4", "identity"), "confirm_arm": None, "done": True,
            "outcome": "collapsed"}] * 28
        + [{"arm": None, "confirm_arm": ("D4", "identity"), "done": True,
            "outcome": "escaped"}] * 1
        + [{"arm": None, "confirm_arm": ("D4", "identity"), "done": True,
            "outcome": "collapsed"}] * 4
    )
    rc = E.report_confirmation(rows)
    out = capsys.readouterr().out
    assert rc == 0
    assert "weaker than confirmation" in out
    assert "confirmed" not in out.lower().replace("confirmation", "")
