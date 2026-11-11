"""The D-066 to D-069 configs are their base configs plus exactly the
registered changes, and none of them can be counted where it does not belong.

Three ways a revision run could corrupt something already sealed: entering
the 2x2 (escape_rates.arm_of), entering D-062's tally (why_e1e2 selects by
run-name stem), or entering D-061's (curriculum_confirm, likewise). Each is
checked here against the real modules rather than a copy of their rules.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from loopvision.analysis import curriculum_confirm, escape_rates, why_e1e2
from loopvision.train import cli

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "configs" / "rev"

_spec = importlib.util.spec_from_file_location("mrc", ROOT / "scripts" / "make_revision_configs.py")
mrc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mrc)

CHANGES = mrc.changes()


def _load(name: str) -> dict:
    return yaml.safe_load((REV / f"{name}.yaml").read_text(encoding="utf-8"))


def test_every_registered_config_exists_and_nothing_else_does() -> None:
    on_disk = {p.stem for p in REV.glob("*.yaml")}
    assert on_disk == set(CHANGES), sorted(on_disk ^ set(CHANGES))
    assert len(CHANGES) == 12 + 2 + 2 + 30


@pytest.mark.parametrize("name", sorted(CHANGES))
def test_a_config_is_its_base_plus_exactly_the_registered_changes(name) -> None:
    base_name, delta, _ = CHANGES[name]
    want = mrc.base(base_name)
    want.update(delta)
    assert _load(name) == want


@pytest.mark.parametrize("name", sorted(CHANGES))
def test_no_revision_config_can_enter_the_sealed_2x2(name) -> None:
    assert escape_rates.arm_of(_load(name)) is None


@pytest.mark.parametrize("name", sorted(CHANGES))
def test_every_revision_config_loads_and_has_early_stop_on(name) -> None:
    cfg = cli.load_config(str(REV / f"{name}.yaml"), {})
    assert cfg["early_stop_tau"] == 0.95 and cfg["early_stop_window"] == 20000


def _stem(name: str) -> str:
    """A run of this config is named <config stem>_s<seed> for c1 to c3, and
    the config name already carries the seed for c4."""
    run_id = name if name.startswith("c4_") else f"{name}_s300"
    return why_e1e2.stem(run_id)


@pytest.mark.parametrize("name", sorted(CHANGES))
def test_no_run_name_collides_with_a_registered_tally(name) -> None:
    stem = _stem(name)
    assert stem not in set(why_e1e2.ARMS.values()), "would enter D-062"
    tables = list(curriculum_confirm.ROWS.values()) + list(curriculum_confirm.CLAIMS.values())
    registered = {a for pair in tables for a in pair}
    assert registered, "D-061's tables were not read, so this test proves nothing"
    assert stem not in registered, "would enter D-061"


@pytest.mark.parametrize("name", sorted(CHANGES))
def test_the_stem_survives_seed_stripping_intact(name) -> None:
    """stem() strips trailing digits and then any trailing '_' or 's', so an
    arm whose name ended in 's' would lose letters and could merge with
    another arm."""
    expected = name.rsplit("_s", 1)[0] if name.startswith("c4_") else name
    assert _stem(name) == expected


def test_each_c4_recipient_starts_from_the_donor_of_its_own_seed() -> None:
    for name in CHANGES:
        if not name.startswith("c4_"):
            continue
        cfg = _load(name)
        seed = name.rsplit("_s", 1)[1]
        assert cfg["init_from"].endswith(f"_s{seed}")
        assert cfg["confirmation"] is True
        assert cfg["init_from"].startswith(mrc.DONOR_STEM[name.split("_")[1]])
