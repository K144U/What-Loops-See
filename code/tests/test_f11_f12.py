"""F11 and F12: the training curves show every seed and only the arm named.

F11's claim is that nothing was selected, so the ways it could lie are
picking up another arm's runs because the names share a prefix, dropping a
seed, drawing a per-cell accuracy in place of the stored overall one, and
inventing a reference level. F12 reads no data, so it is tested only for
drawing at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from loopvision.analysis.plots import f11_training as F11  # noqa: E402
from loopvision.analysis.plots import f12_schematic as F12  # noqa: E402


def _run(root: Path, name: str, overall=(0.12, 0.5, 1.0), cell=0.9) -> Path:
    d = root / name
    d.mkdir(parents=True)
    rows = []
    for i, v in enumerate(overall):
        step = (i + 1) * 1000
        rows.append(dict(step=step, k=8, split="iid_val", depth=-1, breadth=-1,
                         metric_name="accuracy", value=v, seed=0))
        rows.append(dict(step=step, k=8, split="iid_val", depth=2, breadth=1,
                         metric_name="accuracy", value=cell, seed=0))
        rows.append(dict(step=step, k=8, split="train", depth=-1, breadth=-1,
                         metric_name="loss", value=2.0, seed=0))
    pd.DataFrame(rows).to_parquet(d / "metrics.parquet")
    return d


def test_an_arm_takes_its_own_seeds_and_not_a_longer_arms(tmp_path) -> None:
    for n in ("famA_d2_d4colour_s0", "famA_d2_d4colour_s12", "famA_d2_d4colour_s3",
              "famA_d2_d4colour_curr_s0", "famA_d2_d4colour_curr_confirm_s0"):
        _run(tmp_path, n)
    (tmp_path / "famA_d2_d4colour_s9").mkdir()  # no metrics yet
    got = [p.name for p in F11.seeds(tmp_path, "famA_d2_d4colour")]
    assert got == ["famA_d2_d4colour_s0", "famA_d2_d4colour_s3", "famA_d2_d4colour_s12"]
    assert [p.name for p in F11.seeds(tmp_path, "famA_d2_d4colour_curr")] == [
        "famA_d2_d4colour_curr_s0"]


def test_the_curve_is_the_stored_overall_accuracy_not_a_cell(tmp_path) -> None:
    run = _run(tmp_path, "x_s0", overall=(0.12, 0.5, 1.0), cell=0.9)
    steps, values = F11.curve(run)
    assert steps == [1000, 2000, 3000] and values == [0.12, 0.5, 1.0]


def test_two_overall_values_at_one_step_are_refused(tmp_path) -> None:
    run = _run(tmp_path, "x_s0")
    df = pd.read_parquet(run / "metrics.parquet")
    pd.concat([df, df.iloc[[0]]]).to_parquet(run / "metrics.parquet")
    with pytest.raises(ValueError, match="more than one"):
        F11.curve(run)


def test_the_reference_levels_are_read_not_typed(tmp_path) -> None:
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps({"escape_tau": 0.93}), encoding="utf-8")
    reg = tmp_path / "registered.txt"
    reg.write_text("T3  held  5 of 8 E2a runs that left chance finished between "
                   "0.31 and 0.37\n", encoding="utf-8")
    assert F11.escape_level(rates) == 0.93
    assert F11.trap_band(reg) == (0.31, 0.37)


def test_a_registration_without_a_band_is_refused(tmp_path) -> None:
    reg = tmp_path / "registered.txt"
    reg.write_text("T3 held\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no registered band"):
        F11.trap_band(reg)


def test_the_figure_draws_every_seed_and_refuses_an_empty_arm(tmp_path) -> None:
    runs = tmp_path / "runs"
    for arm, _ in F11.ARMS:
        for s in range(3):
            _run(runs, f"{arm}_s{s}")
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps({"escape_tau": 0.95}), encoding="utf-8")
    reg = tmp_path / "registered.txt"
    reg.write_text("finished between 0.24 and 0.26", encoding="utf-8")
    out = F11.draw(runs, rates, reg, str(tmp_path / "f11"))
    assert {p.suffix for p in out} == {".pdf", ".png"}
    import matplotlib.pyplot as plt
    plt.close("all")
    for p in runs.glob(f"{F11.ARMS[2][0]}_s*"):
        for f in p.iterdir():
            f.unlink()
        p.rmdir()
    with pytest.raises(FileNotFoundError, match=F11.ARMS[2][0]):
        F11.draw(runs, rates, reg, str(tmp_path / "f11b"))
    plt.close("all")


def test_the_schematic_draws(tmp_path) -> None:
    out = F12.draw(str(tmp_path / "f12"))
    assert all(p.is_file() for p in out)
    import matplotlib.pyplot as plt
    plt.close("all")

