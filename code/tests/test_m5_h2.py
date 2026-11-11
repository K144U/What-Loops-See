"""The two registered H2 tests have to survive the ways they can quietly fail.

Three of those ways are specific to this instrument. Censoring removes exactly
the items that would embarrass a criterion, so it has to be counted rather than
dropped. The sign of the error decides the engineering fix, so a swapped
subtraction is not a cosmetic error. And items inside one run share weights and
share thresholds fitted on that run's own training split, so resampling items
instead of seeds would turn one model into hundreds of observations.

The Holm bound is the fourth. Six of the fourteen registered tests do not
exist, and the only defensible statements are the two that hold whatever those
six turn out to be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loopvision.analysis import m5_h2 as M5
from loopvision.instruments.m5_halting import CENSORED, CRITERIA


def _frame(n_seeds=5, items=40, offset=3.0, slope=0.0, k_tasks=(1, 2, 3, 4),
           noise=0.0, seed=0):
    """Items whose signed error is known by construction.

    k_hat is built as k_model + offset + slope * k_task, so both registered
    quantities have an answer the test can check against.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_seeds):
        for i in range(items):
            kt = k_tasks[i % len(k_tasks)]
            km = 2 + (i % 3)
            err = offset + slope * kt + rng.normal(0, noise)
            row = {"run_id": f"r{s}", "seed": s, "task_key": "A d(2,) b(1, 2)",
                   "k_task": kt, "k_model": km}
            for c in CRITERIA:
                row[f"k_hat_{c}"] = km + err
            rows.append(row)
    return pd.DataFrame(rows)


def test_censored_rows_leave_the_test_and_are_counted() -> None:
    """An item the model never solves has no signed error. Dropping it is
    right and dropping it silently is not, because those items are the hard
    ones rather than a random sample."""
    df = _frame(n_seeds=2, items=10)
    df.loc[df.index[:4], "k_model"] = CENSORED
    use, counts = M5.usable(df, "kl")
    assert counts["model_never_correct"] == 4
    assert counts["n_used"] == len(df) - 4
    assert (use.k_model != CENSORED).all()


def test_a_criterion_that_never_fires_is_counted_separately() -> None:
    """A criterion firing on a third of items and looking well calibrated on
    those is not well calibrated. Only the count shows it."""
    df = _frame(n_seeds=2, items=9)
    df.loc[df.index[:6], "k_hat_entropy"] = CENSORED
    use, counts = M5.usable(df, "entropy")
    assert counts["criterion_never_fired"] == 6
    assert counts["n_used"] == len(df) - 6


def test_a_criterion_that_never_fires_at_all_reports_nothing_rather_than_zero() -> None:
    """The degenerate case, which is a real possibility: Zhang et al. report
    standard instruments saturating at the fixed points trained loops
    converge to, and a criterion that never crosses its threshold produces
    no signed error at all. Reporting 0.0 there would enter the family as a
    perfectly calibrated criterion, which is the opposite of what happened.
    """
    df = _frame(n_seeds=3, items=10)
    df["k_hat_kl"] = CENSORED
    rows = {r["criterion"]: r for r in M5.analyse(df, n_boot=100)}
    kl = rows["kl"]
    assert kl["counts"]["n_used"] == 0
    assert kl["counts"]["criterion_never_fired"] == 30
    assert kl["signed_error"]["point"] is None
    assert kl["signed_error"]["p"] is None
    assert M5.holm_bound(kl["signed_error"]["p"]) == "no test"
    # The other three criteria are untouched by it.
    assert rows["update_norm"]["signed_error"]["point"] is not None


def test_censoring_on_either_side_removes_the_row() -> None:
    df = _frame(n_seeds=1, items=6)
    df.loc[df.index[0], "k_model"] = CENSORED
    df.loc[df.index[1], "k_hat_kl"] = CENSORED
    use, counts = M5.usable(df, "kl")
    assert counts["n_used"] == 4


def test_the_signed_error_recovers_a_known_offset() -> None:
    """Mutation checked: subtracting the other way round flips the sign and
    fails here. The sign is the whole point, since halting early and halting
    late have opposite fixes."""
    ps = M5.per_seed(_frame(n_seeds=4, offset=3.0), "update_norm")
    assert ps.median_error.median() == pytest.approx(3.0, abs=1e-9)


def test_a_negative_offset_stays_negative() -> None:
    ps = M5.per_seed(_frame(n_seeds=4, offset=-2.0), "update_norm")
    assert ps.median_error.median() == pytest.approx(-2.0, abs=1e-9)


def test_the_slope_recovers_a_known_dependence_on_k_task() -> None:
    """A criterion whose error grows with task depth is measuring the wrong
    thing and no threshold rescues it, so the slope has to be real."""
    ps = M5.per_seed(_frame(n_seeds=4, offset=1.0, slope=2.0), "kl")
    assert ps.slope.median() == pytest.approx(2.0, abs=1e-9)


def test_a_flat_criterion_has_zero_slope() -> None:
    """The falsification arm. A fit that always found a slope would support
    H2 no matter what the data said."""
    ps = M5.per_seed(_frame(n_seeds=4, offset=3.0, slope=0.0), "kl")
    assert ps.slope.median() == pytest.approx(0.0, abs=1e-9)


def test_a_seed_trained_on_one_depth_yields_no_slope() -> None:
    """A slope needs two distinct k*_task values. Returning 0.0 there would
    enter the bootstrap as evidence of no depth dependence."""
    ps = M5.per_seed(_frame(n_seeds=2, k_tasks=(3,)), "kl")
    assert ps.slope.isna().all()
    assert ps.median_error.notna().all()


def test_the_bootstrap_resamples_seeds_and_not_items() -> None:
    """Mutation checked: passing item level errors instead of per seed
    statistics makes n_seeds equal the item count and collapses the interval.

    Five seeds is five observations however many items each contributed.
    """
    df = _frame(n_seeds=5, items=200)
    rows = M5.analyse(df, n_boot=200)
    assert rows[0]["signed_error"]["n_seeds"] == 5, "items were treated as units"


def test_one_seed_gets_no_interval() -> None:
    """With one seed there is nothing to resample, so there must be no
    interval rather than a falsely narrow one built from items."""
    r = M5.bootstrap_stat(np.array([2.0]), n_boot=50)
    assert r["ci"] is None and r["p"] is None and r["n_seeds"] == 1


def test_the_interval_excludes_zero_for_a_real_offset() -> None:
    r = M5.bootstrap_stat(np.array([3.0, 3.2, 2.8, 3.1, 2.9]), n_boot=2000)
    lo, hi = r["ci"]
    assert lo > 0 and r["p"] < 0.05


def test_the_interval_covers_zero_for_a_calibrated_criterion() -> None:
    r = M5.bootstrap_stat(np.array([0.2, -0.3, 0.1, -0.1, 0.05]), n_boot=2000)
    lo, hi = r["ci"]
    assert lo < 0 < hi and r["p"] > 0.05


def test_the_p_value_never_claims_more_resolution_than_the_bootstrap_has() -> None:
    """10000 resamples cannot distinguish a probability below 1/10000, and
    printing 0.0000 would be a claim the procedure cannot make."""
    r = M5.bootstrap_stat(np.array([9.0, 9.1, 9.2, 8.9, 9.3]), n_boot=1000)
    assert r["p"] >= 1.0 / 1000


def test_the_sign_test_cannot_reach_alpha_at_five_seeds() -> None:
    """The reason both tests are reported. At five seeds the distribution
    free test cannot return below 0.0625, so it cannot reject at alpha 0.05
    even before Holm, however large the effect is."""
    assert M5.sign_test_floor(5) == pytest.approx(0.0625)
    assert M5.sign_test_floor(5) > M5.ALPHA
    p = M5.sign_test_p(np.array([5.0, 6.0, 7.0, 8.0, 9.0]))
    assert p == pytest.approx(0.0625)


def test_the_sign_test_can_reach_holm_at_ten_seeds() -> None:
    """Which is why the preregistration's ten seed rule for headline numbers
    is a statistical requirement here and not a stylistic one."""
    assert M5.sign_test_floor(10) < M5.ALPHA / M5.REGISTERED_FAMILY_SIZE


def test_the_sign_test_ignores_exact_zeros() -> None:
    p_with = M5.sign_test_p(np.array([1.0, 1.0, 1.0, 0.0, 0.0]))
    p_without = M5.sign_test_p(np.array([1.0, 1.0, 1.0]))
    assert p_with == pytest.approx(p_without)


def test_holm_rejects_only_below_alpha_over_the_whole_family() -> None:
    """The bound that holds whatever the six missing p-values are.

    Mutation checked: comparing against alpha rather than alpha/14 turns
    every p under 0.05 into a rejection and fails here.
    """
    edge = M5.ALPHA / M5.REGISTERED_FAMILY_SIZE
    assert M5.holm_bound(edge * 0.99) == "rejected under Holm"
    assert M5.holm_bound(edge) == "rejected under Holm"
    assert "undetermined" in M5.holm_bound(edge * 1.01)


def test_holm_says_undetermined_between_the_two_bounds() -> None:
    """The honest middle. A p of 0.01 is under alpha and over alpha/14, so
    whether Holm rejects it depends on tests that have not run."""
    assert "undetermined" in M5.holm_bound(0.01)


def test_holm_never_rejects_above_alpha() -> None:
    assert M5.holm_bound(0.051) == "not rejected"
    assert M5.holm_bound(0.9) == "not rejected"


def test_holm_handles_a_test_that_could_not_be_computed() -> None:
    assert M5.holm_bound(None) == "no test"


def _fake_run(root, name, family, depths, breadths, seed, n=12, offset=2.0):
    """A run directory holding the two files collect() reads."""
    import yaml

    d = root / name
    d.mkdir(parents=True)
    rows = []
    for i in range(n):
        km = 2 + (i % 3)
        row = {"item": i, "k": 32, "k_task": 1 + (i % 4), "k_model": km}
        for c in CRITERIA:
            row[f"k_hat_{c}"] = km + offset
        rows.append(row)
    pd.DataFrame(rows).to_parquet(d / "m5_halting.parquet", index=False)
    (d / "config.yaml").write_text(
        yaml.safe_dump({"family": family, "seed": seed, "depths": list(depths),
                        "breadths": list(breadths)}),
        encoding="utf-8",
    )
    return d


def test_collect_separates_runs_that_differ_only_in_breadth(tmp_path) -> None:
    """The fault M1 had, checked here before it can happen again.

    Grouping by family alone pooled three task definitions into one fit
    there. The key is derived from each run's own config, so it cannot fall
    behind a decision that changes a range.

    This pair differs only in breadth, so it guards only the breadth half of
    the key. The depth half needs its own fixture below: a key built from
    family and depth alone passes this test, which is exactly how that gap
    was found.
    """
    runs = tmp_path / "runs"
    _fake_run(runs, "old_split", "B", (1, 2), (6, 7), seed=0)
    _fake_run(runs, "new_split", "B", (1, 2), (12, 13), seed=1)
    df = M5.collect(runs)
    assert df.task_key.nunique() == 2, "two breadth ranges were pooled"


def test_collect_separates_runs_that_differ_only_in_depth(tmp_path) -> None:
    """The half the breadth fixture cannot see.

    Mutation checked: dropping breadth from the key fails the test above and
    passes this one, and dropping depth does the reverse. Both halves need a
    fixture or half the key is untested.
    """
    runs = tmp_path / "runs"
    _fake_run(runs, "shallow", "B", (1, 2), (12, 13), seed=0)
    _fake_run(runs, "deep", "B", (1, 2, 3, 4), (12, 13), seed=1)
    df = M5.collect(runs)
    assert df.task_key.nunique() == 2, "two depth ranges were pooled"


def test_collect_pools_runs_that_are_the_same_task(tmp_path) -> None:
    """The other direction. A key that separated everything would also pass
    both tests above while making every cell one seed."""
    runs = tmp_path / "runs"
    _fake_run(runs, "s0", "B", (1, 2), (12, 13), seed=0)
    _fake_run(runs, "s1", "B", (1, 2), (12, 13), seed=1)
    df = M5.collect(runs)
    assert df.task_key.nunique() == 1, "two seeds of one task were split apart"


def test_collect_reads_seeds_off_each_run(tmp_path) -> None:
    """Seeds are the unit of independence, so a collect that lost them would
    make every interval wrong in the same direction."""
    runs = tmp_path / "runs"
    for s in range(3):
        _fake_run(runs, f"r{s}", "A", (2,), (1, 2), seed=s)
    df = M5.collect(runs)
    assert sorted(df.seed.unique()) == [0, 1, 2]
    assert M5.analyse(df, n_boot=200)[0]["signed_error"]["n_seeds"] == 3


def test_collect_is_empty_rather_than_failing_when_nothing_has_run(tmp_path) -> None:
    """M5 needs a GPU pass and none has happened, so this is the state the
    module is in today. It has to return an empty frame with the right
    columns rather than raise."""
    (tmp_path / "runs").mkdir()
    df = M5.collect(tmp_path / "runs")
    assert len(df) == 0
    assert "k_model" in df.columns


def test_alpha_cannot_be_changed_from_the_command_line() -> None:
    """It is fixed by the preregistration. A flag that silently accepted a
    looser alpha is how a registered analysis stops being one."""
    assert M5.main(["--alpha", "0.10"]) == 2


def test_main_reports_nothing_to_do_rather_than_inventing_a_result(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    assert M5.main(["--runs", str(tmp_path / "runs")]) == 1


def test_k_model_against_k_task_is_descriptive_only() -> None:
    """The preregistration reports this and does not test it, so it must not
    acquire a p-value on the way through."""
    df = _frame(n_seeds=2, items=20)
    d = M5.k_model_vs_k_task(df)
    assert set(d) == {"n", "pearson"}


def test_the_descriptive_correlation_skips_censored_items() -> None:
    df = _frame(n_seeds=1, items=8)
    df.loc[df.index[:4], "k_model"] = CENSORED
    assert M5.k_model_vs_k_task(df)["n"] == 4


# ---------------------------------------------------------------------------
# Claim 3, the blind axis. Descriptive, and the ways it can quietly lie.
#
# The failure this set exists for is the one that reads as a result: a breadth
# slope of zero means "the criterion cannot see breadth" only if breadth was
# recorded and varied. A run passed before b_task existed, and a family whose
# breadth is fixed by design, both also produce nothing, and neither is
# evidence about the criterion. So the tests below check that the three stay
# apart, and that nothing here reaches the registered family.
# ---------------------------------------------------------------------------


def _frame_breadth(n_seeds=5, items=40, offset=2.0, bslope=0.0,
                   b_tasks=(4, 5, 6, 7, 8), k_task=1, noise=0.0, seed=0,
                   task_key="C d(1,) b(4, 5, 6, 7, 8)"):
    """Family C shape: breadth varies, depth is held at 1.

    k_hat is built as k_model + offset + bslope * b_task, so the breadth
    slope has an answer the test can check against.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_seeds):
        for i in range(items):
            bt = b_tasks[i % len(b_tasks)]
            km = 2 + (i % 3)
            err = offset + bslope * bt + rng.normal(0, noise)
            row = {"run_id": f"c{s}", "seed": s, "task_key": task_key,
                   "k_task": k_task, "b_task": bt, "k_model": km}
            for c in CRITERIA:
                row[f"k_hat_{c}"] = km + err
            rows.append(row)
    return pd.DataFrame(rows)


def test_a_planted_breadth_dependence_comes_back_out() -> None:
    """The positive control. Without it every other test here would pass on
    a breadth slope hardwired to zero."""
    df = _frame_breadth(n_seeds=5, bslope=0.5)
    ps = M5.per_seed(df, "kl")
    assert np.allclose(ps.slope_breadth.to_numpy(), 0.5)
    rows = M5.analyse(df, n_boot=200)
    row = next(r for r in rows if r["criterion"] == "kl")
    assert row["breadth_slope"]["point"] == pytest.approx(0.5)
    assert row["n_seeds_with_breadth_slope"] == 5


def test_the_breadth_slope_is_the_breadth_slope_and_not_the_depth_one() -> None:
    """A frame where the two axes carry different effects. If the fit ever
    reads k_task where it means b_task the two come back swapped, and with
    both slopes nonzero neither value can pass for the other."""
    # 30 items per seed crosses the 3 depths against the 5 breadths twice
    # over, so the two axes are balanced and neither slope can absorb the
    # other. An unbalanced count returns 1.98 and 0.475 here instead, which
    # is the marginal fit doing what _slope says it does rather than a bug.
    df = _frame_breadth(n_seeds=4, items=30, bslope=0.5)
    df["k_task"] = 1 + (np.arange(len(df)) % 3)
    df.loc[:, [f"k_hat_{c}" for c in CRITERIA]] = (
        df[[f"k_hat_{c}" for c in CRITERIA]].to_numpy()
        + 2.0 * df.k_task.to_numpy()[:, None])
    ps = M5.per_seed(df, "kl")
    assert np.allclose(ps.slope.to_numpy(), 2.0)
    assert np.allclose(ps.slope_breadth.to_numpy(), 0.5)


def test_breadth_recorded_but_constant_gives_no_slope_rather_than_zero() -> None:
    """Family B holds breadth fixed. A zero there would say the criterion is
    blind to breadth, which is a claim the data cannot support: the axis
    never moved, so there is nothing to be blind to."""
    df = _frame_breadth(n_seeds=3, b_tasks=(6,), bslope=0.5)
    ps = M5.per_seed(df, "kl")
    assert ps.slope_breadth.isna().all()
    axes = M5.axes_present(df)
    assert axes["breadth_levels"] == [6]
    assert axes["runs_without_breadth"] == []


def test_a_run_passed_before_b_task_existed_is_named_not_flattened() -> None:
    """The case that motivated the column. A parquet without b_task must
    read as an absent axis and must say which runs need a re-pass, because
    breadth cannot be recovered once the pass has ended."""
    df = _frame_breadth(n_seeds=3, bslope=0.5).drop(columns=["b_task"])
    ps = M5.per_seed(df, "kl")
    assert ps.slope_breadth.isna().all()
    axes = M5.axes_present(df.assign(b_task=np.nan))
    assert axes["breadth_levels"] == []
    assert axes["runs_without_breadth"] == ["c0", "c1", "c2"]


def test_collect_fills_a_missing_breadth_column_with_nan(tmp_path) -> None:
    """Read back from parquet rather than constructed in memory, because the
    column is missing on disk and that is where the compatibility has to
    hold. A sentinel such as 0 or -1 would fit as a real breadth level."""
    run = tmp_path / "old_run"
    run.mkdir()
    df = _frame_breadth(n_seeds=1, items=5).drop(columns=["b_task", "run_id",
                                                          "seed", "task_key"])
    df.to_parquet(run / "m5_halting.parquet", index=False)
    (run / "config.yaml").write_text(
        "family: C\ndepths: [1]\nbreadths: [4, 5]\nseed: 0\n", encoding="utf-8")
    got = M5.collect(tmp_path)
    assert "b_task" in got.columns
    assert got.b_task.isna().all()


def test_claim_3_needs_two_families_and_says_which_gave_which() -> None:
    """Neither family can answer it alone. Family B varies depth and fixes
    breadth, family C the reverse, so the contrast only exists once both
    have been passed and the report has to name the source of each half."""
    fam_b = _frame(n_seeds=4, slope=0.4, k_tasks=(1, 2, 3, 4, 5, 6))
    fam_b["b_task"] = 6
    fam_c = _frame_breadth(n_seeds=4, bslope=0.05)

    rows_b = M5.analyse(fam_b, n_boot=200)
    rows_c = M5.analyse(fam_c, n_boot=200)
    rep = M5.blind_axis([("B", rows_b), ("C", rows_c)])

    assert rep["has_depth"] and rep["has_breadth"]
    kl = rep["criteria"]["kl"]
    assert [e["task"] for e in kl["depth"]] == ["B"]
    assert [e["task"] for e in kl["breadth"]] == ["C"]
    assert kl["depth"][0]["point"] == pytest.approx(0.4)
    assert kl["breadth"][0]["point"] == pytest.approx(0.05)
    assert kl["breadth"][0]["levels"] == [4, 5, 6, 7, 8]


def test_with_no_breadth_anywhere_claim_3_refuses_and_explains(capsys) -> None:
    """An absent result, not a null one, and it has to read as one at the
    terminal. This is the state the repository was in until b_task existed."""
    fam_b = _frame(n_seeds=3, slope=0.4, k_tasks=(1, 2, 3, 4))
    rows_b = M5.analyse(fam_b.assign(b_task=np.nan), n_boot=200)
    rep = M5.blind_axis([("B", rows_b)])
    assert rep["has_breadth"] is False
    assert rep["runs_without_breadth"] == ["r0", "r1", "r2"]

    M5.print_blind_axis(rep)
    out = capsys.readouterr().out
    assert "NO BREADTH AXIS IS PRESENT" in out
    assert "absent result and not a flat one" in out
    assert "FORCE=1" in out


def test_claim_3_never_enters_the_registered_family(capsys) -> None:
    """The whole point of the labelling. A breadth slope printed next to
    eight registered tests is the thing that gets quoted as a ninth, so the
    heading must carry its status and no Holm verdict may appear under it."""
    assert M5.REGISTERED_FAMILY_SIZE == 14

    fam_c = _frame_breadth(n_seeds=8, bslope=0.5)
    rows = M5.analyse(fam_c, n_boot=500)
    rep = M5.blind_axis([("C", rows)])
    M5.print_blind_axis(rep)
    out = capsys.readouterr().out

    assert "DESCRIPTIVE" in out
    assert "post-hoc" in out
    # A planted effect this clean would clear any threshold, so if the
    # breadth slope were ever routed through holm_bound this is where the
    # verdict would surface.
    assert "rejected under Holm" not in out
    for entry in rep["criteria"].values():
        for e in entry["depth"] + entry["breadth"]:
            assert "holm" not in e


def test_a_repass_is_offered_only_where_breadth_actually_varies(capsys) -> None:
    """Family B fixes breadth, so re-passing it would spend a workq slot to
    record a constant. The config knows that and the parquet does not, so the
    two cases have to be told apart before a re-pass is recommended."""
    fixed = _frame(n_seeds=3, k_tasks=(1, 2, 3, 4)).assign(
        b_task=np.nan, cfg_breadth_levels=1)
    axes = M5.axes_present(fixed)
    assert axes["repass_would_add_an_axis"] == []
    assert axes["repass_would_add_nothing"] == ["r0", "r1", "r2"]

    rep = M5.blind_axis([("B", M5.analyse(fixed, n_boot=200))])
    M5.print_blind_axis(rep)
    out = capsys.readouterr().out
    assert "add no axis" in out
    assert "FORCE=1" not in out


def test_a_repass_is_offered_where_breadth_was_swept_and_lost(capsys) -> None:
    """The opposite case, and the one that costs a result if it goes unsaid:
    the run swept breadth, the pass predates b_task, and the axis is
    recoverable only by running it again."""
    swept = _frame_breadth(n_seeds=3, bslope=0.5).assign(
        b_task=np.nan, cfg_breadth_levels=5)
    axes = M5.axes_present(swept)
    assert axes["repass_would_add_an_axis"] == ["c0", "c1", "c2"]
    assert axes["repass_would_add_nothing"] == []

    rep = M5.blind_axis([("C", M5.analyse(swept, n_boot=200))])
    M5.print_blind_axis(rep)
    out = capsys.readouterr().out
    assert "FORCE=1" in out
    assert "missing an axis it could have" in out


def test_the_config_breadth_count_is_read_from_the_config(tmp_path) -> None:
    """The wiring the two tests above assume. They hand cfg_breadth_levels in
    ready made, so without this one collect could stop reading the config
    entirely and every re-pass recommendation would silently become the same
    recommendation."""
    for rid, breadths in (("swept", "[4, 5, 6, 7, 8]"), ("fixed", "[6]")):
        d = tmp_path / rid
        d.mkdir()
        df = _frame_breadth(n_seeds=1, items=5).drop(
            columns=["b_task", "run_id", "seed", "task_key"])
        df.to_parquet(d / "m5_halting.parquet", index=False)
        (d / "config.yaml").write_text(
            f"family: C\ndepths: [1]\nbreadths: {breadths}\nseed: 0\n",
            encoding="utf-8")

    got = M5.collect(tmp_path)
    levels = got.groupby("run_id").cfg_breadth_levels.max().to_dict()
    assert levels == {"swept": 5, "fixed": 1}

    axes = M5.axes_present(got)
    assert axes["repass_would_add_an_axis"] == ["swept"]
    assert axes["repass_would_add_nothing"] == ["fixed"]
