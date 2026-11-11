"""k_min has to be read off a curve exactly as the preregistration defines it.

H1's whole contribution is a dissociation between two coefficients, so an
error in how k_min is extracted propagates straight into the verdict. The
definition has three edges that are easy to get wrong and impossible to
notice later: the threshold is a floor and not a strict inequality, the
range stops at 32 even though the sweep runs to 64, and a cell that never
reaches tau is censored at 33 rather than dropped or clamped.

The bootstrap has its own trap. Seeds are the unit of independence because
thirty cells inside one run share one set of weights, and a seed drawn
twice has to become two groups or the random intercept quietly merges them.
"""

from __future__ import annotations

import numpy as np
import pytest

from loopvision.analysis import m1_curves as M1

KS = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])


def test_k_min_is_the_first_k_reaching_tau() -> None:
    acc = np.array([0.1, 0.5, 0.89, 0.91, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert M1.k_min_from_curve(KS, acc) == 4


def test_tau_is_a_floor_not_a_strict_inequality() -> None:
    """Exactly 0.90 counts. A strict > would push every borderline cell one
    grid point later and bias the depth slope upward."""
    acc = np.full(12, 0.0)
    acc[KS == 3] = M1.TAU
    acc[KS > 3] = 1.0
    assert M1.k_min_from_curve(KS, acc) == 3


def test_a_cell_rescued_only_past_32_is_censored() -> None:
    """The sweep runs to 64 and the registered range stops at 32. A cell
    that first reaches tau at 48 is censored, not credited with 48."""
    acc = np.zeros(12)
    acc[KS >= 48] = 1.0
    assert M1.k_min_from_curve(KS, acc) == M1.CENSORED


def test_a_cell_that_never_reaches_tau_is_censored():
    assert M1.k_min_from_curve(KS, np.full(12, 0.5)) == M1.CENSORED


def test_censoring_value_is_outside_the_measurable_range() -> None:
    """33 must not be mistakable for a real k_min, or censored cells and
    slow cells become the same number in the fit."""
    assert M1.CENSORED > 32
    assert M1.TAU == 0.90


def test_unsorted_input_is_handled() -> None:
    """metrics.parquet has no guaranteed row order, and reading k_min off
    an unsorted curve would return whichever row happened to be first."""
    idx = np.array([5, 0, 3, 1, 2, 4, 6, 7, 8, 9, 10, 11])
    acc = np.zeros(12)
    acc[KS >= 6] = 1.0
    assert M1.k_min_from_curve(KS[idx], acc[idx]) == 6


def _frame(n_seeds=4, a=1.0, b=0.0, noise=0.0, seed=0):
    """Synthetic cells whose true coefficients are known."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    # A real per-seed offset, because a random intercept model fitted to
    # data with no seed level variance sits on the variance boundary and
    # the optimiser does not converge. findings.md records real seed
    # variance as large, so zero is the unrealistic case, not this.
    offsets = rng.normal(0, 1.5, size=n_seeds)
    for s in range(n_seeds):
        for d in (1, 2, 3, 4, 5, 6):
            for br in (12, 13, 14, 15, 16):
                k = a * d + b * br + 1.0 + offsets[s] + rng.normal(0, noise)
                rows.append({"run_id": f"r{s}", "family": "B", "seed": s,
                             "d_model": 384, "depth": d, "breadth": br,
                             "k_min": k, "censored": False})
    return pd.DataFrame(rows)


def test_fit_recovers_a_known_depth_slope() -> None:
    """Mutation checked: swapping x_depth and x_breadth in the formula, or
    dropping the log for the saturating fit, both break this."""
    r = M1.fit(_frame(a=2.0, b=0.0, noise=0.05), n_boot=200)
    assert r["a_depth"] == pytest.approx(2.0, abs=0.1)
    assert r["b_breadth"] == pytest.approx(0.0, abs=0.1)


def test_fit_recovers_a_known_breadth_slope() -> None:
    """The dissociation is the contribution, so a breadth effect must be
    detectable when it is really there. A fit that always reported b near
    zero would support H1 no matter what the data said."""
    r = M1.fit(_frame(a=0.0, b=0.5, noise=0.05), n_boot=200)
    assert r["b_breadth"] == pytest.approx(0.5, abs=0.1)


def test_the_breadth_interval_covers_zero_when_breadth_does_nothing() -> None:
    r = M1.fit(_frame(a=1.0, b=0.0, noise=0.3), n_boot=400)
    lo, hi = r["b_ci"]
    assert lo < 0 < hi


def test_the_depth_interval_excludes_zero_when_depth_matters() -> None:
    r = M1.fit(_frame(a=1.0, b=0.0, noise=0.3), n_boot=400)
    lo, hi = r["a_ci"]
    assert lo > 0


def test_bootstrap_resamples_seeds_not_cells() -> None:
    """With one seed there is no seed to resample, so there must be no
    interval rather than a falsely narrow one built from cells."""
    r = M1.fit(_frame(n_seeds=1), n_boot=50)
    assert r["a_ci"] is None and r["n_seeds"] == 1


def test_a_repeated_seed_becomes_a_distinct_group() -> None:
    """If a seed drawn twice kept its label the random intercept would
    merge the copies and the bootstrap would understate the spread."""
    df = _frame(n_seeds=3, noise=0.2)
    r = M1.fit(df, n_boot=300, seed=1)
    assert r["n_boot_ok"] > 150, "most bootstrap fits should succeed"
    assert r["a_ci"][0] < r["a_depth"] < r["a_ci"][1]


def test_tau_cannot_be_changed_from_the_command_line() -> None:
    """It is fixed by the preregistration. A flag that silently accepted a
    different threshold is how a registered analysis stops being one."""
    assert M1.main(["--tau", "0.8"]) == 2


def _fake_run(root, name, family, depths, breadths, ks, acc_fn, seed=0,
              extra=None):
    """A run directory with the two files collect() actually reads.

    `extra` lands on top of the config, for a run trained differently or
    for the full resolved config a trainer stores."""
    import json

    import pandas as pd

    d = root / name
    d.mkdir(parents=True)
    rows = []
    for dep in depths:
        for br in breadths:
            for k in ks:
                rows.append({"metric_name": "sweep_accuracy", "k": k,
                             "depth": dep, "breadth": br,
                             "value": acc_fn(dep, br, k)})
    pd.DataFrame(rows).to_parquet(d / "metrics.parquet", index=False)
    (d / "summary.json").write_text(json.dumps({
        "final_accuracy": 1.0,
        "config": {"family": family, "seed": seed, "d_model": 384,
                   "depths": list(depths), "breadths": list(breadths),
                   **(extra or {})},
    }), encoding="utf-8")
    return d


def test_collect_separates_runs_from_different_task_definitions(tmp_path) -> None:
    """The fault this caught in the real data.

    Every breadth 6 to 9 cell came from runs D-026 invalidated, and the
    first version of collect pooled them with the current family B curve
    runs into one H1 fit. Grouping is by the task a run was trained on,
    derived from its own config, so it cannot fall behind the way a list
    of run ids would.

    Mutation checked: grouping by family alone collapses these two into
    one key and fails here.
    """
    runs = tmp_path / "runs"
    ks = [1, 2, 4, 8]
    always = lambda d, b, k: 1.0
    _fake_run(runs, "old_split", "B", (1, 2), (6, 7), ks, always, seed=0)
    _fake_run(runs, "new_split", "B", (1, 2), (12, 13), ks, always, seed=1)

    df = M1.collect(runs)
    assert df.task_key.nunique() == 2, "two task definitions were pooled"
    keys = sorted(df.task_key.unique())
    assert "b(6, 7)" in keys[1] or "b(6, 7)" in keys[0]


def test_collect_separates_runs_that_differ_only_in_depth(tmp_path) -> None:
    """The half the fixture above cannot see.

    Both runs there differ only in breadth, so a key built from family and
    depth alone separates them and the test passes while half the key is
    unguarded. Found on 2026-09-09 by mutating the equivalent key in M5 and
    watching every test still pass. The same gap was here.
    """
    runs = tmp_path / "runs"
    ks = [1, 2, 4, 8]
    always = lambda d, b, k: 1.0
    _fake_run(runs, "shallow", "B", (1, 2), (12, 13), ks, always, seed=0)
    _fake_run(runs, "deep", "B", (1, 2, 3, 4), (12, 13), ks, always, seed=1)

    df = M1.collect(runs)
    assert df.task_key.nunique() == 2, "two depth ranges were pooled"


def test_collect_separates_setups_that_share_a_task(tmp_path) -> None:
    """The fault this caught in the real data, on 2026-10-22.

    Family B's curve pooled ten arms that share its task, feedforward and
    untied baselines among them, because only the task was keyed. A
    feedforward run's k_min is 1 at every depth, so pooling it flattened the
    depth curve the figure drew. D-074.
    """
    runs = tmp_path / "runs"
    ks = [1, 2, 4, 8]
    always = lambda d, b, k: 1.0
    _fake_run(runs, "looped_s0", "B", (1, 2), (12, 13), ks, always, seed=0)
    _fake_run(runs, "ffwd_s0", "B", (1, 2), (12, 13), ks, always, seed=0,
              extra={"arch": "feedforward"})

    df = M1.collect(runs)
    assert df.task_key.nunique() == 1
    assert df.setup.nunique() == 2, "a baseline was pooled into the curve"


def test_a_jitter_width_or_a_model_width_is_a_different_setup() -> None:
    """The two kinds of arm, besides the baselines, that sat in the pool."""
    base = {"family": "B", "seed": 0, "depths": [1, 2], "breadths": [12, 13]}
    key = M1.setup_key(base)
    assert M1.setup_key({**base, "jitter_sigma": 0.0}) != key
    assert M1.setup_key({**base, "d_model": 256}) != key


def test_repeats_of_one_setup_share_a_key() -> None:
    """A key that split one arm would refuse every summary, so the fields
    that differ between repeats must not count: the seed, the data loader's
    worker count, which seed of the donor arm a recipient started from, and
    a key the run predates, which it trained at its default."""
    a = {"family": "A", "seed": 0, "num_workers": 3,
         "init_from": "famA_d1_d4colour_s0"}
    b = {"family": "A", "seed": 7, "num_workers": 0,
         "init_from": "famA_d1_d4colour_s7", "substrate": "native"}
    assert M1.setup_key(a) == M1.setup_key(b)


def test_a_different_donor_arm_is_a_different_setup() -> None:
    a = {"family": "A", "init_from": "famA_d1_d4colour_s3"}
    b = {"family": "A", "init_from": "famA_d1_s3only_s3"}
    assert M1.setup_key(a) != M1.setup_key(b)


def test_the_summary_refuses_a_task_two_setups_share(tmp_path) -> None:
    """F2 reads one entry per task and draws it as one curve."""
    runs = tmp_path / "runs"
    ks = [1, 2, 4, 8]
    always = lambda d, b, k: 1.0
    _fake_run(runs, "looped_s0", "B", (1, 2), (12, 13), ks, always, seed=0)
    _fake_run(runs, "wide_s0", "B", (1, 2), (12, 13), ks, always, seed=0,
              extra={"d_model": 768})
    with pytest.raises(ValueError, match="different setups"):
        M1.curve_summary(M1.collect(runs))


def test_main_writes_no_summary_that_pools_two_setups(tmp_path) -> None:
    runs = tmp_path / "runs"
    ks = [1, 2, 4, 8]
    always = lambda d, b, k: 1.0
    _fake_run(runs, "looped_s0", "B", (1, 2), (12, 13), ks, always, seed=0)
    _fake_run(runs, "ffwd_s0", "B", (1, 2), (12, 13), ks, always, seed=0,
              extra={"arch": "feedforward"})
    summary = tmp_path / "s.json"
    rc = M1.main(["--runs", str(runs), "--summary", str(summary), "--no-fit",
                  "--out", str(tmp_path / "k.parquet")])
    assert rc == 2 and not summary.exists()


def _registered(tmp_path, d_model=384):
    """A config file, and the resolved config a run trained from it stores."""
    import yaml

    from loopvision.train.cli import load_config

    p = tmp_path / f"arm_{d_model}.yaml"
    p.write_text(yaml.safe_dump({"family": "B", "d_model": d_model,
                                 "depths": [1, 2], "breadths": [12, 13],
                                 "arch": "looped", "k_schedule": "sampled"}),
                 encoding="utf-8")
    return p, load_config(str(p), {})


def test_config_keeps_only_the_arm_it_trains(tmp_path) -> None:
    """How F2's summary is made: name the registered configs, and the
    baselines and other widths sharing the task drop out."""
    import json

    runs = tmp_path / "runs"
    ks = [1, 2, 4, 8]
    p, cfg = _registered(tmp_path)
    for s in (0, 1):
        _fake_run(runs, f"arm_s{s}", "B", (1, 2), (12, 13), ks,
                  lambda d, b, k: 1.0 if k >= d else 0.0, seed=s,
                  extra={**cfg, "seed": s})
    _fake_run(runs, "ffwd_s0", "B", (1, 2), (12, 13), ks,
              lambda d, b, k: 1.0, seed=0, extra={**cfg, "arch": "feedforward"})
    summary = tmp_path / "s.json"
    rc = M1.main(["--runs", str(runs), "--summary", str(summary), "--no-fit",
                  "--out", str(tmp_path / "k.parquet"), "--config", str(p)])
    assert rc == 0
    (task,) = json.loads(summary.read_text(encoding="utf-8"))["tasks"].values()
    assert task["runs"] == 2, "the feedforward run was kept"
    assert [r["median"] for r in task["axes"]["depth"]] == [1.0, 2.0]


def test_a_config_that_trained_no_run_is_refused(tmp_path) -> None:
    """Otherwise a typo in a config path selects nothing, and with nothing
    selected from one arm the figure loses that arm."""
    runs = tmp_path / "runs"
    p, cfg = _registered(tmp_path, d_model=384)
    other, _ = _registered(tmp_path, d_model=512)
    _fake_run(runs, "arm_s0", "B", (1, 2), (12, 13), [1, 2, 4, 8],
              lambda d, b, k: 1.0, seed=0, extra=cfg)
    rc = M1.main(["--runs", str(runs), "--no-fit", "--out",
                  str(tmp_path / "k.parquet"), "--config", str(p),
                  "--config", str(other)])
    assert rc == 2


def test_collect_reads_k_min_off_the_stored_sweep(tmp_path) -> None:
    """End to end on the real file layout, since collect is what feeds
    every number in the H1 table."""
    runs = tmp_path / "runs"
    ks = [1, 2, 4, 8]

    def acc(depth, breadth, k):
        # Needs k >= depth to clear tau. Breadth is irrelevant, which is
        # the H1 prediction, so the fixture cannot accidentally support it.
        return 1.0 if k >= depth else 0.0

    _fake_run(runs, "r0", "B", (1, 2, 4), (12, 13), ks, acc, seed=0)
    df = M1.collect(runs)
    got = df.set_index(["depth", "breadth"]).k_min
    assert got[(1, 12)] == 1
    assert got[(2, 12)] == 2
    assert got[(4, 13)] == 4
    assert not df.censored.any()


def test_a_seed_drawn_twice_becomes_two_groups() -> None:
    """The invariant the bootstrap rests on, asserted directly.

    Mutation checked: dropping the relabelling survives every statistical
    test in this file, because merging a repeated draw barely moves the
    slope and only narrows the interval. It fails here immediately.
    """
    df = _frame(n_seeds=3, noise=0.2)
    by_seed = {s: df[df.seed == s] for s in df.seed.unique()}
    pick = [0, 0, 2]
    sub = M1.resample_frame(by_seed, pick)
    assert sub.seed.nunique() == len(pick), "a repeated draw was merged"
    assert len(sub) == sum(len(by_seed[s]) for s in pick)


def test_resampling_preserves_every_row_of_each_draw() -> None:
    """A resample that silently dropped rows would shrink the data as well
    as merging it, and the two faults would partly cancel in the output."""
    df = _frame(n_seeds=2, noise=0.1)
    by_seed = {s: df[df.seed == s] for s in df.seed.unique()}
    sub = M1.resample_frame(by_seed, [1, 1, 1])
    assert len(sub) == 3 * len(by_seed[1])
    assert sub.seed.nunique() == 3
