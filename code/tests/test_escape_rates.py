"""A generator that replaces a hand-typed number has to be harder to fool
than the hand was.

Three things it must not do. It must not sort a run that is neither escaped
nor collapsed into one of those buckets, because that bucket is the only
thing standing between the paper's dichotomy and a quiet counterexample. It
must not count an unfinished run in either direction, because escape
arrives late and the one run that did so would have been recorded as a
failure at any earlier budget. And it must not put an arm of the registered
2x2 in the wrong cell, which would transpose the table while every count in
it stayed correct.
"""

from __future__ import annotations

import json

import pytest
import yaml

from loopvision.analysis import escape_rates as ER


def _cfg(**over):
    """A config that is the registered baseline arm unless told otherwise."""
    base = dict(ER.BASELINE)
    base.update({"family": "A", "breadths": [1, 2, 3, 4], "factor": "full",
                 "seed": 0})
    base.update(over)
    return base


def _run(tmp_path, name, *, done=True, acc=None, n_distinct=None,
         factor="full", substrate=None, seed=0, family="A", depths=(2,),
         **cfg_over):
    d = tmp_path / "runs" / name
    d.mkdir(parents=True)
    cfg = _cfg(family=family, depths=list(depths), factor=factor, seed=seed,
               **cfg_over)
    if substrate is not None:
        cfg["substrate"] = substrate
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    if done:
        (d / "DONE").write_text("done\n", encoding="utf-8")
    summary = {}
    if acc is not None:
        summary["final_accuracy"] = acc
    if n_distinct is not None:
        summary["n_distinct_predictions"] = n_distinct
    if summary:
        (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return d


def test_a_solved_run_is_escaped() -> None:
    assert ER.classify({"final_accuracy": 1.0})[0] == "escaped"
    assert ER.classify({"final_accuracy": 0.9999})[0] == "escaped"


def test_a_single_class_run_is_collapsed_and_says_so() -> None:
    """The measured path. n_distinct == 1 means the accuracy is a class
    frequency rather than a score, and that is a fact rather than a
    threshold."""
    out, why = ER.classify({"final_accuracy": 0.1254, "n_distinct_predictions": 1})
    assert out == "collapsed" and why == "measured"


def test_an_old_run_is_collapsed_by_accuracy_and_marked_inferred() -> None:
    """Runs finishing before 2026-09-08 carry no class count. Classifying
    them is an inference and the output has to say so, or a later reader
    treats it as a measurement."""
    out, why = ER.classify({"final_accuracy": 0.1254})
    assert out == "collapsed" and "inferred" in why


def test_a_run_between_the_two_is_neither(tmp_path) -> None:
    """The bucket the whole module exists for.

    A two bucket classifier would have to file this as escaped or
    collapsed, and the claim that nothing lands in between would survive
    its own counterexample.

    Mutation checked: making the final branch return "collapsed" fails here
    and nowhere else, because no real run is in this state today.
    """
    out, why = ER.classify({"final_accuracy": 0.5039, "n_distinct_predictions": 6})
    assert out == "neither"
    assert "6 distinct" in why


def test_a_mid_accuracy_run_with_no_class_count_is_neither() -> None:
    """0.5039 with no recorded count is not evidence of collapse. Falling
    back to the accuracy rule here would invent a classification for the
    one case where the data cannot support one."""
    assert ER.classify({"final_accuracy": 0.5039})[0] == "neither"


def test_a_run_with_no_accuracy_is_neither_rather_than_assumed() -> None:
    assert ER.classify({})[0] == "neither"


def test_the_escape_threshold_is_the_one_the_documents_already_used() -> None:
    """Inherited, not chosen here. A threshold that drifts inside a
    generator is a tuned threshold nobody can see."""
    assert ER.ESCAPE_TAU == 0.95
    assert ER.COLLAPSE_ACC == 0.20


def test_the_two_thresholds_do_not_meet() -> None:
    """There has to be a gap for `neither` to be reachable at all. If they
    met, every run would be escaped or collapsed by construction and the
    dichotomy would be an artefact of the classifier."""
    assert ER.COLLAPSE_ACC < ER.ESCAPE_TAU


def test_each_arm_of_the_registered_2x2_lands_in_its_own_cell() -> None:
    """Transposing the table would leave every count correct and every
    conclusion backwards. From experiment-substrate.md: d4_position and
    s3_identity are the built native arms, s3_position and d4_identity move
    each group onto the other medium."""
    assert ER.arm_of(_cfg(factor="d4")) == ("D4", "position")
    assert ER.arm_of(_cfg(factor="s3")) == ("S3", "identity")
    assert ER.arm_of(_cfg(factor="full", substrate="d4_colour")) == ("D4", "identity")
    assert ER.arm_of(_cfg(factor="full", substrate="s3_spatial")) == ("S3", "position")


@pytest.mark.parametrize("field,value", [
    ("d_model", 768),        # wide and big
    ("arch", "feedforward"),  # ffwd and ffwd_mp
    ("depths", [1]),          # the depth 1 donors
    ("prelude_blocks", 4),    # depth4, the parameter matched depth arm
    ("core_blocks", 4),
    ("steps", 1000000),       # the long budget vark runs
    ("init_from", "famA_d1_d4colour_s0"),  # the curriculum arms
    ("lr", 1e-3),             # D-066, the learning-rate grid
    ("weight_decay", 0.5),    # D-066, the weight-decay grid
    ("warmup", 500),
    ("early_stop_tau", 0.95), # D-065, any run that may stop early
    ("early_stop_window", 20000),
    ("pixel_scale", 2),       # D-067, glyphs spanning several patches
])
def test_a_variant_of_an_arm_is_not_that_arm(field, value) -> None:
    """The fault the first version of this module shipped with.

    Every one of these shares the factor and substrate of a registered arm
    and varies something experiment-substrate.md held fixed, so counting it
    into the table inflates the arm with a different experiment. The first
    version did exactly that and reported D4 acting on identity as 6 of 19.

    Mutation checked: dropping the baseline check from arm_of makes every
    case here return an arm.
    """
    cfg = _cfg(factor="full", substrate="d4_colour", **{field: value})
    assert ER.arm_of(cfg) is None, f"a run varying {field} entered the 2x2"


def test_a_sealed_seed_predating_the_new_keys_is_still_in_its_cell() -> None:
    """Every 2x2 seed was written before early stop and pixel scale existed,
    so their stored configs lack those keys. Absence must read as the
    default, or adding the keys would empty the sealed grid at a stroke."""
    cfg = _cfg(factor="full", substrate="d4_colour")
    for key in ("early_stop_tau", "early_stop_window", "pixel_scale"):
        cfg.pop(key)
    assert ER.arm_of(cfg) == ("D4", "identity")


def test_the_baseline_arm_itself_is_not_excluded() -> None:
    """The other direction. A baseline check that rejected everything would
    pass every test above and empty the table."""
    assert ER.is_baseline(_cfg(factor="d4"))
    assert ER.arm_of(_cfg(factor="d4")) is not None


def test_a_seed_is_not_part_of_the_baseline() -> None:
    """Arms differ from each other in factor and substrate and from their
    own seeds in nothing else, so seed must not be checked."""
    assert ER.is_baseline(_cfg(factor="d4", seed=4))


def test_the_four_arms_are_four_distinct_cells() -> None:
    """A mapping that sent two arms to one cell would silently pool them."""
    assert len(set(ER.SUBSTRATE_ARMS.values())) == 4


def test_a_config_outside_the_2x2_has_no_arm() -> None:
    """A factor and substrate pairing that names no arm is not in the table
    however baseline the rest of it is."""
    assert ER.arm_of(_cfg(factor="full", substrate="native")) is None
    assert ER.arm_of(_cfg(factor="full")) is None


def test_an_unfinished_run_is_counted_in_neither_direction(tmp_path) -> None:
    """famA_d2_d4only_s2 read 0.5039 at step 210000 and finished at 0.9999.
    Counting an unfinished run early would have been wrong in the direction
    that helped."""
    _run(tmp_path, "done_ok", acc=1.0, factor="d4", seed=0)
    _run(tmp_path, "still_going", done=False, acc=0.5, factor="d4", seed=1)
    r = ER.rates(ER.scan(tmp_path / "runs"), "arm")[("D4", "position")]
    assert r["escaped"] == 1
    assert r["finished"] == 1, "an unfinished run entered the denominator"
    assert r["unfinished"] == 1


def test_a_live_run_at_the_ceiling_is_still_not_counted(tmp_path) -> None:
    """Reading 1.0000 with steps left is not a finished result, and the
    tempting direction is exactly the one to refuse."""
    _run(tmp_path, "live_high", done=False, acc=1.0, factor="d4", seed=0)
    r = ER.rates(ER.scan(tmp_path / "runs"), "arm")[("D4", "position")]
    assert r["escaped"] == 0 and r["finished"] == 0 and r["unfinished"] == 1


def test_the_rate_counts_escaped_over_finished(tmp_path) -> None:
    for s, a in enumerate([1.0, 1.0, 0.1681, 0.1681]):
        _run(tmp_path, f"sp_s{s}", acc=a, n_distinct=1 if a < 0.5 else 8,
             factor="full", substrate="s3_spatial", seed=s)
    r = ER.rates(ER.scan(tmp_path / "runs"), "arm")[("S3", "position")]
    assert (r["escaped"], r["finished"]) == (2, 4)


def test_floors_are_reported_as_a_range_from_the_runs(tmp_path) -> None:
    """The floor values in the paper table are read off the runs, not
    typed, which is the whole point of the module."""
    for s, a in enumerate([0.1254, 0.1294, 0.1254]):
        _run(tmp_path, f"id_s{s}", acc=a, n_distinct=1, factor="full",
             substrate="d4_colour", seed=s)
    r = ER.rates(ER.scan(tmp_path / "runs"), "arm")[("D4", "identity")]
    assert min(r["floors"]) == pytest.approx(0.1254)
    assert max(r["floors"]) == pytest.approx(0.1294)


def test_seeds_of_one_config_group_into_one_cell(tmp_path) -> None:
    for s in range(3):
        _run(tmp_path, f"c_s{s}", acc=1.0, factor="d4", seed=s)
    assert len(ER.rates(ER.scan(tmp_path / "runs"), "cell")) == 1


def test_different_configs_do_not_share_a_cell(tmp_path) -> None:
    """Mutation checked: keying the cell on family alone merges these and
    reports one rate over two different experiments."""
    _run(tmp_path, "narrow", acc=1.0, factor="d4", seed=0)
    _run(tmp_path, "colour", acc=0.1254, n_distinct=1, factor="full",
         substrate="d4_colour", seed=0)
    assert len(ER.rates(ER.scan(tmp_path / "runs"), "cell")) == 2


def test_a_run_without_a_config_is_skipped_not_guessed(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "_index.json").mkdir()
    _run(tmp_path, "real", acc=1.0, factor="d4", seed=0)
    assert len(ER.scan(tmp_path / "runs")) == 1


def test_main_names_every_exception_rather_than_a_count(tmp_path) -> None:
    """A count of exceptions is useless. The run ids are what someone has
    to go and look at."""
    _run(tmp_path, "oddball", acc=0.5039, n_distinct=6, factor="d4", seed=0)
    assert ER.main(["--runs", str(tmp_path / "runs")]) == 0


def test_main_writes_the_numbers_for_checking(tmp_path) -> None:
    _run(tmp_path, "a_s0", acc=1.0, factor="d4", seed=0)
    out = tmp_path / "rates.json"
    ER.main(["--runs", str(tmp_path / "runs"), "--json", str(out)])
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["escape_tau"] == ER.ESCAPE_TAU
    assert d["arms"]["D4/position"]["escaped"] == 1
    assert d["neither"] == []


def test_main_refuses_a_missing_runs_directory(tmp_path) -> None:
    assert ER.main(["--runs", str(tmp_path / "nope")]) == 2


def _two_by_two(tmp_path):
    """A 2x2 with known counts: D4 position 2 of 2, D4 identity 0 of 3."""
    for s in range(2):
        _run(tmp_path, f"dp{s}", acc=1.0, factor="d4", seed=s)
    for s in range(3):
        _run(tmp_path, f"di{s}", acc=0.1254 if s else 0.1294, n_distinct=1,
             factor="full", substrate="d4_colour", seed=s)
    return ER.rates(ER.scan(tmp_path / "runs"), "arm")


def _doc(tmp_path, body, name="paper.md", annotate=True):
    """A document with counts on both sides of the markers.

    The outside counts carry a `count-ok` annotation by default. They are
    still not checked against the runs, which is what the tests below were
    written to hold; since 2026-09-11 they must merely declare a source, so
    the fixture declares one. `annotate=False` gives the pre-2026-09-11
    shape, which is now a failure and is tested as one below.
    """
    note = f"<!-- {ER.COUNT_OK_MARKER}: fixture -->" if annotate else ""
    p = tmp_path / name
    p.write_text(
        f"Prose before, mentioning 99 of 99 outside the region. {note}\n"
        f"<!-- {ER.BEGIN_MARKER} -->\n{body}\n<!-- {ER.END_MARKER} -->\n"
        f"Prose after, mentioning 77 of 77 also outside. {note}\n",
        encoding="utf-8",
    )
    return p


def test_a_document_matching_the_runs_passes(tmp_path) -> None:
    by_arm = _two_by_two(tmp_path)
    doc = _doc(tmp_path, "D4 position 2 of 2, D4 identity 0 of 3, "
                         "floor 0.1254 to 0.1294. 5 of 5 seeds finished.")
    r = ER.check_document(doc, by_arm)
    assert r["ok"], r


def test_a_stale_count_is_caught_and_named(tmp_path) -> None:
    """The failure that actually happens. The 2x2 in paper.md drifted three
    times on 2026-09-09, twice with the generator already written."""
    by_arm = _two_by_two(tmp_path)
    doc = _doc(tmp_path, "D4 identity 0 of 2, floor 0.1254 to 0.1294.")
    r = ER.check_document(doc, by_arm)
    assert not r["ok"]
    assert r["stale"][0]["quoted"] == "0 of 2"


def test_a_stale_floor_is_caught(tmp_path) -> None:
    """0.1249 and 0.1277 are the values the 8 September audit found in no
    stored file. A checker that read only the counts would pass them."""
    by_arm = _two_by_two(tmp_path)
    doc = _doc(tmp_path, "0 of 3, floor 0.1249 to 0.1277.")
    r = ER.check_document(doc, by_arm)
    assert not r["ok"]
    assert r["bad_floors"] == ["floor 0.1249 to 0.1277"]


def test_counts_outside_the_region_are_not_checked(tmp_path) -> None:
    """The region is opt-in, and it has to be, or every unrelated "one in
    three" in the document becomes a false failure that trains the reader
    to ignore the check.

    Mutation checked: returning the whole document from marked_region makes
    the 99 of 99 and 77 of 77 outside the markers fail here.
    """
    by_arm = _two_by_two(tmp_path)
    doc = _doc(tmp_path, "2 of 2 and 0 of 3.")
    r = ER.check_document(doc, by_arm)
    assert r["ok"], r["stale"]
    assert r["found"] == 2, "counts outside the markers were scanned"


def test_a_document_with_no_markers_is_an_error_not_a_pass(tmp_path) -> None:
    """Silently passing an unmarked document is the worst outcome: the
    check appears green while checking nothing at all."""
    by_arm = _two_by_two(tmp_path)
    p = tmp_path / "nomark.md"
    p.write_text("0 of 999 seeds, quite wrong.\n", encoding="utf-8")
    r = ER.check_document(p, by_arm)
    assert not r["ok"] and r["reason"] == "markers"


def test_the_medium_totals_are_allowed_counts(tmp_path) -> None:
    """The sentence under the table sums across groups, and those sums are
    not any single cell. Omitting them would flag correct prose.

    Mutation checked: dropping the per-medium loop from expected_claims
    fails here.
    """
    by_arm = _two_by_two(tmp_path)
    claims = ER.expected_claims(by_arm)
    assert (2, 2) in claims          # D4 on position, and the position total
    assert (0, 3) in claims          # D4 on identity, and the identity total
    assert (5, 5) in claims          # finished over total in the 2x2
    assert any("both groups" in w for ws in claims.values() for w in ws)


def test_main_check_returns_one_on_drift_and_zero_when_clean(tmp_path) -> None:
    """The exit code is what a pre-commit hook reads."""
    _two_by_two(tmp_path)
    runs = str(tmp_path / "runs")
    good = _doc(tmp_path, "2 of 2, 0 of 3, floor 0.1254 to 0.1294.", "ok.md")
    bad = _doc(tmp_path, "2 of 2, 0 of 9.", "bad.md")
    assert ER.main(["--runs", runs, "--check", str(good)]) == 0
    assert ER.main(["--runs", runs, "--check", str(bad)]) == 1


def test_main_check_refuses_a_missing_document(tmp_path) -> None:
    _two_by_two(tmp_path)
    assert ER.main(["--runs", str(tmp_path / "runs"),
                    "--check", str(tmp_path / "nope.md")]) == 2


# ---------------------------------------------------------------------------
# Counts in prose outside the markers, added 2026-09-11.
#
# The rule was already in CLAUDE.md and enforced by nothing, and it cost a
# real error: paper.md carried "7 of 7" hand typed when seven such runs
# existed, and there were eighteen by the time anyone read it. The marked
# region twenty lines above was checked on every commit and never drifted.
# ---------------------------------------------------------------------------


def _marked_doc(body: str) -> str:
    return ("intro\n\n<!-- escape-rates: begin -->\n\n"
            "| a | b |\n|---|---|\n| 1 of 2 | x |\n\n"
            "<!-- escape-rates: end -->\n\n" + body)


def test_a_bare_count_outside_the_markers_is_reported() -> None:
    from loopvision.analysis.escape_rates import unannotated_counts
    got = unannotated_counts(_marked_doc("Every finished failure, 7 of 7, collapses.\n"))
    assert len(got) == 1
    assert got[0]["quoted"] == "7 of 7"
    assert got[0]["line"] == 11


def test_counts_inside_the_markers_are_left_alone() -> None:
    """Those are the generator's own, checked against the runs a different
    way. Reporting them here would fire on every correct document."""
    from loopvision.analysis.escape_rates import unannotated_counts
    assert unannotated_counts(_marked_doc("nothing here\n")) == []


@pytest.mark.parametrize("body", [
    "Every finished failure, 7 of 7, collapses. <!-- count-ok: survey -->\n",
    "<!-- count-ok: survey -->\nEvery finished failure, 7 of 7, collapses.\n",
    "<!-- count-ok: survey -->\n\nEvery finished failure, 7 of 7, collapses.\n",
])
def test_an_annotated_count_is_allowed(body: str) -> None:
    """Same line, the line above, or two above. Plenty of counts in prose are
    fixed forever, and failing on those would make the check noise."""
    from loopvision.analysis.escape_rates import unannotated_counts
    assert unannotated_counts(_marked_doc(body)) == []


def test_the_annotation_does_not_reach_three_lines_down() -> None:
    """Otherwise one annotation silences a whole section."""
    from loopvision.analysis.escape_rates import unannotated_counts
    body = "<!-- count-ok: survey -->\n\n\nEvery finished failure, 7 of 7.\n"
    assert len(unannotated_counts(_marked_doc(body))) == 1


# ---------------------------------------------------------------------------
# Fisher exact on arm contrasts, D-050.
# ---------------------------------------------------------------------------


def test_fisher_reproduces_the_contrasts_that_retired_a_claim() -> None:
    """The three that matter, computed by hand on 2026-09-10 before this
    function existed. If it disagrees with them, it is wrong."""
    from loopvision.analysis.escape_rates import fisher_two_sided
    assert fisher_two_sided(1, 4, 0, 2) == pytest.approx(1.000, abs=5e-4)
    assert fisher_two_sided(1, 4, 0, 6) == pytest.approx(0.455, abs=5e-4)
    assert fisher_two_sided(2, 8, 0, 6) == pytest.approx(0.500, abs=5e-4)


def test_fisher_separates_what_is_actually_separable() -> None:
    """A positive control. Without it every assertion above passes on a
    function hardwired to return 1.0."""
    from loopvision.analysis.escape_rates import fisher_two_sided
    assert fisher_two_sided(10, 0, 0, 10) < 0.001
    assert fisher_two_sided(8, 2, 1, 9) < 0.05


def test_fisher_is_symmetric_and_bounded() -> None:
    from loopvision.analysis.escape_rates import fisher_two_sided
    assert fisher_two_sided(1, 4, 0, 6) == pytest.approx(fisher_two_sided(0, 6, 1, 4))
    for t in [(0, 0, 0, 0), (5, 0, 5, 0), (0, 5, 0, 5)]:
        assert 0.0 <= fisher_two_sided(*t) <= 1.0


def test_an_arm_with_few_seeds_is_flagged_where_its_rate_is_printed() -> None:
    """D-050's rule lives here rather than in a document, because the number
    is what gets copied into a claim and the warning has to travel with it."""
    from loopvision.analysis.escape_rates import SMALL_N, contrast
    by_arm = {"thin": {"escaped": 0, "finished": 2},
              "fat": {"escaped": 1, "finished": 5}}
    c = contrast(by_arm, "thin", "fat")
    assert c["small"] is True
    assert c["p"] == pytest.approx(1.000, abs=5e-4)
    assert SMALL_N == 5

    by_arm["thin"]["finished"] = 6
    assert contrast(by_arm, "thin", "fat")["small"] is False


def test_an_unannotated_document_now_fails_the_check(tmp_path) -> None:
    """The shape paper.md was in when it carried a stale 7 of 7. Same body,
    same marked region, the only difference being whether the outside counts
    declare where they came from."""
    by_arm = _two_by_two(tmp_path)
    body = "D4 position 2 of 2, D4 identity 0 of 3, floor 0.1254 to 0.1294."

    clean = ER.check_document(_doc(tmp_path, body, annotate=True), by_arm)
    assert clean["ok"], clean
    assert clean["unannotated"] == []

    loose = ER.check_document(
        _doc(tmp_path, body, name="loose.md", annotate=False), by_arm)
    assert not loose["ok"], "an unannotated count outside the markers passed"
    assert [u["quoted"] for u in loose["unannotated"]] == ["99 of 99", "77 of 77"]
    # and it is not a stale-count failure: those counts were never compared
    # against the runs, which is the older contract this must not break.
    assert loose["stale"] == []


def test_main_exits_nonzero_on_an_unannotated_count(tmp_path, capsys) -> None:
    """It has to fail the command, not just populate a dict, or CLAUDE.md's
    'run it before committing' does nothing."""
    _two_by_two(tmp_path)
    doc = _doc(tmp_path, "D4 position 2 of 2, D4 identity 0 of 3, "
                         "floor 0.1254 to 0.1294.", annotate=False)
    rc = ER.main(["--runs", str(tmp_path / "runs"), "--check", str(doc)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "UNANNOTATED" in out
    assert ER.COUNT_OK_MARKER in out


def test_contrast_names_arms_the_way_the_listing_prints_them(tmp_path) -> None:
    """The first version keyed by 2x2 cell, which cannot express the
    comparison D-050 is about: the looped d4colour arm against the two
    feedforward ones are three configs inside one cell."""
    rows = [
        {"run_id": "famA_d2_d4colour_s0", "outcome": "collapsed"},
        {"run_id": "famA_d2_d4colour_s1", "outcome": "collapsed"},
        {"run_id": "famA_d2_d4colour_ffwd_mp_s0", "outcome": "collapsed"},
        {"run_id": "famA_d2_d4colour_ffwd_mp_s3", "outcome": "escaped"},
        {"run_id": "famA_d2_d4colour_ffwd_mp_s4", "outcome": "unfinished"},
    ]
    by = ER.stem_rates(rows)
    assert by["famA_d2_d4colour"] == {"escaped": 0, "finished": 2}
    assert by["famA_d2_d4colour_ffwd_mp"] == {"escaped": 1, "finished": 2}
    c = ER.contrast(by, "famA_d2_d4colour_ffwd_mp", "famA_d2_d4colour")
    assert c["rate_a"] == "1 of 2" and c["rate_b"] == "0 of 2"
    assert c["small"] is True


# --------------------------------------------------------------------------
# Closure. "Nothing unfinished" is not closure: an empty runs directory has
# nothing unfinished too. On 2026-09-17 a GPU watch reported all-clear off
# exactly that shape, an ssh failure being indistinguishable from an empty
# card, so closure here has to be earned by a positive count per cell.
# --------------------------------------------------------------------------

def _row(arm, done: bool) -> dict:
    return {"run_id": f"r{id(arm)}{done}", "cell": arm, "arm": arm, "seed": 0,
            "done": done, "outcome": "escaped" if done else "unfinished",
            "why": "", "accuracy": 1.0 if done else None,
            "n_distinct": 8 if done else None, "modal": None}


def _full(finished: int = ER.SEED_MIN, unfinished: int = 0) -> list[dict]:
    rows = []
    for g in ER.GROUPS:
        for m in ER.MEDIA:
            rows += [_row((g, m), True) for _ in range(finished)]
            rows += [_row((g, m), False) for _ in range(unfinished)]
    return rows


def test_an_empty_runs_directory_is_not_a_closed_experiment() -> None:
    """The whole reason this function exists. Zero unfinished, zero finished,
    and announcing closure here would announce an experiment that never ran."""
    st = ER.cell_closure([])
    assert st["unfinished"] == 0
    assert st["closed"] is False
    assert len(st["short"]) == 4


def test_every_cell_at_its_registered_minimum_and_idle_is_closed() -> None:
    assert ER.cell_closure(_full())["closed"] is True


def test_one_cell_still_running_is_not_closed() -> None:
    rows = _full()
    rows.append(_row(("D4", "identity"), False))
    st = ER.cell_closure(rows)
    assert st["closed"] is False
    assert st["busy"] == [("D4", "identity")]


def test_one_cell_below_the_registered_minimum_is_not_closed() -> None:
    """D-042 requires ten seeds of any cell whose number reaches a headline,
    and every cell of this 2x2 does."""
    rows = [r for r in _full() if r["arm"] != ("S3", "position")]
    rows += [_row(("S3", "position"), True) for _ in range(ER.SEED_MIN - 1)]
    st = ER.cell_closure(rows)
    assert st["closed"] is False
    assert st["short"] == [("S3", "position")]


def test_runs_outside_the_2x2_do_not_count_toward_closure() -> None:
    """A feedforward or curriculum run is not a seed of a registered cell
    however many of them finish.

    The cell is left one seed short and then buried under 99 out-of-design
    runs. Asserting only "not closed" on the 99 alone would pass whether or
    not they were counted, because the other three cells would be empty
    either way, so the short cell has to be the only thing missing."""
    rows = [r for r in _full() if r["arm"] != ("D4", "identity")]
    rows += [_row(("D4", "identity"), True) for _ in range(ER.SEED_MIN - 1)]
    rows += [dict(_row(("D4", "identity"), True), arm=None) for _ in range(99)]

    st = ER.cell_closure(rows)
    assert st["closed"] is False
    assert st["short"] == [("D4", "identity")]
    assert st["cells"][("D4", "identity")]["finished"] == ER.SEED_MIN - 1


def test_the_closed_flag_exits_zero_only_when_closed(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(ER, "queued_2x2", lambda _c: ([], None))
    assert ER.report_closure(_full()) == 0
    assert "THE 2x2 HAS CLOSED" in capsys.readouterr().out
    assert ER.report_closure(_full(unfinished=1)) == 1
    assert "NOT closed" in capsys.readouterr().out
    assert ER.report_closure([]) == 1
    assert "NOT closed" in capsys.readouterr().out


def test_a_queued_seed_blocks_closure_even_though_it_has_no_run_directory(monkeypatch, capsys) -> None:
    """The nine deep seeds, D-043. A queued run has no directory, so scan
    cannot see it and every cell can read finished and idle while members of
    one are still waiting for a slot."""
    monkeypatch.setattr(ER, "queued_2x2",
                        lambda _c: (["famA_d2_d4colour_deep_s1"], None))
    assert ER.report_closure(_full()) == 1
    out = capsys.readouterr().out
    assert "still QUEUED" in out
    assert "famA_d2_d4colour_deep_s1" in out
    assert "HAS CLOSED" not in out


def test_an_unreadable_queue_refuses_to_declare_closure(monkeypatch, capsys) -> None:
    """"Nothing queued" and "could not ask" must not be the same answer. An
    ssh failure looking like an idle GPU is what cost a false all-clear."""
    monkeypatch.setattr(ER, "queued_2x2", lambda _c: ([], "the queue could not be read"))
    assert ER.report_closure(_full()) == 1
    out = capsys.readouterr().out
    assert "NOT declaring" in out
    assert "HAS CLOSED" not in out


def test_closure_is_declared_when_the_queue_is_readable_and_empty(monkeypatch, capsys) -> None:
    """The control. Without this the two tests above would pass against a
    function that can never declare closure at all."""
    monkeypatch.setattr(ER, "queued_2x2", lambda _c: ([], None))
    assert ER.report_closure(_full()) == 0
    assert "THE 2x2 HAS CLOSED" in capsys.readouterr().out


def test_queued_2x2_distinguishes_an_empty_queue_from_an_unreachable_one(monkeypatch) -> None:
    from pathlib import Path as P
    from loopvision.analysis import check_queue as Q
    monkeypatch.setattr(Q, "queued_text", lambda: "")
    runs, why = ER.queued_2x2(P("configs"))
    assert runs == []
    assert why is not None, "an unreadable queue must report a reason, not silence"


def test_a_queue_that_parses_badly_reports_a_reason_rather_than_nothing(monkeypatch) -> None:
    """A malformed qstat blob is not an empty queue. Swallowing the exception
    and returning [] would let a parser bug declare the experiment closed."""
    from pathlib import Path as P
    from loopvision.analysis import check_queue as Q
    monkeypatch.setattr(Q, "queued_text", lambda: "Job Id: 1\n  job_state = Q\n")

    def boom(*_a, **_k):
        raise ValueError("malformed")

    monkeypatch.setattr(Q, "headline_backlog", boom)
    runs, why = ER.queued_2x2(P("configs"))
    assert runs == []
    assert why is not None and "malformed" in why
