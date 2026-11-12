"""F13 to F16 and the tiles of F12: the illustrations say only what is true.

These figures teach the premise, so the ways they could mislead are the ones a
reader cannot check by eye: a worked answer that is not the generator's label,
an example picked because it looked good, a tile that is not the renderer's, a
count that is not the stored one, and a sentence in a panel that the group's
arithmetic does not support. Each test below is aimed at one of those.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from loopvision.analysis.plots import f12_schematic as F12  # noqa: E402
from loopvision.analysis.plots import f13_anatomy as F13  # noqa: E402
from loopvision.analysis.plots import f14_renderings as F14  # noqa: E402
from loopvision.analysis.plots import f15_rescue as F15  # noqa: E402
from loopvision.analysis.plots import f16_trap as F16  # noqa: E402
from loopvision.analysis.plots import pictures as P  # noqa: E402
from loopvision.data import dataset as D  # noqa: E402
from loopvision.data import groups as G  # noqa: E402

D4 = G.subgroup_members("d4")


def _close():
    import matplotlib.pyplot as plt
    plt.close("all")


# Pictures -----------------------------------------------------------------

def test_moves_are_named_by_their_code() -> None:
    assert P.d4_words(0) == "do nothing"
    assert P.d4_words(G.element_index((1, 0), G.S3_ELEMENTS[0])) == "a quarter turn"
    assert P.d4_words(G.element_index((0, 1), G.S3_ELEMENTS[0])) == "a flip"
    assert (P.d4_words(G.element_index((3, 1), G.S3_ELEMENTS[0]))
            == "a flip, then three quarter turns")


def test_f12_shows_the_outline_change_its_text_claims() -> None:
    assert F12.outline(F12.IDENTITY, "native") != F12.outline(F12.QUARTER_TURN, "native")
    assert F12.outline(F12.IDENTITY, "d4_colour") == F12.outline(F12.QUARTER_TURN, "d4_colour")
    assert G.d4_of(F12.QUARTER_TURN) == (1, 0)


# F13 ----------------------------------------------------------------------

def test_the_worked_answer_is_the_generators_label() -> None:
    for i in range(20):
        s = D.generate("A", i, F13.SPLIT, F13.config())
        start, ops, states, answer = F13.worked(s)
        assert answer == s.label and states[-1] == s.label
        acc = start
        for op, st in zip(ops, states):
            acc = G.multiply(op, acc)
            assert st == acc


def test_a_worked_answer_that_is_not_the_label_is_refused() -> None:
    import dataclasses
    s = D.generate("A", 1, F13.SPLIT, F13.config())
    wrong = dataclasses.replace(s, label=G.multiply(s.label, D4[1]))
    with pytest.raises(ValueError, match="not the label"):
        F13.worked(wrong)


def test_the_sample_is_the_first_with_no_step_that_does_nothing() -> None:
    def clear(i):
        start, ops, _, answer = F13.worked(D.generate("A", i, F13.SPLIT, F13.config()))
        return G.IDENTITY not in (start, *ops, answer)

    k = F13.pick()
    assert clear(k) and not any(clear(i) for i in range(k))


def test_f13_draws(tmp_path) -> None:
    assert all(p.is_file() for p in F13.draw(str(tmp_path / "f13")))
    _close()


# F14 ----------------------------------------------------------------------

def _salience(tmp_path, drop=None, rates_extra=False) -> Path:
    subs = {"D4 position": ("native", "d4"), "D4 appearance": ("d4_colour", "d4"),
            "E1a": ("d4_square", "d4"), "E1b": ("d4_scrambled", "d4"),
            "E2b": ("d4_colour_decoy", "d4"), "S3 position": ("s3_spatial", "s3"),
            "S3 appearance": ("native", "s3"), "E2a": ("d4_colour_cue", "d4")}
    m = {k: {"substrate": s, "subgroup": g, "n_elements": len(G.subgroup_members(g)),
             "m_pix": 1.0 + i / 10} for i, (k, (s, g)) in enumerate(subs.items())}
    if drop:
        m.pop(drop)
    rates = {k: [i + 1, 10] for i, k in enumerate(m)}
    if rates_extra:
        rates["extra"] = [1, 2]
    d = {"measures": m, "rates": rates, "S1": "held", "rho_pix": 0.5, "p_pix": 0.2,
         "tested": [k for k in m if k != "E2a"]}
    p = tmp_path / "d070.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_the_gallery_sorts_by_the_stored_distance_and_keeps_e2a_apart(tmp_path) -> None:
    d = F14.load(_salience(tmp_path))
    tested, rest = F14.order(d)
    dist = [d["measures"][k]["m_pix"] for k in tested]
    assert dist == sorted(dist, reverse=True) and rest == ["E2a"]


def test_a_rendering_with_no_plain_name_or_no_rate_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="plain name"):
        F14.load(_salience(tmp_path, drop="E1b"))
    with pytest.raises(ValueError, match="no rate"):
        F14.load(_salience(tmp_path, rates_extra=True))


def test_the_tiles_are_the_renderers_and_their_count_is_the_stored_one(tmp_path) -> None:
    d = F14.load(_salience(tmp_path))
    ts = F14.tiles(d["measures"]["S3 position"])
    assert len(ts) == 6 and (ts[1] == P.tile(G.subgroup_members("s3")[1], "s3_spatial")).all()
    bad = dict(d["measures"]["D4 position"], n_elements=7)
    with pytest.raises(ValueError, match="stored 7"):
        F14.tiles(bad)


def test_f14_draws(tmp_path) -> None:
    assert all(p.is_file() for p in F14.draw(_salience(tmp_path), str(tmp_path / "f14")))
    _close()


# F15 ----------------------------------------------------------------------

def _timing(tmp_path, tamper=None) -> Path:
    rows, summary = [], {}
    outcomes = ("escaped", "collapsed", "neither")
    for _, keys in F15.SETS:
        for key in keys.values():
            if key in summary:
                continue
            n = 12
            got = [outcomes[(i * 7 + len(key)) % 3] for i in range(n)]
            rows += [{"run_id": f"{key.split(' ')[0]}_s{i}", "group": key, "outcome": o}
                     for i, o in enumerate(got)]
            summary[key] = {"runs": n, "escaped": got.count("escaped")}
    if tamper == "count":
        k = next(iter(summary))
        summary[k]["escaped"] += 1
    if tamper == "outcome":
        rows[0]["outcome"] = "unfinished"
    p = tmp_path / "timing.json"
    p.write_text(json.dumps({"rows": rows, "summary": summary, "unfinished": 0}),
                 encoding="utf-8")
    return p


def test_an_arms_marks_are_in_seed_order_and_match_the_stored_count(tmp_path) -> None:
    t = json.loads(_timing(tmp_path).read_text(encoding="utf-8"))
    key = F15.SETS[0][1]["same"]
    outcomes, k, n = F15.arm(t, key)
    ordered = [r["outcome"] for r in sorted(
        (r for r in t["rows"] if r["group"] == key), key=lambda r: int(r["run_id"].rsplit("_s", 1)[1]))]
    assert outcomes == ordered and n == 12 and k == outcomes.count("escaped")


def test_a_count_that_disagrees_with_the_marks_is_refused(tmp_path) -> None:
    t = json.loads(_timing(tmp_path, tamper="count").read_text(encoding="utf-8"))
    key = next(iter(t["summary"]))
    with pytest.raises(ValueError, match="stored"):
        F15.arm(t, key)


def test_an_outcome_with_no_mark_or_a_missing_arm_is_refused(tmp_path) -> None:
    t = json.loads(_timing(tmp_path, tamper="outcome").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="no mark"):
        F15.arm(t, t["rows"][0]["group"])
    with pytest.raises(KeyError):
        F15.arm(t, "no_such_arm [init_from]")


def test_f15_draws(tmp_path) -> None:
    assert all(p.is_file() for p in F15.draw(_timing(tmp_path), str(tmp_path / "f15")))
    _close()


# F16 ----------------------------------------------------------------------

def test_the_cue_bits_of_every_problem_add_up_to_its_answers() -> None:
    for s, a, b in itertools.product(D4, repeat=3):
        answer = G.multiply(b, G.multiply(a, s))
        assert (F16.cue_bit(s) + F16.cue_bit(a) + F16.cue_bit(b)) % 2 == F16.cue_bit(answer)


def test_the_cue_leaves_exactly_four_answers() -> None:
    assert sorted(sum(F16.cue_bit(g) == b for g in D4) for b in (0, 1)) == [4, 4]


def test_the_decoy_gives_away_no_composing_bit_and_fixes_no_answer_pixel() -> None:
    for name, chi in G.D4_CHARACTERS.items():
        for key in itertools.product((0, 1), repeat=3):
            got = [chi(G.multiply(b, G.multiply(a, s)))
                   for s, a, b in itertools.product(D4, repeat=3)
                   if (F16.decoy_bit(s), F16.decoy_bit(a), F16.decoy_bit(b)) == key]
            assert got.count(0) == got.count(1), (name, key)
    one, two = F16.decoy_pair()
    assert [F16.decoy_bit(g) for g in one[:3]] == [F16.decoy_bit(g) for g in two[:3]]
    assert F16.decoy_bit(one[3]) != F16.decoy_bit(two[3])
    for s, a, b, answer in (one, two):
        assert answer == G.multiply(b, G.multiply(a, s))
        assert G.IDENTITY not in (s, a, b, answer)


def test_the_ring_sits_on_the_cue_pixel() -> None:
    from loopvision.data import render
    grey = render.COLOURS[G.NEUTRAL_COLOUR]
    for sub, bit in ((F16.CUE, F16.cue_bit), (F16.DECOY, F16.decoy_bit)):
        for g in D4:
            r, c = F16.cue_cell(g, sub)
            lit = tuple(int(v) for v in P.tile(g, sub)[r, c]) == grey
            assert lit == bool(bit(g)), (sub, g)


def test_the_cue_problem_is_the_first_clear_one_and_adds_up() -> None:
    cfg = D.TaskConfig(depths=(2,), breadths=(4,), factor="d4", substrate=F16.CUE)

    def clear(i):
        start, ops, _, answer = F13.worked(D.generate("A", i, "iid_val", cfg))
        steps = (start, *ops)
        return G.IDENTITY not in (*steps, answer) and any(F16.cue_bit(g) for g in steps)

    i, start, ops, answer = F16.cue_problem()
    assert clear(i) and not any(clear(j) for j in range(i))
    assert sum(F16.cue_bit(g) for g in (start, *ops)) % 2 == F16.cue_bit(answer)


def test_f16_draws(tmp_path) -> None:
    assert all(p.is_file() for p in F16.draw(str(tmp_path / "f16")))
    _close()
