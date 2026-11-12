"""The release snapshot: nothing private leaves, and nothing it promises is missing.

A release is published once and cannot be recalled, so the failures that
matter are the silent ones: an identifier that survives because it was in a
case or a file type nobody looked at, a run file dropped because it was not
on a list, a figure that is not the one drawn from the stored data, and a
ledger that dates an entry by a later edit. Every identifier here is made up.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_release as R  # noqa: E402

IDS = {"zeta": "<cluster>", "ann.k": "<user>"}


def test_identifiers_come_from_the_cluster_file(tmp_path) -> None:
    p = tmp_path / "cluster"
    p.write_text("# comment\nLOOPVISION_SSH_HOST=zeta@zeta\n"
                 "LOOPVISION_CLUSTER_USER=ann.k\n", encoding="utf-8")
    assert R.load_identifiers(p) == {"zeta": "<cluster>", "ann.k": "<user>"}


def test_a_home_directory_becomes_the_tilde() -> None:
    text, n = R.scrub("-e /home/ann.k/loopvision", IDS)
    assert text == "-e ~/loopvision" and n == 1


def test_a_token_is_replaced_in_any_case() -> None:
    """A node name is written zeta-gpu01 in one file and ZETA in another."""
    text, _ = R.scrub("pbsnodes zeta-gpu01 on ZETA, Job Id: 5.Zeta-master", IDS)
    assert text == "pbsnodes <cluster>-gpu01 on <cluster>, Job Id: 5.<cluster>-master"


def test_a_token_inside_a_word_is_left_alone() -> None:
    """A whole token only, or scrubbing would edit words that merely contain
    the letters, and the text would no longer be the text."""
    text, n = R.scrub("zetaform and betazeta", IDS)
    assert text == "zetaform and betazeta" and n == 0


def test_another_home_directory_is_left_alone() -> None:
    """Test fixtures write /home/u and /home/x on purpose."""
    text, n = R.scrub("/home/u/runs", IDS)
    assert text == "/home/u/runs" and n == 0


def test_residue_in_text_counts_any_case() -> None:
    assert R.residue(b"left ZETA here", IDS, any_case=True) == ["<cluster>"]


def test_residue_in_binary_counts_only_the_exact_case() -> None:
    """Base64 and compressed bytes hold a short token in some case by chance:
    the paper's own images do."""
    assert R.residue(b"\x00zEtA\x01", IDS, any_case=False) == []
    assert R.residue(b"\x00zeta\x01", IDS, any_case=False) == ["<cluster>"]


def _run(root: Path, name: str, files: dict[str, bytes]) -> Path:
    d = root / name
    d.mkdir(parents=True)
    for f, data in files.items():
        (d / f).write_bytes(data)
    return d


def test_a_run_keeps_its_listed_files_and_drops_the_excluded(tmp_path) -> None:
    d = _run(tmp_path, "famA_d2_d4only_s0", {
        "config.yaml": b"a: 1", "metrics.parquet": b"PAR1", "state_probe_d4.json": b"{}",
        "m4_geometry.parquet": b"PAR1"})
    (d / "checkpoints").mkdir()
    keep, unknown = R.plan_run(d)
    assert sorted(p.name for p in keep) == ["config.yaml", "metrics.parquet",
                                            "state_probe_d4.json"]
    assert unknown == []


def test_a_run_file_nobody_listed_is_named_not_dropped(tmp_path) -> None:
    """A list that silently drops what a new instrument writes would release
    a paper whose evidence is quietly incomplete."""
    d = _run(tmp_path, "famA_d2_d4only_s0", {"config.yaml": b"", "train.log": b""})
    (d / "extra").mkdir()
    _, unknown = R.plan_run(d)
    assert unknown == ["famA_d2_d4only_s0/extra/", "famA_d2_d4only_s0/train.log"]


def test_every_shipped_test_reaches_only_shipped_scripts() -> None:
    """A test of a script the release leaves out cannot run inside the release.

    On 2026-11-12 the two tests of the TMLR build were about to ship without
    the scripts they import, and the release's own suite would have stopped
    at collection.
    """
    root = Path(__file__).resolve().parents[1]
    scripts = {p.stem: p.name for p in (root / "scripts").glob("*.py")}
    shipped = {Path(p).name for p in R.CODE_PATHS if p.startswith("scripts/")}
    bad = []
    for t in sorted((root / "tests").glob("test_*.py")):
        if t.name in R.EXCLUDED_TESTS:
            continue
        text = t.read_text(encoding="utf-8")
        used = {m.group(1) for m in re.finditer(r"^\s*(?:import|from)\s+(\w+)", text, re.M)}
        used = {scripts[u] for u in used if u in scripts}
        used |= set(re.findall(r"\"scripts\"\s*/\s*\"([\w.]+)\"", text))
        bad += [f"{t.name}: {u}" for u in sorted(used - shipped)]
    assert not bad, bad


def test_the_jitter_arms_are_excluded_and_nothing_else() -> None:
    assert R.arm_of("famB_curve111_jit025_s3") in R.EXCLUDED_ARMS
    assert R.arm_of("famB_curve111_s3") not in R.EXCLUDED_ARMS


def test_stored_directories_are_the_ones_the_draft_names() -> None:
    draft = "read from `runs/_why_e1e2_20261009/` and runs/_figures_20261022/x.json"
    assert R.stored_dirs(draft) == sorted({"_why_e1e2_20261009", "_figures_20261022"}
                                          | set(R.EXTRA_STORED))


LOG = """commit {a} 2026-09-01T10:00:00+05:30
+### D-001. First title
commit {b} 2026-09-05T10:00:00+05:30
-### D-001. First title
+### D-001. Retitled
+### D-002. Second
""".format(a="a" * 40, b="b" * 40)


def test_the_ledger_dates_an_entry_by_the_commit_that_first_added_it() -> None:
    """A retitled heading is removed and added again in a later diff; dating
    it by that would make a registration look newer than it is."""
    rows = R.ledger_rows([("D-001", "Retitled"), ("D-002", "Second")], LOG)
    assert [(r["entry"], r["commit"][0]) for r in rows] == [("D-001", "a"), ("D-002", "b")]
    assert rows[0]["committed_at"].startswith("2026-09-01")


def test_the_ledger_refuses_an_entry_no_commit_adds() -> None:
    with pytest.raises(ValueError, match="D-003"):
        R.ledger_rows([("D-003", "Never committed")], LOG)


def test_headings_are_read_in_order() -> None:
    text = "### D-001. One\nbody\n### D-002. Two  \n"
    assert R.headings_of(text) == [("D-001", "One"), ("D-002", "Two")]


def test_a_figure_is_released_only_if_it_is_the_stored_one(tmp_path) -> None:
    stored, figs = tmp_path / "stored", tmp_path / "figs"
    stored.mkdir()
    figs.mkdir()
    (figs / "f2_curves.png").write_bytes(b"drawn from the stored input")
    (figs / "f9_binding.png").write_bytes(b"redrawn later")
    good = hashlib.sha256(b"drawn from the stored input").hexdigest()
    (stored / "figures.sha256").write_text(
        f"{good}  figures/f2_curves.png\n{'0' * 64}  figures/f9_binding.png\n"
        f"{'1' * 64}  figures/f3_patching.png\n", encoding="utf-8")
    ok, bad = R.check_figures(stored, figs)
    assert [p.name for p, _ in ok] == ["f2_curves.png"]
    assert bad == ["f9_binding.png: not the figure drawn from the stored input",
                   "f3_patching.png: not found"]


def test_text_is_scrubbed_on_the_way_in_and_binary_is_not(tmp_path) -> None:
    rel = R.Release(tmp_path, IDS)
    rel.put("runs/r/env.txt", b"-e /home/ann.k/loopvision\n")
    rel.put("runs/r/metrics.parquet", b"PAR1 zeta")
    assert (tmp_path / "runs/r/env.txt").read_bytes() == b"-e ~/loopvision\n"
    assert (tmp_path / "runs/r/metrics.parquet").read_bytes() == b"PAR1 zeta"
    assert rel.scrubbed == {"runs/r/env.txt": 1}


def test_anything_left_is_found_wherever_it_is(tmp_path) -> None:
    """The last gate before a file is published: a binary file carrying the
    exact token fails the build by name."""
    rel = R.Release(tmp_path, IDS)
    rel.put("runs/r/env.txt", b"clean\n")
    rel.put("runs/r/metrics.parquet", b"PAR1 zeta")
    assert rel.leftovers() == ["runs/r/metrics.parquet: <cluster>"]


def test_the_mit_licence_names_its_holder_and_year() -> None:
    text = R.mit_licence("A. Author and B. Author", "2026")
    assert text.startswith("MIT License")
    assert "Copyright (c) 2026 A. Author and B. Author" in text


def test_the_mit_licence_refuses_an_empty_holder() -> None:
    with pytest.raises(ValueError):
        R.mit_licence("  ", "2026")


def test_a_release_without_a_licence_holder_is_refused(tmp_path) -> None:
    """Code released under a licence with no holder is a licence nobody can
    enforce or grant. A trial may leave it out; a release may not."""
    ids = tmp_path / "cluster"
    ids.write_text("LOOPVISION_SSH_HOST=zeta@zeta\nLOOPVISION_CLUSTER_USER=ann.k\n",
                   encoding="utf-8")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    rc = R.main(["--runs", str(tmp_path), "--pdf", str(pdf), "--figures",
                 str(tmp_path), "--out", str(tmp_path / "release"),
                 "--identifiers", str(ids)])
    assert rc == 2 and not (tmp_path / "release").exists()


def test_an_existing_release_is_never_written_over(tmp_path) -> None:
    out = tmp_path / "release"
    out.mkdir()
    ids = tmp_path / "cluster"
    ids.write_text("LOOPVISION_SSH_HOST=zeta@zeta\nLOOPVISION_CLUSTER_USER=ann.k\n",
                   encoding="utf-8")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    # A commit no repository has: if git were asked before the refusal, this
    # would raise, as it does in any copy of the release that is not a checkout.
    rc = R.main(["--runs", str(tmp_path), "--pdf", str(pdf), "--figures",
                 str(tmp_path), "--out", str(out), "--identifiers", str(ids),
                 "--licence-holder", "A. Author", "--commit", "no-such-commit-anywhere"])
    assert rc == 2


def test_the_checksum_file_ends_its_lines_with_lf_alone(tmp_path) -> None:
    """`sha256sum -c` reads a CR as part of the file name."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_bytes(b"one")
    (tmp_path / "b.txt").write_bytes(b"two")
    assert R.write_sums(tmp_path) == 2
    raw = (tmp_path / "SHA256SUMS").read_bytes()
    assert bytes([13]) not in raw and raw.endswith(bytes([10]))
    lines = raw.decode().splitlines()
    assert [ln.split("  ")[1] for ln in lines] == ["b.txt", "sub/a.txt"]
    assert lines[0].split("  ")[0] == hashlib.sha256(b"two").hexdigest()
    # Run again and the file does not hash itself.
    assert R.write_sums(tmp_path) == 2
