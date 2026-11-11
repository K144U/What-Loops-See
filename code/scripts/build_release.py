"""Assemble the release snapshot: the paper, what it was read from, scrubbed.

    python scripts/build_release.py --runs ../zenodo/mirror/runs \\
        --pdf ../paper.pdf --figures ../figures \\
        --out ../zenodo/loopvision-release-20261022 --zip

**WHY A SNAPSHOT AND NOT THE REPOSITORY.** Two commits in the history name
the cluster, and history cannot be cleaned without rewriting it. So the
release is the files the paper reads, taken from one commit, with a ledger of
when each decision log entry was first committed standing in for the history.
D-075.

**WHAT IS LEFT OUT, AND WHY.** Checkpoints, about 96 GB; every instrument the
paper reports was run on them and its output is in. The three jitter arms and
the per-run geometry files, which belong to the H4 work the paper leaves to
concurrent work in submission. The PBS scripts and shell helpers that drive
the cluster, and the tests of them. Every other run the project trained is in,
cited or not, so that nothing reads as a file drawer.

**A RUN FILE THIS DOES NOT KNOW STOPS THE BUILD.** Run files are copied from
a list, because a list cannot leak a log nobody looked at. A list also drops
whatever a new instrument writes, silently, so a file that is neither listed
nor excluded is an error naming it rather than a quiet omission.

**SCRUBBING.** The identifiers come from the cluster file kept outside the
repository, never from this one. In text, a home directory becomes ~ and each
identifier, as a whole token in any case, becomes a placeholder. Then every
file, text or binary, is searched again and the build fails naming the file if
anything is left. In a binary file only an exact-case token counts: a short
token turns up by chance in compressed bytes in some case or other, as it
does in the paper's embedded images.

**THE FIGURES ARE THE STORED ONES.** Each figure is copied only if its hash is
the one `scripts/build_figures.sh` stored beside its inputs, so the figures in
the release are provably the ones drawn from the data in it.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Arms left out, with the reason the README gives.
EXCLUDED_ARMS = {
    "famB_curve111_jit000": "the loop-count jitter sweep, concurrent work in submission",
    "famB_curve111_jit025": "the loop-count jitter sweep, concurrent work in submission",
    "famB_curve111_jit100": "the loop-count jitter sweep, concurrent work in submission",
}

#: Per-run files that say how a run was made, what it scored, and what each
#: instrument read off it.
RUN_FILES = (
    "config.yaml", "summary.json", "metrics.parquet", "DONE", "git.txt",
    "env.txt", "ablate.json", "blank_control.json", "coda_lens.json",
    "factor_lens_*.json", "factor_probe*.json", "state_probe*.json",
    "m2_patching.parquet", "m2_confirm.parquet",
    "m5_halting.json", "m5_halting.parquet",
)

#: Per-run files left out on purpose. The geometry pass fed the figure the
#: paper cut for its overlap with concurrent work.
EXCLUDED_RUN_FILES = ("m4_geometry.json", "m4_geometry.parquet")

#: Stored directories the draft does not name but its text reads from.
EXTRA_STORED = {"_gate": "the validity gate of Section 4"}

#: Code at the commit, as paths for `git ls-tree`.
CODE_PATHS = ("src", "tests", "configs", "pyproject.toml",
              "scripts/build_figures.sh", "scripts/build_paper.py",
              "scripts/build_release.py", "scripts/make_revision_configs.py")

#: Tests of the cluster tooling, which read scripts the release leaves out:
#: job chaining, card picking and its lease and ledger, retries, walltime,
#: memory floors and the launch check.
EXCLUDED_TESTS = (
    "test_chain.py", "test_check_launched.py", "test_gpu_retry.py",
    "test_mem_floor.py", "test_pick_gpus.py", "test_pick_lease.py",
    "test_pick_promise.py", "test_pytest_job.py", "test_wall_margin.py",
)

BINARY_SUFFIXES = {".parquet", ".png", ".pdf", ".pt", ".zip"}

#: The code is MIT and everything else CC BY 4.0, the author's choice of
#: 2026-10-22. The repository has no licence of its own, so without this the
#: code in the release would be all rights reserved.
MIT = """MIT License

Copyright (c) {year} {holder}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def mit_licence(holder: str, year: str) -> str:
    if not holder.strip():
        raise ValueError("the MIT licence needs a copyright holder")
    return MIT.format(year=year, holder=holder.strip())


def load_identifiers(path: Path) -> dict[str, str]:
    """Identifier to placeholder, read from the cluster file."""
    vals = dict(re.findall(r"^(\w+)=(.*)$", path.read_text(encoding="utf-8"), re.M))
    user_at_host = vals["LOOPVISION_SSH_HOST"].strip().split("@")
    ids = {user_at_host[-1]: "<cluster>"}
    for user in (vals["LOOPVISION_CLUSTER_USER"].strip(), user_at_host[0]):
        ids.setdefault(user, "<user>")
    return {k: v for k, v in ids.items() if k}


def _token(value: str, flags=0) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", flags)


def scrub(text: str, ids: dict[str, str]) -> tuple[str, int]:
    """Home directories to ~, then each identifier to its placeholder.

    Longest first, so an identifier that contains another is replaced whole.
    Returns the text and the number of replacements.
    """
    n = 0
    for value in sorted(ids, key=len, reverse=True):
        text, k = re.subn(r"/home/" + re.escape(value) + r"(?![A-Za-z0-9])", "~", text)
        n += k
    for value in sorted(ids, key=len, reverse=True):
        text, k = _token(value, re.I).subn(ids[value], text)
        n += k
    return text, n


def residue(data: bytes, ids: dict[str, str], any_case: bool) -> list[str]:
    """Placeholders whose identifier is still in data, as a whole token."""
    flags = re.I if any_case else 0
    return sorted({ph for value, ph in ids.items()
                   if re.search(_token(value, flags).pattern.encode(), data, flags)})


def is_text(path: Path, data: bytes) -> bool:
    if path.suffix in BINARY_SUFFIXES:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def arm_of(run_id: str) -> str:
    return re.sub(r"_s\d+$", "", run_id)


def plan_run(run_dir: Path) -> tuple[list[Path], list[str]]:
    """Files of one run to release, and names this does not know."""
    keep, unknown = [], []
    for p in sorted(run_dir.iterdir()):
        if p.is_dir():
            if p.name != "checkpoints":
                unknown.append(f"{run_dir.name}/{p.name}/")
            continue
        if p.name in EXCLUDED_RUN_FILES:
            continue
        if any(fnmatch.fnmatch(p.name, pat) for pat in RUN_FILES):
            keep.append(p)
        else:
            unknown.append(f"{run_dir.name}/{p.name}")
    return keep, unknown


def stored_dirs(draft: str) -> list[str]:
    """Every runs/_name the draft names, and the ones its text reads without
    naming."""
    named = set(re.findall(r"runs/(_[A-Za-z0-9_]+)", draft))
    return sorted(named | set(EXTRA_STORED))


def ledger_rows(headings: list[tuple[str, str]], log: str) -> list[dict]:
    """The commit that first added each decision log heading.

    `log` is `git log --reverse --format="commit %H %cI" -p -U0`. A heading
    retitled later shows as removed and added again; the first addition is
    the one that dates the entry.
    """
    first: dict[str, tuple[str, str]] = {}
    commit = date = None
    for line in log.splitlines():
        m = re.match(r"^commit ([0-9a-f]{40}) (\S+)$", line)
        if m:
            commit, date = m.groups()
            continue
        m = re.match(r"^\+### (D-\d+)\.", line)
        if m and m.group(1) not in first and commit:
            first[m.group(1)] = (commit, date)
    missing = [e for e, _ in headings if e not in first]
    if missing:
        raise ValueError("no commit adds " + ", ".join(missing))
    return [{"entry": e, "title": t, "commit": first[e][0],
             "committed_at": first[e][1]} for e, t in headings]


def headings_of(decisions: str) -> list[tuple[str, str]]:
    return re.findall(r"^### (D-\d+)\.\s*(.*?)\s*$", decisions, re.M)


def check_figures(stored: Path, figures: Path) -> tuple[list[tuple[Path, str]], list[str]]:
    """The figures whose hash is the stored one, and every mismatch."""
    ok, bad = [], []
    for line in (stored / "figures.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        src = figures / Path(name.strip().lstrip("*")).name
        if not src.is_file():
            bad.append(f"{src.name}: not found")
        elif hashlib.sha256(src.read_bytes()).hexdigest() != digest:
            bad.append(f"{src.name}: not the figure drawn from the stored input")
        else:
            ok.append((src, digest))
    return ok, bad


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True).stdout


class Release:
    """Writes files into a staging directory and keeps the tally."""

    def __init__(self, staging: Path, ids: dict[str, str]):
        self.staging, self.ids = staging, ids
        self.scrubbed: dict[str, int] = {}

    def put(self, rel: str, data: bytes) -> None:
        dest = self.staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if is_text(dest, data):
            text, n = scrub(data.decode("utf-8"), self.ids)
            if n:
                self.scrubbed[rel] = n
            data = text.encode("utf-8")
        dest.write_bytes(data)

    def leftovers(self) -> list[str]:
        out = []
        for p in sorted(self.staging.rglob("*")):
            if p.is_file():
                data = p.read_bytes()
                found = residue(data, self.ids, any_case=is_text(p, data))
                if found:
                    out.append(f"{p.relative_to(self.staging).as_posix()}: "
                               f"{', '.join(found)}")
        return out


def readme(info: dict) -> str:
    excluded = "\n".join(f"- `{a}`: {why}" for a, why in sorted(EXCLUDED_ARMS.items()))
    return f"""# Release snapshot

This archive accompanies the preprint *{info['title']}*, in `paper.pdf`. It
was taken from commit
`{info['commit']}` of the project repository on {info['date']}. Appendix A of
the paper names, for each claim, the decision log entry behind it and the
stored file it is read from; both are here.

## Contents

- `paper.pdf`, the preprint.
- `registrations/preregistration.md`, frozen at tag `prereg-v1`, commit
  `{info['prereg_commit']}`, {info['prereg_date']}.
- `registrations/decisions.md`, the decision log, and
  `registrations/experiment-substrate.md`, the design of the substrate
  experiment.
- `registrations/ledger.csv`, the commit and time at which each decision log
  entry was first committed, {info['n_entries']} entries.
- `runs/`, {info['n_runs']} training runs in {info['n_arms']} arms: each run's
  config, metrics over training and evaluation, summary, and the output of
  every instrument run on it.
- `runs/_*/`, the stored outputs the verdicts and figures are read from,
  {info['n_stored']} directories, each with the commit that wrote it.
- `figures/`, the {info['n_figures']} figures of the paper, each identical to
  the hash stored beside its inputs in `{info['figures_dir']}`.
- `code/`, the `loopvision` package, its tests and its configs at the same
  commit.
- `SHA256SUMS`, a hash of every other file here.

## Licences

The code under `code/` is under the MIT licence in `code/LICENSE`. Everything
else, the paper, the registrations and the data, is under Creative Commons
Attribution 4.0 International.

## Left out, and why

- Model checkpoints, about 96 GB. Every instrument the paper reports was run
  on them and its output is under `runs/`.
- {info['n_excluded_runs']} runs of three arms, and the per-run geometry files,
  which belong to concurrent work in submission:
{excluded}
- The cluster's job scripts and their tests.
- The repository's history, which is not public. `ledger.csv` records when
  each registration was committed.

## Scrubbing

Host and user names of the cluster were replaced in text: home directories
by `~`, the cluster's name by `<cluster>`, user names by `<user>`. That
changed {info['n_scrubbed_files']} files, {info['n_scrubbed_env']} of them the per-run
`env.txt` package listings, and nothing else was edited.

## Rerunning

Python 3.10 or later, PyTorch 2 (the runs used the cu121 build).

    pip install -e code
    cd code
    python -m loopvision.analysis.escape_rates --runs ../runs
    python -m loopvision.analysis.m1_curves --runs ../runs --no-fit \\
        --summary m1_summary.json \\
        --config configs/m3/famB_curve_111.yaml \\
        --config configs/m3/famC_breadth_111.yaml

Each figure script under `code/src/loopvision/analysis/plots/` reads one of
the stored inputs in `{info['figures_dir']}` and computes nothing;
`code/scripts/build_figures.sh` lists the command for each.
"""


def build(args) -> int:
    ids = load_identifiers(args.identifiers)
    commit = git("rev-parse", args.commit).decode().strip()
    out = args.out
    if out.exists():
        print(f"refusing: {out} exists")
        return 2
    staging = out.with_name(out.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    rel = Release(staging, ids)

    # Runs.
    errors: list[str] = []
    n_runs, arms, n_excluded = 0, set(), 0
    for run_dir in sorted(p for p in args.runs.iterdir() if p.is_dir()):
        if run_dir.name.startswith("_"):
            continue
        if arm_of(run_dir.name) in EXCLUDED_ARMS:
            n_excluded += 1
            continue
        keep, unknown = plan_run(run_dir)
        errors += [f"unknown run file {u}" for u in unknown]
        for p in keep:
            rel.put(f"runs/{run_dir.name}/{p.name}", p.read_bytes())
        n_runs += 1
        arms.add(arm_of(run_dir.name))

    # Stored outputs, named by the draft at the commit. A draft from the
    # working tree makes a trial, which says so and is never zipped.
    if args.draft:
        draft = args.draft.read_text(encoding="utf-8")
    else:
        draft = git("show", f"{commit}:docs/draft.md").decode("utf-8")
    stored = stored_dirs(draft)
    for name in stored:
        src = args.runs / name
        if not src.is_dir():
            errors.append(f"the draft names runs/{name}, which is not in {args.runs}")
            continue
        for p in sorted(src.rglob("*")):
            if p.is_file():
                rel.put(f"runs/{name}/{p.relative_to(src).as_posix()}", p.read_bytes())

    # Figures, only as drawn from the stored inputs.
    fig_dirs = [n for n in stored if n.startswith("_figures_")]
    if len(fig_dirs) != 1:
        errors.append(f"the draft names {len(fig_dirs)} stored figure directories, not one")
        figs = []
    else:
        figs, bad = check_figures(args.runs / fig_dirs[0], args.figures)
        errors += bad
        for src, _ in figs:
            rel.put(f"figures/{src.name}", src.read_bytes())

    # Registrations, and the ledger that dates them.
    prereg = git("rev-parse", "prereg-v1^{commit}").decode().strip()
    rel.put("registrations/preregistration.md",
            git("show", f"{prereg}:docs/preregistration.md"))
    decisions = git("show", f"{commit}:docs/decisions.md")
    rel.put("registrations/decisions.md", decisions)
    rel.put("registrations/experiment-substrate.md",
            git("show", f"{commit}:docs/experiment-substrate.md"))
    log = git("log", "--reverse", "--follow", "--format=commit %H %cI", "-p",
              "-U0", commit, "--", "docs/decisions.md").decode("utf-8", "replace")
    rows = ledger_rows(headings_of(decisions.decode("utf-8")), log)
    prereg_date = git("show", "-s", "--format=%cI", prereg).decode().strip()
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["entry", "title", "commit", "committed_at"],
                       lineterminator="\n")
    w.writeheader()
    w.writerow({"entry": "prereg-v1", "title": "docs/preregistration.md, frozen",
                "commit": prereg, "committed_at": prereg_date})
    w.writerows(rows)
    rel.put("registrations/ledger.csv", buf.getvalue().encode("utf-8"))

    # Code at the commit.
    for path in git("ls-tree", "-r", "--name-only", commit, "--",
                    *CODE_PATHS).decode().splitlines():
        if Path(path).name in EXCLUDED_TESTS:
            continue
        rel.put(f"code/{path}", git("show", f"{commit}:{path}"))
    date = git("show", "-s", "--format=%cs", commit).decode().strip()
    holder = args.licence_holder or "[copyright holders to be named]"
    rel.put("code/LICENSE", mit_licence(holder, date[:4]).encode("utf-8"))

    rel.put("paper.pdf", args.pdf.read_bytes())

    title = re.search(r"^#\s+(.+)$", draft, re.M)
    trial = ("TRIAL BUILD. The draft was read from the working tree and not "
             f"from commit {commit}. Not for release.\n\n" if args.draft else "")
    rel.put("README.md", trial.encode("utf-8") + readme({
        "title": title.group(1) if title else "loopvision",
        "commit": commit, "date": date,
        "prereg_commit": prereg, "prereg_date": prereg_date,
        "n_entries": len(rows), "n_runs": n_runs, "n_arms": len(arms),
        "n_stored": len(stored), "n_figures": len(figs),
        "figures_dir": f"runs/{fig_dirs[0]}/" if len(fig_dirs) == 1 else "runs/",
        "n_excluded_runs": n_excluded,
        "n_scrubbed_files": len(rel.scrubbed),
        "n_scrubbed_env": sum(1 for r in rel.scrubbed if r.endswith("/env.txt")),
    }).encode("utf-8"))

    errors += [f"identifier left in {x}" for x in rel.leftovers()]
    if errors:
        print(f"NOT RELEASED. {len(errors)} problem(s); the staging copy is "
              f"left at {staging} for inspection:")
        for e in errors[:40]:
            print("  " + e)
        return 1

    write_sums(staging)
    staging.rename(out)

    print(f"wrote {out}")
    print(f"  {n_runs} runs in {len(arms)} arms, {n_excluded} left out; "
          f"{len(stored)} stored directories; {len(figs)} figures; "
          f"{len(rows)} ledger entries")
    print(f"  scrubbed {len(rel.scrubbed)} files: "
          + ", ".join(f"{r} ({n})" for r, n in sorted(rel.scrubbed.items())
                      if not r.endswith("/env.txt"))
          + f", and {sum(1 for r in rel.scrubbed if r.endswith('/env.txt'))} env.txt")
    if args.draft:
        print("  TRIAL: the draft came from the working tree, so this is not "
              "zipped and is not a release")
    elif args.zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=out.parent,
                                      base_dir=out.name)
        print(f"  zipped to {archive}")
    return 0


def write_sums(staging: Path) -> int:
    """Hash every file under `staging` into its SHA256SUMS; return the count.

    Written as bytes so the line ends are LF on every machine. Until
    2026-11-12 this went through `write_text`, which on Windows ends each
    line with CRLF, and `sha256sum -c` then looks for a file whose name ends
    in a carriage return: all 4413 lines of the first GitHub snapshot failed
    that way although every hash in it was right.
    """
    sums = []
    for p in sorted(staging.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  "
                        f"{p.relative_to(staging).as_posix()}")
    text = "\n".join(sums) + "\n"
    (staging / "SHA256SUMS").write_bytes(text.encode("utf-8"))
    return len(sums)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Assemble the release snapshot")
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--figures", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--identifiers", type=Path,
                    default=Path.home() / ".loopvision-cluster")
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("--draft", type=Path, default=None,
                    help="read the draft from this file: a trial, not a release")
    ap.add_argument("--licence-holder", default=None,
                    help="the copyright holders named in the code's MIT licence")
    args = ap.parse_args(argv)
    if not args.draft and not args.licence_holder:
        print("refusing: a release names the code's copyright holders, "
              "--licence-holder. Only a trial may leave them out.")
        return 2
    for p in (args.runs, args.figures):
        if not p.is_dir():
            print(f"no directory at {p}")
            return 2
    for p in (args.pdf, args.identifiers):
        if not p.is_file():
            print(f"no file at {p}")
            return 2
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
