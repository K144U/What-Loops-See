"""Every config must be able to generate its data and build its model.

A config that asks for data outside its split fails seconds after the job
starts, and the job stays in `qstat` looking healthy. On 2026-09-08 job
5275 held a four core slot for thirteen hours with three of its four runs
dead that way, while the highest priority queued job waited. Nothing reads
per-run logs, so the only signal was a stalled queue. See D-044.

Three checks, in the order they fail in real life:

  parses    load_config can read it at all
  data      split_spec accepts its depths and breadths on train and iid_val
  model     build_model can instantiate it, which is where a d_model that
            does not divide the head count fails

HISTORICAL CONFIGS ARE NOT FAILURES, AND THE DIFFERENCE IS DERIVED RATHER
THAN LISTED. Three configs in this repository ask for family B breadths 6
to 9, which was correct under D-020 and stopped being correct when D-028
moved the range to 12 to 16. They are the only record of what seven
finished runs trained on, so they stay. A hand maintained allowlist would
be the same fault one level up, the kind that let the run watcher fall
behind twice in a day, so instead a broken config is called historical
exactly when a finished run exists that matches it. Evidence that it once
worked is the run itself.

RUN SNAPSHOTS ARE CHECKED TOO, AND ARE NOT ERRORS. Every instrument reads
`runs/<id>/config.yaml` to regenerate evaluation batches, so a run whose
snapshot names a range the current code rejects can never be re-analysed.
That is worth knowing before someone queues a geometry pass over it, but
the run's recorded numbers are untouched by it and the run is not invalid.

    python -m loopvision.analysis.check_configs
    python -m loopvision.analysis.check_configs --skip-models
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

#: Configs that cannot run, are known not to run, and are kept on purpose.
#: D-044 decided this on 2026-09-08 and gave the reason: these three were
#: "written for a depth axis family A does not have", and deleting a config
#: that was never runnable "does not stop the next person writing it again;
#: the entry does".
#:
#: **They are listed so this check can exit 0, and that is the whole point.**
#: Before this list the command failed every single time it was run, and
#: CLAUDE.md tells you to run it before every submission. A check that is
#: always red is a check that stops being read, which is L-029 exactly: the
#: count rule was prose-only until a stale "7 of 7" survived eleven runs
#: because nobody was reading the thing that would have caught it.
#:
#: **Listing them makes the check STRICTER, not weaker.** A config here that
#: starts passing is now a failure too, because family A's depth axis would
#: have had to change for that to happen and no such change is recorded.
#: Paths are relative to the configs root, posix separators.
RECORDED_UNRUNNABLE = frozenset({
    "m3/famA_d4_d4only.yaml",
    "m3/famA_d5_d4only.yaml",
    "m3/famA_d6_d4only.yaml",
})

#: Config keys that identify which experiment a run was, ignoring the ones
#: that legitimately vary between runs of the same config.
IDENTITY_KEYS = (
    "family", "depths", "breadths", "factor", "substrate",
    "arch", "d_model", "prelude_blocks", "core_blocks", "coda_blocks",
    "k_schedule", "k_train", "init_from",
)


def _identity(cfg: dict) -> tuple:
    def norm(v):
        return tuple(v) if isinstance(v, list) else v
    return tuple(norm(cfg.get(k)) for k in IDENTITY_KEYS)


def check_data(cfg: dict) -> str | None:
    """None if the config can generate train and iid_val, else the reason."""
    from loopvision.data import dataset as D

    family = cfg.get("family")
    if not family:
        return None
    tc = D.TaskConfig(
        master_seed=cfg.get("master_seed", 0),
        depths=tuple(cfg["depths"]) if cfg.get("depths") else None,
        breadths=tuple(cfg["breadths"]) if cfg.get("breadths") else None,
        factor=cfg.get("factor", "full"),
        substrate=cfg.get("substrate", "native"),
    )
    for split in ("train", "iid_val"):
        try:
            D.split_spec(family, split, tc)
        except Exception as e:
            return f"{split}: {str(e).split('. ')[0]}"
    return None


def check_model(cfg: dict) -> str | None:
    """None if the model instantiates, else the reason."""
    import torch

    from loopvision.train.cli import build_model

    try:
        build_model(cfg, torch.device("cpu"))
    except Exception as e:
        return f"model: {type(e).__name__}: {e}"
    return None


def finished_run_identities(runs_root: Path) -> set:
    """Identities of runs that finished, so a config can be shown to have
    worked once even if it cannot run today."""
    from loopvision.train.cli import load_config

    out = set()
    for p in glob.glob(str(runs_root / "*" / "config.yaml")):
        if not os.path.exists(os.path.join(os.path.dirname(p), "DONE")):
            continue
        try:
            out.add(_identity(load_config(p, {})))
        except Exception:
            continue
    return out


def check(configs_root: Path, runs_root: Path, models: bool) -> dict:
    from loopvision.train.cli import load_config

    done_ids = finished_run_identities(runs_root)
    broken, historical, checked = [], [], 0
    recorded, revived = [], []

    for p in sorted(glob.glob(str(configs_root / "**" / "*.yaml"), recursive=True)):
        try:
            cfg = load_config(p, {})
        except Exception as e:
            broken.append((p, f"parse: {e}"))
            continue
        checked += 1
        rel = Path(p).relative_to(configs_root).as_posix()
        reason = check_data(cfg)
        if reason is None and models:
            reason = check_model(cfg)
        if reason is None:
            # A config D-044 recorded as unrunnable that now runs means the
            # data definition moved and nobody wrote it down. Reported as a
            # failure, because a silent change to what a split contains is
            # how two runs end up incomparable without anyone noticing.
            if rel in RECORDED_UNRUNNABLE:
                revived.append((p, "recorded as unrunnable in D-044 and it runs"))
            continue
        # It once worked if a finished run matches it. Derived, not listed.
        if _identity(cfg) in done_ids:
            historical.append((p, reason))
        elif rel in RECORDED_UNRUNNABLE:
            recorded.append((p, reason))
        else:
            broken.append((p, reason))

    closed = []
    for p in sorted(glob.glob(str(runs_root / "*" / "config.yaml"))):
        if not os.path.exists(os.path.join(os.path.dirname(p), "DONE")):
            continue
        # RAW, not load_config, because that is how every instrument reads
        # a run snapshot: m2, m4, m5, the lenses and the probes all call
        # yaml.safe_load directly. load_config layers DEFAULTS over the
        # file, so it invents a prelude_blocks for a snapshot that has
        # none and the check passes while the instrument dies on a
        # KeyError. That is what happened on 2026-09-10: nine vark_* runs
        # passed this check and then failed a geometry job all at once.
        #
        # Reading it the way the instruments do is also the only honest
        # check. A default silently substituted here is a value the run
        # may never have trained with, so the checker would be validating
        # a configuration that never existed.
        try:
            import yaml as _yaml

            cfg = _yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
        except Exception as e:
            closed.append((Path(p).parent.name, f"parse: {e}"))
            continue
        # Data AND model, because every instrument does both. Checking
        # only the data missed nine runs on 2026-09-10: the vark_* configs
        # predate prelude_blocks, core_blocks and coda_blocks, so they
        # generate their data perfectly well and then die in build_model
        # with a KeyError. A geometry job found that by failing on all nine
        # at once, which is the wrong place to find it.
        reason = check_data(cfg)
        if reason is None and models:
            reason = check_model(cfg)
        if reason is not None:
            closed.append((Path(p).parent.name, reason))

    return {
        "checked": checked,
        "broken": broken,
        "historical": historical,
        "recorded": recorded,
        "revived": revived,
        "closed": closed,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check configs can run")
    ap.add_argument("--configs", default=Path("configs"), type=Path)
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    ap.add_argument("--skip-models", action="store_true",
                    help="skip instantiation, which is the slow check")
    args = ap.parse_args(argv)

    if not args.configs.is_dir():
        print(f"no such configs directory: {args.configs}", file=sys.stderr)
        return 2

    r = check(args.configs, args.runs, models=not args.skip_models)
    print(f"checked {r['checked']} config(s)"
          f"{'' if args.skip_models else ', data and model'}")

    if r["historical"]:
        print(f"\n{len(r['historical'])} historical config(s). These cannot run "
              f"today and a finished run proves they once did, so they are kept:")
        for p, why in r["historical"]:
            print(f"  {p}\n      {why}")

    if r["closed"]:
        print(f"\n{len(r['closed'])} finished run(s) closed to re-analysis. Their "
              f"recorded numbers stand; no instrument can regenerate their batches:")
        for run, why in r["closed"]:
            print(f"  {run}\n      {why}")

    if r["recorded"]:
        print(f"\n{len(r['recorded'])} config(s) recorded as unrunnable in D-044 "
              f"and kept on purpose. NOT failures:")
        for p, why in r["recorded"]:
            print(f"  {p}\n      {why}")
        print("  D-044 keeps these because deleting a config that was never "
              "runnable does not")
        print("  stop the next person writing it again. The entry does.")

    if r["revived"]:
        print(f"\n{len(r['revived'])} config(s) recorded as unrunnable NOW RUN, "
              f"which is a failure:")
        for p, why in r["revived"]:
            print(f"  {p}\n      {why}")
        print("  Family A's depth axis would have had to change for this and no "
              "such change is")
        print("  recorded. Write the decision entry, or find out what moved the "
              "split.")

    if not r["broken"] and not r["revived"]:
        print("\nNo config is broken. The D-044 set above is expected, and "
              "listing it is what lets")
        print("this command exit 0 instead of failing every time it is run.")
        return 0

    if r["broken"]:
        print(f"\n{len(r['broken'])} BROKEN config(s). These have never produced a "
              f"finished run and cannot produce one:")
        for p, why in r["broken"]:
            print(f"  {p}\n      {why}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
