"""Which D4 character a run learns first. E0.4 of docs/why-experiments.md.

Reads the `character_flip`, `character_rot2` and `character_rotflip` metrics
that training logs at every eval when a config sets `log_characters: true`.
Each is the share of eval items whose PREDICTED element has the same value of
that character as the label, so a model that knows one character and nothing
else reads 1.0 on it and about 0.5 on the other two.

    python -m loopvision.analysis.character_stage
    python -m loopvision.analysis.character_stage --registered

A SINGLE-CHARACTER STAGE is an eval at which exactly one character is at or
above STAGE_HIGH and both others are at or below STAGE_LOW, held by the same
character for STAGE_MIN_EVALS consecutive evals. Two readings cannot pass for
one:

  solved          all three near 1.0, so none is alone
  location stage  an identity run that has the group but not the query sits
                  near 0.76 on all three together (why-experiments.md,
                  observation 4), so none is high

The stage says which character a model gets right, from its outputs. It does
not say how the model computes it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from loopvision.analysis import escape_rates as ER
from loopvision.analysis import escape_timing as T
from loopvision.data import groups as G

STAGE_HIGH = 0.9
STAGE_LOW = 0.6
STAGE_MIN_EVALS = 2

#: The config key and value that mark an E0.4 run. Nothing else is read.
EXPERIMENT = "E0.4"
#: What the aspect-ratio account predicts D4 position learns first.
PREDICTED_FIRST = "rotflip"
#: Seeds per arm and the fewest informative D4 position seeds for a verdict.
POSITION_SEEDS = 4
IDENTITY_SEEDS = 8
MIN_INFORMATIVE = 3


def character_series(metrics: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Eval steps, and each character's accuracy at them, aligned by step."""
    rows = metrics[metrics.metric_name.str.startswith("character_")]
    if rows.empty:
        return np.array([], dtype=int), {}
    wide = (rows.drop_duplicates(subset=["step", "metric_name"], keep="last")
                .pivot(index="step", columns="metric_name", values="value")
                .sort_index())
    names = [f"character_{n}" for n in G.D4_CHARACTERS]
    if not set(names) <= set(wide.columns):
        return np.array([], dtype=int), {}
    wide = wide[names].dropna()
    return (wide.index.to_numpy(),
            {n.removeprefix("character_"): wide[n].to_numpy() for n in names})


def first_stage(steps: np.ndarray, series: dict[str, np.ndarray]) -> dict | None:
    """The first single-character stage, or None if the run never had one."""
    run_name, run_start, run_len = None, None, 0
    for i in range(len(steps)):
        high = [n for n, v in series.items() if v[i] >= STAGE_HIGH]
        # Exactly one high follows from the second condition: any other
        # character at STAGE_HIGH is above STAGE_LOW. Writing len(high) == 1
        # as well left a check no test could break, found by mutation.
        alone = bool(high) and all(
            v[i] <= STAGE_LOW for n, v in series.items() if n != high[0])
        name = high[0] if alone else None
        if name is not None and name == run_name:
            run_len += 1
        elif name is not None:
            run_name, run_start, run_len = name, int(steps[i]), 1
        else:
            run_name, run_start, run_len = None, None, 0
        if run_len >= STAGE_MIN_EVALS:
            return {"character": run_name, "step": run_start}
    return None


def scan(runs_root: Path) -> list[dict]:
    """Every finished E0.4 run, with its arm, its departure and its stage."""
    rows = []
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        cp = d / "config.yaml"
        if not cp.is_file():
            continue
        cfg = yaml.safe_load(cp.read_text(encoding="utf-8")) or {}
        if cfg.get("experiment") != EXPERIMENT or not (d / "DONE").is_file():
            continue
        arm = ER.SUBSTRATE_ARMS.get((cfg.get("factor", "full"),
                                     cfg.get("substrate") or "native"))
        m = pd.read_parquet(d / "metrics.parquet")
        steps, series = character_series(m)
        tsteps, tvalues = T.train_loss(m)
        order = T.group_order(cfg)
        rows.append({
            "run_id": d.name,
            "arm": list(arm) if arm else None,
            "leave_step": T.leave_step(tsteps, tvalues, order),
            "evals": int(len(steps)),
            "stage": first_stage(steps, series) if len(steps) else None,
        })
    return rows


def registered(rows: list[dict]) -> dict[str, tuple[str, str]]:
    """Evaluate D-060 on the E0.4 runs.

    Q1  every D4 position seed with a single-character stage has it on
        rotflip. Needs MIN_INFORMATIVE such seeds for a verdict.
    Q2  no D4 identity seed has a single-character stage.
    """
    out = {}
    pos = [r for r in rows if r["arm"] == ["D4", "position"]]
    staged = [r for r in pos if r["stage"]]
    wrong = [f"{r['run_id']} learned {r['stage']['character']} first"
             for r in staged if r["stage"]["character"] != PREDICTED_FIRST]
    if wrong:
        out["Q1"] = ("failed", "; ".join(wrong))
    elif len(pos) < POSITION_SEEDS:
        out["Q1"] = ("pending", f"{len(pos)} of {POSITION_SEEDS} finished, "
                                f"{len(staged)} with a stage so far")
    elif len(staged) < MIN_INFORMATIVE:
        out["Q1"] = ("undetermined", f"only {len(staged)} of {len(pos)} seeds had a "
                                     f"single-character stage; {MIN_INFORMATIVE} needed")
    else:
        out["Q1"] = ("held", f"{len(staged)} of {len(pos)} seeds had a stage, "
                             f"all on {PREDICTED_FIRST}")

    ident = [r for r in rows if r["arm"] == ["D4", "identity"]]
    stepped = [f"{r['run_id']} on {r['stage']['character']}" for r in ident if r["stage"]]
    left = sum(1 for r in ident if r["leave_step"] is not None)
    if stepped:
        out["Q2"] = ("failed", "; ".join(stepped))
    elif len(ident) < IDENTITY_SEEDS:
        out["Q2"] = ("pending", f"{len(ident)} of {IDENTITY_SEEDS} finished")
    else:
        # Held with no departures is vacuous: a run at chance has no stage
        # to show. The count is printed so that cannot pass unnoticed.
        out["Q2"] = ("held", f"no stage in {len(ident)} seeds; "
                             f"{left} of {len(ident)} left chance at all")
    return out


EXIT = {"failed": 1, "pending": 2, "undetermined": 3, "held": 0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="First D4 character learned, E0.4")
    ap.add_argument("--runs", default=Path("runs"), type=Path)
    ap.add_argument("--json", default=None, type=Path)
    ap.add_argument("--registered", action="store_true",
                    help="evaluate D-060: exit 1 failed, 2 pending, 3 undetermined, 0 held")
    args = ap.parse_args(argv)
    if not args.runs.is_dir():
        print(f"no runs directory at {args.runs}")
        return 2
    rows = scan(args.runs)
    if args.registered:
        verdicts = registered(rows)
        for k, (v, detail) in verdicts.items():
            print(f"{k}  {v:12s} {detail}")
        states = {v for v, _ in verdicts.values()}
        for worst in ("failed", "pending", "undetermined"):
            if worst in states:
                return EXIT[worst]
        return 0
    print(f"stage: one character >= {STAGE_HIGH}, both others <= {STAGE_LOW}, "
          f"for {STAGE_MIN_EVALS} consecutive evals")
    for r in rows:
        st = f"{r['stage']['character']} at {r['stage']['step']}" if r["stage"] else "none"
        print(f"{r['run_id']:32s} {str(r['arm']):24s} evals {r['evals']:4d}  "
              f"left chance {r['leave_step']}  stage {st}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
