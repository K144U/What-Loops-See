"""The run index. Every count that appears in prose comes from here.

CLAUDE.md makes this a non-negotiable: counts and totals in the documents
are read off this output rather than typed by hand, because a hand-typed
total is correct on the day it is written and silently wrong afterwards.

    python -m loopvision.analysis.registry
    python -m loopvision.analysis.registry --family A --json out.json

A run is a directory under `runs/` holding a `config.yaml`. Status is read
from the filesystem rather than tracked separately, so it cannot drift:

    done     a DONE sentinel is present
    live     metrics exist and no sentinel
    empty    a directory was made and nothing was written

`empty` is reported rather than skipped. A run that was submitted and
produced nothing is the case most worth seeing, and filtering it out would
make a failed launch look like a launch that never happened.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import yaml


def _read_metrics(run_dir: Path) -> tuple[int | None, float | None, float | None]:
    """Latest step, latest iid_val accuracy, latest loss. None if unreadable.

    A partially written parquet is normal: a live run may be flushing while
    this reads. That is not an error worth failing the whole index over.
    """
    path = run_dir / "metrics.parquet"
    if not path.is_file():
        return None, None, None
    try:
        import pandas as pd

        df = pd.read_parquet(path)
    except Exception:
        return None, None, None
    if df.empty:
        return None, None, None

    step = int(df.step.max())
    acc = df[(df.metric_name == "accuracy") & (df.split == "iid_val")]
    loss = df[df.metric_name == "loss"]
    a = float(acc[acc.step == acc.step.max()].value.mean()) if len(acc) else None
    l = float(loss[loss.step == loss.step.max()].value.mean()) if len(loss) else None
    return step, a, l


def scan(runs_root: Path) -> list[dict]:
    out = []
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        cfg_path = d / "config.yaml"
        cfg = {}
        if cfg_path.is_file():
            try:
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            except Exception:
                cfg = {}
        step, acc, loss = _read_metrics(d)
        status = (
            "done" if (d / "DONE").is_file()
            else "live" if step is not None
            else "empty"
        )
        out.append({
            "run_id": d.name,
            "status": status,
            "family": cfg.get("family"),
            "arch": cfg.get("arch"),
            "factor": cfg.get("factor", "full"),
            "depths": cfg.get("depths"),
            "breadths": cfg.get("breadths"),
            "target_steps": cfg.get("steps"),
            "step": step,
            "accuracy": acc,
            "loss": loss,
            "checkpoints": len(list((d / "checkpoints").glob("step_*.pt")))
            if (d / "checkpoints").is_dir() else 0,
        })
    return out


def counts(rows: list[dict]) -> dict:
    by_status = collections.Counter(r["status"] for r in rows)
    by_family = collections.Counter(r["family"] or "?" for r in rows)
    return {
        "total": len(rows),
        "by_status": dict(sorted(by_status.items())),
        "by_family": dict(sorted(by_family.items())),
        "families_present": sorted({r["family"] for r in rows if r["family"]}),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--family", help="filter to one family")
    p.add_argument("--status", choices=["done", "live", "empty"])
    p.add_argument("--json", help="write the full index here")
    args = p.parse_args()

    root = Path(args.runs_root)
    if not root.is_dir():
        print(f"no runs directory at {root}")
        return 1

    rows = scan(root)
    shown = [
        r for r in rows
        if (args.family is None or r["family"] == args.family)
        and (args.status is None or r["status"] == args.status)
    ]

    print()
    print(f"{'run_id':<34} {'st':<5} {'fam':<4} {'factor':<6} {'step':>9} "
          f"{'of':>9}  {'acc':>7} {'loss':>7}")
    print("-" * 96)
    for r in shown:
        step = f"{r['step']:,}" if r["step"] is not None else "-"
        tgt = f"{r['target_steps']:,}" if r["target_steps"] else "-"
        acc = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "-"
        loss = f"{r['loss']:.4f}" if r["loss"] is not None else "-"
        print(f"{r['run_id']:<34} {r['status']:<5} {str(r['family'] or '?'):<4} "
              f"{str(r['factor']):<6} {step:>9} {tgt:>9}  {acc:>7} {loss:>7}")

    c = counts(rows)
    print("-" * 96)
    print(f"  {c['total']} run(s): "
          + ", ".join(f"{k} {v}" for k, v in c["by_status"].items()))
    print("  by family: " + ", ".join(f"{k} {v}" for k, v in c["by_family"].items()))
    print()

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps({"counts": c, "runs": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
