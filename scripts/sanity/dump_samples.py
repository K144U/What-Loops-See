"""Dump sample images for a human to look at.

Milestone 1's definition of done includes eyeballing 64 samples per depth.
That is not ceremony: a rendering bug that produces plausible-looking
tensors will pass every numerical test and quietly ruin the dataset, and
the cheapest detector is a person glancing at a contact sheet.

    python scripts/sanity/dump_samples.py
    python scripts/sanity/dump_samples.py --family A --out runs/_gate/samples
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loopvision.data import dataset as D
from loopvision.data import render


def dump_family(family: str, out_dir: Path, per_depth: int, cfg: D.TaskConfig) -> None:
    module = D.get_family(family)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_depth: dict[int, list] = {}
    programs: dict[int, list[str]] = {}

    # Walk the index space until every depth has enough samples. Depth is
    # drawn per sample, so we cannot ask for a specific one directly.
    for split in sorted(module.SUPPORTED_SPLITS):
        # Per-family ranges, not the global defaults. Using D.SPLITS here
        # made this look for family A's depths (4, 5, 6) inside family B,
        # so family B depth 3 was never dumped and two splits spun to the
        # index cap looking for depths that do not exist.
        spec = D.split_spec(family, split)
        wanted = {d for d in spec.depth}
        idx = 0
        while wanted and idx < 20_000:
            sample = D.generate(family, D.global_index(split, idx), split, cfg)
            bucket = by_depth.setdefault(sample.depth, [])
            if sample.depth in wanted and len(bucket) < per_depth:
                bucket.append(sample.image)
                programs.setdefault(sample.depth, []).append(sample.program)
                if len(bucket) >= per_depth:
                    wanted.discard(sample.depth)
            idx += 1

    for depth in sorted(by_depth):
        images = by_depth[depth][:per_depth]
        if not images:
            continue
        sheet = out_dir / f"family{family}_depth{depth}.png"
        render.save_contact_sheet(images, sheet, cols=8)
        listing = out_dir / f"family{family}_depth{depth}_programs.txt"
        listing.write_text("\n".join(programs[depth][:per_depth]), encoding="utf-8")
        print(f"{sheet}  ({len(images)} samples)")

    # One large single sample per family, so the layout is legible at all.
    first = D.generate(family, D.global_index("train", 0), "train", cfg)

    render.save_png(first.image, out_dir / f"family{family}_single.png", scale=16)
    print(f"{out_dir / f'family{family}_single.png'}  (program: {first.program})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", default="A", help="A, B, C, or 'all'")
    p.add_argument("--out", default="runs/_gate/samples")
    p.add_argument("--per-depth", type=int, default=64)
    p.add_argument("--presentation", default="strip")
    args = p.parse_args()

    cfg = D.TaskConfig(presentation=args.presentation)
    families = ["A", "B", "C"] if args.family == "all" else [args.family]
    for family in families:
        try:
            dump_family(family, Path(args.out), args.per_depth, cfg)
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"family {family} not implemented yet: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
