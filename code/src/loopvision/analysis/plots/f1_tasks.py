"""F1. What the benchmark actually looks like, with depth and breadth crossed.

Three rows of REAL samples from the generator, not a drawing. A schematic
drawn by hand can flatter the design; these are the images the models were
trained on, produced by `data.dataset.generate` at a fixed seed.

  row 1   family A, composition depth 1, 2 and 3. Sequential work rises and
          the scene does not.
  row 2   family C, breadth 4, 6 and 8 at depth 1. Parallel work rises and
          the sequential work does not. **The two rows are the benchmark's
          whole claim: the axes move independently.**
  row 3   the same group acting on position and on colour, which is the
          decisive comparison of Section 5.3 and the one a reader has to see
          to believe that only the substrate differs.

**GREYSCALE, and why the third row is a special case.** Section 6 requires
every figure to survive greyscale. Rows 1 and 2 do: the structure is layout.
Row 3 cannot, and that is the point rather than a defect. The appearance
condition puts the task in the glyph colours, so a greyscale reader loses
exactly the information the model has to compose, which is a fair visual
statement of why the two conditions are not interchangeable. The caption says
so, and the row is labelled in text as well as colour so the panel is still
identifiable without it.

    python -m loopvision.analysis.plots.f1_tasks --out figures/f1_tasks.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loopvision.analysis.plots import style

#: Fixed so the figure is identical on every regeneration and a reader
#: comparing two drafts sees only real changes.
SAMPLE_INDEX = 0
SPLIT = "iid_val"


def _db(sample) -> str:
    """Depth and breadth read off the sample's own program string.

    Typed captions drift. This figure exists to assert that the two axes move
    independently, so the numbers asserting it come from the generated sample
    rather than from what the caller asked for. The program is the record the
    renderer itself used.
    """
    head = sample.program.split(";")
    d = next((x.split("=")[1] for x in head if x.startswith("d=")), "?")
    b = next((x.split("=")[1] for x in head if x.startswith("b=")), "?")
    return f"depth {d}, breadth {b}"


def _panels():
    """(row label, [(caption, Sample)]) for each of the three rows."""
    from loopvision.data import dataset as D

    depth_row = [D.generate("A", SAMPLE_INDEX, SPLIT, D.TaskConfig(depths=(d,)))
                 for d in (1, 2, 3)]
    breadth_row = [D.generate("C", SAMPLE_INDEX, SPLIT,
                              D.TaskConfig(depths=(1,), breadths=(b,)))
                   for b in (4, 6, 8)]
    substrate_row = [
        ("D4 on position",
         D.generate("A", SAMPLE_INDEX, SPLIT,
                    D.TaskConfig(depths=(2,), factor="d4", substrate="native"))),
        ("D4 on colour",
         D.generate("A", SAMPLE_INDEX, SPLIT,
                    D.TaskConfig(depths=(2,), factor="full",
                                 substrate="d4_colour"))),
        ("S3 on position",
         D.generate("A", SAMPLE_INDEX, SPLIT,
                    D.TaskConfig(depths=(2,), factor="full",
                                 substrate="s3_spatial"))),
    ]
    return [
        ("family A", [(_db(s), s) for s in depth_row]),
        ("family C", [(_db(s), s) for s in breadth_row]),
        ("substrate", [(lbl + chr(10) + _db(s), s)
                       for lbl, s in substrate_row]),
    ]


def draw(out: Path | None = None):
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _panels()
    style.use()
    fig, axes = plt.subplots(
        len(rows), 3, figsize=(style.WIDTH_FULL * 0.62, 4.2),
        gridspec_kw={"hspace": 0.62, "wspace": 0.06})

    for r, (row_label, panels) in enumerate(rows):
        for c, (caption, sample) in enumerate(panels):
            ax = axes[r][c]
            # (3, H, W) uint8 to (H, W, 3) for imshow.
            ax.imshow(np.transpose(sample.image, (1, 2, 0)),
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#bbbbbb")
            ax.set_title(caption, fontsize=6, pad=3)
        axes[r][0].set_ylabel(row_label, fontsize=6.5, labelpad=4)

    fig.suptitle("Depth and breadth are crossed, and both are known exactly",
                 x=0.008, ha="left", fontsize=8)
    # The caption lives in the draft. A note drawn here as well printed the
    # caption twice under the figure, which a reader caught on 2026-10-22.
    fig.subplots_adjust(top=0.90, bottom=0.04)

    if out is not None:
        fig.savefig(out, dpi=220, bbox_inches="tight")
        print(f"wrote {out}")
    else:
        for p in style.save(fig, "f1_tasks"):
            print(f"wrote {p}")
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F1 task schematic, real samples")
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args(argv)

    draw(args.out)
    for label, panels in _panels():
        flat = ", ".join(f"{c} label={s.label}" for c, s in panels)
        print(f"  {label.splitlines()[0]:<40} {flat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
