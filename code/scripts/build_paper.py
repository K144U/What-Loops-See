"""Build docs/draft.md into one self-contained HTML, figures included.

    python scripts/build_paper.py
    python scripts/build_paper.py --out /tmp/paper.html --figures ../figures

**WHY NOT LATEX.** Nothing on this machine or the cluster has pandoc,
pdflatex or a PDF library. A build that needs a toolchain nobody has is a
build that does not run, and the point of this one is that it reruns every
time a result lands. The output is a single HTML file with the figures
embedded as data URIs, so it opens anywhere, survives being emailed, and
prints to PDF from any browser. A NeurIPS LaTeX submission is a milestone 12
job and this is not that.

**WHERE THE FIGURES GO.** The draft cites figures in prose, as "Figure 9".
This inserts each figure immediately after the paragraph that first cites it,
so no placement lives in the markdown and the draft stays readable as text.
A figure cited but missing from disk becomes a visible placeholder rather
than a silent omission: a paper that quietly drops a figure it refers to is
worse than one that says the figure is not built yet.

The figure numbering comes from the citations themselves, so adding F7 later
needs no change here. The draft cites by file number and the reader sees the
figures numbered in order of first citation; captions are written in the draft
as HTML comments, <!-- figure 9: ... -->, beside the text they belong to.
"""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

#: Citation to file stem. The draft says "Figure 9" and the plot module is
#: f9_binding, so the mapping is by number and the stem is discovered on
#: disk rather than hardcoded twice.
CITATION = re.compile(r"\*\*Figure (\d+)\*\*")

CSS = """
:root { color-scheme: light; }
body { max-width: 46rem; margin: 0 auto; padding: 2.5rem 1.5rem 6rem;
       font: 15px/1.65 Georgia, 'Times New Roman', serif; color: #1a1a1a; }
h1 { font-size: 1.75rem; line-height: 1.25; margin: 0 0 0.4rem; }
h2 { font-size: 1.2rem; margin: 2.4rem 0 0.6rem;
     border-bottom: 1px solid #ddd; padding-bottom: 0.25rem; }
h3 { font-size: 1.02rem; margin: 1.8rem 0 0.4rem; }
p, li { margin: 0.6rem 0; }
code { font: 12.5px/1.4 Consolas, monospace; background: #f4f4f4;
       padding: 0.1em 0.3em; border-radius: 2px; }
pre { background: #f6f6f6; padding: 0.8rem; overflow-x: auto; }
table { border-collapse: collapse; margin: 1rem 0; font-size: 13.5px;
        width: 100%; }
th, td { border: 1px solid #ccc; padding: 0.35rem 0.6rem; text-align: left; }
th { background: #f0f0f0; }
blockquote { border-left: 3px solid #ccc; margin: 1rem 0; padding: 0.2rem 1rem;
             color: #444; }
figure { margin: 1.6rem 0; page-break-inside: avoid; }
figure img { display: block; max-width: 100%; margin: 0 auto;
             border: 1px solid #e2e2e2; }
figcaption { font-size: 12.5px; color: #555; margin-top: 0.4rem; }
td, th { overflow-wrap: break-word; }
/* A path in a cell may break anywhere, and words elsewhere may not. */
td code { display: inline-block; max-width: 100%; box-sizing: border-box;
          overflow-wrap: anywhere; }
.missing { border: 1px dashed #c00; color: #c00; padding: 1rem;
           font-size: 13px; }
hr { border: 0; border-top: 1px solid #ddd; margin: 2rem 0; }
@page { size: A4; margin: 20mm 18mm; }
@media print {
  body { max-width: none; padding: 0; font-size: 10.5pt; }
  h2, h3 { page-break-after: avoid; }
  /* A collapsed table's outer border straddles the column edge, and on
     2026-10-22 a reader's copy lost every table's right border to it. */
  table { width: calc(100% - 2px); font-size: 9pt; }
  /* A figure taller than what is left of a page jumps to the next one and
     leaves the rest blank; capped, most fit below the text that cites them.
     The best cap moves with the text, so after a prose edit sweep 90 to 110
     and keep the one that leaves the least blank space: 100mm on the final
     22 October draft, after it had been 95mm for one revision that day. */
  figure img { max-height: 100mm; width: auto; }
  tr { page-break-inside: avoid; }
}
"""


def find_figures(figdir: Path) -> dict[int, Path]:
    """Number to file, read off the filenames rather than a second list."""
    out: dict[int, Path] = {}
    for p in sorted(figdir.glob("f*.png")):
        m = re.match(r"f(\d+)_", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def default_figures(root: Path) -> Path:
    """Where the figures are, which is not the same place on both machines.

    On the Windows machine the repository sits beside its figures, in a
    parent directory that holds both. On the cluster the figures are inside
    the repository. The default was `root.parent / "figures"`, which is the
    local layout, so the build ran there and died on the cluster with
    "no figure directory" **even though nine figures were sitting in the
    checkout**. That matters more than an ordinary path bug: the cluster is
    the only machine that has the runs, and this script exists to rebuild
    the draft as results land.

    Inside the repository wins when both exist, because that copy is the one
    next to the metrics the figures were drawn from.
    """
    inside = root / "figures"
    return inside if inside.is_dir() else root.parent / "figures"


def embed(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


#: A caption lives in the draft, beside the text it belongs to, as an HTML
#: comment that no renderer shows: <!-- figure 9: what the reader sees -->.
#: The number is the file's, like every citation in the draft.
CAPTION = re.compile(r"<!--\s*figure\s+(\d+):\s*(.*?)\s*-->", re.S)


def captions(md: str) -> dict[int, str]:
    return {int(n): " ".join(text.split()) for n, text in CAPTION.findall(md)}


def display_order(md: str) -> dict[int, int]:
    """File number to the number a reader sees, in order of first citation.

    The draft cites figures by their file number, f9_binding is "Figure 9",
    so a citation never changes when a figure is added or cut. Readers see
    them numbered 1, 2, 3 in the order the text first mentions them. Until
    2026-10-22 the file number was printed as is, and the 9 October draft
    showed its figures as 1, 9, 8, 2, 6, 3, 10, 4, 5.
    """
    order: dict[int, int] = {}
    for m in CITATION.finditer(md):
        n = int(m.group(1))
        if n not in order:
            order[n] = len(order) + 1
    return order


def renumber(text: str, order: dict[int, int]) -> str:
    return CITATION.sub(lambda m: f"**Figure {order[int(m.group(1))]}**", text)


def insert_figures(md: str, figs: dict[int, Path],
                   caps: dict[int, str] | None = None) -> tuple[str, list[int], list[int]]:
    """Put each figure after the paragraph that first cites it, numbered in
    order of first citation, with the caption the draft gives it.

    Returns the markdown, and the FILE numbers placed and cited but not found
    on disk, so a report can name the file a reader's number came from.
    """
    caps = captions(md) if caps is None else caps
    order = display_order(md)
    paragraphs = md.split("\n\n")
    placed: list[int] = []
    missing: list[int] = []
    out: list[str] = []

    for para in paragraphs:
        out.append(renumber(para, order))
        # A figure goes in once, after its FIRST citation, so a figure
        # referred to three times does not appear three times.
        for num in sorted({int(n) for n in CITATION.findall(para)}, key=order.get):
            if num in placed or num in missing:
                continue
            shown = order[num]
            path = figs.get(num)
            if path is None:
                missing.append(num)
                out.append(
                    f'<p class="missing">Figure {shown} (file f{num}) is cited '
                    f"here and has not been built yet.</p>")
                continue
            placed.append(num)
            text = caps.get(num)
            body = (text if text else
                    f'<span class="missing">No caption written for f{num}.</span>')
            out.append(
                f"<figure><img alt='Figure {shown}' src='{embed(path)}'>"
                f"<figcaption><b>Figure {shown}.</b> {body}</figcaption></figure>")
    return "\n\n".join(out), placed, missing


def build(draft: Path, figdir: Path, out: Path) -> int:
    import markdown

    figs = find_figures(figdir)
    md_text, placed, missing = insert_figures(
        draft.read_text(encoding="utf-8"), figs)

    body = markdown.markdown(
        md_text, extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])

    title = "What a Loop Sees"
    m = re.search(r"^#\s+(.+)$", draft.read_text(encoding="utf-8"), re.M)
    if m:
        title = m.group(1)

    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title><style>{CSS}</style></head><body>"
        f"{body}</body></html>")
    out.write_text(html, encoding="utf-8")

    size = out.stat().st_size / 1_000_000
    print(f"wrote {out}  ({size:.1f} MB, {len(placed)} figure(s) embedded)")
    if placed:
        text = draft.read_text(encoding="utf-8")
        order, caps = display_order(text), captions(text)
        print("  placed, as the reader numbers them: "
              + ", ".join(f"Figure {order[n]} from f{n}" for n in placed))
        nocap = [n for n in placed if n not in caps]
        if nocap:
            print("  NO CAPTION WRITTEN: " + ", ".join(f"f{n}" for n in nocap))
    if missing:
        print("  CITED BUT NOT BUILT: " + ", ".join(f"F{n}" for n in missing))
        print("  Those appear in the document as visible placeholders, not "
              "silent gaps.")
    unused = sorted(set(figs) - set(placed))
    if unused:
        print("  built but never cited: " + ", ".join(f"F{n}" for n in unused))
        print("  A figure nobody refers to is either missing a citation or "
              "does not belong in the paper.")
    return 0


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Build the paper draft to HTML")
    ap.add_argument("--draft", default=root / "docs" / "draft.md", type=Path)
    ap.add_argument("--figures", default=default_figures(root), type=Path)
    ap.add_argument("--out", default=root / "docs" / "paper.html", type=Path)
    args = ap.parse_args(argv)

    if not args.draft.is_file():
        print(f"no draft at {args.draft}")
        return 2
    if not args.figures.is_dir():
        print(f"no figure directory at {args.figures}")
        return 2
    return build(args.draft, args.figures, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
