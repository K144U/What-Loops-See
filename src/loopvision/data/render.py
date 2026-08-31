"""Canvas rendering primitives.

The canvas is 32x32 RGB, patchified by the model with a 4x4 stride 4
convolution into an 8x8 grid of 64 patch tokens. Rendering is aligned to
that grid on purpose: every glyph occupies exactly one 4x4 patch, so a
patch token corresponds to exactly one semantic element. Without that
alignment the M2 patching instrument would be patching over positions that
straddle two glyphs, and its spatial axis would be uninterpretable.

That alignment is a measurement decision, not an aesthetic one. Keep it.

A group element is drawn as its own effect on the reference glyph, so an
operator icon is literally the state that operator produces from the
identity. Object states and operator icons therefore share one renderer,
and a model cannot distinguish them by drawing style, only by position.
"""

from __future__ import annotations

import numpy as np

from loopvision.data import groups as G

CANVAS = 32
PATCH = 4
GRID = CANVAS // PATCH  # 8 patches per side, 64 patch tokens

# Column 0 holds an object's current state, columns 1..depth hold its
# operator strip, and the last column holds the query marker. That caps
# family A depth at GRID - 2 = 6, which matches the spec's depth range.
STATE_COL = 0
FIRST_OP_COL = 1
MARKER_COL = GRID - 1
MAX_DEPTH = MARKER_COL - FIRST_OP_COL  # 6
MAX_BREADTH = GRID  # 8 rows, spec uses at most 6

BACKGROUND = (15, 15, 15)
MARKER_COLOUR = (255, 214, 40)

#: Index by colour id. G.NEUTRAL_COLOUR maps to the grey entry.
COLOURS: dict[int, tuple[int, int, int]] = {
    0: (222, 58, 58),
    1: (54, 200, 88),
    2: (62, 104, 232),
    G.NEUTRAL_COLOUR: (205, 205, 205),
}


def blank_canvas(size: int = CANVAS) -> np.ndarray:
    """(3, size, size) uint8 filled with the background colour."""
    canvas = np.empty((3, size, size), dtype=np.uint8)
    for c in range(3):
        canvas[c, :, :] = BACKGROUND[c]
    return canvas


def render_element(index: int, patch: int = PATCH) -> np.ndarray:
    """Draw a group element as a (3, patch, patch) uint8 tile.

    The element is applied to the reference glyph and the result is
    centred in the tile by its bounding box. Centring is deterministic
    given the cell set, so distinct elements give distinct tiles. That is
    asserted rather than assumed, in tests/test_render.py.
    """
    cells = G.act_on_glyph(index)
    xs = [x for x, _, _ in cells]
    ys = [y for _, y, _ in cells]
    width = max(xs) + 1
    height = max(ys) + 1
    if width > patch or height > patch:
        raise ValueError(
            f"glyph is {width}x{height} and does not fit a {patch}x{patch} tile"
        )

    ox = (patch - width) // 2
    oy = (patch - height) // 2

    tile = np.empty((3, patch, patch), dtype=np.uint8)
    for c in range(3):
        tile[c, :, :] = BACKGROUND[c]
    for x, y, colour in cells:
        rgb = COLOURS[colour]
        for c in range(3):
            # Row index is y, column index is x.
            tile[c, oy + y, ox + x] = rgb[c]
    return tile


def render_marker(patch: int = PATCH) -> np.ndarray:
    """The query marker: a filled square inset by one pixel.

    Deliberately unlike any glyph tile, since it is a pointer rather than a
    group element and must not be confusable with one.
    """
    tile = np.empty((3, patch, patch), dtype=np.uint8)
    for c in range(3):
        tile[c, :, :] = BACKGROUND[c]
        tile[c, 1 : patch - 1, 1 : patch - 1] = MARKER_COLOUR[c]
    return tile


def paste_patch(canvas: np.ndarray, tile: np.ndarray, row: int, col: int) -> None:
    """Write a tile into the (row, col) cell of the patch grid, in place."""
    patch = tile.shape[-1]
    if not (0 <= row < canvas.shape[1] // patch and 0 <= col < canvas.shape[2] // patch):
        raise IndexError(f"patch cell ({row}, {col}) is outside the canvas")
    y0, x0 = row * patch, col * patch
    canvas[:, y0 : y0 + patch, x0 : x0 + patch] = tile


def patch_index(row: int, col: int, grid: int = GRID) -> int:
    """Flat patch token index for a grid cell, row major.

    This is the coordinate the M2 heatmaps are indexed by, so it has one
    definition and everything reads it from here.
    """
    return row * grid + col


def to_hwc(canvas: np.ndarray) -> np.ndarray:
    """(3, H, W) to (H, W, 3), for saving or displaying only."""
    return np.transpose(canvas, (1, 2, 0))


def save_png(canvas: np.ndarray, path, scale: int = 8) -> None:
    """Write a canvas to a PNG, nearest-neighbour upscaled to be legible.

    Sample dumps exist to be looked at by a human, and a 32x32 image on a
    modern display is about the size of a full stop.
    """
    from PIL import Image

    img = Image.fromarray(to_hwc(canvas), mode="RGB")
    img = img.resize((canvas.shape[2] * scale, canvas.shape[1] * scale), Image.NEAREST)
    img.save(path)


def save_contact_sheet(canvases, path, cols: int = 8, scale: int = 4, pad: int = 2):
    """Tile many canvases into one PNG so a whole depth can be eyeballed at once."""
    from PIL import Image

    n = len(canvases)
    if n == 0:
        raise ValueError("nothing to render")
    rows = (n + cols - 1) // cols
    h, w = canvases[0].shape[1], canvases[0].shape[2]
    sheet = np.zeros((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), dtype=np.uint8)
    for i, canvas in enumerate(canvases):
        r, c = divmod(i, cols)
        y0 = pad + r * (h + pad)
        x0 = pad + c * (w + pad)
        sheet[y0 : y0 + h, x0 : x0 + w, :] = to_hwc(canvas)
    img = Image.fromarray(sheet, mode="RGB")
    img = img.resize((sheet.shape[1] * scale, sheet.shape[0] * scale), Image.NEAREST)
    img.save(path)
