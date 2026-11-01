"""Write layers out: per-part PNGs, a Live2D-ready PSD, a parts sheet, and an editable label map."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .parts import COLOR, ID, NAMES, PARTS

GROUPS = [  # PSD folders, front to back
    ("Head", lambda n: NAMES.index(n) <= NAMES.index("headwear")),
    ("Body", lambda n: NAMES.index("headwear") < NAMES.index(n) < NAMES.index("back_hair")),
    ("Back hair", lambda n: n == "back_hair"),
]


def _bbox(a):
    ys, xs = np.nonzero(a[..., 3])
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def composite(layers, size):
    w, h = size
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for name in reversed([n for n in NAMES if n in layers]):
        canvas.alpha_composite(Image.fromarray(layers[name]))
    return canvas


def save_pngs(layers, out: Path):
    (out / "layers").mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    manifest = []
    order = [n for n in NAMES if n in layers] + [n for n in layers if n not in NAMES]
    for i, name in enumerate(order):
        arr = layers[name]
        Image.fromarray(arr).save(out / "layers" / f"{i:02d}_{name}.png")
        x0, y0, x1, y1 = _bbox(arr)
        Image.fromarray(arr[y0:y1, x0:x1]).save(out / "parts" / f"{name}.png")
        manifest.append({"name": name, "draw_order": len(order) - i, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                         "file": f"parts/{name}.png"})
    h, w = next(iter(layers.values())).shape[:2]
    (out / "manifest.json").write_text(json.dumps({"canvas": [w, h], "layers": manifest}, indent=2))
    return order


def save_psd(layers, path: Path):
    from psd_tools import PSDImage
    from psd_tools.api.layers import Group, PixelLayer

    h, w = next(iter(layers.values())).shape[:2]
    psd = PSDImage.new("RGB", (w, h), color=(255, 255, 255))
    names = [n for n in NAMES if n in layers] + [n for n in layers if n not in NAMES]
    # psd layer lists are bottom -> top, so add groups back to front
    for gname, pick in reversed(GROUPS):
        members = [n for n in names if (n in NAMES and pick(n))]
        if gname == "Body":
            members += [n for n in names if n not in NAMES]
        if not members:
            continue
        group = Group.new(psd, name=gname)
        for n in reversed(members):
            x0, y0, x1, y1 = _bbox(layers[n])
            im = Image.fromarray(layers[n][y0:y1, x0:x1])
            group.append(PixelLayer.frompil(im, psd, n, top=y0, left=x0))
        psd.append(group)
    psd.save(str(path))


def save_atlas(layers, original_rgba, path: Path, bg=(246, 240, 248), gap=14, captions=False):
    """Parts sheet like a Live2D material sheet: full character on the left, parts packed on the right."""
    h, w = original_rgba.shape[:2]
    crops = []
    for n in [n for n in NAMES if n in layers] + [n for n in layers if n not in NAMES]:
        x0, y0, x1, y1 = _bbox(layers[n])
        crops.append((n, Image.fromarray(layers[n][y0:y1, x0:x1])))
    crops.sort(key=lambda t: -t[1].height)
    area = sum((c.width + gap) * (c.height + gap) for _, c in crops)
    shelf_w = max(int(1.15 * area ** 0.5 * max(1.0, h / max(w, 1)) ** 0.25), max(c.width for _, c in crops) + gap)
    # shelf packing
    x = y = gap
    row_h = 0
    places = []
    for n, c in crops:
        if x + c.width + gap > shelf_w:
            x, y = gap, y + row_h + gap
            row_h = 0
        places.append((n, c, x, y))
        x += c.width + gap
        row_h = max(row_h, c.height)
    sheet_h = max(h + 2 * gap, y + row_h + gap)
    sheet = Image.new("RGBA", (w + 2 * gap + shelf_w, sheet_h), bg + (255,))
    sheet.alpha_composite(Image.fromarray(original_rgba), (gap, gap))
    draw = ImageDraw.Draw(sheet)
    for n, c, px, py in places:
        sheet.alpha_composite(c, (w + 2 * gap + px, py))
        if captions:
            draw.text((w + 2 * gap + px, py), n, fill=(90, 90, 90, 255))
    sheet.convert("RGB").save(path)


def save_label_map(labels, alpha, path: Path):
    pal = np.zeros((256, 3), np.uint8)
    for n, c in PARTS:
        pal[ID[n]] = c
    rgb = pal[labels]
    rgba = np.dstack([rgb, np.where(labels > 0, 255, 0).astype(np.uint8)])
    Image.fromarray(rgba).save(path)


def save_legend(path: Path):
    row = 18
    im = Image.new("RGB", (190, row * len(PARTS) + 8), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for i, (n, c) in enumerate(PARTS):
        d.rectangle([6, 4 + i * row, 26, 4 + i * row + 14], fill=c)
        d.text((34, 5 + i * row), f"{n}  {c}", fill=(30, 30, 30))
    im.save(path)


def load_label_map(path, size):
    """Read a (hand-edited) label map back: each pixel snaps to the nearest palette colour."""
    im = Image.open(path).convert("RGBA").resize(size, Image.NEAREST)
    arr = np.asarray(im)
    pal = np.array([c for _, c in PARTS], np.int32)
    d = ((arr[..., None, :3].astype(np.int32) - pal[None, None]) ** 2).sum(-1)
    labels = (np.argmin(d, -1) + 1).astype(np.uint8)
    labels[(arr[..., 3] < 128) | (d.min(-1) > 60 ** 2)] = 0
    return labels
