"""Command line: `python -m l2dsplit split image.png -o out/` or `... from-psd file.psd -o out/`."""
import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import background, export, layers as layermod, segment


def _write_all(layers, original_rgba, out: Path, stem: str, captions: bool):
    out.mkdir(parents=True, exist_ok=True)
    order = export.save_pngs(layers, out)
    export.save_psd(layers, out / f"{stem}.psd")
    export.save_atlas(layers, original_rgba, out / "atlas.png", captions=captions)
    size = original_rgba.shape[1], original_rgba.shape[0]
    export.composite(layers, size).save(out / "recomposed.png")
    print(f"[l2dsplit] {len(order)} layers -> {out}")
    for n in order:
        print("   ", n)


def cmd_split(a):
    t = time.time()
    src = Image.open(a.image)
    if src.mode in ("RGBA", "LA") and np.asarray(src.convert("RGBA"))[..., 3].min() < 250:
        rgba = np.asarray(src.convert("RGBA"))
        rgb, alpha = rgba[..., :3].copy(), rgba[..., 3].copy()
        print("[l2dsplit] background: using the image's own transparency")
    else:
        rgb = np.asarray(src.convert("RGB")).copy()
        alpha = background.remove_background(rgb, a.bg)
    if a.max_size and max(rgb.shape[:2]) > a.max_size:
        s = a.max_size / max(rgb.shape[:2])
        sz = (int(rgb.shape[1] * s), int(rgb.shape[0] * s))
        rgb = np.asarray(Image.fromarray(rgb).resize(sz, Image.LANCZOS))
        alpha = np.asarray(Image.fromarray(alpha).resize(sz, Image.LANCZOS))
    out = Path(a.out or Path(a.image).with_suffix("")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    labels, geo = segment.segment(rgb, alpha)
    if a.labels:
        print(f"[l2dsplit] using hand-edited label map {a.labels}")
        labels = export.load_label_map(a.labels, (rgb.shape[1], rgb.shape[0]))
    export.save_label_map(labels, alpha, out / "labels.png")
    export.save_legend(out / "labels_legend.png")

    layers = layermod.build_layers(rgb, alpha, labels, geo, fill_hidden=not a.no_fill)
    original = np.dstack([rgb, alpha])
    _write_all(layers, original, out, Path(a.image).stem, a.captions)
    print(f"[l2dsplit] done in {time.time() - t:.1f}s")


def cmd_from_psd(a):
    from .psd_import import load_psd

    layers, comp = load_psd(a.psd)
    if comp is None:
        comp = np.asarray(export.composite(layers, next(iter(layers.values())).shape[1::-1]))
    out = Path(a.out or Path(a.psd).with_suffix("")).resolve()
    _write_all(layers, comp, out, Path(a.psd).stem + "_parts", a.captions)


def main(argv=None):
    p = argparse.ArgumentParser(prog="l2dsplit", description="Cut an anime character into Live2D-style PNG/PSD layers.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split", help="split a character image with the built-in CPU engine")
    s.add_argument("image")
    s.add_argument("-o", "--out", help="output folder (default: next to the image)")
    s.add_argument("--bg", default="auto", choices=["auto", "flat", "isnet"],
                   help="background removal: flat colour key, ISNet anime model, or auto-pick")
    s.add_argument("--labels", help="hand-edited labels.png from a previous run: re-cut using it")
    s.add_argument("--no-fill", action="store_true", help="do not paint hidden areas under front layers")
    s.add_argument("--max-size", type=int, default=2048, help="downscale if the long side is bigger (0 = off)")
    s.add_argument("--captions", action="store_true", help="write part names on the atlas sheet")
    s.set_defaults(fn=cmd_split)

    f = sub.add_parser("from-psd", help="export PNG parts + atlas from a layered PSD (e.g. See-through output)")
    f.add_argument("psd")
    f.add_argument("-o", "--out")
    f.add_argument("--captions", action="store_true")
    f.set_defaults(fn=cmd_from_psd)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
