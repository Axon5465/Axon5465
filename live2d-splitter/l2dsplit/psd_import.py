"""Import a layered PSD (e.g. See-through output) so it gets the same PNG/atlas/PSD exports."""
import re

import numpy as np
from PIL import Image

# See-through tag -> our layer name (anything unknown keeps a cleaned-up version of its own name)
TAGS = {
    "front hair": "front_hair", "back hair": "back_hair", "headwear": "headwear", "face": "face",
    "irides": "iris", "eyebrow": "eyebrow", "eyewhite": "eyewhite", "eyelash": "eyelash",
    "ears": "headwear", "nose": "nose", "mouth": "mouth", "neck": "neck", "neckwear": "neckwear",
    "topwear": "topwear", "handwear": "sleeve", "bottomwear": "bottomwear", "legwear": "legwear",
    "footwear": "footwear", "tail": "tail", "wings": "objects", "objects": "objects",
}


def _clean(name):
    return re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower() or "layer"


def load_psd(path):
    from psd_tools import PSDImage

    psd = PSDImage.open(path)
    w, h = psd.width, psd.height
    layers = {}
    for layer in psd.descendants():
        if layer.is_group() or not layer.has_pixels() or not layer.is_visible():
            continue
        im = layer.topil()
        if im is None:
            continue
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas.paste(im.convert("RGBA"), (layer.left, layer.top))
        raw = layer.name.strip()
        base = raw.lower()
        side = ""
        m = re.match(r"(.*?)[\s_-]*(left|right|l|r)$", base)
        if m and m.group(1) in TAGS:
            base, side = m.group(1), "_L" if m.group(2).startswith("l") else "_R"
        name = TAGS.get(base, _clean(raw)) + side
        while name in layers:
            name += "_2"
        layers[name] = np.asarray(canvas).copy()
    # psd-tools lists bottom -> top; our exporters want front -> back
    layers = dict(reversed(list(layers.items())))
    comp = psd.composite()
    comp = np.asarray(comp.convert("RGBA")) if comp is not None else None
    return layers, comp
