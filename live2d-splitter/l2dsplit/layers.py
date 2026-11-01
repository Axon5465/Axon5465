"""Turn a label map into RGBA layers, painting in the parts hidden behind other layers.

Live2D needs every part to continue under the parts in front of it (face under the
bangs, back hair behind the head, skirt under the top...) so nothing tears open when
the model moves. We grow each part's shape under the layers in front of it and fill
those pixels from the part's own colours with OpenCV inpainting.
"""
import cv2
import numpy as np

from .parts import ID, NAMES, front_of


def _hull(mask):
    pts = cv2.findNonZero(mask.astype(np.uint8))
    out = np.zeros(mask.shape, np.uint8)
    if pts is not None:
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out > 0


def _close(mask, k):
    k = max(3, int(k) | 1)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, el) > 0


def extend_labels_to_soft_edges(labels, alpha):
    """Give anti-aliased edge pixels (alpha 1..127) the label of the nearest labelled pixel."""
    todo = (alpha > 0) & (labels == 0)
    if not todo.any() or not (labels > 0).any():
        return labels
    _, idx = cv2.distanceTransformWithLabels((labels == 0).astype(np.uint8), cv2.DIST_L2, 3,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(idx.max() + 1, labels.dtype)
    lut[idx[labels > 0]] = labels[labels > 0]
    out = labels.copy()
    out[todo] = lut[idx[todo]]
    return out


def _fill_colors(rgb, known, fill, radius=4):
    """Inpaint `fill` pixels using only `known` pixels, working on a crop for speed."""
    ys, xs = np.nonzero(known | fill)
    pad = 8
    y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad + 1)
    crop = rgb[y0:y1, x0:x1].copy()
    k = known[y0:y1, x0:x1]
    f = fill[y0:y1, x0:x1]
    # big fills: inpaint at reduced size, then upsample (much faster, and smooth is fine there)
    scale = 1.0
    if f.sum() > 40000:
        scale = (40000 / f.sum()) ** 0.5
    unknown = (~k).astype(np.uint8)
    if scale < 1.0:
        sz = (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale)))
        small = cv2.resize(crop, sz, interpolation=cv2.INTER_AREA)
        sk = cv2.resize(k.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST)
        small = cv2.inpaint(small, (1 - sk).astype(np.uint8), radius, cv2.INPAINT_TELEA)
        filled = cv2.resize(small, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        filled = cv2.inpaint(crop, unknown, radius, cv2.INPAINT_TELEA)
    out = rgb.copy()
    region = out[y0:y1, x0:x1]
    region[f] = filled[f]
    return out


def build_layers(rgb, alpha, labels, geo=None, fill_hidden=True):
    """Return {name: RGBA uint8 full-canvas array}, front-to-back order."""
    labels = extend_labels_to_soft_edges(labels, alpha)
    fg = alpha > 0
    out = {}
    for name in NAMES:
        m = labels == ID[name]
        if m.sum() < 8:
            continue
        a = np.where(m, alpha, 0).astype(np.uint8)
        layer_rgb = rgb
        if fill_hidden:
            ahead = np.isin(labels, [ID[n] for n in front_of(name)])
            core = m & (alpha > 127)
            area = core.sum()
            if name == "face" and geo is not None:
                ext = geo.face_shape | _hull(core)
            elif name == "back_hair" and geo is not None:
                ext = _close(m, 0.15 * np.sqrt(area)) | (geo.skull & ~_hull(m)) | _hull(m & (np.arange(m.shape[0])[:, None] < geo.chin_y))
                ext &= fg | geo.skull
            elif name.startswith(("eyewhite", "iris")):
                side = name[-1]
                ext = _hull(m | (labels == ID[f"iris_{side}"])) if name.startswith("eyewhite") else _hull(m)
            elif name.startswith(("eyelash", "eyebrow", "mouth", "nose")):
                ext = m                                  # line art: nothing hidden to paint
            else:
                ext = _close(m, 0.12 * np.sqrt(area))
            hidden = ext & ~m & (ahead | ~fg if name == "back_hair" else ahead)
            if name != "back_hair":
                hidden &= fg
            if hidden.any() and core.any():
                layer_rgb = _fill_colors(rgb, core, hidden)
                a = np.where(hidden, 255, a).astype(np.uint8)
                # soft 1px edge on the painted part so it does not look cut out when exposed
                edge = hidden & ~(cv2.erode((a > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
                a[edge] = 160
        out[name] = np.dstack([layer_rgb, a]).astype(np.uint8)
    return out
