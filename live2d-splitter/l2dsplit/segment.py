"""Heuristic part segmentation for clean, front-facing anime character art (CPU only).

Pipeline:
  1. find the face (anime LBP cascade, silhouette fallback)
  2. sample skin + hair colours from the face box
  3. skin / hair masks by colour distance, kept only where connected to the right seeds
  4. facial features (eyes, brows, mouth, nose) = non-skin blobs inside the face
  5. hair -> front / side L-R / back by geometry around the face
  6. skin -> face / neck / arms / legs, clothing -> garments by position
Returns an integer label map (see parts.ID) plus geometry used later for inpainting.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import models
from .parts import ID

K = np.ones((3, 3), np.uint8)
DEBUG = False


@dataclass
class Geometry:
    face_box: tuple            # x, y, w, h
    face_shape: np.ndarray     # bool mask of the full face incl. area under bangs
    skull: np.ndarray          # bool mask of the head behind the face (filled by back hair)
    eye_boxes: dict = field(default_factory=dict)
    cx: int = 0
    chin_y: int = 0


# ---------------------------------------------------------------- helpers

def _lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _dist(lab, ref, wl=0.5):
    d = lab - np.asarray(ref, np.float32)
    d[..., 0] *= wl
    return np.sqrt((d ** 2).sum(-1))


def _components(mask, min_area=1):
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lab == i, stats[i], cents[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def _keep_touching(mask, seed):
    """Components of `mask` that overlap `seed`."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    ids = np.unique(lab[seed & mask])
    ids = ids[ids > 0]
    return np.isin(lab, ids)


def _clean(mask, k=3):
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return m > 0


def _fill_holes(mask):
    m = mask.astype(np.uint8) * 255
    h, w = m.shape
    flood = m.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return mask | (flood == 0)


def _hull(mask):
    pts = cv2.findNonZero(mask.astype(np.uint8))
    out = np.zeros(mask.shape, np.uint8)
    if pts is not None:
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out > 0


def _mode_color(lab, region, valid=None):
    """Dominant colour in a region (median of the largest coarse colour bin)."""
    sel = region if valid is None else region & valid
    px = lab[sel]
    if len(px) == 0:
        return None
    q = (px // 12).astype(np.int32)
    keys = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    top = vals[np.argmax(counts)]
    return np.median(px[keys == top], axis=0)


# ---------------------------------------------------------------- face

def detect_face(rgb, fg):
    casc = cv2.CascadeClassifier(str(models.get("lbpcascade_animeface.xml")))
    g = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    ys, xs = np.nonzero(fg)
    top, bottom = ys.min(), ys.max()
    faces = casc.detectMultiScale(g, 1.05, 3, minSize=(24, 24))
    # prefer large faces in the top 45% of the silhouette
    best, score = None, -1
    for (x, y, w, h) in faces:
        cy = y + h / 2
        if cy > top + 0.45 * (bottom - top) or fg[int(cy), int(x + w / 2)] == 0:
            continue
        s = w * h * (1.0 - (cy - top) / (bottom - top))
        if s > score:
            best, score = (int(x), int(y), int(w), int(h)), s
    if best is None:
        # fallback: head ~ top 1/6 of the silhouette, centred on its widest early row
        hh = int((bottom - top) / 6)
        band = fg[top: top + hh]
        cols = np.nonzero(band.any(0))[0]
        cx = int(cols.mean())
        best = (cx - hh // 2, top + hh // 4, hh, hh)
        print("[l2dsplit] face not detected - using silhouette estimate")
    return best


# ---------------------------------------------------------------- main

def segment(rgb, alpha):
    h, w = alpha.shape
    fg = alpha > 127
    smooth = cv2.bilateralFilter(rgb, 9, 40, 9)
    lab = _lab(smooth)
    L = lab[..., 0]
    chroma = np.sqrt((lab[..., 1] - 128) ** 2 + (lab[..., 2] - 128) ** 2)

    fx, fy, fw, fh = detect_face(rgb, fg)
    cx = fx + fw // 2
    yy, xx = np.mgrid[0:h, 0:w]
    ys, xs = np.nonzero(fg)
    top, bottom = ys.min(), ys.max()

    def box(x0, y0, x1, y1):
        m = np.zeros((h, w), bool)
        m[max(0, int(fy + y0 * fh)): int(fy + y1 * fh), max(0, int(fx + x0 * fw)): int(fx + x1 * fw)] = True
        return m & fg

    # ---- skin: dominant light colour on the cheeks
    cheek = box(0.25, 0.55, 0.75, 0.75)
    skin_ref = _mode_color(lab, cheek, L > np.percentile(L[cheek], 30))
    skin_d = _dist(lab, skin_ref)
    skin = fg & (skin_d < 13)
    skin = _clean(skin)

    # ---- hair: dominant colour(s) on the forehead band and above the face
    hair_zone = (box(0.15, 0.05, 0.85, 0.30) | box(0.0, -0.25, 1.0, 0.05)) & ~skin
    hair_ref = _mode_color(lab, hair_zone)
    # second shade (highlight / shadow) from what the first one does not explain
    rest = hair_zone & (_dist(lab, hair_ref, 0.35) > 14)
    refs = [hair_ref]
    if rest.sum() > 0.25 * hair_zone.sum():
        refs.append(_mode_color(lab, rest))
    hair_d = np.min([_dist(lab, r, 0.35) for r in refs], axis=0)
    hair = fg & ~skin & (hair_d < 16)
    # include lineart / dark shading surrounded by hair
    hair = _clean(hair, 3)
    seed = box(0.1, -0.3, 0.9, 0.3)
    hair = _keep_touching(hair, seed & hair)
    hair = _fill_holes(hair) & fg & ~skin

    # ---- face skin: skin component under the face box centre
    face_seed = box(0.3, 0.45, 0.7, 0.8) & skin
    face_skin = _keep_touching(skin, face_seed)

    # chin: going down, the first row where the face-connected skin becomes narrow (neck)
    widths = face_skin.sum(1)
    ref_w = max(1, widths[int(fy + 0.5 * fh): int(fy + 0.7 * fh)].max(initial=1))
    chin_y = int(fy + 0.95 * fh)
    for y in range(int(fy + 0.6 * fh), min(h, int(fy + 1.4 * fh))):
        if widths[y] < 0.55 * ref_w:
            chin_y = y
            break
    neck_skin = face_skin & (yy >= chin_y)
    face_skin = face_skin & (yy < chin_y)

    # full face shape = hull of visible face skin, extended up into an ellipse cap under the bangs
    cols = np.nonzero(face_skin.any(0))[0]
    fl, fr = (cols.min(), cols.max()) if len(cols) else (fx, fx + fw)
    fcx, rx = (fl + fr) / 2, (fr - fl) / 2
    face_top = fy + 0.12 * fh
    eye_y = fy + 0.5 * fh
    cap = (((xx - fcx) / max(rx, 1)) ** 2 + ((yy - eye_y) / max(eye_y - face_top, 1)) ** 2) <= 1
    face_shape = (_hull(face_skin) | (cap & (yy <= eye_y))) & fg & (yy < chin_y)
    cx = int(fcx)

    # ---- facial features: non-skin, non-hair inside the visible face hull
    face_hull = _hull(face_skin) & (yy < chin_y)
    labels = np.zeros((h, w), np.uint8)
    geo = Geometry((fx, fy, fw, fh), face_shape, None, {}, cx, chin_y)

    # eyes: pick the blob on each side with the most eye-like pixels (white + saturated iris + dark lash)
    skin_full = _dist(lab, skin_ref, 1.0)
    eye_band = face_hull & fg & (yy > fy + 0.25 * fh) & (yy < fy + 0.75 * fh)
    nonskin = eye_band & (skin_full > 16)
    whiteish = (L > 190) & (chroma < 14)
    vivid = (chroma > 30) & (L < skin_ref[0] - 45)
    darkish = L < min(90, skin_ref[0] - 90)
    grow = max(3, fw // 40)
    blobs = _components(cv2.dilate(nonskin.astype(np.uint8), np.ones((grow, grow), np.uint8)) > 0, 10)
    eyes = {}
    for m, st, ct in blobs:
        m = m & nonskin
        nw, nv, nd = (m & whiteish).sum(), (m & vivid).sum(), (m & darkish).sum()
        if nd == 0 or (nw + nv) == 0 or abs(ct[0] - cx) < 0.06 * fw:
            continue
        # bangs touching the blob from above do not make it an eye
        score = nw + nv + 0.5 * nd - (m & hair).sum() * 0.5
        side = "L" if ct[0] < cx else "R"
        if score > 0 and score > eyes.get(side, (None, None, 0))[2]:
            core = m & (whiteish | vivid | darkish) & ~(hair & ~(whiteish | vivid))
            eyes[side] = (core, cv2.boundingRect(core.astype(np.uint8)), score)
    eye_masks = []
    for side, (core, st, _) in list(eyes.items()):
        x, y, bw, bh = st
        roi = np.zeros((h, w), bool)
        roi[max(0, int(y - 0.15 * bh)): int(y + 1.15 * bh) + 1, max(0, int(x - 0.1 * bw)): int(x + 1.1 * bw)] = True
        em = _fill_holes(_keep_touching(nonskin & roi & (skin_full > 22), core))
        # stop just under the iris so blush strokes below the eye stay on the face
        rows = np.nonzero((em & (vivid | darkish)).any(1))[0]
        if len(rows):
            em &= yy <= rows.max() + max(1, int(0.08 * (rows.max() - rows.min() + 1)))
        eyes[side] = (em, cv2.boundingRect(em.astype(np.uint8)))
        eye_masks.append(em)
    for em in eye_masks:
        hair &= ~em
    eye_bottom = fy + 0.5 * fh
    for side, (em, _) in eyes.items():
        ex, ey, ebw, ebh = cv2.boundingRect(em.astype(np.uint8))
        geo.eye_boxes[side] = (ex, ey, ebw, ebh)
        eye_bottom = max(eye_bottom, ey + ebh)
        hull = _hull(em)
        lash = em & darkish & (yy < ey + 0.4 * ebh)
        lash = _keep_touching(em & (L < 110) & (yy < ey + 0.5 * ebh), lash)
        white = hull & ~lash & whiteish
        iris = hull & ~lash & ~white & em
        labels[hull & ~lash] = ID[f"eyewhite_{side}"]
        labels[iris] = ID[f"iris_{side}"]
        labels[lash] = ID[f"eyelash_{side}"]
        # eyebrows: dark-ish blobs a little above this eye
        above = face_hull & fg & ~hair & (skin_full > 25) & (yy < ey - 1) & (yy > ey - 0.25 * fh) & \
            (xx > ex - 0.15 * ebw) & (xx < ex + 1.15 * ebw)
        labels[above & (labels == 0)] = ID[f"eyebrow_{side}"]

    low = face_hull & fg & ~hair & (labels == 0) & (skin_full > 22) & (yy > eye_bottom) & \
        (np.abs(xx - cx) < 0.2 * fw) & ~((L > skin_ref[0] - 25) & (chroma < 45)) & ~hair
    mouth_y = None
    comps = sorted(_components(low, 3), key=lambda c: -c[1][cv2.CC_STAT_AREA])
    if comps:
        mouth_y = max(comps[0][2][1], max(c[2][1] for c in comps) if len(comps) > 1 else 0)
    for m, st, ct in comps:
        is_mouth = mouth_y is not None and ct[1] >= mouth_y - 0.06 * fh
        labels[m] = ID["mouth" if is_mouth else "nose"]

    # ---- hair split
    skull = cap & (yy <= eye_y) | face_shape
    skull = cv2.dilate(skull.astype(np.uint8), np.ones((max(3, fw // 12),) * 2, np.uint8)) > 0
    geo.skull = skull
    near_face = cv2.dilate(face_shape.astype(np.uint8), np.ones((max(3, fw // 30),) * 2, np.uint8)) > 0
    front = hair & (near_face | ((np.abs(xx - fcx) < rx) & (yy > face_top - 0.15 * fh) & (yy < eye_y)))
    front = _keep_touching(hair & (np.abs(xx - fcx) < 1.05 * rx) & (yy < chin_y), front)
    side_band = (np.abs(xx - fcx) >= 0.8 * rx) & (np.abs(xx - fcx) < 1.45 * rx) & (yy > eye_y - 0.1 * fh) & (yy < chin_y + 0.9 * fh)
    labels[hair & ~front & side_band & (xx < fcx) & (labels == 0)] = ID["side_hair_L"]
    labels[hair & ~front & side_band & (xx >= fcx) & (labels == 0)] = ID["side_hair_R"]
    labels[front & (labels == 0)] = ID["front_hair"]
    labels[hair & (labels == 0)] = ID["back_hair"]

    labels[(face_skin | (face_hull & fg)) & (labels == 0)] = ID["face"]

    # ---- body skin
    body_skin = skin & (labels == 0) & ~hair
    neck_band = (yy >= chin_y) & (yy < chin_y + 0.6 * fh) & (np.abs(xx - cx) < 0.45 * fw)
    labels[(neck_skin | (body_skin & neck_band)) & (labels == 0)] = ID["neck"]

    # crotch: highest row from the bottom where the centre column is background between two legs
    crotch_y = bottom
    for y in range(bottom, int(chin_y + fh), -1):
        if fg[y, max(0, cx - 2): cx + 3].any():
            if y < bottom - 0.05 * (bottom - top):
                break
            continue
        crotch_y = y
    leg_zone = yy > min(crotch_y, bottom - 0.25 * (bottom - top))

    bgpx = ~fg
    for m, st, ct in _components(skin & (labels == 0) & ~hair, max(30, int(0.002 * fg.sum()))):
        side = "L" if ct[0] < cx else "R"
        ring = (cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~m
        if (ring & bgpx).sum() < 0.04 * ring.sum() and m.sum() < 0.01 * fg.sum():
            continue                                        # enclosed skin-coloured pattern, not a limb
        if ct[1] > crotch_y - 0.3 * fh or (m & leg_zone).sum() > 0.5 * m.sum():
            labels[m] = ID[f"leg_{side}"]
        elif abs(ct[0] - cx) > 0.5 * fw:
            labels[m] = ID[f"arm_{side}"]
        else:
            labels[m] = ID["neck"]  # chest / shoulder skin joins the neck layer

    # ---- clothing / accessories
    rest = fg & (labels == 0)
    garments = _split_garments(smooth, rest, fg)
    torso_half = 0.75 * fw
    for m, st, ct in garments:
        x0, y0, bw, bh = st[:4]
        y1 = y0 + bh
        area = m.sum()
        side = "L" if ct[0] < cx else "R"
        lateral = abs(ct[0] - cx)
        in_legs = (m & leg_zone).sum() > 0.6 * area
        if y1 < fy + 0.5 * fh:
            name = "headwear"                              # above the eyes: hats, horns, animal ears
        elif ct[1] < chin_y + 0.5 * fh and lateral < 0.5 * fw and area < 0.02 * fg.sum():
            name = "neckwear"
        elif (in_legs or y0 > crotch_y - 0.5 * fh) and lateral < 0.9 * fw:
            name = "footwear" if (y0 > bottom - 0.12 * (bottom - top)) else "legwear"
            for sd, half in (("L", xx < cx), ("R", xx >= cx)):
                labels[m & half & (labels == 0)] = ID[f"{name}_{sd}"]
            continue
        elif y0 < crotch_y - 0.3 * fh and lateral > torso_half and ct[1] < crotch_y + fh:
            name = f"sleeve_{side}"
        elif y0 > crotch_y - 1.5 * fh and lateral > 0.9 * fw:
            name = "tail"
        elif ct[1] < crotch_y - 1.2 * fh:
            name = "topwear"
        else:
            name = "bottomwear"
        labels[m & (labels == 0)] = ID[name]

    # animal ears: grow headwear blobs above the face into the hair right next to them
    hw = labels == ID["headwear"]
    if hw.any():
        r = max(3, int(0.07 * fw))
        near = cv2.dilate(hw.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
        hb = np.nonzero(hw.any(1))[0].max()
        grow = near & hair & (yy <= hb) & (yy < face_top) & (np.abs(xx - fcx) > 0.25 * fw)
        labels[grow] = ID["headwear"]

    # legs never reach much wider than the feet: wide leg-zone blobs are a tail / object
    feet = np.isin(labels, [ID["footwear_L"], ID["footwear_R"]])
    spread = (np.abs(xx[feet] - cx).max() + 0.2 * fw) if feet.any() else 0.6 * fw
    legw = np.isin(labels, [ID["legwear_L"], ID["legwear_R"]])
    for m, st, ct in _components(legw & (np.abs(xx - cx) > spread), int(0.004 * fg.sum())):
        labels[m] = ID["tail"]

    # a "tail" blob that mostly borders a sleeve is just more sleeve (patterned kimono sleeves)
    for m, st, ct in _components(labels == ID["tail"], 1):
        ring = (cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~m & fg
        ids, cnt = np.unique(labels[ring], return_counts=True)
        if len(ids):
            best = ids[np.argmax(cnt)]
            if best in (ID["sleeve_L"], ID["sleeve_R"]):
                labels[m] = best

    # anything left (lineart slivers) -> nearest labelled neighbour
    labels = _fill_unlabelled(labels, fg)
    return labels, geo


def _split_garments(smooth, rest, fg):
    """Over-segment by colour, then merge small or similar regions into garments."""
    lab = _lab(smooth)
    px = lab[rest]
    if len(px) < 50:
        return []
    k = 8
    sample = px[np.random.default_rng(0).choice(len(px), min(len(px), 40000), replace=False)]
    _, _, centers = cv2.kmeans(sample, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3, cv2.KMEANS_PP_CENTERS)
    cl = np.argmin(((lab[..., None, :] - centers[None, None]) ** 2).sum(-1), -1)
    regions = np.zeros(rest.shape, np.int32)
    nid = 0
    for c in range(k):
        n, cc = cv2.connectedComponents((rest & (cl == c)).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            nid += 1
            regions[cc == i] = nid
    min_area = max(30, int(0.0025 * fg.sum()))
    regions = _merge_small(regions, rest, min_area)
    regions = _merge_similar(regions, lab, 9.0)
    out = []
    for r in np.unique(regions):
        if r == 0:
            continue
        m = regions == r
        x, y, bw, bh = cv2.boundingRect(m.astype(np.uint8))
        M = cv2.moments(m.astype(np.uint8))
        out.append((m, (x, y, bw, bh), (M["m10"] / M["m00"], M["m01"] / M["m00"])))
    return out


def _neighbours(regions, r):
    m = regions == r
    ring = (cv2.dilate(m.astype(np.uint8), K) > 0) & ~m
    ids, counts = np.unique(regions[ring], return_counts=True)
    keep = ids > 0
    return ids[keep], counts[keep]


def _merge_small(regions, rest, min_area):
    changed = True
    while changed:
        changed = False
        ids, areas = np.unique(regions[rest], return_counts=True)
        for r, a in sorted(zip(ids, areas), key=lambda t: t[1]):
            if r == 0 or a >= min_area:
                continue
            nb, cnt = _neighbours(regions, r)
            nb_ok = [(n, c) for n, c in zip(nb, cnt) if n != r]
            if nb_ok:
                regions[regions == r] = max(nb_ok, key=lambda t: t[1])[0]
                changed = True
            # isolated tiny region stays (accessory)
    return regions


def _merge_similar(regions, lab, thr):
    changed = True
    while changed:
        changed = False
        ids = [i for i in np.unique(regions) if i > 0]
        means = {i: lab[regions == i].mean(0) for i in ids}
        for r in ids:
            if r not in means or not (regions == r).any():
                continue
            nb, cnt = _neighbours(regions, r)
            for n, c in sorted(zip(nb, cnt), key=lambda t: -t[1]):
                if n == r or n not in means:
                    continue
                if np.linalg.norm(means[r] - means[n]) < thr:
                    regions[regions == n] = r
                    means[r] = lab[regions == r].mean(0)
                    del means[n]
                    changed = True
    return regions


def _fill_unlabelled(labels, fg):
    """Give leftover foreground pixels the label of the nearest labelled pixel."""
    todo = fg & (labels == 0)
    if not todo.any() or not (labels > 0).any():
        return labels
    src = (labels == 0).astype(np.uint8)
    _, idx = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(labels > 0)
    lut = np.zeros(idx.max() + 1, np.uint8)
    # distanceTransformWithLabels numbers zero-pixels in raster order
    zero_ids = idx[labels > 0]
    lut[zero_ids] = labels[ys, xs]
    out = labels.copy()
    out[todo] = lut[idx[todo]]
    return out
