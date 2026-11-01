"""Background removal: flat-colour keying or the ISNet anime model."""
import cv2
import numpy as np

from . import models


def _largest_component(mask: np.ndarray, keep_ratio: float = 0.02) -> np.ndarray:
    """Keep the main character blob plus any blob attached to it (drops stray parts on a sheet)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = 1 + int(np.argmax(areas))
    keep = lab == big
    return keep


def flat_key(rgb: np.ndarray, tol: float = 18.0) -> np.ndarray:
    """Alpha for images on a plain background: flood the border colour, then soften the edge."""
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    near = (dist < tol).astype(np.uint8)
    # flood from the border only, so bg-coloured pixels inside the character survive
    n, cc = cv2.connectedComponents(near, connectivity=4)
    edge_ids = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
    edge_ids = edge_ids[edge_ids > 0]
    bgmask = np.isin(cc, edge_ids)
    fg = _largest_component(~bgmask)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    # soft edge: alpha ramps with colour distance in a 1-2 px band around the silhouette
    band = cv2.dilate(fg.astype(np.uint8), np.ones((3, 3), np.uint8)) & ~cv2.erode(fg.astype(np.uint8), np.ones((3, 3), np.uint8))
    alpha = fg.astype(np.float32)
    soft = np.clip((dist - tol * 0.5) / (tol * 1.5), 0, 1)
    alpha = np.where(band > 0, np.maximum(alpha * soft, 0), alpha)
    alpha[~cv2.dilate(fg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = 0
    return (alpha * 255).astype(np.uint8)


def isnet(rgb: np.ndarray, size: int = 1024) -> np.ndarray:
    """Alpha from the ISNet anime segmentation model (works on busy backgrounds)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(models.get("isnet-anime.onnx")), providers=["CPUExecutionProvider"])
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    alpha = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    fg = _largest_component(alpha > 0.5)
    alpha = np.where(cv2.dilate(fg.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0, alpha, 0)
    return (alpha * 255).astype(np.uint8)


def is_flat_background(rgb: np.ndarray, tol: float = 12.0) -> bool:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    return float(np.percentile(np.linalg.norm(border - np.median(border, 0), axis=1), 90)) < tol


def remove_background(rgb: np.ndarray, method: str = "auto") -> np.ndarray:
    if method == "auto":
        method = "flat" if is_flat_background(rgb) else "isnet"
    print(f"[l2dsplit] background: {method}")
    return flat_key(rgb) if method == "flat" else isnet(rgb)
