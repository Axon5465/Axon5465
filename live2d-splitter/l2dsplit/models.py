"""Download and cache the small model files the local engine uses."""
import os
import urllib.request
from pathlib import Path

CACHE = Path(os.environ.get("L2DSPLIT_CACHE", Path.home() / ".cache" / "l2dsplit"))

MODELS = {
    # Anime character background removal (ISNet, from SkyTNT/anime-segmentation, packaged by rembg)
    "isnet-anime.onnx": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx",
    # Anime face detector (OpenCV LBP cascade by nagadomi)
    "lbpcascade_animeface.xml": "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml",
}


def get(name: str) -> Path:
    """Return the local path of a model, downloading it on first use."""
    path = CACHE / name
    if path.exists() and path.stat().st_size > 0:
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"[l2dsplit] downloading {name} ...")
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(MODELS[name], tmp)
    tmp.rename(path)
    return path
