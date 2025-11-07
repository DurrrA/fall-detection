from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def load_dataset(
    dataset_root: str | Path,
    image_dir: str = "images",
    img_size: Tuple[int, int] = (224, 224),
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads images from <dataset_root>/<image_dir>/<class_name>/... into memory.
    Returns (X, y):
      - X: float32 array [N, H, W, 3]
      - y: int64 labels [N] with classes sorted alphabetically.
    """
    root = Path(dataset_root)
    data_dir = root / image_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {data_dir}")

    class_names: List[str] = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not class_names:
        raise RuntimeError(f"No class folders found in {data_dir}")

    images: List[np.ndarray] = []
    labels: List[int] = []

    for idx, cname in enumerate(class_names):
        cls_dir = data_dir / cname
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                img = load_img(p.as_posix(), target_size=img_size)
                arr = img_to_array(img)  # float32 RGB
                images.append(arr)
                labels.append(idx)

    if not images:
        raise RuntimeError(f"No images found under {data_dir} with extensions {sorted(IMAGE_EXTS)}")
    X = np.stack(images).astype("float32")
    y = np.array(labels, dtype="int64")
    return X, y