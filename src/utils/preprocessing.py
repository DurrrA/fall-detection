import os
import cv2
import numpy as np
from typing import Iterable, List, Tuple
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

def preprocess_frame(
    frame: np.ndarray,
    img_size: Tuple[int, int] = (224, 224),
    to_batch: bool = False,
) -> np.ndarray:
    """
    Resize BGR frame -> RGB, resize, convert to float32 (0..255), apply MobileNetV2 preprocess_input.
    Returns array shaped (H,W,3) or (1,H,W,3) if to_batch=True.
    """
    if frame is None:
        raise ValueError("frame is None")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, img_size, interpolation=cv2.INTER_LINEAR)
    arr = resized.astype("float32")  # keep 0..255
    arr = mobilenet_preprocess(arr)   # model expects this preprocessing
    if to_batch:
        return np.expand_dims(arr, axis=0)
    return arr

def preprocess_video(
    video_path: str,
    img_size: Tuple[int, int] = (224, 224),
    max_frames: int | None = None,
) -> np.ndarray:
    """
    Load video and return preprocessed frames as np.ndarray shape (N,H,W,3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames: List[np.ndarray] = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame, img_size=img_size, to_batch=False))
        count += 1
        if max_frames and count >= max_frames:
            break
    cap.release()
    if not frames:
        return np.zeros((0, img_size[1], img_size[0], 3), dtype="float32")
    return np.stack(frames, axis=0)

def preprocess_images(
    image_folder: str,
    img_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Load all images in folder and return preprocessed frames (N,H,W,3).
    """
    files = sorted(
        [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
    )
    images: List[np.ndarray] = []
    for p in files:
        img = cv2.imread(p)
        if img is None:
            continue
        images.append(preprocess_frame(img, img_size=img_size, to_batch=False))
    if not images:
        return np.zeros((0, img_size[1], img_size[0], 3), dtype="float32")
    return np.stack(images, axis=0)