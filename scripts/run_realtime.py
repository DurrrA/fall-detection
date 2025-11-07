import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import tensorflow as tf


def load_labels(labels_path: Path | None) -> list[str]:
    # Returns label list ordered by index. Fallback to ["fall","not-fall"].
    if labels_path and labels_path.exists():
        mapping = json.loads(labels_path.read_text())
        idxs = sorted(map(int, mapping.keys()))
        return [mapping[str(i)] for i in idxs]
    return ["fall", "not-fall"]


def find_fall_index(labels: list[str]) -> int:
    # Find "fall" label index; fallback to 0 if not found
    for i, name in enumerate(labels):
        if name.lower().replace(" ", "").replace("-", "") == "fall":
            return i
    return 0


def resolve_source(source_str: str) -> int | str:
    # Webcam index
    if source_str.isdigit():
        return int(source_str)
    p = Path(source_str)
    if p.exists():
        return p.as_posix()
    # Try common dataset folder
    candidates = [
        Path("data/fall_dataset/videos") / source_str,
        Path("data/fall_dataset/videos") / Path(source_str).name,
    ]
    for c in candidates:
        if c.exists():
            return c.as_posix()
    raise RuntimeError(
        f"Could not open source: {source_str}\n"
        f"Checked: {p.as_posix()}, {candidates[0].as_posix()}, {candidates[1].as_posix()}\n"
        f"CWD: {Path.cwd().as_posix()}"
    )


def main():
    parser = argparse.ArgumentParser("Realtime Fall Detection")
    parser.add_argument("--checkpoint", type=str, default="models/best.keras")
    parser.add_argument("--labels", type=str, default="models/labels.json")
    parser.add_argument("--source", type=str, default="0", help="0 for webcam or path to video")
    parser.add_argument("--img-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--threshold", type=float, default=0.6, help="probability threshold")
    parser.add_argument("--smooth", type=float, default=0.8, help="EMA smoothing factor (0..1)")
    parser.add_argument("--min-frames", type=int, default=3, help="consecutive frames to trigger")
    parser.add_argument("--latch-sec", type=float, default=2.0, help="keep FALL overlay this many seconds")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.checkpoint)
    labels = load_labels(Path(args.labels))
    fall_idx = find_fall_index(labels)

    src = resolve_source(args.source)
    print(f"Opening source: {src}")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {src}")

    # Optional: set a reasonable FPS cap/hints
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    smoothed = 0.0
    consec = 0
    last_trigger = 0.0
    last_print_state = False  # to avoid spamming "fall" every frame

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Preprocess: DO NOT divide by 255; model graph handles preprocess_input
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, tuple(args.img_size), interpolation=cv2.INTER_LINEAR)
        inp = resized.astype("float32")[None, ...]  # [1,H,W,3]

        preds = model.predict(inp, verbose=0)
        if preds.ndim == 2 and preds.shape[1] == 1:
            p_fall = float(preds[0, 0])
            pred_idx = 0 if p_fall >= 0.5 else 1
        else:
            probs = preds[0].astype(float)
            pred_idx = int(np.argmax(probs))
            p_fall = float(probs[fall_idx])

        alpha = float(np.clip(args.smooth, 0.0, 1.0))
        smoothed = alpha * smoothed + (1.0 - alpha) * p_fall
        if smoothed >= args.threshold:
            consec += 1
        else:
            consec = 0

        now = time.time()
        triggered = (consec >= args.min_frames) or (pred_idx == fall_idx and smoothed >= args.threshold)
        if triggered:
            last_trigger = now
            if not last_print_state:
                print("fall")
                try:
                    from src.notifications.notifier import Notifier  # optional
                    Notifier().notify("fall")
                except Exception:
                    pass
                last_print_state = True
        else:
            if (now - last_trigger) > args.latch_sec:
                last_print_state = False

        show_fall = (now - last_trigger) <= args.latch_sec
        color = (0, 0, 255) if show_fall else (0, 255, 0)
        text = "FALL" if show_fall else labels[pred_idx] if 0 <= pred_idx < len(labels) else str(pred_idx)
        info = f"p_fall={p_fall:.2f}  smooth={smoothed:.2f}"

        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, info, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Fall Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()