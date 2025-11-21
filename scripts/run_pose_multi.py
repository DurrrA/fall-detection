import argparse
import sys
from pathlib import Path
import time
import cv2
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.realtime.pose_fall import FallParams  
from src.realtime.pose_fall_multi import (
    ensure_pose_task,
    PoseTrackerTasks,
    MultiPoseFall,
    draw_pose_landmarks,
)

def resolve_source(source_str: str) -> int | str:
    if source_str.isdigit():
        return int(source_str)
    p = Path(source_str)
    if p.exists():
        return p.as_posix()
    for c in [
        Path("data/fall_dataset/videos") / source_str,
        Path("data/fall_dataset/videos") / Path(source_str).name,
    ]:
        if c.exists():
            return c.as_posix()
    raise RuntimeError(f"Source not found: {source_str}. CWD={Path.cwd().as_posix()}")

POSE_TASK_URLS = [
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
]

def main():
    ap = argparse.ArgumentParser("Multi-person Pose-based Fall Detection")
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--num-poses", type=int, default=5)
    ap.add_argument("--angle", type=float, default=55.0)
    ap.add_argument("--speed", type=float, default=0.04)
    ap.add_argument("--gap", type=float, default=0.15)
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--latch-sec", type=float, default=2.0)
    ap.add_argument("--smooth", type=float, default=0.8)
    ap.add_argument("--max-dist", type=float, default=0.15, help="match distance in normalized coords")
    ap.add_argument("--ttl-sec", type=float, default=1.0, help="track time-to-live")
    ap.add_argument("--task-model", type=str, default="models/pose/pose_landmarker_full.task")
    args = ap.parse_args()

    model_path = ensure_pose_task(Path(args.task_model))
    tracker = PoseTrackerTasks(model_path, num_poses=args.num_poses)
    params = FallParams(
        angle_thresh_deg=args.angle,
        speed_thresh=args.speed,
        head_ankle_gap=args.gap,
        min_frames=args.min_frames,
        latch_sec=args.latch_sec,
        smooth=args.smooth,
    )
    multi = MultiPoseFall(params, max_dist=args.max_dist, ttl_sec=args.ttl_sec)

    cap_src = resolve_source(args.source)
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {cap_src}")

    print("Press 'q' to quit.")
    last_print = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        poses = tracker.process(frame) or []
        for person in poses:
            draw_pose_landmarks(frame, person)
        any_latched, infos = multi.update(poses)
        now = time.time()
        if any_latched and now - last_print > 0.5:
            print("fall")
            last_print = now

        y0 = 40
        for tid, info in sorted(infos, key=lambda x: x[0]):
            color = (0, 0, 255) if info.get("latched") else (0, 200, 0)
            txt = f"ID {tid} a={info.get('torso_angle',0):.1f} spd={info.get('hip_down_speed',0):.3f} gap={info.get('head_ankle_gap',0):.2f}"
            cv2.putText(frame, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y0 += 25

        title = "Multi-Person Pose Fall Detection"
        cv2.imshow(title, frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()