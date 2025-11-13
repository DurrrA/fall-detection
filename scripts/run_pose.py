import argparse
from pathlib import Path
import time
import cv2
import sys

# Add project root so 'src' package is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.realtime.pose_fall import PoseTracker, PoseFallDetector, FallParams, draw_pose

def resolve_source(source_str: str) -> int | str:
    if source_str.isdigit():
        return int(source_str)
    p = Path(source_str)
    if p.exists():
        return p.as_posix()
    # dataset guesses
    for c in [
        Path("data/fall_dataset/videos") / source_str,
        Path("data/fall_dataset/videos") / Path(source_str).name,
    ]:
        if c.exists():
            return c.as_posix()
    raise RuntimeError(f"Source not found: {source_str}. CWD={Path.cwd().as_posix()}")

def main():
    ap = argparse.ArgumentParser("Pose-based Realtime Fall Detection")
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--angle", type=float, default=55.0, help="torso angle threshold (deg)")
    ap.add_argument("--speed", type=float, default=0.04, help="downward hip speed threshold")
    ap.add_argument("--gap", type=float, default=0.15, help="head-ankle gap after fall")
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--latch-sec", type=float, default=2.0)
    ap.add_argument("--smooth", type=float, default=0.8)
    args = ap.parse_args()

    params = FallParams(
        angle_thresh_deg=args.angle,
        speed_thresh=args.speed,
        head_ankle_gap=args.gap,
        min_frames=args.min_frames,
        latch_sec=args.latch_sec,
        smooth=args.smooth,
    )
    detector = PoseFallDetector(params)
    tracker = PoseTracker()

    cap_src = resolve_source(args.source)
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {cap_src}")

    last_print = 0.0
    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = tracker.process(frame)
        latched = False
        info = {}
        if res.pose_landmarks:
            latched, info = detector.update(res.pose_landmarks.landmark)
            draw_pose(frame, res)

        now = time.time()
        if latched and now - last_print > 0.5:
            print("fall")
            last_print = now

        color = (0, 0, 255) if latched else (0, 200, 0)
        text = "FALL" if latched else "OK"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        if info:
            dbg = f"a={info.get('torso_angle',0):.1f} spd={info.get('hip_down_speed',0):.3f} gap={info.get('head_ankle_gap',0):.2f}"
            cv2.putText(frame, dbg, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Pose Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()