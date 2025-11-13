from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import requests
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from .pose_fall import PoseFallDetector, FallParams

mp_tasks = mp.tasks
vision = mp_tasks.vision
BaseOptions = mp_tasks.BaseOptions
VisionRunningMode = vision.RunningMode

POSE_TASK_URLS = [
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
]


def ensure_pose_task(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return model_path
    last_err = None
    for url in POSE_TASK_URLS:
        try:
            print(f"Downloading pose task model from {url} ...")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"Saved model to {model_path}")
            return model_path
        except Exception as e:
            last_err = e
            print(f"Failed: {e}")
            continue
    raise RuntimeError(f"Failed to download model. Tried: {POSE_TASK_URLS}. Last error: {last_err}")


def lm_mid_hip(landmarks) -> Tuple[float, float]:
    l_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
    r_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
    return ((l_hip.x + r_hip.x) * 0.5, (l_hip.y + r_hip.y) * 0.5)


def draw_pose_landmarks(frame_bgr: np.ndarray, landmarks) -> None:
    lm_list = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks:
        lm_list.landmark.add(
            x=float(lm.x),
            y=float(lm.y),
            z=float(getattr(lm, "z", 0.0)),
            visibility=float(getattr(lm, "visibility", 0.0)),
            presence=float(getattr(lm, "presence", 0.0)),
        )
    mp.solutions.drawing_utils.draw_landmarks(
        frame_bgr,
        lm_list,
        mp.solutions.pose.POSE_CONNECTIONS,
    )


@dataclass
class Track:
    detector: PoseFallDetector
    last_pos: Tuple[float, float]
    last_seen: float


class MultiPoseFall:
    def __init__(self, params: FallParams, max_dist: float = 0.15, ttl_sec: float = 1.0):
        self.params = params
        self.max_dist = float(max_dist)
        self.ttl_sec = float(ttl_sec)
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def _match_tracks(self, positions: List[Tuple[float, float]]) -> Dict[int, int]:
        unmatched = set(self.tracks.keys())
        assignment: Dict[int, int] = {}
        for i, p in enumerate(positions):
            best_id = -1
            best_d = 1e9
            for tid in list(unmatched):
                q = self.tracks[tid].last_pos
                d = np.hypot(p[0] - q[0], p[1] - q[1])
                if d < best_d:
                    best_d, best_id = d, tid
            if best_id != -1 and best_d <= self.max_dist:
                assignment[i] = best_id
                unmatched.discard(best_id)
        return assignment

    def update(self, poses: List[List]) -> Tuple[bool, List[Tuple[int, dict]]]:
        now = time.time()
        positions = [lm_mid_hip(p) for p in poses]
        assignment = self._match_tracks(positions)

        for i, pos in enumerate(positions):
            if i not in assignment:
                tid = self.next_id
                self.next_id += 1
                assignment[i] = tid
                self.tracks[tid] = Track(
                    detector=PoseFallDetector(self.params),
                    last_pos=pos,
                    last_seen=now,
                )

        any_latched = False
        infos: List[Tuple[int, dict]] = []
        for i, tid in assignment.items():
            latched, info = self.tracks[tid].detector.update(poses[i])
            self.tracks[tid].last_pos = positions[i]
            self.tracks[tid].last_seen = now
            info["latched"] = latched
            infos.append((tid, info))
            any_latched = any_latched or latched

        # prune stale
        for tid in [tid for tid, t in self.tracks.items() if (now - t.last_seen) > self.ttl_sec]:
            del self.tracks[tid]

        return any_latched, infos


class PoseTrackerTasks:
    def __init__(self, model_path: Path, num_poses: int = 5):
        self._ts_ms = 0
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=int(num_poses),
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray) -> List:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33  # ~30 FPS
        result = self.landmarker.detect_for_video(mp_image, self._ts_ms)
        return result.pose_landmarks or []

    def close(self):
        self.landmarker.close()