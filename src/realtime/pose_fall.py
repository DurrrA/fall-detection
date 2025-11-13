import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
try:
    mp_style = mp.solutions.drawing_styles
except Exception:
    mp_style = None  # fallback


@dataclass
class PoseState:
    last_time: float = 0.0
    last_hip_y: Optional[float] = None
    consec: int = 0
    last_trigger: float = 0.0


@dataclass
class FallParams:
    angle_thresh_deg: float = 55.0     
    speed_thresh: float = 0.04         
    head_ankle_gap: float = 0.15        
    min_frames: int = 3                 
    latch_sec: float = 2.0             
    smooth: float = 0.8                 


class PoseFallDetector:
    def __init__(self, params: FallParams):
        self.params = params
        self.state = PoseState()
        self.ema = 0.0

    @staticmethod
    def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a + b) * 0.5

    @staticmethod
    def _angle_from_vertical_deg(vec: np.ndarray) -> float:
        if np.linalg.norm(vec) < 1e-6:
            return 0.0
        v = vec / (np.linalg.norm(vec) + 1e-8)
        vertical_up = np.array([0.0, -1.0], dtype=np.float32)  # image y goes down
        cosang = float(np.clip(np.dot(v, vertical_up), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))

    def _compute_features(self, lm) -> Tuple[float, float, float]:
        nose = np.array([lm[mp_pose.PoseLandmark.NOSE].x, lm[mp_pose.PoseLandmark.NOSE].y], dtype=np.float32)
        l_sh = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y], dtype=np.float32)
        r_sh = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y], dtype=np.float32)
        l_hip = np.array([lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y], dtype=np.float32)
        r_hip = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y], dtype=np.float32)
        l_ank = np.array([lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y], dtype=np.float32)
        r_ank = np.array([lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y], dtype=np.float32)

        sh_mid = self._midpoint(l_sh, r_sh)
        hip_mid = self._midpoint(l_hip, r_hip)
        ank_mid = self._midpoint(l_ank, r_ank)

        torso_vec = sh_mid - hip_mid
        torso_angle = self._angle_from_vertical_deg(torso_vec)

        now = time.time()
        hip_down_speed = 0.0
        if self.state.last_time > 0 and self.state.last_hip_y is not None:
            dt = max(1e-3, now - self.state.last_time)
            hip_down_speed = float((hip_mid[1] - self.state.last_hip_y) / dt)  
        self.state.last_time = now
        self.state.last_hip_y = float(hip_mid[1])

        head_ankle_gap = float(abs(nose[1] - ank_mid[1]))
        return torso_angle, hip_down_speed, head_ankle_gap

    def update(self, landmarks) -> Tuple[bool, dict]:
        vis = [lm.visibility for lm in landmarks]
        if max(vis) < 0.5:
            self.state.consec = max(0, self.state.consec - 1)
            return False, {"p": 0.0, "reason": "low_visibility"}

        torso_angle, hip_down_speed, head_ankle_gap = self._compute_features(landmarks)

        a = torso_angle / max(1.0, self.params.angle_thresh_deg)
        s = max(0.0, hip_down_speed)
        h = max(0.0, (self.params.head_ankle_gap - head_ankle_gap) / max(self.params.head_ankle_gap, 1e-3))
        score = float(0.5 * a + 0.3 * s + 0.2 * h)
        self.ema = self.params.smooth * self.ema + (1.0 - self.params.smooth) * score

        cond_angle = torso_angle >= self.params.angle_thresh_deg
        cond_speed = hip_down_speed >= self.params.speed_thresh
        cond_head = head_ankle_gap <= self.params.head_ankle_gap
        triggered_frame = (cond_angle and cond_speed) or (cond_angle and cond_head)

        if self.ema >= 1.0 or triggered_frame:
            self.state.consec += 1
        else:
            self.state.consec = 0

        now = time.time()
        triggered = self.state.consec >= self.params.min_frames
        if triggered:
            self.state.last_trigger = now

        latched = (now - self.state.last_trigger) <= self.params.latch_sec
        return latched, {
            "torso_angle": float(torso_angle),
            "hip_down_speed": float(hip_down_speed),
            "head_ankle_gap": float(head_ankle_gap),
            "score": float(score),
            "ema": float(self.ema),
            "consec": int(self.state.consec),
            "triggered_frame": bool(triggered_frame),
        }


def draw_pose(frame_bgr, results):
    try:
        lm_style = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
    except Exception:
        lm_style = None
    mp_drawing.draw_landmarks(
        frame_bgr,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=lm_style,
    )


class PoseTracker:
    def __init__(self, static_image_mode=False, model_complexity=1, enable_segmentation=False):
        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.pose.process(frame_rgb)

    def close(self):
        self.pose.close()