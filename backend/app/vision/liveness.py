"""Server-observed, randomized active challenge. Not a certified PAD model."""

import hashlib
import secrets
import time
import uuid
from threading import RLock
import cv2
import numpy as np
from app.core.errors import KycError

EYES = ((33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380))
PROMPTS = {
    "NEUTRAL": "Look straight ahead with both eyes open.",
    "CENTER": "Look straight. Eyes open.",
    "BLINK": "Close BOTH eyes for about 2 seconds, then open them.",
    "TURN_LEFT": "Turn slowly toward your LEFT shoulder and hold.",
    "TURN_RIGHT": "Turn slowly toward your RIGHT shoulder and hold.",
}


class Landmarker:
    def __init__(self, settings):
        import mediapipe as mp

        self.mp = mp
        self.lock = RLock()
        path = settings.weights_dir / "face_landmarker.task"
        if not path.is_file():
            raise FileNotFoundError(
                "Model not loaded: face_landmarker.task. Run model-setup."
            )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
        )
        self.model = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def analyze(self, bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with self.lock:
            result = self.model.detect(
                self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            )
        if len(result.face_landmarks) != 1:
            raise KycError(
                "landmark_face_count",
                f"MediaPipe detected {len(result.face_landmarks)} faces; exactly one is required.",
            )
        points = np.array([[v.x, v.y] for v in result.face_landmarks[0]])
        pixels = points * np.array([bgr.shape[1], bgr.shape[0]])

        def ear(indices):
            p = pixels[list(indices)]
            return float(
                (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4]))
                / (2 * max(np.linalg.norm(p[0] - p[3]), 1))
            )

        ears = [ear(eye) for eye in EYES]
        eye_width = max(abs(pixels[263, 0] - pixels[33, 0]), 1)
        offset = float(
            (pixels[1, 0] - (pixels[33, 0] + pixels[263, 0]) / 2) / eye_width
        )
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return {
            "landmarks": points.round(5).tolist(),
            "ear": float(np.mean(ears)),
            "ear_left": ears[0],
            "ear_right": ears[1],
            "nose_offset": offset,
            "brightness": float(gray.mean()),
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "model": "MediaPipe Face Landmarker",
        }

    def close(self):
        self.model.close()


class Challenge:
    def __init__(self, settings, now=None):
        self.settings = settings
        self.id = str(uuid.uuid4())
        self.started = now if now is not None else time.monotonic()
        actions = ["BLINK", "TURN_LEFT", "TURN_RIGHT"]
        secrets.SystemRandom().shuffle(actions)
        self.steps = ["NEUTRAL"] + [
            step for action in actions for step in (action, "CENTER")
        ]
        self.index = 0
        self.sequence = 0
        self.status = "IN_PROGRESS"
        self.issue = None
        self.streak = 0
        self.streak_start = None
        self.baselines = []
        self.baseline = None
        self.nose_baseline = 0
        self.closed = False
        self.closed_at = None
        self.seen = set()
        self.last_received = None
        self.last = None
        self.events = []
        self.reference = None
        self.best_embedding = None
        self.selfie = None
        self.passed_at = None
        self.continuity_min = 1.0
        self.frames = 0

    def response(self):
        return {
            "challenge_id": self.id,
            "status": self.status,
            "step": self.steps[self.index]
            if self.index < len(self.steps)
            else "COMPLETE",
            "prompt": PROMPTS[self.steps[self.index]]
            if self.index < len(self.steps)
            else "Challenge complete.",
            "steps": self.steps,
            "completed_steps": self.index,
            "progress": self.index / len(self.steps),
            "next_sequence": self.sequence,
            "remaining_seconds": max(
                0,
                round(
                    self.settings.challenge_ttl_seconds
                    - (time.monotonic() - self.started)
                ),
            ),
            "hold_progress": min(1, self.streak / (4 if self.index == 0 else 2)),
            "issue": self.issue,
            "metrics": self.last,
            "events": self.events,
            "baseline_ear": self.baseline,
            "nose_baseline": self.nose_baseline,
            "assurance": "Active challenge only; print/screen/replay resistance must be evaluated.",
        }

    def accept_frame(self, image, sequence, now):
        if self.status != "IN_PROGRESS":
            raise KycError("challenge_finished", "Start a new challenge.", 409)
        if now - self.started > self.settings.challenge_ttl_seconds:
            self.status = "FAILED"
            self.issue = "Challenge timed out."
            raise KycError("challenge_expired", self.issue, 410)
        if sequence != self.sequence:
            raise KycError(
                "frame_sequence",
                "Unexpected frame sequence. Restart the challenge.",
                409,
            )
        if self.last_received is not None and now - self.last_received < 0.08:
            raise KycError("frame_rate", "Frames arrived too quickly.", 429)
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest in self.seen:
            self.status = "FAILED"
            self.issue = "Repeated frame detected. Use the live camera."
            raise KycError("replayed_frame", self.issue, 409)
        if self.last_received is not None and now - self.last_received > 6:
            self.reset_partial()
            self.issue = "Long frame gap; repeat this step."
        self.seen.add(digest)
        self.sequence += 1
        self.frames += 1
        self.last_received = now

    def reset_partial(self):
        self.streak = 0
        self.streak_start = None
        self.closed = False
        self.closed_at = None
        if self.baseline is None:
            self.baselines = []

    def advance(self, metrics, now):
        self.last = metrics
        self.issue = None
        if (
            metrics["brightness"] < 35
            or metrics["brightness"] > 235
            or metrics["sharpness"] < 15
        ):
            self.reset_partial()
            self.issue = "Improve lighting/focus before continuing."
            return
        step = self.steps[self.index]
        ear = metrics["ear"]
        offset = metrics["nose_offset"]
        condition = False
        if step == "NEUTRAL":
            if ear < 0.18 or abs(offset) > 0.18:
                self.reset_partial()
                self.issue = "Look straight ahead, eyes fully open."
                return
            self.baselines.append((ear, offset))
            condition = True
        elif step == "BLINK":
            # Hysteresis prevents a noisy single measurement counting as a blink.
            if not self.closed:
                condition = (
                    max(metrics["ear_left"], metrics["ear_right"])
                    < self.baseline * 0.65
                )
                if condition:
                    self.streak += 1
                    if self.streak >= 2:
                        self.closed = True
                        self.closed_at = now
                        self.streak = 0
                        self.issue = "Now open both eyes."
                else:
                    self.streak = 0
                return
            if now - self.closed_at > 6:
                self.reset_partial()
                self.issue = "Blink too slow. Repeat: close, then open."
                return
            condition = (
                min(metrics["ear_left"], metrics["ear_right"]) > self.baseline * 0.85
            )
        elif step == "CENTER":
            condition = (
                abs(offset - self.nose_baseline) < 0.09 and ear > self.baseline * 0.8
            )
        elif step == "TURN_LEFT":
            condition = offset - self.nose_baseline > 0.16
        elif step == "TURN_RIGHT":
            condition = offset - self.nose_baseline < -0.16
        if condition:
            if self.streak_start is None:
                self.streak_start = now
            self.streak += 1
        else:
            self.streak = 0
            self.streak_start = None
        required = 4 if step == "NEUTRAL" else 2
        if self.streak >= required and now - self.streak_start >= 0.35:
            if step == "NEUTRAL":
                self.baseline = float(np.median([p[0] for p in self.baselines]))
                self.nose_baseline = float(np.median([p[1] for p in self.baselines]))
            self.events.append(
                {
                    "step": step,
                    "elapsed_seconds": round(now - self.started, 2),
                    "ear": ear,
                    "nose_offset": offset,
                }
            )
            self.index += 1
            self.reset_partial()
            if self.index == len(self.steps):
                self.status = "PASSED"
                self.passed_at = now
