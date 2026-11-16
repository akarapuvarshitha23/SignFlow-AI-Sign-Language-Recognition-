"""
Landmark Extractor Module
Handles hand landmark detection and geometric invariant feature engineering.
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# 21 Hand Landmark Connections for Skeleton Rendering
HAND_CONNECTIONS = [
    # Palm base & knuckles
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (17, 18), (18, 19), (19, 20)
]

FINGERTIP_INDICES = [4, 8, 12, 16, 20]


class LandmarkExtractor:
    """Extracts and normalizes 3D hand landmarks from camera frames."""

    def __init__(self, model_path=None):
        if model_path is None:
            # Check likely model paths
            candidates = [
                os.path.join(os.getcwd(), "hand_landmarker.task"),
                os.path.join(os.getcwd(), "models", "hand_landmarker.task"),
                os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")
            ]
            model_path = next((p for p in candidates if os.path.exists(p)), None)
            if model_path is None:
                # Automatically download if missing
                os.makedirs("models", exist_ok=True)
                model_path = os.path.join("models", "hand_landmarker.task")
                print(f"[LandmarkExtractor] Model not found. Downloading to {model_path}...")
                urllib.request.urlretrieve(MODEL_URL, model_path)
                print("[LandmarkExtractor] Download complete.")

        self.model_path = os.path.abspath(model_path)
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def extract(self, bgr_frame):
        """
        Process a BGR frame from OpenCV.
        Returns:
            features: np.ndarray of shape (76,) or None if no hand detected.
            landmarks_raw: list of 21 (x, y, z) tuples or None.
            bounding_box: (min_x, min_y, max_x, max_y) in pixel coordinates or None.
        """
        h, w, _ = bgr_frame.shape
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.detector.detect(mp_image)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return None, None, None

        landmarks = result.hand_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        # Compute bounding box in pixel coordinates
        xs = pts[:, 0] * w
        ys = pts[:, 1] * h
        bbox = (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys)))

        # Feature Engineering: Translation and scale invariant coordinates
        features = self.compute_features(pts)
        return features, pts, bbox

    @staticmethod
    def compute_features(pts):
        """
        Calculates 76-dimensional invariant feature vector:
        - 63 normalized relative coordinates (21 x 3)
        - 5 wrist-to-fingertip Euclidean distances
        - 4 adjacent fingertip distances
        - 4 thumb-to-other-fingertip distances
        """
        wrist = pts[0]
        # Shift relative to wrist (landmark 0)
        rel_pts = pts - wrist

        # Scale normalization by max Euclidean distance from wrist
        d_max = np.max(np.linalg.norm(rel_pts, axis=1))
        if d_max > 1e-6:
            norm_pts = rel_pts / d_max
        else:
            norm_pts = rel_pts

        # Geometric distances
        wrist_to_tips = [np.linalg.norm(norm_pts[t]) for t in FINGERTIP_INDICES]
        adj_tips = [
            np.linalg.norm(norm_pts[FINGERTIP_INDICES[i]] - norm_pts[FINGERTIP_INDICES[i + 1]])
            for i in range(4)
        ]
        thumb_to_tips = [
            np.linalg.norm(norm_pts[4] - norm_pts[t]) for t in [8, 12, 16, 20]
        ]

        feature_vector = np.concatenate([
            norm_pts.flatten(),
            wrist_to_tips,
            adj_tips,
            thumb_to_tips
        ]).astype(np.float32)

        return feature_vector

    @staticmethod
    def draw_hand(frame, pts, bbox=None, label=None, score=None):
        """Draws glowing skeleton connections, keypoints, and HUD box on frame."""
        h, w, _ = frame.shape
        pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in pts]

        # Draw bone connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = pixel_pts[start_idx]
            pt2 = pixel_pts[end_idx]
            cv2.line(frame, pt1, pt2, (40, 220, 240), 3, cv2.LINE_AA)
            cv2.line(frame, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw joints
        for i, pt in enumerate(pixel_pts):
            if i in FINGERTIP_INDICES:
                # Fingertips highlighted in neon pink/red
                cv2.circle(frame, pt, 7, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 9, (255, 255, 255), 1, cv2.LINE_AA)
            elif i == 0:
                # Wrist
                cv2.circle(frame, pt, 8, (255, 100, 0), -1, cv2.LINE_AA)
            else:
                # Knuckles
                cv2.circle(frame, pt, 5, (0, 255, 150), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 6, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw bounding box and label if provided
        if bbox is not None:
            min_x, min_y, max_x, max_y = bbox
            # Add padding
            pad = 20
            min_x = max(0, min_x - pad)
            min_y = max(0, min_y - pad)
            max_x = min(w, max_x + pad)
            max_y = min(h, max_y + pad)

            # Draw stylish rounded corner box
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 230, 115), 2, cv2.LINE_AA)

            if label:
                tag = f"{label}" + (f" ({score:.1%})" if score is not None else "")
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
                tag_y = max(min_y - 10, th + 10)
                cv2.rectangle(frame, (min_x, tag_y - th - 6), (min_x + tw + 12, tag_y + 4), (0, 200, 100), -1)
                cv2.putText(frame, tag, (min_x + 6, tag_y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        return frame
