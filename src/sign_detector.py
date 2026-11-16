"""
Sign Detector Module
Integrates Landmark Extraction, Deep Learning Model Inference,
and Temporal Smoothing for Continuous Sentence Translation.
"""

import os
import sys

# Ensure root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["KERAS_HOME"] = os.path.join(ROOT_DIR, "model", ".keras_cache")
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)

import json
import time
from collections import deque
import numpy as np
import cv2
import keras

from src.landmark_extractor import LandmarkExtractor


class SignDetector:
    """Real-time Sign Language Detector and Sentence Translator."""

    def __init__(self, model_path="models/sign_model.keras", label_map_path="models/label_map.json"):
        # Resolve paths
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.getcwd(), model_path)
        if not os.path.isabs(label_map_path):
            label_map_path = os.path.join(os.getcwd(), label_map_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Please run train_model first.")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"Label map not found at: {label_map_path}.")

        print(f"[SignDetector] Loading deep learning model from {model_path}...")
        self.model = keras.models.load_model(model_path)

        with open(label_map_path, "r") as f:
            raw_map = json.load(f)
            self.label_map = {int(k): v for k, v in raw_map.items()}

        self.extractor = LandmarkExtractor()

        # Temporal Smoothing State
        self.buffer_size = 7
        self.recent_predictions = deque(maxlen=self.buffer_size)
        self.sentence = []
        self.last_added_sign = None
        self.last_added_time = 0.0
        self.cooldown_seconds = 1.2  # Minimum pause before re-accepting the same sign
        self.min_confidence = 0.70

    def reset_sentence(self):
        """Clears the accumulated sentence."""
        self.sentence = []
        self.last_added_sign = None

    def delete_last_char(self):
        """Deletes the last character or word from sentence."""
        if self.sentence:
            self.sentence.pop()

    def add_space(self):
        """Adds a space to the sentence."""
        self.sentence.append(" ")

    def get_sentence(self):
        """Returns the current translated sentence string."""
        return "".join(self.sentence)

    def process_frame(self, frame, draw=True):
        """
        Processes a single camera frame.
        Returns:
            dict containing prediction metadata and the annotated frame.
        """
        h, w, _ = frame.shape
        start_t = time.time()

        features, pts, bbox = self.extractor.extract(frame)
        latency_ms = (time.time() - start_t) * 1000

        result = {
            "hand_detected": False,
            "prediction": "None",
            "confidence": 0.0,
            "top3": [],
            "sentence": self.get_sentence(),
            "latency_ms": round(latency_ms, 1),
            "annotated_frame": frame
        }

        if features is None:
            # Hand departed, reset debounce tracker
            if len(self.recent_predictions) > 0:
                self.recent_predictions.clear()
            return result

        result["hand_detected"] = True

        # Deep Learning Inference
        input_data = features.reshape(1, -1)
        pred_probs = self.model.predict(input_data, verbose=0)[0]

        top_idx = int(np.argmax(pred_probs))
        top_label = self.label_map.get(top_idx, "Unknown")
        top_conf = float(pred_probs[top_idx])

        # Top 3 Candidates
        top3_indices = np.argsort(pred_probs)[-3:][::-1]
        result["top3"] = [
            {"label": self.label_map.get(int(i), "Unknown"), "prob": round(float(pred_probs[i]), 3)}
            for i in top3_indices
        ]

        result["prediction"] = top_label
        result["confidence"] = round(top_conf, 3)

        # Temporal Smoothing & Gesture Translation Logic
        now = time.time()
        if top_conf >= self.min_confidence:
            self.recent_predictions.append(top_label)
            # Check if recent frames consistently agree on this gesture
            if self.recent_predictions.count(top_label) >= (self.buffer_size - 2):
                should_add = False
                if top_label != self.last_added_sign:
                    should_add = True
                elif (now - self.last_added_time) > self.cooldown_seconds:
                    should_add = True

                if should_add:
                    self._commit_sign(top_label)
                    self.last_added_sign = top_label
                    self.last_added_time = now
        else:
            self.recent_predictions.append("None")

        result["sentence"] = self.get_sentence()

        # Render skeleton and HUD on the frame
        if draw:
            frame = self.extractor.draw_hand(
                frame, pts, bbox, label=top_label, score=top_conf
            )
            result["annotated_frame"] = frame

        return result

    def _commit_sign(self, sign):
        """Translates gestures into sentence tokens."""
        if sign == "SPACE":
            self.add_space()
        elif sign == "DELETE":
            self.delete_last_char()
        elif sign in ["HELLO", "THANK_YOU", "YES", "NO", "I_LOVE_YOU"]:
            # Word signs append the word followed by a space
            clean_word = sign.replace("_", " ")
            if self.sentence and not self.sentence[-1].endswith(" "):
                self.sentence.append(" ")
            self.sentence.append(f"{clean_word} ")
        else:
            # Single alphabet letter
            self.sentence.append(sign)
