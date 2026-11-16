"""
SignFlow AI - Web Application & Interactive Dashboard
Flask Backend with Live Video Streaming, REST APIs, and Gesture Translation.
"""

import os
import sys

# Ensure root directory is in sys.path and configure Keras cache
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["KERAS_HOME"] = os.path.join(ROOT_DIR, "model", ".keras_cache")
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)

import json
import base64
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS

from backend.sign_detector import SignDetector
from dataset.dataset_generator import CLASSES

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Global Detector Instance (lazy-loaded)
detector = None


def get_detector():
    global detector
    if detector is None:
        try:
            detector = SignDetector()
        except Exception as e:
            print(f"[App] Could not initialize detector on startup: {e}")
    return detector


@app.route("/")
def index():
    """Serves the main interactive 5-page dashboard."""
    metadata = {}
    meta_path = os.path.join(ROOT_DIR, "model", "model_metadata.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(ROOT_DIR, "models", "model_metadata.json")

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    return render_template("index.html", classes=CLASSES, metadata=metadata)


@app.route("/api/info")
def api_info():
    """Returns model metadata and supported classes."""
    meta_path = os.path.join(ROOT_DIR, "model", "model_metadata.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(ROOT_DIR, "models", "model_metadata.json")

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"classes": CLASSES, "status": "Model ready"})


@app.route("/api/predict_frame", methods=["POST"])
def api_predict_frame():
    """
    Receives a Base64-encoded frame from client browser webcam,
    runs deep learning recognition, and returns annotations and prediction.
    """
    det = get_detector()
    if det is None:
        return jsonify({"error": "Model not loaded. Please train model first."}), 500

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image payload"}), 400

    # Decode base64 image
    img_data = data["image"]
    if "," in img_data:
        img_data = img_data.split(",")[1]

    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image data"}), 400

    # Flip horizontally for selfie mirror effect if requested
    if data.get("flip", True):
        frame = cv2.flip(frame, 1)

    result = det.process_frame(frame, draw=True)

    # Encode annotated frame back to JPEG Base64
    _, buffer = cv2.imencode(".jpg", result["annotated_frame"], [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64_annotated = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "hand_detected": result["hand_detected"],
        "message": result.get("message", ""),
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "top3": result["top3"],
        "sentence": result["sentence"],
        "latency_ms": result["latency_ms"],
        "annotated_image": f"data:image/jpeg;base64,{b64_annotated}"
    })


@app.route("/api/sentence/action", methods=["POST"])
def api_sentence_action():
    """Handles sentence manipulation (clear, space, backspace)."""
    det = get_detector()
    if det is None:
        return jsonify({"error": "Detector offline"}), 500

    req = request.get_json(silent=True) or {}
    action = req.get("action", "")

    if action == "clear":
        det.reset_sentence()
    elif action == "space":
        det.add_space()
    elif action == "backspace":
        det.delete_last_char()

    return jsonify({"sentence": det.get_sentence()})


def generate_camera_stream():
    """MJPEG stream generator for desktop camera access."""
    det = get_detector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Webcam unavailable. Use Browser Mode above!", (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        _, buffer = cv2.imencode(".jpg", blank)
        frame_bytes = buffer.tobytes()
        while True:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.5)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            if det is not None:
                res = det.process_frame(frame, draw=True)
                frame = res["annotated_frame"]
            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        cap.release()


@app.route("/video_feed")
def video_feed():
    """MJPEG video streaming route."""
    return Response(generate_camera_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    get_detector()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[SignFlow Server] Starting on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
