"""
SignFlow AI - Real-Time Desktop Application
Runs live Sign Language Recognition with OpenCV HUD and Skeletal Overlay.
"""

import time
import os
import sys

# Ensure root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np
from backend.sign_detector import SignDetector


def draw_hud(frame, detector_result, fps):
    """Draws sleek modern HUD overlay on the OpenCV window."""
    h, w, _ = frame.shape

    # 1. Top HUD Banner (semi-transparent dark background)
    banner_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 24, 33), -1)
    # Bottom Sentence Banner
    cv2.rectangle(overlay, (0, h - 90), (w, h), (20, 24, 33), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Accent top border
    cv2.line(frame, (0, banner_h), (w, banner_h), (0, 220, 130), 2)
    cv2.line(frame, (0, h - 90), (w, h - 90), (0, 220, 130), 2)

    # Top Left: System Title & FPS
    cv2.putText(frame, "SignFlow AI", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 170), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {detector_result['latency_ms']}ms", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 205), 1, cv2.LINE_AA)

    # Top Center/Right: Current Sign & Confidence
    pred = detector_result["prediction"]
    conf = detector_result["confidence"]
    hand_detected = detector_result["hand_detected"]

    if hand_detected and pred != "None":
        sign_text = f"Sign: {pred}"
        cv2.putText(frame, sign_text, (w // 2 - 80, 32), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        # Confidence Bar
        bar_w = 160
        bar_h = 14
        bar_x = w // 2 - 80
        bar_y = 44
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 65, 80), -1)
        fill_w = int(bar_w * conf)
        # Color based on confidence
        bar_color = (0, 220, 100) if conf >= 0.75 else (0, 180, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.putText(frame, f"{conf:.0%}", (bar_x + bar_w + 8, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    else:
        cv2.putText(frame, "Waiting for hand...", (w // 2 - 80, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 130, 150), 1, cv2.LINE_AA)

    # Bottom Banner: Translated Sentence
    sentence = detector_result["sentence"]
    disp_sentence = sentence if sentence else "(Perform signs to build sentence...)"
    sentence_color = (255, 255, 255) if sentence else (140, 150, 165)

    cv2.putText(frame, "TRANSLATION:", (20, h - 60), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 230, 140), 1, cv2.LINE_AA)
    cv2.putText(frame, disp_sentence, (20, h - 25), cv2.FONT_HERSHEY_DUPLEX, 0.85, sentence_color, 2, cv2.LINE_AA)

    # Bottom Right: Keybind Helpers
    cv2.putText(frame, "[C] Clear | [SPACE] Space | [Q] Quit", (w - 280, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 180, 195), 1, cv2.LINE_AA)

    return frame


def main():
    print("=" * 60)
    print("       SignFlow AI - Real-Time Live Recognition")
    print("=" * 60)
    print("[Info] Initializing deep learning model & hand detector...")

    try:
        detector = SignDetector()
    except Exception as e:
        print(f"[Error] Failed to initialize detector: {e}")
        print("[Action] Make sure to run 'python src/trainer.py' first!")
        return

    print("[Info] Connecting to webcam (ID: 0)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] Webcam could not be accessed. (Make sure your camera is connected and not used by another app).")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[Controls]")
    print("  'C'         - Clear sentence")
    print("  'SPACE'     - Add manual space")
    print("  'BACKSPACE' - Delete last character")
    print("  'Q' / ESC   - Exit application\n")

    prev_time = time.time()
    fps = 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Frame read failed. Retrying...")
            time.sleep(0.05)
            continue

        # Flip horizontally for natural mirror interaction
        frame = cv2.flip(frame, 1)

        # FPS Calculation
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
        prev_time = curr_time

        # Run Deep Learning Sign Detection
        result = detector.process_frame(frame, draw=True)

        # Draw HUD overlays
        annotated_frame = draw_hud(result["annotated_frame"], result, fps)

        # Show frame
        cv2.imshow("SignFlow AI - Live Sign Recognition", annotated_frame)

        # Handle Keyboard Inputs
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:  # 'q' or ESC
            print("[Info] Closing application...")
            break
        elif key in [ord('c'), ord('C')]:
            detector.reset_sentence()
            print("[Action] Cleared translation sentence.")
        elif key == 32:  # Spacebar
            detector.add_space()
        elif key == 8:  # Backspace
            detector.delete_last_char()

    cap.release()
    cv2.destroyAllWindows()
    print("[Info] Application closed successfully.")


if __name__ == "__main__":
    main()
