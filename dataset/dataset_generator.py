"""
Dataset Generator & Real-Time Gesture Collector
Synthesizes high-fidelity 3D landmark sign dataset based on ASL biomechanics
and supports capturing live custom gestures using a webcam.
"""

import os
import sys

# Ensure root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import argparse
import numpy as np
import cv2
import pandas as pd
from src.landmark_extractor import LandmarkExtractor, FINGERTIP_INDICES

# List of 19 Standard Sign Classes
CLASSES = [
    "A", "B", "C", "D", "E", "F", "I", "L", "O", "V", "W", "Y",
    "HELLO", "THANK_YOU", "YES", "NO", "I_LOVE_YOU", "SPACE", "DELETE"
]


def create_base_skeleton():
    """Generates a neutral base 3D hand skeleton in normalized coordinates."""
    # 21 keypoints
    pts = np.zeros((21, 3), dtype=np.float32)
    # 0: Wrist
    pts[0] = [0.0, 0.0, 0.0]

    # Thumb (1 to 4)
    pts[1] = [-0.08, -0.15, 0.02]
    pts[2] = [-0.15, -0.28, 0.04]
    pts[3] = [-0.22, -0.38, 0.05]
    pts[4] = [-0.27, -0.48, 0.06]

    # Index finger (5 to 8)
    pts[5] = [-0.10, -0.42, 0.0]
    pts[6] = [-0.11, -0.58, -0.01]
    pts[7] = [-0.12, -0.72, -0.02]
    pts[8] = [-0.12, -0.86, -0.02]

    # Middle finger (9 to 12)
    pts[9] = [0.0, -0.45, 0.0]
    pts[10] = [0.0, -0.62, -0.01]
    pts[11] = [0.0, -0.77, -0.02]
    pts[12] = [0.0, -0.92, -0.02]

    # Ring finger (13 to 16)
    pts[13] = [0.10, -0.42, 0.0]
    pts[14] = [0.11, -0.58, -0.01]
    pts[15] = [0.11, -0.72, -0.02]
    pts[16] = [0.12, -0.85, -0.02]

    # Pinky finger (17 to 20)
    pts[17] = [0.20, -0.38, 0.0]
    pts[18] = [0.22, -0.51, -0.01]
    pts[19] = [0.23, -0.63, -0.02]
    pts[20] = [0.24, -0.75, -0.02]

    return pts


def curl_finger(pts, base_idx, curl_amount=1.0):
    """
    Curls a 4-joint finger towards the palm.
    curl_amount: 0.0 (extended) to 1.0 (fully curled into fist).
    """
    mcp = pts[base_idx]
    # Curl trajectory: bends inward along Z and down along Y
    pip_target = mcp + np.array([0.0, 0.12, -0.15])
    dip_target = mcp + np.array([0.0, 0.22, -0.05])
    tip_target = mcp + np.array([0.0, 0.18, 0.06])

    pts[base_idx + 1] = (1 - curl_amount) * pts[base_idx + 1] + curl_amount * pip_target
    pts[base_idx + 2] = (1 - curl_amount) * pts[base_idx + 2] + curl_amount * dip_target
    pts[base_idx + 3] = (1 - curl_amount) * pts[base_idx + 3] + curl_amount * tip_target


def fold_thumb_across(pts, fold_amount=1.0):
    """Folds thumb flat across palm."""
    target_1 = np.array([-0.05, -0.15, 0.04])
    target_2 = np.array([-0.02, -0.22, 0.06])
    target_3 = np.array([0.02, -0.25, 0.07])
    target_4 = np.array([0.08, -0.24, 0.08])

    pts[1] = (1 - fold_amount) * pts[1] + fold_amount * target_1
    pts[2] = (1 - fold_amount) * pts[2] + fold_amount * target_2
    pts[3] = (1 - fold_amount) * pts[3] + fold_amount * target_3
    pts[4] = (1 - fold_amount) * pts[4] + fold_amount * target_4


def generate_sign_template(sign_name):
    """Returns the base 3D hand landmark configuration for a given sign."""
    pts = create_base_skeleton()

    if sign_name == "A":
        # Fist with thumb resting alongside index MCP
        for idx in [5, 9, 13, 17]:
            curl_finger(pts, idx, 1.0)
        pts[1] = [-0.08, -0.16, 0.04]
        pts[2] = [-0.14, -0.28, 0.06]
        pts[3] = [-0.17, -0.38, 0.07]
        pts[4] = [-0.18, -0.48, 0.06]  # thumb up beside index

    elif sign_name == "B":
        # 4 fingers straight up touching; thumb folded across palm
        fold_thumb_across(pts, 1.0)
        # Tighten fingers together
        pts[5:9, 0] -= 0.02
        pts[13:17, 0] -= 0.02
        pts[17:21, 0] -= 0.05

    elif sign_name == "C":
        # Hand curved in a C shape
        for idx in [5, 9, 13, 17]:
            curl_finger(pts, idx, 0.45)
            pts[idx + 1:idx + 4, 2] -= 0.12  # curve forward
        # Curve thumb out and forward to meet fingers
        pts[2] = [-0.18, -0.25, -0.06]
        pts[3] = [-0.22, -0.35, -0.08]
        pts[4] = [-0.15, -0.45, -0.10]

    elif sign_name == "D":
        # Index straight up; thumb touches middle finger forming a circle; ring/pinky curled
        curl_finger(pts, 9, 0.7)
        curl_finger(pts, 13, 1.0)
        curl_finger(pts, 17, 1.0)
        # Thumb touches middle tip
        pts[4] = pts[12] + np.array([-0.02, 0.02, 0.02])
        pts[3] = pts[4] + np.array([-0.06, 0.08, 0.01])

    elif sign_name == "E":
        # All 4 fingers curled down resting on folded thumb
        for idx in [5, 9, 13, 17]:
            curl_finger(pts, idx, 0.95)
        fold_thumb_across(pts, 0.9)
        # Thumb tucked under fingertip pads
        pts[4] = [0.0, -0.28, -0.04]

    elif sign_name == "F":
        # Thumb and index tips touch (OK sign); middle, ring, pinky straight up
        curl_finger(pts, 5, 0.65)
        pts[4] = pts[8] + np.array([-0.01, 0.01, 0.01])
        pts[3] = pts[4] + np.array([-0.05, 0.08, 0.02])

    elif sign_name == "I":
        # Pinky extended straight up; other 3 fingers curled into fist; thumb across
        curl_finger(pts, 5, 1.0)
        curl_finger(pts, 9, 1.0)
        curl_finger(pts, 13, 1.0)
        fold_thumb_across(pts, 0.85)

    elif sign_name == "L":
        # Index extended up, thumb extended sideways (90 deg); other 3 fingers curled
        curl_finger(pts, 9, 1.0)
        curl_finger(pts, 13, 1.0)
        curl_finger(pts, 17, 1.0)
        pts[1] = [-0.10, -0.12, 0.0]
        pts[2] = [-0.22, -0.15, 0.0]
        pts[3] = [-0.35, -0.15, 0.0]
        pts[4] = [-0.46, -0.15, 0.0]  # thumb sticking directly left

    elif sign_name == "O":
        # All fingertips touch thumb forming an 'O'
        for idx in [5, 9, 13, 17]:
            curl_finger(pts, idx, 0.65)
        o_center = np.array([-0.05, -0.42, -0.12])
        for tip in [4, 8, 12, 16, 20]:
            pts[tip] = o_center + np.random.normal(0, 0.01, 3)

    elif sign_name == "V":
        # Index and middle extended in a 'V' shape (peace sign); ring, pinky, thumb curled
        curl_finger(pts, 13, 1.0)
        curl_finger(pts, 17, 1.0)
        fold_thumb_across(pts, 0.9)
        # Spread index and middle
        pts[5:9, 0] -= 0.07
        pts[9:13, 0] += 0.07

    elif sign_name == "W":
        # Index, middle, and ring extended and spread; pinky curled; thumb holds pinky
        curl_finger(pts, 17, 1.0)
        fold_thumb_across(pts, 0.9)
        pts[5:9, 0] -= 0.08
        pts[9:13, 0] += 0.0
        pts[13:17, 0] += 0.08

    elif sign_name == "Y":
        # Thumb and pinky extended out sideways; middle 3 curled (hang loose)
        curl_finger(pts, 5, 1.0)
        curl_finger(pts, 9, 1.0)
        curl_finger(pts, 13, 1.0)
        # Thumb extended left
        pts[1] = [-0.10, -0.12, 0.0]
        pts[2] = [-0.22, -0.16, 0.0]
        pts[3] = [-0.33, -0.18, 0.0]
        pts[4] = [-0.43, -0.20, 0.0]
        # Pinky spread outward
        pts[17:21, 0] += 0.12

    elif sign_name == "HELLO":
        # Open upright flat hand with fingers spread naturally (waving/greeting)
        pts[5:9, 0] -= 0.06
        pts[9:13, 0] -= 0.01
        pts[13:17, 0] += 0.04
        pts[17:21, 0] += 0.09
        pts[4] = [-0.30, -0.38, 0.04]

    elif sign_name == "THANK_YOU":
        # Flat hand extending forward from mouth towards viewer (Z negative projection)
        pts[5:9, 2] -= 0.35
        pts[9:13, 2] -= 0.38
        pts[13:17, 2] -= 0.35
        pts[17:21, 2] -= 0.30
        pts[5:21, 1] *= 0.65  # perspective foreshortening

    elif sign_name == "YES":
        # Clenched fist nodding gesture (all fingers curled into fist, thumb tucked)
        for idx in [5, 9, 13, 17]:
            curl_finger(pts, idx, 1.0)
        fold_thumb_across(pts, 0.95)
        # Wrist flexed slightly forward
        pts[:, 2] += 0.05

    elif sign_name == "NO":
        # Index and middle fingers together tapping down on thumb tip
        curl_finger(pts, 13, 1.0)
        curl_finger(pts, 17, 1.0)
        curl_finger(pts, 5, 0.55)
        curl_finger(pts, 9, 0.55)
        pts[5:9, 0] -= 0.02
        pts[9:13, 0] -= 0.02
        contact = np.array([-0.06, -0.38, -0.05])
        pts[8] = contact
        pts[12] = contact + np.array([0.02, 0.0, 0.0])
        pts[4] = contact + np.array([-0.01, 0.03, 0.0])

    elif sign_name == "I_LOVE_YOU":
        # ASL ILY sign: Thumb, Index, and Pinky extended; Middle and Ring curled
        curl_finger(pts, 9, 1.0)
        curl_finger(pts, 13, 1.0)
        # Thumb extended outward
        pts[1] = [-0.10, -0.12, 0.0]
        pts[2] = [-0.22, -0.16, 0.0]
        pts[3] = [-0.33, -0.20, 0.0]
        pts[4] = [-0.42, -0.22, 0.0]
        # Index up
        pts[5:9, 0] -= 0.04
        # Pinky extended up & slightly out
        pts[17:21, 0] += 0.08

    elif sign_name == "SPACE":
        # Open flat hand oriented horizontally (90 deg rotated)
        # Rotate all points 90 degrees around Z axis
        cos_theta = 0.0
        sin_theta = 1.0
        x_rot = pts[:, 0] * cos_theta - pts[:, 1] * sin_theta
        y_rot = pts[:, 0] * sin_theta + pts[:, 1] * cos_theta
        pts[:, 0] = x_rot
        pts[:, 1] = y_rot

    elif sign_name == "DELETE":
        # Index finger pointing horizontally to the left; other fingers curled
        curl_finger(pts, 9, 1.0)
        curl_finger(pts, 13, 1.0)
        curl_finger(pts, 17, 1.0)
        fold_thumb_across(pts, 0.9)
        # Bend index pointing to left
        pts[6] = [-0.22, -0.42, 0.0]
        pts[7] = [-0.38, -0.42, 0.0]
        pts[8] = [-0.52, -0.42, 0.0]

    return pts.astype(np.float32)


def apply_3d_transformations(base_pts):
    """
    Applies realistic 3D rotations, anatomical joint variation,
    perspective jitter, and natural hand scale changes.
    """
    pts = base_pts.copy()

    # 1. 3D Random Euler Rotations: Pitch (X), Yaw (Y), Roll (Z)
    roll = np.radians(np.random.uniform(-18, 18))
    pitch = np.radians(np.random.uniform(-14, 14))
    yaw = np.radians(np.random.uniform(-16, 16))

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    pts = pts @ R.T

    # 2. Hand scale variation (person distance & hand size)
    scale = np.random.uniform(0.75, 1.35)
    pts = pts * scale

    # 3. Individual finger length & span variance
    finger_scales = np.random.uniform(0.92, 1.08, size=(5, 1))
    pts[1:5] *= finger_scales[0]
    pts[5:9] *= finger_scales[1]
    pts[9:13] *= finger_scales[2]
    pts[13:17] *= finger_scales[3]
    pts[17:21] *= finger_scales[4]

    # 4. Realistic landmark sensor Gaussian noise
    noise = np.random.normal(0, 0.012, size=pts.shape).astype(np.float32)
    pts += noise

    return pts


def generate_synthetic_dataset(samples_per_class=450, output_csv="dataset/sign_data.csv"):
    """
    Generates high-diversity dataset for all 19 sign classes.
    """
    print(f"[Dataset] Generating synthetic sign language dataset ({len(CLASSES)} classes, {samples_per_class} samples/class)...")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs("models", exist_ok=True)

    data_rows = []
    labels = []

    for class_idx, class_name in enumerate(CLASSES):
        base_template = generate_sign_template(class_name)
        for _ in range(samples_per_class):
            augmented_pts = apply_3d_transformations(base_template)
            features = LandmarkExtractor.compute_features(augmented_pts)
            data_rows.append(features)
            labels.append(class_name)

    # Convert to DataFrame
    num_features = data_rows[0].shape[0]
    col_names = [f"f_{i}" for i in range(num_features)]
    df = pd.DataFrame(data_rows, columns=col_names)
    df["label"] = labels

    df.to_csv(output_csv, index=False)
    print(f"[Dataset] Saved {len(df)} samples across {len(CLASSES)} classes to {output_csv}.")

    # Save Label Map JSON
    label_map = {idx: name for idx, name in enumerate(CLASSES)}
    label_map_path = "models/label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[Dataset] Saved label map to {label_map_path}.")

    return df


def record_live_gesture(sign_label, num_samples=100, output_csv="dataset/sign_data.csv"):
    """
    Interactive live webcam tool to record real hand gestures.
    """
    print(f"\n[Webcam Collector] Preparing to record '{sign_label}'. Press 'SPACE' to record, 'Q' to quit.")
    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] Could not open webcam.")
        return

    recorded = []
    recording = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        features, pts, bbox = extractor.extract(frame)

        if pts is not None:
            extractor.draw_hand(frame, pts, bbox, label=sign_label)

        status_text = f"Class: {sign_label} | Recorded: {len(recorded)}/{num_samples}"
        if recording:
            status_text += " [RECORDING...]"
            if features is not None:
                recorded.append(features)
                if len(recorded) >= num_samples:
                    print(f"[Collector] Completed {num_samples} samples for {sign_label}!")
                    break

        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: Start/Stop | Q: Quit", (20, 75), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("SignFlow Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            recording = not recording
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(recorded) > 0:
        new_df = pd.DataFrame(recorded, columns=[f"f_{i}" for i in range(len(recorded[0]))])
        new_df["label"] = sign_label
        if os.path.exists(output_csv):
            existing_df = pd.read_csv(output_csv)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(output_csv, index=False)
        else:
            new_df.to_csv(output_csv, index=False)
        print(f"[Collector] Appended {len(recorded)} samples to {output_csv}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Dataset Tool")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Generate synthetic dataset")
    parser.add_argument("--samples", type=int, default=450, help="Samples per class")
    parser.add_argument("--record", type=str, default=None, help="Record live webcam samples for class name")
    args = parser.parse_args()

    if args.record:
        record_live_gesture(args.record)
    else:
        generate_synthetic_dataset(samples_per_class=args.samples)
