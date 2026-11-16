"""
Sign Gesture Image Generator
Generates high-definition anatomical reference pictures for all ASL Alphabets (A-Z)
and communicative gestures, saving them into static/images/signs/ and creating a master chart.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

# Ensure root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset.dataset_generator import create_base_skeleton, curl_finger, fold_thumb_across
from backend.landmark_extractor import HAND_CONNECTIONS, FINGERTIP_INDICES

OUTPUT_DIR = os.path.join(ROOT_DIR, "static", "images", "signs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_alphabet_landmarks(char):
    """
    Returns accurate anatomical 3D landmark coordinates for any letter A-Z or gesture.
    """
    pts = create_base_skeleton()

    if char == "A":
        for idx in [5, 9, 13, 17]:
            curl_finger(pts, idx, 1.0)
        pts[1] = [-0.08, -0.16, 0.04]
        pts[2] = [-0.14, -0.28, 0.06]
        pts[3] = [-0.17, -0.38, 0.07]
        pts[4] = [-0.18, -0.48, 0.06]

    elif char == "B":
        fold_thumb_across(pts, 1.0)
        pts[5:9, 0] -= 0.02
        pts[13:17, 0] -= 0.02
        pts[17:21, 0] -= 0.05

    elif char == "C":
        for idx in [5, 9, 13, 17]:
            curl_finger(pts, idx, 0.45)
            pts[idx + 1:idx + 4, 2] -= 0.12
        pts[2] = [-0.18, -0.25, -0.06]
        pts[3] = [-0.22, -0.35, -0.08]
        pts[4] = [-0.15, -0.45, -0.10]

    elif char == "D":
        curl_finger(pts, 9, 0.7)
        curl_finger(pts, 13, 1.0)
        curl_finger(pts, 17, 1.0)
        pts[4] = pts[12] + np.array([-0.02, 0.02, 0.02])
        pts[3] = pts[4] + np.array([-0.06, 0.08, 0.01])

    elif char == "E":
        for idx in [5, 9, 13, 17]:
            curl_finger(pts, idx, 0.95)
        fold_thumb_across(pts, 0.9)
        pts[4] = [0.0, -0.28, -0.04]

    elif char == "F":
        curl_finger(pts, 5, 0.65)
        pts[4] = pts[8] + np.array([-0.01, 0.01, 0.01])
        pts[3] = pts[4] + np.array([-0.05, 0.08, 0.02])

    elif char == "G":
        for idx in [9, 13, 17]:
            curl_finger(pts, idx, 1.0)
        pts[6] = [-0.18, -0.42, 0.0]
        pts[7] = [-0.30, -0.42, 0.0]
        pts[8] = [-0.42, -0.42, 0.0]
        pts[2] = [-0.15, -0.30, 0.0]
        pts[3] = [-0.26, -0.32, 0.0]
        pts[4] = [-0.38, -0.32, 0.0]

    elif char == "H":
        for idx in [13, 17]:
            curl_finger(pts, idx, 1.0)
        fold_thumb_across(pts, 0.8)
        pts[6] = [-0.18, -0.44, 0.0]
        pts[7] = [-0.30, -0.44, 0.0]
        pts[8] = [-0.42, -0.44, 0.0]
        pts[10] = [-0.16, -0.36, 0.0]
        pts[11] = [-0.28, -0.36, 0.0]
        pts[12] = [-0.40, -0.36, 0.0]

    elif char == "I":
        curl_finger(pts, 5, 1.0)
        curl_finger(pts, 9, 1.0)
        curl_finger(pts, 13, 1.0)
        fold_thumb_across(pts, 0.85)

    elif char == "J":
        curl_finger(pts, 5, 1.0)
        curl_finger(pts, 9, 1.0)
        curl_finger(pts, 13, 1.0)
        fold_thumb_across(pts, 0.85)
        pts[18] += np.array([0.04, -0.05, 0.0])
        pts[19] += np.array([0.08, -0.02, 0.0])
        pts[20] += np.array([0.12, 0.05, 0.0])

    elif char == "K":
        curl_finger(pts, 13, 1.0)
        curl_finger(pts, 17, 1.0)
        pts[10] = [0.0, -0.58, -0.10]
        pts[11] = [0.0, -0.70, -0.20]
        pts[12] = [0.0, -0.80, -0.30]
        pts[4] = [0.0, -0.50, 0.0]

    elif char == "L":
        curl_finger(pts, 9, 1.0)
        curl_finger(pts, 13, 1.0)
        curl_finger(pts, 17, 1.0)
        pts[1] = [-0.10, -0.12, 0.0]
        pts[2] = [-0.22, -0.15, 0.0]
        pts[3] = [-0.35, -0.15, 0.0]
        pts[4] = [-0.46, -0.15, 0.0]

    elif char == "M":
        for idx in [5, 9, 13, 17]:
            curl_finger(pts, idx, 1.0)
        pts[4] = [0.12, -0.30, -0.05]

    elif char == "N":
        for idx in [5, 9, 13, 17]:
            curl_finger(pts, idx, 1.0)
        pts[4] = [0.02, -0.30, -0.05]

    elif char == "O":
        for idx in [5, 9, 13, 17]:
            curl_finger(pts, idx, 0.65)
        o_center = np.array([-0.05, -0.42, -0.12])
        for tip in [4, 8, 12, 16, 20]:
            pts[tip] = o_center

    elif char == "P":
        curl_finger(pts, 13, 1.0)
        curl_finger(pts, 17, 1.0)
        pts[6:9, 0] -= 0.25
        pts[6:9, 1] += 0.30
        pts[10:13, 1] += 0.45
        pts[4] = [-0.15, -0.25, 0.0]

    elif char == "Q":
        for idx in [9, 13, 17]:
            curl_finger(pts, idx, 1.0)
        pts[6:9, 1] += 0.40
        pts[2:5, 1] += 0.35

    elif char == "R":
        curl_finger(pts, 13, 1.0)
        curl_finger(pts, 17, 1.0)
        fold_thumb_across(pts, 0.9)
        pts[5:9, 0] += 0.03
        pts[9:13, 0] -= 0.03

    elif char == "S":
        for idx in [5, 9, 13, 17]:
            curl_finger(pts, idx, 1.0)
        pts[1] = [-0.04, -0.15, 0.06]
        pts[2] = [0.0, -0.24, 0.08]
        pts[3] = [0.04, -0.32, 0.09]
        pts[4] = [0.08, -0.35, 0.09]

    elif char == "T":
        for idx in [5, 9, 13, 17]:
            curl_finger(pts, idx, 1.0)
        pts[4] = [-0.08, -0.38, 0.06]

    elif char == "U":
        curl_finger(pts, 13, 1.0)
        curl_finger(pts, 17, 1.0)
        fold_thumb_across(pts, 0.9)
        pts[5:9, 0] = -0.04
        pts[9:13, 0] = 0.04

    elif char == "V":
        curl_finger(pts, 13, 1.0)
        curl_finger(pts, 17, 1.0)
        fold_thumb_across(pts, 0.9)
        pts[5:9, 0] -= 0.07
        pts[9:13, 0] += 0.07

    elif char == "W":
        curl_finger(pts, 17, 1.0)
        fold_thumb_across(pts, 0.9)
        pts[5:9, 0] -= 0.08
        pts[9:13, 0] += 0.0
        pts[13:17, 0] += 0.08

    elif char == "X":
        for idx in [9, 13, 17]:
            curl_finger(pts, idx, 1.0)
        fold_thumb_across(pts, 0.9)
        pts[6] = [-0.10, -0.55, -0.05]
        pts[7] = [-0.10, -0.62, -0.15]
        pts[8] = [-0.10, -0.56, -0.20]

    elif char == "Y":
        curl_finger(pts, 5, 1.0)
        curl_finger(pts, 9, 1.0)
        curl_finger(pts, 13, 1.0)
        pts[1] = [-0.10, -0.12, 0.0]
        pts[2] = [-0.22, -0.16, 0.0]
        pts[3] = [-0.33, -0.18, 0.0]
        pts[4] = [-0.43, -0.20, 0.0]
        pts[17:21, 0] += 0.12

    elif char == "Z":
        for idx in [9, 13, 17]:
            curl_finger(pts, idx, 1.0)
        fold_thumb_across(pts, 0.9)
        pts[6] = [-0.12, -0.58, 0.0]
        pts[7] = [-0.13, -0.72, 0.0]
        pts[8] = [-0.14, -0.86, 0.0]

    elif char == "HELLO":
        pts[5:9, 0] -= 0.06
        pts[9:13, 0] -= 0.01
        pts[13:17, 0] += 0.04
        pts[17:21, 0] += 0.09
        pts[4] = [-0.30, -0.38, 0.04]

    elif char == "THANK_YOU":
        pts[5:9, 2] -= 0.35
        pts[9:13, 2] -= 0.38
        pts[13:17, 2] -= 0.35
        pts[17:21, 2] -= 0.30
        pts[5:21, 1] *= 0.65

    elif char == "YES":
        for idx in [5, 9, 13, 17]:
            curl_finger(pts, idx, 1.0)
        fold_thumb_across(pts, 0.95)
        pts[:, 2] += 0.05

    elif char == "NO":
        curl_finger(pts, 13, 1.0)
        curl_finger(pts, 17, 1.0)
        curl_finger(pts, 5, 0.55)
        curl_finger(pts, 9, 0.55)
        contact = np.array([-0.06, -0.38, -0.05])
        pts[8] = contact
        pts[12] = contact + np.array([0.02, 0.0, 0.0])
        pts[4] = contact + np.array([-0.01, 0.03, 0.0])

    elif char == "I_LOVE_YOU":
        curl_finger(pts, 9, 1.0)
        curl_finger(pts, 13, 1.0)
        pts[1] = [-0.10, -0.12, 0.0]
        pts[2] = [-0.22, -0.16, 0.0]
        pts[3] = [-0.33, -0.20, 0.0]
        pts[4] = [-0.42, -0.22, 0.0]
        pts[5:9, 0] -= 0.04
        pts[17:21, 0] += 0.08

    elif char == "SPACE":
        cos_theta = 0.0
        sin_theta = 1.0
        x_rot = pts[:, 0] * cos_theta - pts[:, 1] * sin_theta
        y_rot = pts[:, 0] * sin_theta + pts[:, 1] * cos_theta
        pts[:, 0] = x_rot
        pts[:, 1] = y_rot

    elif char == "DELETE":
        curl_finger(pts, 9, 1.0)
        curl_finger(pts, 13, 1.0)
        curl_finger(pts, 17, 1.0)
        fold_thumb_across(pts, 0.9)
        pts[6] = [-0.22, -0.42, 0.0]
        pts[7] = [-0.38, -0.42, 0.0]
        pts[8] = [-0.52, -0.42, 0.0]

    return pts.astype(np.float32)


POSTURE_TIPS = {
    "A": "Fist with thumb resting upright beside the index finger.",
    "B": "Four fingers straight up touching; thumb folded flat across palm.",
    "C": "Fingers and thumb curved into an open 'C' shape.",
    "D": "Index finger points straight up; thumb touches middle finger forming a circle.",
    "E": "All 4 fingers curled down, resting on top of folded thumb.",
    "F": "Thumb and index tips touch ('OK' circle); remaining 3 fingers extended up.",
    "G": "Index finger and thumb extend horizontally parallel (pincher gesture).",
    "H": "Index and middle fingers extended together horizontally pointing sideways.",
    "I": "Pinky finger points straight up; other fingers curled into fist.",
    "J": "Pinky finger points up and draws a 'J' hook in the air.",
    "K": "Index points up; middle extends forward; thumb rests between them.",
    "L": "Index finger points up, thumb extends horizontally (90° 'L' shape).",
    "M": "Fist with thumb tucked under the first 3 fingers (Index, Middle, Ring).",
    "N": "Fist with thumb tucked under the first 2 fingers (Index and Middle).",
    "O": "All 5 fingertips meet the thumb tip forming an 'O' ring.",
    "P": "Hand shaped like 'K' tilted downwards towards the floor.",
    "Q": "Hand shaped like 'G' tilted downwards towards the floor.",
    "R": "Index and middle fingers extended upright and crossed over each other.",
    "S": "Fist with thumb wrapped tightly across the front of all curled fingers.",
    "T": "Thumb tucked under the index finger only (peeking between 1st & 2nd knuckle).",
    "U": "Index and middle fingers extended straight up touching together.",
    "V": "Index and middle fingers extended up and spread apart in a 'V' (Peace).",
    "W": "Index, middle, and ring fingers extended up and spread ('W' shape).",
    "X": "Index finger bent like a small hook; other fingers curled in fist.",
    "Y": "Thumb and pinky extended out ('Hang Loose' / 'Call Me'); middle 3 curled.",
    "Z": "Index finger traces the letter 'Z' in the air.",
    "HELLO": "Open flat hand upright waving or saluting from temple.",
    "THANK_YOU": "Flat hand touches chin and extends forward towards the viewer.",
    "YES": "Closed fist nodding up and down from the wrist.",
    "NO": "Index and middle fingers snap together against thumb tip.",
    "I_LOVE_YOU": "Thumb, index, and pinky extended; middle and ring curled down (ASL ILY).",
    "SPACE": "Open flat hand oriented horizontally to trigger a space break.",
    "DELETE": "Index finger pointing left to trigger a backspace character delete."
}


def render_sign_card(char, output_path):
    """
    Renders an aesthetic, high-resolution posture card with hand skeleton,
    joints, bones, palm outline, and step-by-step instructions.
    """
    pts = get_alphabet_landmarks(char)
    # Scale to canvas coordinates
    # Center around (250, 270)
    canvas_w, canvas_h = 500, 560
    xs = pts[:, 0] * 320 + 250
    ys = -pts[:, 1] * 320 + 200  # flip Y so fingers point up

    fig, ax = plt.subplots(figsize=(5, 5.6), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafc')

    # Draw rounded background card
    card_box = FancyBboxPatch(
        (10, 10), canvas_w - 20, canvas_h - 20,
        boxstyle="round,pad=10,rounding_size=20",
        facecolor='#f8f9ff', edgecolor='#e2e8f0', linewidth=1.5, zorder=0
    )
    ax.add_patch(card_box)

    # Draw Top Header Banner
    header_box = FancyBboxPatch(
        (25, canvas_h - 90), canvas_w - 50, 65,
        boxstyle="round,pad=6,rounding_size=12",
        facecolor='#6366f1', edgecolor='#4f46e5', linewidth=1, zorder=1
    )
    ax.add_patch(header_box)

    clean_name = char.replace('_', ' ')
    ax.text(canvas_w / 2, canvas_h - 52, f"SIGN: {clean_name}",
            color='#ffffff', fontsize=16, fontweight='bold',
            ha='center', va='center', zorder=2)
    ax.text(canvas_w / 2, canvas_h - 74, "American Sign Language (ASL) Guide",
            color='#e0e7ff', fontsize=8.5, ha='center', va='center', zorder=2)

    # Draw Hand Bones
    for start_idx, end_idx in HAND_CONNECTIONS:
        x_pair = [xs[start_idx], xs[end_idx]]
        y_pair = [ys[start_idx], ys[end_idx]]
        # Outer glow
        ax.plot(x_pair, y_pair, color='#818cf8', linewidth=7, alpha=0.35, zorder=3)
        # Inner clean bone
        ax.plot(x_pair, y_pair, color='#4f46e5', linewidth=3.5, zorder=4)

    # Draw Joints
    for i in range(21):
        if i in FINGERTIP_INDICES:
            # Fingertips highlighted
            ax.scatter(xs[i], ys[i], s=120, color='#ec4899', edgecolors='#ffffff', linewidth=2, zorder=6)
        elif i == 0:
            # Wrist
            ax.scatter(xs[i], ys[i], s=140, color='#0284c7', edgecolors='#ffffff', linewidth=2.5, zorder=6)
        else:
            # Knuckles
            ax.scatter(xs[i], ys[i], s=75, color='#10b981', edgecolors='#ffffff', linewidth=1.5, zorder=5)

    # Legend at bottom right of viewport
    ax.scatter(40, canvas_h - 110, s=40, color='#ec4899', edgecolors='#fff', zorder=5)
    ax.text(50, canvas_h - 110, "Fingertip", fontsize=7.5, color='#64748b', va='center')
    ax.scatter(120, canvas_h - 110, s=40, color='#10b981', edgecolors='#fff', zorder=5)
    ax.text(130, canvas_h - 110, "Knuckle", fontsize=7.5, color='#64748b', va='center')
    ax.scatter(195, canvas_h - 110, s=40, color='#0284c7', edgecolors='#fff', zorder=5)
    ax.text(205, canvas_h - 110, "Wrist Base", fontsize=7.5, color='#64748b', va='center')

    # Posture Instructions Box at Bottom
    tip_box = FancyBboxPatch(
        (25, 25), canvas_w - 50, 75,
        boxstyle="round,pad=6,rounding_size=12",
        facecolor='#ffffff', edgecolor='#cbd5e1', linewidth=1, zorder=1
    )
    ax.add_patch(tip_box)

    tip_text = POSTURE_TIPS.get(char, "Perform gesture steadily facing webcam.")
    ax.text(canvas_w / 2, 80, "HOW TO POSE YOUR HAND:",
            color='#4338ca', fontsize=8.5, fontweight='bold', ha='center', va='center', zorder=2)
    ax.text(canvas_w / 2, 50, tip_text,
            color='#334155', fontsize=8, ha='center', va='center', wrap=True, zorder=2)

    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def generate_all_sign_images():
    """Generates pictures for all alphabets and creates the master chart."""
    all_chars = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + [
        "HELLO", "THANK_YOU", "YES", "NO", "I_LOVE_YOU", "SPACE", "DELETE"
    ]

    print(f"[Image Generator] Creating {len(all_chars)} gesture reference pictures in '{OUTPUT_DIR}'...")
    for char in all_chars:
        out_file = os.path.join(OUTPUT_DIR, f"{char}.png")
        render_sign_card(char, out_file)

    print(f"[Image Generator] Successfully created {len(all_chars)} individual sign pictures.")

    # Now create Master Alphabet Poster / Chart (6 columns x 5 rows)
    print("[Image Generator] Generating Master ASL Alphabet Chart poster...")
    alphabet_list = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    cols, rows = 6, 5

    fig, axes = plt.subplots(rows, cols, figsize=(15, 13), dpi=150)
    fig.patch.set_facecolor('#fafafc')
    fig.suptitle("ASL Fingerspelling Alphabet Reference Chart (A - Z)",
                 fontsize=20, fontweight='bold', color='#1e1b4b', y=0.98)

    for i, char in enumerate(alphabet_list):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        ax.set_facecolor('#ffffff')

        pts = get_alphabet_landmarks(char)
        xs = pts[:, 0]
        ys = -pts[:, 1]

        # Draw bones
        for start_idx, end_idx in HAND_CONNECTIONS:
            ax.plot([xs[start_idx], xs[end_idx]], [ys[start_idx], ys[end_idx]],
                    color='#6366f1', linewidth=2.5, zorder=2)

        # Draw joints
        for j in range(21):
            if j in FINGERTIP_INDICES:
                ax.scatter(xs[j], ys[j], s=40, color='#ec4899', edgecolors='#fff', zorder=4)
            elif j == 0:
                ax.scatter(xs[j], ys[j], s=45, color='#0284c7', edgecolors='#fff', zorder=4)
            else:
                ax.scatter(xs[j], ys[j], s=25, color='#10b981', edgecolors='#fff', zorder=3)

        ax.set_title(char, fontsize=18, fontweight='bold', color='#4338ca', pad=4)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.2, 1.1)
        ax.axis('off')

    # Hide unused subplots in last row
    for i in range(len(alphabet_list), rows * cols):
        r = i // cols
        c = i % cols
        axes[r, c].axis('off')

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    poster_path = os.path.join(OUTPUT_DIR, "ASL_Alphabet_Chart.png")
    plt.savefig(poster_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Image Generator] Saved Master Chart to '{poster_path}'.")


if __name__ == "__main__":
    generate_all_sign_images()

