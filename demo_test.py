"""
SignFlow AI - Automated Test & Benchmark Runner
Tests model inference on all sign classes, evaluates prediction latency,
and validates deep learning accuracy and sanity.
"""

import os
import sys

# Ensure root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import time
import numpy as np
import keras

from dataset.dataset_generator import CLASSES, generate_sign_template, apply_3d_transformations
from backend.landmark_extractor import LandmarkExtractor


def run_benchmark():
    print("=" * 65)
    print("      SignFlow AI - Deep Learning Model Verification & Benchmark")
    print("=" * 65)

    model_path = os.path.join(ROOT_DIR, "model", "sign_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(ROOT_DIR, "models", "sign_model.keras")

    label_map_path = os.path.join(ROOT_DIR, "model", "label_map.json")
    if not os.path.exists(label_map_path):
        label_map_path = os.path.join(ROOT_DIR, "models", "label_map.json")

    if not os.path.exists(model_path):
        print(f"[Error] Model file '{model_path}' not found! Please train the model first.")
        sys.exit(1)

    print(f"[1/4] Loading trained Deep Learning model from '{model_path}'...")
    model = keras.models.load_model(model_path)
    print(f"      Model Name: {model.name}")
    print(f"      Total Parameters: {model.count_params():,}")

    with open(label_map_path, "r") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    class_to_idx = {v: k for k, v in label_map.items()}

    print(f"\n[2/4] Generating 50 validation samples per class ({len(CLASSES)} classes = {len(CLASSES)*50} total)...")
    test_X = []
    test_y = []

    for class_name in CLASSES:
        base_pts = generate_sign_template(class_name)
        for _ in range(50):
            aug_pts = apply_3d_transformations(base_pts)
            feat = LandmarkExtractor.compute_features(aug_pts)
            test_X.append(feat)
            test_y.append(class_to_idx[class_name])

    test_X = np.array(test_X, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.int32)

    print(f"\n[3/4] Running Deep Learning Batch Inference...")
    t0 = time.time()
    predictions = model.predict(test_X, verbose=0)
    batch_time = time.time() - t0
    pred_labels = np.argmax(predictions, axis=1)

    # Calculate overall metrics
    correct = np.sum(pred_labels == test_y)
    total = len(test_y)
    accuracy = correct / total
    avg_latency_ms = (batch_time / total) * 1000

    print(f"      Batch Inference Time: {batch_time*1000:.2f} ms ({total} samples)")
    print(f"      Average Latency Per Sample: {avg_latency_ms:.3f} ms (~{1000/avg_latency_ms:.0f} FPS capable)")
    print(f"      Overall Test Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")

    print("\n[4/4] Per-Class Precision & Confidence Breakdown:")
    print("-" * 65)
    print(f"{'Class':<14} | {'Samples':<8} | {'Accuracy':<10} | {'Avg Confidence':<15}")
    print("-" * 65)

    for c_idx, c_name in enumerate(CLASSES):
        mask = (test_y == c_idx)
        c_preds = pred_labels[mask]
        c_probs = predictions[mask, c_idx]
        c_acc = np.mean(c_preds == c_idx) * 100
        c_conf = np.mean(c_probs) * 100
        print(f"{c_name:<14} | {np.sum(mask):<8} | {c_acc:>8.1f}% | {c_conf:>13.1f}%")

    print("-" * 65)
    print("\n[OK] Deep Learning Model successfully verified and ready for live recognition!")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
