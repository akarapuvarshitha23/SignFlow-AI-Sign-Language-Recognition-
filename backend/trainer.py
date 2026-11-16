"""
Model Trainer Module
Trains, validates, evaluates, and exports the Deep Learning Sign Recognition Model.
"""

import os
import sys

# Ensure root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from backend.model_builder import build_sign_classifier
from dataset.dataset_generator import generate_synthetic_dataset, CLASSES


def train_sign_model(
    dataset_csv="dataset/sign_data.csv",
    model_save_path="model/sign_model.keras",
    epochs=40,
    batch_size=32
):
    """Orchestrates end-to-end dataset loading, training, and evaluation."""
    print("=" * 60)
    print("      SignFlow AI - Deep Learning Model Training")
    print("=" * 60)

    # 1. Ensure dataset exists
    if not os.path.exists(dataset_csv):
        print(f"[Trainer] Dataset {dataset_csv} not found. Generating...")
        generate_synthetic_dataset(samples_per_class=450, output_csv=dataset_csv)

    print(f"[Trainer] Loading dataset from {dataset_csv}...")
    df = pd.read_csv(dataset_csv)
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    X = df[feature_cols].values.astype(np.float32)
    y_labels = df["label"].values

    # Map labels to integers
    label_to_idx = {name: idx for idx, name in enumerate(CLASSES)}
    y = np.array([label_to_idx[l] for l in y_labels], dtype=np.int32)

    # Train / Val / Test Split: 70% / 15% / 15%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"[Trainer] Dataset Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    print(f"[Trainer] Features per sample: {X.shape[1]} | Classes: {len(CLASSES)}")

    # 2. Build Model
    model = build_sign_classifier(
        input_dim=X.shape[1],
        num_classes=len(CLASSES),
        learning_rate=0.001
    )
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 3. Train
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    training_duration = time.time() - start_time

    # 4. Evaluate on Test Set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n" + "=" * 60)
    print(f"      EVALUATION RESULTS (Held-out Test Set)")
    print("=" * 60)
    print(f"Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")
    print(f"Total Training Time: {training_duration:.2f}s")

    # 5. Save Model and Metadata
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"[Trainer] Trained deep learning model saved to {model_save_path}")

    # Save Label Map
    idx_to_label = {idx: name for idx, name in enumerate(CLASSES)}
    with open("model/label_map.json", "w") as f:
        json.dump(idx_to_label, f, indent=2)

    # Save Metadata
    metadata = {
        "model_name": "SignFlow Deep Multilayer Perceptron",
        "input_dimension": int(X.shape[1]),
        "num_classes": len(CLASSES),
        "classes": CLASSES,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "total_samples": int(len(df)),
        "epochs_trained": len(history.history["loss"]),
        "training_time_seconds": round(training_duration, 2),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1])
    }
    with open("model/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return model, metadata


if __name__ == "__main__":
    train_sign_model()

