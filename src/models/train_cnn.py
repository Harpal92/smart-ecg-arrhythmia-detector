"""
Train a 1D CNN directly on raw beat waveform segments (no handcrafted
features) to classify beats as Normal (0) or Arrhythmic (1).

Usage:
    python src/models/train_cnn.py --epochs 20
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "models")


def load_data():
    segments = np.load(os.path.join(DATA_DIR, "segments.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
    # Per-beat z-normalization (each waveform independently)
    mean = segments.mean(axis=1, keepdims=True)
    std = segments.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    segments = (segments - mean) / std
    return segments[..., np.newaxis], labels  # add channel dim for Conv1D


def build_cnn(input_len):
    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.Conv1D(32, kernel_size=7, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    model = build_cnn(X.shape[1])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "cnn_model.keras"))
    np.savez(os.path.join(MODEL_DIR, "cnn_split.npz"), X_test=X_test, y_test=y_test)

    print(f"Saved model to {MODEL_DIR}/cnn_model.keras")
    print("Run evaluate.py to see metrics on the held-out test split.")


if __name__ == "__main__":
    main()
