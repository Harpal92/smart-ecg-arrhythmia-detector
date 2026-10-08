"""
Evaluate a trained model (rf, svm, or cnn) on its held-out test split.

Usage:
    python src/models/evaluate.py --model rf
    python src/models/evaluate.py --model svm
    python src/models/evaluate.py --model cnn
    python src/models/evaluate.py --model all      # compare all three
"""

import os
import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "models")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "figures")


def evaluate_sklearn_model(name):
    model = joblib.load(os.path.join(MODEL_DIR, f"{name}.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, f"{name}_scaler.joblib"))
    split = np.load(os.path.join(MODEL_DIR, f"{name}_split.npz"))
    X_test, y_test = split["X_test"], split["y_test"]

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    return y_test, y_pred


def evaluate_cnn_model():
    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "cnn_model.keras"))
    split = np.load(os.path.join(MODEL_DIR, "cnn_split.npz"))
    X_test, y_test = split["X_test"], split["y_test"]

    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    return y_test, y_pred


def report(name, y_true, y_pred):
    print(f"\n{'=' * 50}\n{name.upper()} — Test Set Performance\n{'=' * 50}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Arrhythmic"]))

    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Arrhythmic"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Arrhythmic"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{name.upper()} Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, f"{name}_confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "svm", "cnn", "all"], default="all")
    args = parser.parse_args()

    targets = ["rf", "svm", "cnn"] if args.model == "all" else [args.model]

    for name in targets:
        try:
            if name in ("rf", "svm"):
                y_true, y_pred = evaluate_sklearn_model(name)
            else:
                y_true, y_pred = evaluate_cnn_model()
            report(name, y_true, y_pred)
        except FileNotFoundError:
            print(f"\n[{name.upper()}] No trained model found — run train_baseline.py "
                  f"or train_cnn.py first.")


if __name__ == "__main__":
    main()
