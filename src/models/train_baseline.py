"""
Train classical ML baselines (Random Forest, SVM) on handcrafted
features (RR-interval, QRS duration, peak amplitude, signal energy).

Usage:
    python src/models/train_baseline.py --model rf
    python src/models/train_baseline.py --model svm
"""

import os
import argparse
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "models")


def load_data():
    features = np.load(os.path.join(DATA_DIR, "features.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
    return features, labels


def build_model(name):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
    if name == "svm":
        return SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=42,
        )
    raise ValueError(f"Unknown model: {name}. Choose 'rf' or 'svm'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "svm"], default="rf")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = build_model(args.model)
    print(f"Training {args.model.upper()} on {len(X_train)} beats "
          f"({X_train.shape[1]} features) ...")
    model.fit(X_train_scaled, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, f"{args.model}.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{args.model}_scaler.joblib"))

    print(f"Saved model to {MODEL_DIR}/{args.model}.joblib")
    print("Run evaluate.py to see metrics on the held-out test split.")

    # Save the exact test split used, for consistent evaluation
    np.savez(
        os.path.join(MODEL_DIR, f"{args.model}_split.npz"),
        X_test=X_test, y_test=y_test,
    )


if __name__ == "__main__":
    main()
