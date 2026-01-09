"""
Sign Language Recognition System (Terminal-friendly)
- Uses Kaggle-style ASL image dataset (folder-per-class).
- Works on Windows / PowerShell.
- Avoids notebook-only functions and fixes keras import resolution.

Expected dataset folder structure:
  <project_root>/data/asl/
      A/
        img1.jpg
        img2.jpg
      B/
      ...
      space/
      nothing/
      del/

How to run:
  1) Activate venv:
        .venv\Scripts\activate
  2) Install deps:
        python -m pip install --upgrade pip
        python -m pip install numpy opencv-python scikit-learn tensorflow keras
  3) Run:
        python .\Sign_Language_Recognition_System_final.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2

# Use standalone Keras imports to avoid "tensorflow.keras could not be resolved" in VS Code.
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "asl"  # put Kaggle ASL folders here
IMG_SIZE = 64
MAX_IMAGES_PER_CLASS = 1500  # lower (e.g., 300-800) if your PC is slow / low RAM
RANDOM_SEED = 42
EPOCHS = 8
BATCH_SIZE = 32


def load_asl_image_dataset(
    data_path: Path,
    img_size: int = 64,
    max_images_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Loads images from:
        data_path/<label_name>/*.jpg|png|jpeg
    Returns:
        X: (N, img_size, img_size, 3) float32 in [0,1]
        y: (N,) int labels
        labels: list of label names in index order
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_path}\n"
            f"Create it and place ASL folders inside, e.g. {data_path}/A, {data_path}/B, ..."
        )

    labels = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    if not labels:
        raise FileNotFoundError(
            f"No class folders found inside: {data_path}\n"
            "Expected structure like data/asl/A/*.jpg, data/asl/B/*.jpg ..."
        )

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for idx, label in enumerate(labels):
        folder = data_path / label
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in valid_ext]

        if not files:
            print(f"[WARN] No images in class folder: {folder}")
            continue

        if max_images_per_class is not None:
            files = files[: max_images_per_class]

        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            X_list.append(img)      # shape (H,W,3)
            y_list.append(idx)

        print(f"Loaded {len(files):>4} images from class '{label}'")

    if not X_list:
        raise RuntimeError("No images were loaded. Check your dataset path and file extensions.")

    X = np.array(X_list, dtype=np.float32) / 255.0
    y = np.array(y_list, dtype=np.int64)

    return X, y, labels


def build_cnn(num_classes: int, img_size: int = 64) -> Sequential:
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 3)),
            MaxPooling2D(),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> int:
    print("Sign Language Recognition System Scope:")
    print("- targeted_sign_languages: ASL gestures based on Kaggle folder labels")
    print("- operational_environment: Offline image classification")
    print("- intended_application: Communication assistance and educational tools")
    print()

    print(f"Dataset path: {DATA_PATH}")
    X, y, labels = load_asl_image_dataset(
        DATA_PATH, img_size=IMG_SIZE, max_images_per_class=MAX_IMAGES_PER_CLASS
    )

    print("\nDataset summary:")
    print(f"- Total samples: {len(X)}")
    print(f"- Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"- Num classes: {len(labels)}")
    print(f"- Example classes: {labels[:10]}{' ...' if len(labels) > 10 else ''}")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    model = build_cnn(num_classes=len(labels), img_size=IMG_SIZE)
    model.summary()

    callbacks = [EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)]

    print("\nTraining...")
    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nEvaluating...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")

    print("\nClassification report (test set):")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

    print("\nConfusion matrix (top-left 10x10 if many classes):")
    cm = confusion_matrix(y_test, y_pred)
    max_show = min(10, cm.shape[0])
    print(cm[:max_show, :max_show])

    out_path = PROJECT_ROOT / "asl_cnn_model.keras"
    model.save(out_path)
    print(f"\nSaved model to: {out_path}")
    print("\nDone ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
