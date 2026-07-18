"""
Heart Sound Classification Model
Trains a lightweight 1D CNN on MFCC features from PhysioNet 2016 Challenge data.
Outputs a TensorFlow.js-compatible model for in-browser inference.
"""
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from pathlib import Path

# Config
DATA_DIR = Path("/home/ubuntu/heartbeat-app/data/training")
MODEL_DIR = Path("/home/ubuntu/heartbeat-app/tfjs_model")
SAMPLE_RATE = 2000  # Downsample to 2kHz (heart sounds are 25-600Hz)
SEGMENT_DURATION = 3  # seconds
N_MFCC = 20
HOP_LENGTH = 128
MAX_FRAMES = int(SAMPLE_RATE * SEGMENT_DURATION / HOP_LENGTH) + 1


def load_labels(data_dir):
    """Load labels from REFERENCE.csv files in each training subfolder."""
    labels = {}
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        ref_file = subdir / "REFERENCE.csv"
        if not ref_file.exists():
            continue
        with open(ref_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    fname, label = parts
                    # -1 = normal, 1 = abnormal
                    labels[subdir / f"{fname}.wav"] = 1 if int(label) == 1 else 0
    return labels


def extract_mfcc(audio_path):
    """Load audio and extract MFCC features."""
    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, duration=SEGMENT_DURATION)
    except Exception:
        return None

    # Pad if shorter than expected
    target_len = SAMPLE_RATE * SEGMENT_DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    # Transpose to (frames, n_mfcc)
    mfcc = mfcc.T

    # Pad/trim to fixed size
    if mfcc.shape[0] < MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0, MAX_FRAMES - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_FRAMES]

    return mfcc


def build_model(input_shape):
    """Build a lightweight 1D CNN for heart sound classification."""
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv1D(32, 5, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.2),

        keras.layers.Conv1D(64, 5, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.2),

        keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2, activation='softmax')
    ])
    return model


def main():
    print("Loading labels...")
    labels = load_labels(DATA_DIR)
    print(f"Found {len(labels)} labeled recordings")

    if len(labels) == 0:
        print("ERROR: No labels found. Check DATA_DIR path.")
        return

    print("Extracting MFCC features...")
    X, y = [], []
    for i, (audio_path, label) in enumerate(labels.items()):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(labels)}...")
        mfcc = extract_mfcc(audio_path)
        if mfcc is not None:
            X.append(mfcc)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Dataset: {X.shape[0]} samples, {np.sum(y==0)} normal, {np.sum(y==1)} abnormal")

    # Normalize
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - mean) / std

    # Save normalization params for inference
    np.save(str(MODEL_DIR / "mfcc_mean.npy"), mean.squeeze())
    np.save(str(MODEL_DIR / "mfcc_std.npy"), std.squeeze())

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Build and train
    model = build_model((MAX_FRAMES, N_MFCC))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Class weights for imbalanced data
    n_normal = np.sum(y_train == 0)
    n_abnormal = np.sum(y_train == 1)
    weight_normal = len(y_train) / (2 * n_normal)
    weight_abnormal = len(y_train) / (2 * n_abnormal)
    class_weight = {0: weight_normal, 1: weight_abnormal}
    print(f"Class weights: normal={weight_normal:.2f}, abnormal={weight_abnormal:.2f}")

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {accuracy:.4f}")

    # Save Keras model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    keras_path = str(MODEL_DIR / "heart_model.keras")
    model.save(keras_path)
    print(f"Saved Keras model to {keras_path}")

    # Convert to TF.js
    print("Converting to TensorFlow.js format...")
    os.system(f"pip3 install --break-system-packages tensorflowjs 2>/dev/null")
    os.system(f"tensorflowjs_converter --input_format=keras '{keras_path}' '{MODEL_DIR}/web_model'")
    print(f"TF.js model saved to {MODEL_DIR}/web_model/")

    # Save config for the web app
    import json
    config = {
        "sampleRate": SAMPLE_RATE,
        "segmentDuration": SEGMENT_DURATION,
        "nMfcc": N_MFCC,
        "hopLength": HOP_LENGTH,
        "maxFrames": MAX_FRAMES,
        "classes": ["normal", "abnormal"]
    }
    with open(str(MODEL_DIR / "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()
