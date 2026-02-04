import os
import ast
import gc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt, savgol_filter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

# =======================
# CONFIG
# =======================

RAW_DATA_PATH = "data/raw_data.csv"

CSI_DATA_LENGTH = 384          # 192 subcarrier × I/Q
N_SUB = CSI_DATA_LENGTH // 2

SAMPLING_FREQUENCY = 20        # Hz
WINDOW_LENGTH = 200            # samples
LEARNING_RATE = 0.001          #1e-4
MSE_THRESHOLD = 0.01

TRAIN_MODEL = True
SAVE_TRAIN_DATA = False

MODEL_PATH = f"models/csi_hr_{WINDOW_LENGTH}.keras"
CONTINUE_MODEL = f"models/csi_hr_best_{WINDOW_LENGTH}.keras"
BATCH_SIZE = 64

# =======================
# TF GPU SAFE CONFIG
# =======================

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

# =======================
# SIGNAL UTILITIES
# =======================

def iq_to_complex_matrix(csi_raw_series):
    """
    csi_raw_series: iterable di liste lunghezza 384
    restituisce: ndarray (T, 192) complesso64
    """
    valid_rows = []
    for i, row in enumerate(csi_raw_series):
        # controlla che sia lista e abbia lunghezza esatta
        if isinstance(row, list) and len(row) == 384:
            valid_rows.append(np.array(row, dtype=np.float32))
        else:
            print(f"Riga {i} scartata, lunghezza={len(row) if isinstance(row, list) else type(row)}")

    if len(valid_rows) == 0:
        raise ValueError("Nessuna riga CSI valida trovata!")

    # forma (T, 384)
    csi = np.stack(valid_rows, axis=0)

    # separa I/Q in complessi
    I = csi[:, 0::2]
    Q = csi[:, 1::2]
    return I + 1j * Q



def butter_bandpass_filter(x, fs, lowcut=0.8, highcut=2.17, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, x)


# =======================
# FEATURE EXTRACTION
# =======================

def extract_features(df, training=True):
    """
    Return:
        X: (N, W, 192) float32
        y: (N,) float32 or None
    """

    # -----------------------
    # CLEAN INPUT
    # -----------------------
    cols = ["csi_raw", "AVG BPM"] if training else ["csi_raw"]
    df = df[cols].dropna()

    # tieni solo CSI della lunghezza giusta (384)
    df = df[df["csi_raw"].apply(lambda x: isinstance(x, list) and len(x) == CSI_DATA_LENGTH)]

    if len(df) < WINDOW_LENGTH:
        return None, None

    # -----------------------
    # CSI → COMPLEX
    # -----------------------
    A_complex = iq_to_complex_matrix(df["csi_raw"])
    A_complex = A_complex.astype(np.complex64)

    # -----------------------
    # AMPLITUDE
    # -----------------------
    A_amp = np.abs(A_complex).astype(np.float32)
    del A_complex
    gc.collect()

    # -----------------------
    # DC REMOVAL (vectorized)
    # -----------------------
    kernel = np.ones(WINDOW_LENGTH, dtype=np.float32) / WINDOW_LENGTH
    mean_dc = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="same"),
        axis=0,
        arr=A_amp
    )
    A_dc = A_amp - mean_dc
    del A_amp, mean_dc
    gc.collect()

    # -----------------------
    # BANDPASS (parallel)
    # -----------------------
    A_pulse = np.stack(
        Parallel(n_jobs=-1, prefer="processes")(
            delayed(butter_bandpass_filter)(
                A_dc[:, k], SAMPLING_FREQUENCY
            )
            for k in range(N_SUB)
        ),
        axis=1
    ).astype(np.float32)

    del A_dc
    gc.collect()

    # -----------------------
    # SAVITZKY–GOLAY (parallel)
    # -----------------------
    A_smooth = np.stack(
        Parallel(n_jobs=-1)(
            delayed(savgol_filter)(
                A_pulse[:, k], 15, 3
            )
            for k in range(N_SUB)
        ),
        axis=1
    ).astype(np.float32)

    del A_pulse
    gc.collect()

    # -----------------------
    # WINDOWING (zero copy)
    # -----------------------
    X = np.lib.stride_tricks.sliding_window_view(
        A_smooth,
        window_shape=(WINDOW_LENGTH, N_SUB)
    )[:, 0, :, :]

    # -----------------------
    # NORMALIZATION
    # -----------------------
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-6
    X = ((X - mean) / std).astype(np.float32)

    del A_smooth
    gc.collect()

    if not training:
        return X, None

    # -----------------------
    # HR WINDOW
    # -----------------------
    y = None
    if training:
        hr = df["AVG BPM"].to_numpy(dtype=np.float32)
        y = np.convolve(hr, np.ones(WINDOW_LENGTH) / WINDOW_LENGTH, mode="valid")
        y = y.astype(np.float32)

    return X, y


# =======================
# MAIN
# =======================

if __name__ == "__main__":

    print("Loading CSV...")
    df = pd.read_csv(RAW_DATA_PATH)
    df["csi_raw"] = df["csi_raw"].apply(ast.literal_eval)

    print(f"Initial dataset length: {len(df)}")

    print("Extracting features...")
    X, y = extract_features(df, training=TRAIN_MODEL)

    print("X:", X.shape)
    print("y:", None if y is None else y.shape)

    if not TRAIN_MODEL:
        exit()

    print("X shape:", X.shape)
    print("Estimated size (GB):", X.nbytes / 1e9)

    rng = np.random.default_rng(seed=42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # =======================
    # MODEL
    # =======================

    inputs = keras.Input(shape=(WINDOW_LENGTH, N_SUB))
    x = keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)

    model = None
    if CONTINUE_MODEL is not None:
        print("Resuming training...")
        model = keras.models.load_model(CONTINUE_MODEL)
    else:
        model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="mse"
    )

    model.summary()

    # =======================
    # TRAINING
    # =======================

    checkpoint = ModelCheckpoint(
        filepath=f"models/csi_hr_best_{WINDOW_LENGTH}.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    class StopCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs and logs.get("val_loss", 1.0) <= MSE_THRESHOLD:
                print("Reached MSE threshold. Stopping.")
                self.model.stop_training = True

    model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=5000,
        validation_split=0.3,
        callbacks=[checkpoint, StopCallback()],
        verbose=2
    )

    model.save(MODEL_PATH)
    print("Training complete.")
