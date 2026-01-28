import pandas as pd
import os
import ast
import re
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

RAW_DATA_PATH = "data/raw_data.csv" # where to get the raw data
CSI_DATA_LENGTH = 384               # esp32 exposes only 192 subcarriers, each carrier has associated I/Q components, so 192 x 2 = 384
SAMPLING_FREQUENCY = 20
SEGMENTATION_WINDOW_LENGTH = 200
SUBCARRIER_PLOT = None
MSE_THRESHOLD = 0.01
SAVE_TRAIN_DATA = False
TRAIN_MODEL = True
LEARNING_RATE = 1e-4



def extract_features(df, settings):
    training_phase = settings["training_phase"]
    verbose = settings["verbose"]
    csi_data_length = settings["csi_data_length"]
    sampling_frequency = settings["sampling_frequency"]
    segmentation_window_length = settings["segmentation_window_length"]

    #remove useless columns
    if training_phase:
        df_csi = df[["local_timestamp", "csi_raw", "AVG BPM"]].copy()
    else:
        df_csi = df[["local_timestamp", "csi_raw"]].copy()

    # remove corrupted data fields
    df_csi = df_csi.dropna()

    if len(df_csi) < segmentation_window_length:
        if verbose:
            print("not enough data")
        return []

    # extract I/Q components from the raw csi data
    def iq_to_complex(csi_raw):
        csi = np.array(csi_raw, dtype=float)
        I = csi[0::2]
        Q = csi[1::2]
        return I + 1j * Q

    df_csi["csi_complex"] = df_csi["csi_raw"].apply(iq_to_complex)
    df_csi["csi_len_complex"] = df_csi["csi_complex"].apply(len)
    df_csi = df_csi[df_csi["csi_len_complex"] == csi_data_length / 2].copy()

    # remove the rest of the useless columns
    if training_phase:
        df_csi = df_csi[["local_timestamp", "csi_complex", "AVG BPM"]].copy()
    else:
        df_csi = df_csi[["local_timestamp", "csi_complex"]].copy()

    if verbose:
        print(f"initial len: {len(df)}, final len: {len(df_csi)}. Lost {100 * (len(df) - len(df_csi)) / len(df)}%")
        print(df_csi.head())


    # START OF PULSE-FI CSI PROCESSING

    # 1. compute amplitude, ignore phase
    df_csi["csi_amp"] = df_csi["csi_complex"].apply(np.abs)

    # 2. stationary noise removal (remove the dc component)
    buffer = deque(maxlen=segmentation_window_length)
    def dc_remove_online(amp):
        buffer.append(amp)
        mean = np.mean(buffer, axis=0)
        return amp - mean

    df_csi["csi_amp_no_dc"] = df_csi["csi_amp"].apply(dc_remove_online)

    # 3. pulse extraction (Butterworth bandpass filter 0.8 - 2.17 Hz)
    def butter_bandpass(lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass(x, lowcut=0.8, highcut=2.17, fs=50, order=3):
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        return filtfilt(b, a, x)

    A_dc = np.vstack(df_csi["csi_amp_no_dc"].values)
    fs = sampling_frequency
    num_subcarriers = A_dc.shape[1]
    A_pulse = np.zeros_like(A_dc)

    for k in range(num_subcarriers):
        A_pulse[:, k] = apply_bandpass(A_dc[:, k], lowcut=0.8, highcut=2.17, fs=fs, order=3)
    df_csi["csi_pulse"] = list(A_pulse)

    # 4. pulse shaping (Savitzky-Golay filter)
    def pulse_shaping(x):
        return savgol_filter(x, window_length=15, polyorder=3)

    # Applica subcarrier per subcarrier
    A_pulse = np.vstack(df_csi["csi_pulse"].values)
    A_shaped = np.zeros_like(A_pulse)

    for k in range(A_pulse.shape[1]):
        A_shaped[:, k] = pulse_shaping(A_pulse[:, k])

    # Rimetti nel DataFrame
    df_csi["csi_pulse_smooth"] = list(A_shaped)

    subcarrier_idx = SUBCARRIER_PLOT
    if subcarrier_idx is not None:
        signal = A_shaped[:, subcarrier_idx]
        t = np.arange(len(signal)) / fs  # tempo in secondi
        plt.figure(figsize=(12, 4))
        plt.plot(t, signal, color='b')
        plt.title(f'Segnale pulsatile - Subcarrier {subcarrier_idx}')
        plt.xlabel('Tempo [s]')
        plt.ylabel('Ampiezza CSI DC-removed e filtrata')
        plt.grid(True)
        plt.show()

    # 5. data segmentation and normalization
    A_shaped = np.vstack(df_csi["csi_pulse_smooth"].values)
    def segment_multichannel(A):
        """
        Segmenta A (tempo x subcarrier) in finestre sovrapposte.
        Ogni finestra ha shape (window_size, num_subcarrier)
        """
        num_packets, num_subcarriers = A.shape
        num_windows = num_packets - segmentation_window_length + 1
        windows = np.array([A[i:i+segmentation_window_length, :] for i in range(num_windows)])
        return windows

    windows = segment_multichannel(A_shaped)

    def normalize_windows_multichannel(windows):
        """
        Normalizza ogni finestra sul tempo per ciascun subcarrier.
        """
        mean = windows.mean(axis=1, keepdims=True)   # media lungo il tempo (dimensione 1)
        std = windows.std(axis=1, keepdims=True)
        return (windows - mean) / (std + 1e-8)       # epsilon per evitare divisione per 0

    if len(windows) == 0:
        if verbose:
            print("no windows generated")
        return []
    windows_norm = normalize_windows_multichannel(windows)
    if not training_phase:
        if len(windows) == 0:
            if verbose:
                print("no windows generated after norm")
        return windows_norm

    def segment_hr(hr_series):
        """
        Restituisce la media HR per ogni finestra sovrapposta.
        """
        hr_array = hr_series.to_numpy()
        num_windows = len(hr_array) - segmentation_window_length + 1
        hr_windows = np.array([hr_array[i:i+segmentation_window_length].mean() for i in range(num_windows)])
        return hr_windows

    hr_windows = segment_hr(df_csi['AVG BPM'])

    window_duration = segmentation_window_length / fs
    if verbose:
        print(f"each window lasts {window_duration:.2f} seconds")

    # save
    df_windows = pd.DataFrame({
        'window': [np.array2string(w, separator=',', threshold=np.inf) for w in windows_norm],
        'AVG BPM': list(hr_windows)
    })
    if verbose:
        print(df_windows['AVG BPM'].value_counts(bins=10))
    
    bins = [60, 66.717, 72.434, 78.151, 83.868, 89.585, 95.302, 101.019, 106.736, 112.453, 118.17]
    labels = range(len(bins)-1)
    df_windows['BPM_bin'] = pd.cut(df_windows['AVG BPM'], bins=bins, labels=labels, include_lowest=True)
    # trovo la dimensione della classe più grande
    max_count = df_windows['BPM_bin'].value_counts().max()
    # funzione per fare oversampling
    def oversample(df, target_col, max_count):
        dfs = []
        for cls, group in df.groupby(target_col):
            n_repeat = max_count // len(group)
            remainder = max_count % len(group)
            df_rep = pd.concat([group]*n_repeat + [group.sample(remainder, replace=True)])
            dfs.append(df_rep)
        return pd.concat(dfs).sample(frac=1).reset_index(drop=True)  # shuffle
    # applico oversampling
    df_balanced = oversample(df_windows, 'BPM_bin', max_count)
    # ora df_balanced ha tutte le classi bilanciate
    df_windows = df_windows[["window", "AVG BPM"]].copy()
    if verbose:
        print(df_balanced['AVG BPM'].value_counts(bins=10))

    return df_balanced

if __name__ == '__main__':
    settings = {}
    settings["training_phase"] = TRAIN_MODEL
    settings["verbose"] = True
    settings["csi_data_length"] = CSI_DATA_LENGTH
    settings["sampling_frequency"] = SAMPLING_FREQUENCY
    settings["segmentation_window_length"] = SEGMENTATION_WINDOW_LENGTH

    df = pd.read_csv(RAW_DATA_PATH, sep=",")
    print(df.columns)
    df["csi_raw"] = df["csi_raw"].apply(ast.literal_eval)
    df = extract_features(df, settings)
    if SAVE_TRAIN_DATA:
        df.to_csv(f"data/train_data_{SEGMENTATION_WINDOW_LENGTH}.csv", index=False)

    if not TRAIN_MODEL:
        exit()
    
    # Mostra dispositivi disponibili
    print(tf.config.list_physical_devices('GPU'))


    class stopCallback(keras.callbacks.Callback): 
        def on_epoch_end(self, epoch, logs={}): 
            if(logs.get('val_loss') <= MSE_THRESHOLD) or os.path.isfile("training.stop"):   
                print("\nReached {0} MSE; stopping training.".format(MSE_THRESHOLD)) 
                self.model.stop_training = True


    df['window'] = df['window'].apply(lambda x: np.array(ast.literal_eval(x)))
    print(df.dtypes)

    # Build the model.
    main_input = keras.Input(shape=(SEGMENTATION_WINDOW_LENGTH, int(CSI_DATA_LENGTH / 2)), name='main_input')
    layers = keras.layers.LSTM(64, return_sequences=True, name='lstm_1')(main_input)
    layers = keras.layers.Dropout(0.2, name='dropout_1')(layers)
    layers = keras.layers.LSTM(32, name='lstm_2')(layers)
    layers = keras.layers.Dropout(0.2, name='dropout_2')(layers)
    layers = keras.layers.Dense(16, activation='relu', name='dense_1')(layers)
    hr_output = keras.layers.Dense(1, name='hr_output')(layers)
    
    model = keras.Model(inputs=main_input, outputs=hr_output)

    # Compile the model.
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        
    model.compile(optimizer=optimizer, 
        loss={'hr_output': 'mse'}
    )

    # Print the model summary.
    model.summary()

    # Prepare the training data.
    X = np.stack(df['window'].values)  # shape = (num_windows, window_size, num_subcarrier)
    y = df['AVG BPM'].to_numpy()             # target LSTM

    print("Training data X shape: {0}".format(X.shape))
    print("Training data Y shape: {0}".format(y.shape))

    # Train the model.
    checkpoint = ModelCheckpoint(
        filepath=f"models/csi_hr_best_{SEGMENTATION_WINDOW_LENGTH}.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )
    callbacks_list = [
        stopCallback(),
        checkpoint
    ]

    model.fit(X, y, batch_size=128, epochs=5000, verbose=2, validation_split=0.2, callbacks=callbacks_list)
    model.save(f"models/csi_hr_{SEGMENTATION_WINDOW_LENGTH}.keras")
    print("Model training complete!")