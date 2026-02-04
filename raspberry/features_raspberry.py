import gc
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

# =======================
# CONFIG
# =======================
CSI_DATA_LENGTH = 384          # 192 subcarrier × I/Q
N_SUB = CSI_DATA_LENGTH // 2
SAMPLING_FREQUENCY = 20        # Hz
WINDOW_LENGTH = 200            # samples

# =======================
# SIGNAL UTILITIES
# =======================

def iq_to_complex_matrix(csi_raw_series):
    """
    csi_raw_series: iterable di liste lunghezza 384
    restituisce: ndarray (T, 192) complesso64
    """
    csi = np.stack(csi_raw_series.to_numpy(), axis=0).astype(np.float32)
    csi = csi.reshape(-1, N_SUB, 2)
    return csi[..., 0] + 1j * csi[..., 1]



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
    df = df[["csi_raw"]].dropna()

    if len(df) < WINDOW_LENGTH:
        return None

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
    A_pulse = butter_bandpass_filter(A_dc, SAMPLING_FREQUENCY)

    del A_dc
    gc.collect()

    # -----------------------
    # SAVITZKY–GOLAY (parallel)
    # -----------------------
    A_smooth = savgol_filter(A_pulse, 15, 3)

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

    return X
