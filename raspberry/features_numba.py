import gc
import numpy as np
import numba as nb


# =======================
# SIGNAL UTILITIES
# =======================

def iq_to_complex_matrix(csi_raw_series, n_sub):
    """
    csi_raw_series: iterable di liste lunghezza 384
    restituisce: ndarray (T, 192) complesso64
    """
    csi = np.stack(csi_raw_series.to_numpy(), axis=0).astype(np.float32)
    csi = csi.reshape(-1, n_sub, 2)
    return (csi[..., 0] + 1j * csi[..., 1]).astype(np.complex64)


# =======================
# NUMBA DSP KERNELS
# =======================

@nb.njit(parallel=True, fastmath=True)
def remove_dc_parallel(A, win):
    """
    Moving-average DC removal per subcarrier
    A: (T, C)
    """
    T, C = A.shape
    out = np.empty_like(A)

    for c in nb.prange(C):
        s = 0.0
        for i in range(T):
            s += A[i, c]
            if i >= win:
                s -= A[i - win, c]

            if i >= win - 1:
                mean = s / win
            else:
                mean = s / (i + 1)

            out[i, c] = A[i, c] - mean

    return out


@nb.njit(parallel=True, fastmath=True)
def bandpass_fft_parallel(A, fs, low, high):
    """
    FFT bandpass per subcarrier
    """
    T, C = A.shape
    out = np.empty_like(A)

    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    mask = (freqs >= low) & (freqs <= high)

    for c in nb.prange(C):
        X = np.fft.rfft(A[:, c])
        X *= mask
        out[:, c] = np.fft.irfft(X, n=T)

    return out


@nb.njit(parallel=True, fastmath=True)
def smooth_parallel(A):
    """
    FIR smoothing tipo Savitzky-Golay (kernel fisso len=15)
    """
    T, C = A.shape
    out = np.empty_like(A)

    # kernel Savitzky-Golay precomputato (poly=3, win=15)
    k = np.array([
        -3, -2, -1, 0, 1, 2, 3,
        4, 3, 2, 1, 0, -1, -2, -3
    ], dtype=np.float32)

    k = k / np.sum(np.abs(k))

    half = len(k) // 2

    for c in nb.prange(C):
        for i in range(T):
            s = 0.0
            for j in range(len(k)):
                idx = i + j - half
                if idx < 0:
                    idx = 0
                elif idx >= T:
                    idx = T - 1
                s += A[idx, c] * k[j]
            out[i, c] = s

    return out


@nb.njit(fastmath=True)
def normalize_windows(X):
    """
    Normalizzazione per finestra
    X: (N, W, C)
    """
    N, W, C = X.shape

    for n in range(N):
        mean = 0.0
        std = 0.0

        # mean
        for i in range(W):
            for c in range(C):
                mean += X[n, i, c]
        mean /= (W * C)

        # std
        for i in range(W):
            for c in range(C):
                diff = X[n, i, c] - mean
                std += diff * diff

        std = np.sqrt(std / (W * C)) + 1e-6

        # normalize
        for i in range(W):
            for c in range(C):
                X[n, i, c] = (X[n, i, c] - mean) / std

    return X


# =======================
# FEATURE EXTRACTION
# =======================

def extract_features(df, csi_data_length, sampling_frequency, window_length):
    """
    Return:
        X: (N, W, 192) float32
    """
    n_subcarriers = csi_data_length // 2

    # -----------------------
    # CLEAN INPUT
    # -----------------------
    df = df[["csi_raw"]].dropna()

    if len(df) < window_length:
        return None

    # -----------------------
    # 1. CSI → AMPLITUDE
    # -----------------------
    A_complex = iq_to_complex_matrix(df["csi_raw"], n_subcarriers)
    A_amp = np.abs(A_complex).astype(np.float32)
    del A_complex
    gc.collect()

    # -----------------------
    # 2. DC REMOVAL
    # -----------------------
    A_dc = remove_dc_parallel(A_amp, window_length)
    del A_amp
    gc.collect()

    # -----------------------
    # 3. BANDPASS FFT
    # -----------------------
    A_pulse = bandpass_fft_parallel(
        A_dc,
        sampling_frequency,
        0.8,
        2.17
    )
    del A_dc
    gc.collect()

    # -----------------------
    # 4. SMOOTH
    # -----------------------
    A_smooth = smooth_parallel(A_pulse)
    del A_pulse
    gc.collect()

    # -----------------------
    # 5. WINDOWING + NORMALIZATION
    # -----------------------
    X = np.lib.stride_tricks.sliding_window_view(
        A_smooth,
        window_shape=(window_length, n_subcarriers)
    )[:, 0, :, :].astype(np.float32)

    del A_smooth
    gc.collect()

    X = normalize_windows(X)

    return X
