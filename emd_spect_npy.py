import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal as sp_signal
from multiprocessing import Pool, cpu_count


def emd_decompose(signal, max_imfs=8, max_sift_iter=100, tol=1e-6):
    signal = signal.astype(np.float64)
    imfs = []
    residue = signal.copy()
    while len(imfs) < max_imfs and not _is_monotonic(residue):
        h = residue.copy()
        for _ in range(max_sift_iter):
            max_peaks, _ = sp_signal.find_peaks(h)
            min_peaks, _ = sp_signal.find_peaks(-h)
            if len(max_peaks) < 2 or len(min_peaks) < 2:
                break
            upper_env = _interpolate_envelope(h, max_peaks, len(h))
            lower_env = _interpolate_envelope(h, min_peaks, len(h))
            mean_env = (upper_env + lower_env) / 2
            h_new = h - mean_env
            if np.sum(h**2) < 1e-10:
                sd = 0
            else:
                sd = np.sum((h - h_new)**2) / np.sum(h**2)
            h = h_new
            if sd < tol:
                break
        imfs.append(h)
        residue -= h
    return np.array(imfs), residue

def _is_monotonic(x):
    return np.all(np.diff(x) >= 0) or np.all(np.diff(x) <= 0)

def _interpolate_envelope(signal, peaks, n):
    if len(peaks) < 2:
        return np.zeros(n)
    peaks = np.concatenate([[0], peaks, [n-1]])
    from scipy.interpolate import interp1d
    interp_func = interp1d(peaks, signal[peaks], kind='cubic', fill_value="extrapolate")
    return interp_func(np.arange(n))


def compute_spectrogram_resized(signal, sr, freq_bins=224, time_bins=16, nperseg=2048, noverlap=1024):
    try:
        f, t_spec, Z = sp_signal.stft(signal, fs=sr, nperseg=nperseg, noverlap=noverlap)
        S = np.abs(Z)
    except Exception:
        S = np.abs(np.fft.rfft(signal.reshape(1, -1)))
        S = S.reshape(-1, 1)
    if S.shape[0] != freq_bins:
        try:
            S_freq = sp_signal.resample(S, freq_bins, axis=0)
        except Exception:
            old_idx = np.linspace(0, 1, S.shape[0])
            new_idx = np.linspace(0, 1, freq_bins)
            S_freq = np.vstack([np.interp(new_idx, old_idx, S[:, j]) for j in range(S.shape[1])]).T
    else:
        S_freq = S
    if S_freq.shape[1] != time_bins:
        try:
            S_rt = sp_signal.resample(S_freq, time_bins, axis=1)
        except Exception:
            old_idx = np.linspace(0, 1, S_freq.shape[1])
            new_idx = np.linspace(0, 1, time_bins)
            S_rt = np.vstack([np.interp(new_idx, old_idx, S_freq[i, :]) for i in range(S_freq.shape[0])])
    else:
        S_rt = S_freq
    eps = 1e-10
    S_db = 20 * np.log10(np.abs(S_rt) + eps)
    S_db = np.nan_to_num(S_db, nan=-120.0, neginf=-120.0, posinf=120.0)
    vmin = float(np.nanmin(S_db))
    vmax = float(np.nanmax(S_db))
    if vmax - vmin == 0:
        S_norm = np.zeros_like(S_db)
    else:
        S_norm = (S_db - vmin) / (vmax - vmin)
    return S_norm


def process_file(file_name):
    input_dir = 'datasets/MMAUD/Data-M/combine_audio_npy/4/'
    output_dir = 'datasets/MMAUD/Data-M/combine_all_emd_spect_imf/4/'
    sr = 48000
    max_imfs = 8
    file_path = os.path.join(input_dir, file_name)
    try:
        data = np.load(file_path)
        if data.shape[0] == 4:
            channels = data
        elif data.shape[1] == 4:
            channels = data.T
        else:
            print(f"Skip {file_name}，shape error {data.shape}")
            return
        for ch_idx in range(4):
            sig = channels[ch_idx]
            imfs, _ = emd_decompose(sig, max_imfs=max_imfs)
            for imf_idx in range(min(max_imfs, imfs.shape[0])):
                S_norm = compute_spectrogram_resized(imfs[imf_idx], sr, freq_bins=224, time_bins=16)
                out_name = f'file_{file_name[:-4]}_channel_{ch_idx+1}_imf_{imf_idx+1}.npy'
                out_path = os.path.join(output_dir, out_name)
                np.save(out_path, S_norm)
    except Exception as e:
        print(f"Process {file_name} Fail: {e}")

if __name__ == "__main__":
    input_dir = 'datasets/MMAUD/Data-M/combine_audio_npy/4/'
    output_dir = 'datasets/MMAUD/Data-M/combine_all_emd_spect_imf/4/'
    os.makedirs(output_dir, exist_ok=True)
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    file_list.sort(key=lambda x: int(x.split('.')[0]))
    from tqdm import tqdm
    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_file, file_list), total=len(file_list), desc='', ncols=80):
            pass
    print("✅ Finished!")