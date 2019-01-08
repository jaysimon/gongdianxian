#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
this class is used for acustic feature extract
author: hou wei
2018.08.01
"""

import soundfile
import numpy as np
import librosa
import scipy
import h5py

def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    # if audio.ndim > 1:
    #    audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio[:, 0], orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


def get_window_function(n, window_type="hamming_asymmetric"):
        # Windowing function
    if window_type == 'hamming_asymmetric':
        return scipy.signal.hamming(n, sym=False)

    elif window_type == 'hamming_symmetric' or window_type == 'hamming':
        return scipy.signal.hamming(n, sym=True)

    elif window_type == 'hann_asymmetric':
        return scipy.signal.hann(n, sym=False)

    elif window_type == 'hann_symmetric' or window_type == 'hann':
        return scipy.signal.hann(n, sym=True)


def get_spectrogram(
        y, n_fft=2048, fs=16000, center=True, win_length_samples=None,
        hop_length_samples=None, spectrogram_type="magnitude", window=None):

    eps = np.spacing(1)

    if spectrogram_type == "magnitude":
        return np.abs(
            librosa.stft(
                y + eps, n_fft=n_fft, win_length=win_length_samples,
                hop_length=hop_length_samples, center=center, window=window))
    if spectrogram_type == "power":
        return np.abs(
                librosa.stft(
                        y + eps, n_fft=n_fft,
                        win_length=win_length_samples,
                        hop_length=hop_length_samples, center=center,
                        window=window)) ** 2


def get_mel(
        y,  n_fft=2048, fs=16000, center=True, win_length_seconds=0.04,
        hop_length_seconds=0.02, spectrogram_type="magnitude", n_mels=40,
        fmin=0, fmax=16000, htk=False, log=False):

    if(len(y.shape) > 1):
        y = np.mean(y, axis=1)
    eps = np.spacing(1)
    win_length_samples = int(fs * win_length_seconds)
    hop_length_samples = int(fs * hop_length_seconds)
    window = get_window_function(win_length_samples)

    spectrogram = get_spectrogram(
        y=y,
        n_fft=n_fft,
        win_length_samples=win_length_samples,
        hop_length_samples=hop_length_samples,
        spectrogram_type=spectrogram_type,
        center=True,
        window=window
    )

    mel_basis = librosa.filters.mel(
        sr=fs,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        htk=htk
    )
    mel_spectrum = np.dot(mel_basis, spectrogram)

    if log:
        mel_spectrum = np.log(mel_spectrum + eps)

    return mel_spectrum


if __name__ == "__main__":
    """
    path = "./airport-barcelona-1-54-a.wav"
    #path = "./babble.wav"
    audio , fs = read_audio(path)
    result = get_mel(
                audio, spectrogram_type = "magnitude",
                log=True, fs = 48000, n_mels = 40)
    print(result)
    """
    f_path = "/home/houwei/02-dataset/03-fowl/data/ori_fowl/jLDUJyv0qxQ_70.000_80.000.wav"
    speech_audio, fs = read_audio(f_path)

    soundfile.write("./tmp.wav", get_mel(speech_audio), 16000)
