# -*- coding: utf-8 -*-
import logging
import os
import os.path

import numpy as np
import soundfile
from numpy.lib.stride_tricks import as_strided
import random

from python_speech_features import delta
from python_speech_features import logfbank, fbank

logger = logging.getLogger(__name__)

noise_work, sr2 = soundfile.read('resources/noise_work.wav', dtype='float32')


def calc_feat_dim(window, max_freq):
    return int(0.001 * window * max_freq) + 1


def conv_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
    """
    Compute the spectrogram for a real signal.
    The parameters follow the naming convention of
    matplotlib.mlab.specgram
    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).
    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x
    Note:
        This is a truncating computation e.g. if fft_length=10,
        hop_length=5 and the signal has 23 elements, then the
        last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window ** 2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    # This function computes the one-dimensional n-point discrete Fourier Transform (DFT) of a real-valued array by means of an efficient algorithm called the Fast Fourier Transform (FFT).
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x) ** 2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs


def spectrogram_from_file(filename, step=10, window=20, max_freq=None,
                          eps=1e-14, overwrite=False, save_feature_as_csvfile=False,
                          noise_percent=0.4, speed_percent=0, seq_length=-1):
    """ Calculate the log of linear spectrogram from FFT energy
    Params:
        filename (str): Path to the audio file
        step (int): Step size in milliseconds between windows
        window (int): FFT window size in milliseconds
        max_freq (int): Only FFT bins corresponding to frequencies between
            [0, max_freq] are returned
        eps (float): Small value to ensure numerical stability (for ln(x))
    """

    # filename = filename.encode("utf-8")
    csvfilename = filename[:-3] + "csv"
    if (os.path.isfile(csvfilename) is False) or overwrite:
        with soundfile.SoundFile(filename) as sound_file:
            audio = sound_file.read(dtype='float32')
            if random.random() < noise_percent and seq_length > 0:
                audio_length = audio.shape[0]
                max_length_ratio = min(int((float(audio_length) / (seq_length - 100) / sound_file.samplerate) * 10000),
                                       120)
                min_length_ratio = max(
                    int(np.math.ceil((float(audio_length) / seq_length / sound_file.samplerate) * 10000)), 80)
                speed_rate = random.randint(min_length_ratio, max_length_ratio) / 100.
                new_length = int(audio_length / speed_rate)
                old_indices = np.arange(audio_length)
                new_indices = np.linspace(start=0, stop=audio_length, num=new_length)
                audio = np.interp(new_indices, old_indices, audio)
            if random.random() < noise_percent:
                if seq_length != -1:
                    max_length = seq_length * sound_file.samplerate / 100
                    if audio.shape[0] < max_length:
                        bg = np.zeros((max_length,))
                        rand_start = random.randint(0, max_length - audio.shape[0])
                        bg[rand_start:rand_start + audio.shape[0]] = audio
                        audio = bg
                start = random.randint(1, noise_work.shape[0] - audio.shape[0] - 1)
                audio = audio + random.randint(150, 250) / float(100.) * noise_work[start: audio.shape[0] + start]
            sample_rate = sound_file.samplerate
            if audio.ndim >= 2:
                audio = np.mean(audio, 1)
            if max_freq is None:
                max_freq = sample_rate / 2
            if max_freq > sample_rate / 2:
                raise ValueError("max_freq must not be greater than half of "
                                 " sample rate")
            if step > window:
                raise ValueError("step size must not be greater than window size")
            hop_length = int(0.001 * step * sample_rate)
            fft_length = int(0.001 * window * sample_rate)

            pxx, freqs = spectrogram(
                audio, fft_length=fft_length, sample_rate=sample_rate,
                hop_length=hop_length)

            ind = np.where(freqs <= max_freq)[0][-1] + 1
            res = np.transpose(np.log(pxx[:ind, :] + eps))
            if save_feature_as_csvfile:
                np.savetxt(csvfilename, res)
            return res
    else:
        return np.loadtxt(csvfilename)


def fbank_from_file(wav_path, step=10, window=20, max_freq=None,
                    eps=1e-14, overwrite=False, save_feature_as_csvfile=False,
                    noise_percent=0.4, speed_percent=0, seq_length=-1):
    sig1, sr1 = soundfile.read(wav_path, dtype='float32')
    fbank_feat, energy = fbank(sig1, sr1, nfilt=40)  # (407, 40)
    fbank_feat = np.column_stack((np.log(energy), np.log(fbank_feat)))  # (407, 41)
    d_fbank_feat = delta(fbank_feat, 2)
    dd_fbank_feat = delta(d_fbank_feat, 2)
    concat_fbank_feat = np.array([fbank_feat, d_fbank_feat, dd_fbank_feat])  # (3, 407, 41)
    return concat_fbank_feat
